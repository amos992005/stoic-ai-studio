#!/usr/bin/env python3
"""
stoic-ai-studio :: PHASE 1 — Micro-Pilot 1
"THE CHARACTER FORGE" — Golden reference + face-locked scene variations.

Runs on a Kaggle kernel (Dual Tesla T4, 16 GB each, internet enabled).
Everything is written to the kernel working directory so that
`kaggle kernels output` drops the artifacts straight into `outputs/` of the
GitHub repository.

Artifacts
---------
01_master_face.png   1024x1024 golden reference portrait (seed 42)
02_library_scene.png 768x1344 face-locked variation
03_rainy_street.png  768x1344 face-locked variation
04_podcast_studio.png768x1344 face-locked variation
contact_sheet.png    labelled inspection grid of all four
metrics.json         timings, seeds, prompts, environment fingerprint
build.log            (mirrored stdout, so failures are debuggable from the repo)

Design notes
------------
* Defensive by default: every stage is timed, logged and wrapped in clear
  exceptions. Non-fatal degradations (e.g. IP-Adapter unavailable) are flagged
  in metrics.json instead of producing a silently-wrong face.
* Deterministic: fixed seeds, no random shuffling, no torch nondeterminism that
  matters for a still image.
* VRAM aware (T4 / 16 GB): fp16 weights, sliced attention, VAE slicing/tiling,
  sequential scene generation with explicit `torch.cuda.empty_cache()`.
"""

from __future__ import annotations

import gc
import json
import os
import platform
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

#: Photoreal SDXL checkpoint for cinematic portraits.
MODEL_ID = os.environ.get("SDXL_MODEL_ID", "SG161222/RealVisXL_V5.0")

#: CLIP vision tower required by the Face ID IP-Adapter.
CLIP_VISION_ID = os.environ.get(
    "CLIP_VISION_ID", "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
)

#: IP-Adapter repository / weights ("Face ID Plus" — designed for SDXL portraits).
IP_ADAPTER_REPO = os.environ.get("IP_ADAPTER_REPO", "h94/IP-Adapter")
IP_ADAPTER_SUBFOLDER = "sdxl_models"
IP_ADAPTER_WEIGHT = "ip-adapter-plus-face_sdxl_vit-h.safetensors"
IP_ADAPTER_SCALE = 0.65

#: Golden reference.
BASE_SEED = int(os.environ.get("BASE_SEED", "42"))
MASTER_STEPS = 30
MASTER_CFG = 5.5
MASTER_SIZE = (1024, 1024)

#: Face-locked variations (9:16 — Reels / TikTok, still fast on a T4).
SCENE_STEPS = 30
SCENE_CFG = 5.5
SCENE_SIZE = (768, 1344)

#: If the IP-Adapter cannot be loaded, fail loudly by default. Set
#: ALLOW_IP_ADAPTER_FALLBACK=1 to still produce a (non face-locked) contact sheet.
ALLOW_IP_ADAPTER_FALLBACK = os.environ.get("ALLOW_IP_ADAPTER_FALLBACK", "0") in {
    "1",
    "true",
    "True",
    "yes",
}

MASTER_FILENAME = "01_master_face.png"
CONTACT_SHEET_FILENAME = "contact_sheet.png"
METRICS_FILENAME = "metrics.json"

CHARACTER = (
    "Julian, a 42-year-old charismatic male mindset mentor, sharp jawline, "
    "short salt-and-pepper beard, intense calm eyes, wearing a dark turtleneck"
)

#: Shared look, kept identical across every panel so only the *scene* changes.
STYLE_SUFFIX = (
    "moody chiaroscuro cinematic lighting, dramatic rim light, deep shadows, "
    "hyper-realistic skin texture with visible pores, ultra detailed, 85mm lens, "
    "shallow depth of field, shot on ARRI Alexa, editorial photography, "
    "rich filmic color grade, photorealistic, 8k"
)

NEGATIVE_PROMPT = (
    "cartoon, anime, illustration, painting, 3d render, cgi, doll, plastic skin, "
    "airbrushed, waxy, oversaturated, blurry, low quality, jpeg artifacts, "
    "deformed, disfigured, mutated hands, extra fingers, extra limbs, "
    "bad anatomy, bad proportions, cross-eyed, asymmetric eyes, "
    "watermark, signature, text, logo, frame, border, "
    "young, teenager, child, female, androgynous"
)

MASTER_PROMPT = (
    f"Close-up studio portrait of {CHARACTER}. "
    "Front-facing, looking straight into the camera, neutral confident expression, "
    "clean dark charcoal background, dramatic rim lighting separating him from the "
    "background, subtle key light from camera left, head and shoulders, centered "
    f"composition, {STYLE_SUFFIX}"
)

SCENES: List[Dict[str, Any]] = [
    {
        "filename": "02_library_scene.png",
        "title": "02 · LIBRARY",
        "prompt": (
            f"Medium shot of {CHARACTER}, sitting at a dark mahogany desk in an "
            "ancient private library, towering bookshelves filled with leather-bound "
            "volumes fading into shadow, warm candlelight flickering on his face, "
            "looking directly at the camera, glass of whiskey and an open book on the "
            f"desk, vertical composition, {STYLE_SUFFIX}"
        ),
    },
    {
        "filename": "03_rainy_street.png",
        "title": "03 · RAINY STREET",
        "prompt": (
            f"Full-body-leaning medium shot of {CHARACTER}, wearing a dark wool "
            "trench coat, standing in a moody rain-slicked city street at night, "
            "neon signs and wet asphalt reflections in teal and amber, fine rain "
            "droplets catching the light, looking directly at the camera, "
            f"cinematic bokeh streetlights behind him, vertical composition, {STYLE_SUFFIX}"
        ),
    },
    {
        "filename": "04_podcast_studio.png",
        "title": "04 · PODCAST STUDIO",
        "prompt": (
            f"Medium shot of {CHARACTER}, seated in a high-end dark minimalist "
            "podcast studio, broadcast condenser microphone with shock mount softly "
            "out of focus in the foreground, acoustic foam panels and warm practical "
            "lights in the background, looking directly at the camera, cinematic "
            f"lighting, vertical composition, {STYLE_SUFFIX}"
        ),
    },
]


# --------------------------------------------------------------------------- #
# Logging / utilities
# --------------------------------------------------------------------------- #


def log(message: str, level: str = "INFO") -> None:
    """Timestamped, flush-on-write logging that survives a crashed kernel."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{stamp}] [{level:<5}] {message}", flush=True)


@contextmanager
def timed(label: str, timings: Dict[str, float]):
    """Record wall-clock duration of a block in `timings` (seconds)."""
    log(f"▶ start: {label}")
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        timings[label] = round(elapsed, 3)
        log(f"✔ done:  {label} ({elapsed:.1f}s)")


def resolve_output_dir() -> Path:
    """
    Where artifacts are written.

    On Kaggle we deliberately write to the *working directory root*
    (`/kaggle/working`) so that `kaggle kernels output ... -p ./outputs` in CI
    lands the files at `outputs/<file>` rather than `outputs/outputs/<file>`.
    Locally we use `./outputs`.
    """
    explicit = os.environ.get("OUTPUT_DIR")
    if explicit:
        out = Path(explicit).expanduser().resolve()
    elif Path("/kaggle/working").is_dir():
        out = Path("/kaggle/working")
    else:
        out = Path.cwd() / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out


class Tee:
    """Mirror everything printed to stdout into the kernel's build log file."""

    def __init__(self, path: Path, stream):
        self._stream = stream
        self._file = open(path, "a", encoding="utf-8", buffering=1)

    def write(self, data: str) -> int:
        self._stream.write(data)
        try:
            self._file.write(data)
        except Exception:  # never let logging break generation
            pass
        return len(data)

    def flush(self) -> None:
        for target in (self._stream, self._file):
            try:
                target.flush()
            except Exception:
                pass

    def isatty(self) -> bool:  # pragma: no cover - cosmetic
        return False

    def close(self) -> None:
        self.flush()
        try:
            self._file.close()
        except Exception:
            pass


def env_fingerprint() -> Dict[str, Any]:
    """Versions + GPU description for metrics.json."""
    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu_names"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
            info["gpu_total_vram_gb"] = [
                round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 2)
                for i in range(torch.cuda.device_count())
            ]
    except Exception as exc:  # pragma: no cover
        info["torch_error"] = f"{type(exc).__name__}: {exc}"
    for mod in ("diffusers", "transformers", "accelerate", "safetensors", "PIL"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = "unavailable"
    return info


# --------------------------------------------------------------------------- #
# Dependency guard
# --------------------------------------------------------------------------- #


def ensure_dependencies(timings: Dict[str, float]) -> None:
    """
    Kaggle's image usually ships torch/diffusers/transformers, but the diffusers
    version drifts and IP-Adapter needs >= 0.24. Upgrade in place (silently) when
    the installed build is too old, and fail with an actionable message otherwise.
    """
    import subprocess

    def _has_ip_adapter() -> bool:
        try:
            from diffusers import StableDiffusionXLPipeline

            return hasattr(StableDiffusionXLPipeline, "load_ip_adapter") and hasattr(
                StableDiffusionXLPipeline, "set_ip_adapter_scale"
            )
        except Exception:
            return False

    try:
        import diffusers

        log(f"diffusers {diffusers.__version__} detected")
    except Exception:
        log("diffusers is not installed", "WARN")

    if _has_ip_adapter():
        return

    log("installed diffusers lacks IP-Adapter support — upgrading", "WARN")
    with timed("pip_install_diffusers", timings):
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "--no-input",
             "--disable-pip-version-check", "diffusers>=0.27.0"],
            capture_output=True,
            text=True,
        )
        if proc.stdout.strip():
            log(proc.stdout.strip()[-2000:])
        if proc.returncode != 0:
            log(proc.stderr.strip()[-2000:], "ERROR")
    import importlib

    import diffusers as diffusers_module

    importlib.reload(diffusers_module)
    if not _has_ip_adapter():
        raise RuntimeError(
            "diffusers does not expose load_ip_adapter/set_ip_adapter_scale even "
            "after upgrading. Pin `diffusers>=0.27.0` and re-run the kernel."
        )
    log(f"diffusers upgraded to {diffusers_module.__version__}")


# --------------------------------------------------------------------------- #
# Pipeline construction
# --------------------------------------------------------------------------- #


def build_pipeline(timings: Dict[str, float]):
    """Load RealVisXL V5.0 in fp16 on cuda:0 with a Karras DPM-Solver++ scheduler."""
    import torch
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This kernel must run with GPU enabled "
            "(Kaggle: Settings → Accelerator → GPU T4 x2)."
        )

    device = "cuda:0"
    log(f"device={device} · {torch.cuda.get_device_name(0)}")

    # `variant="fp16"` downloads half the bytes; fall back for repos without it.
    load_kwargs: Dict[str, Any] = {
        "torch_dtype": torch.float16,
        "use_safetensors": True,
    }
    with timed("load_sdxl_pipeline", timings):
        try:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                MODEL_ID, variant="fp16", **load_kwargs
            )
        except Exception as exc:
            log(f"fp16 variant unavailable ({type(exc).__name__}: {exc}); retrying", "WARN")
            pipe = StableDiffusionXLPipeline.from_pretrained(MODEL_ID, **load_kwargs)

    with timed("configure_scheduler_and_vram", timings):
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas=True,
            algorithm_type="dpmsolver++",
        )
        pipe.set_progress_bar_config(disable=False, desc="diffusion")
        pipe = pipe.to(device, dtype=torch.float16)
        # T4-friendly memory hygiene.
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
        try:
            pipe.enable_attention_slicing("max")
        except Exception as exc:  # pragma: no cover
            log(f"attention slicing unavailable: {exc}", "WARN")
        try:  # newer diffusers only: never silently crop to a training bucket
            pipe.set_use_resolution_binning(False)
            log("resolution binning disabled (exact 768x1344 output)")
        except Exception:
            pass

    return pipe, device


def attach_face_adapter(pipe, device: str, timings: Dict[str, float]) -> bool:
    """
    Load the CLIP image encoder + IP-Adapter Face weights and set scale 0.65.

    Returns True on success. Raises unless ALLOW_IP_ADAPTER_FALLBACK is set.
    """
    import torch
    from transformers import CLIPVisionModelWithProjection

    log(f"attaching IP-Adapter Face ({IP_ADAPTER_WEIGHT}) @ scale {IP_ADAPTER_SCALE}")
    try:
        with timed("load_clip_vision", timings):
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                CLIP_VISION_ID,
                torch_dtype=torch.float16,
                use_safetensors=True,
            ).to(device)
            image_encoder.eval()
            pipe.image_encoder = image_encoder

        with timed("load_ip_adapter", timings):
            pipe.load_ip_adapter(
                IP_ADAPTER_REPO,
                subfolder=IP_ADAPTER_SUBFOLDER,
                weight_name=IP_ADAPTER_WEIGHT,
            )
            pipe.set_ip_adapter_scale(IP_ADAPTER_SCALE)
        log("IP-Adapter Face ready — character consistency is locked.")
        return True
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        log(f"IP-Adapter setup FAILED — {detail}", "ERROR")
        log(traceback.format_exc(), "ERROR")
        if not ALLOW_IP_ADAPTER_FALLBACK:
            raise RuntimeError(
                "Face-lock adapter is mandatory for this pilot (set "
                "ALLOW_IP_ADAPTER_FALLBACK=1 to degrade gracefully)."
            ) from exc
        log(
            "ALLOW_IP_ADAPTER_FALLBACK=1 → continuing WITHOUT face lock; "
            "identity will drift between panels.",
            "WARN",
        )
        try:
            pipe.set_ip_adapter_scale(0.0)
        except Exception:
            pass
        return False
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #


def _generate(
    pipe,
    device: str,
    *,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    seed: int,
    steps: int,
    cfg: float,
    ip_adapter_image: Optional[Any] = None,
    label: str = "image",
):
    """Run one diffusion pass with a private generator (fully deterministic)."""
    import torch

    generator = torch.Generator(device=device).manual_seed(seed)
    kwargs: Dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "num_inference_steps": steps,
        "guidance_scale": cfg,
        "generator": generator,
    }
    if ip_adapter_image is not None:
        kwargs["ip_adapter_image"] = ip_adapter_image

    log(f"sampling {label}: {width}x{height} · seed={seed} · steps={steps} · cfg={cfg}")
    with torch.inference_mode():
        result = pipe(**kwargs)

    if not result or not getattr(result, "images", None):
        raise RuntimeError(f"Pipeline returned no images for {label}")
    image = result.images[0]
    if image.size != (width, height):
        log(f"{label}: unexpected size {image.size}, expected {(width, height)}", "WARN")
    return image


def save_image(image, path: Path) -> None:
    """Save a PNG atomically and keep the repo light (contact sheet stays small)."""
    tmp = path.with_suffix(path.suffix + ".part")
    image.save(tmp, format="PNG", optimize=True)
    tmp.replace(path)
    size_kb = path.stat().st_size / 1024
    log(f"saved {path.name} ({size_kb:.0f} KB)")


# --------------------------------------------------------------------------- #
# Contact sheet
# --------------------------------------------------------------------------- #


def build_contact_sheet(
    master_path: Path,
    scene_paths: List[Path],
    titles: List[str],
    out_path: Path,
    timings: Dict[str, float],
) -> None:
    """
    Assemble the golden reference (resized to the panel height) plus the three
    vertical scenes into one labelled inspection grid.
    """
    from PIL import Image, ImageDraw, ImageFont

    with timed("build_contact_sheet", timings):
        master = Image.open(master_path)
        scenes = [Image.open(p) for p in scene_paths]
        images = [master, *scenes]
        if any(img.mode != "RGB" for img in images):
            images = [img.convert("RGB") for img in images]

        gutter = 18
        margin = 28
        header_height = 86
        strip_height = 54
        panel_h = MASTER_SIZE[1]  # 1024 — everything is scaled to this height

        panels = []
        for img in images:
            if img.height != panel_h:
                new_w = max(1, round(img.width * panel_h / img.height))
                img = img.resize((new_w, panel_h), Image.LANCZOS)
            panels.append(img)

        total_w = margin * 2 + sum(p.width for p in panels) + gutter * (len(panels) - 1)
        canvas_h = margin * 2 + header_height + strip_height + panel_h
        canvas = Image.new("RGB", (total_w, canvas_h), (12, 12, 14))
        draw = ImageDraw.Draw(canvas)

        def load_font(size: int, bold: bool = False):
            for name in (
                "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
                "LiberationSans-Bold.ttf" if bold else "LiberationSans-Regular.ttf",
                "Arial.ttf",
            ):
                try:
                    return ImageFont.truetype(name, size)
                except Exception:
                    continue
            try:  # Pillow >= 9/10 ships a default bitmap font
                return ImageFont.load_default(size=size)
            except Exception:
                return ImageFont.load_default()

        label_font = load_font(24, bold=True)
        header_font = load_font(30, bold=True)
        sub_font = load_font(19)

        header = "JULIAN · THE CHARACTER FORGE"
        subtitle = "PHASE 1 · MICRO-PILOT 1 · golden reference + IP-Adapter Face-locked variations"
        draw.text((margin, margin + 2), header, fill=(240, 240, 246), font=header_font)
        draw.text((margin, margin + 46), subtitle, fill=(150, 152, 162), font=sub_font)

        def _fit_label(text: str, width: int, start_size: int):
            """Shrink the label font until it fits inside the label plate."""
            for size in range(start_size, 11, -1):
                candidate = load_font(size, bold=True)
                if draw.textlength(text, font=candidate) <= width - 24:
                    return candidate
            return load_font(12, bold=True)

        x = margin
        for panel, title in zip(panels, titles):
            y = margin + header_height + strip_height
            canvas.paste(panel, (x, y))
            draw.rectangle(
                [x, y, x + panel.width - 1, y + panel.height - 1],
                outline=(70, 70, 78),
                width=2,
            )
            # Label plate above the panel: dark bar, accent rule, fitted text.
            plate_top = y - strip_height + 8
            plate_bottom = y - 8
            draw.rectangle([x, plate_top, x + panel.width - 1, plate_bottom], fill=(24, 24, 30))
            draw.rectangle([x, plate_top, x + panel.width - 1, plate_top + 3], fill=(196, 106, 62))
            label = f"{title}  ·  {panel.width}×{panel.height}"
            fitted = _fit_label(label, panel.width, 24)
            text_h = draw.textbbox((0, 0), label, font=fitted)[3]
            draw.text(
                (x + 12, plate_top + (plate_bottom - plate_top - text_h) // 2),
                label,
                fill=(242, 242, 248),
                font=fitted,
            )
            x += panel.width + gutter

        canvas.save(out_path, format="PNG", optimize=True)
        log(f"saved {out_path.name} ({canvas.width}x{canvas.height}, "
            f"{out_path.stat().st_size / 1024:.0f} KB)")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    t_start = time.perf_counter()
    output_dir = resolve_output_dir()
    log_path = output_dir / "build.log"
    tee = Tee(log_path, sys.stdout)
    sys.stdout = tee  # type: ignore[assignment]

    metrics: Dict[str, Any] = {
        "phase": "phase-1-micro-pilot-1",
        "pilot": "the-character-forge",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "config": {
            "model_id": MODEL_ID,
            "clip_vision_id": CLIP_VISION_ID,
            "ip_adapter_repo": IP_ADAPTER_REPO,
            "ip_adapter_weight": IP_ADAPTER_WEIGHT,
            "ip_adapter_scale": IP_ADAPTER_SCALE,
            "allow_ip_adapter_fallback": ALLOW_IP_ADAPTER_FALLBACK,
            "base_seed": BASE_SEED,
            "master": {
                "size": list(MASTER_SIZE),
                "steps": MASTER_STEPS,
                "cfg": MASTER_CFG,
                "seed": BASE_SEED,
            },
            "scenes": {"size": list(SCENE_SIZE), "steps": SCENE_STEPS, "cfg": SCENE_CFG},
            "dtype": "float16",
            "device": "cuda:0",
        },
        "artifacts": [],
        "timings_seconds": {},
        "errors": [],
    }
    timings: Dict[str, float] = metrics["timings_seconds"]

    log("=" * 78)
    log("stoic-ai-studio · THE CHARACTER FORGE · Micro-Pilot 1")
    log(f"output dir: {output_dir}")
    log("=" * 78)

    try:
        ensure_dependencies(timings)
        metrics["environment"] = env_fingerprint()
        log(f"environment: {json.dumps(metrics['environment'], default=str)}")

        # ---------------- Step A: golden reference ---------------- #
        pipe, device = build_pipeline(timings)

        master_path = output_dir / MASTER_FILENAME
        with timed("generate_master_face", timings):
            master = _generate(
                pipe,
                device,
                prompt=MASTER_PROMPT,
                negative_prompt=NEGATIVE_PROMPT,
                width=MASTER_SIZE[0],
                height=MASTER_SIZE[1],
                seed=BASE_SEED,
                steps=MASTER_STEPS,
                cfg=MASTER_CFG,
                label="01_master_face",
            )
            save_image(master, master_path)

        # ---------------- Step B: face-locked variations ---------------- #
        face_locked = attach_face_adapter(pipe, device, timings)
        metrics["ip_adapter_active"] = face_locked

        scene_paths: List[Path] = []
        scene_metrics: List[Dict[str, Any]] = []
        for index, scene in enumerate(SCENES):
            seed = BASE_SEED + index + 1
            path = output_dir / scene["filename"]
            key = f"generate_{scene['filename'].removesuffix('.png')}"
            try:
                with timed(key, timings):
                    image = _generate(
                        pipe,
                        device,
                        prompt=scene["prompt"],
                        negative_prompt=NEGATIVE_PROMPT,
                        width=SCENE_SIZE[0],
                        height=SCENE_SIZE[1],
                        seed=seed,
                        steps=SCENE_STEPS,
                        cfg=SCENE_CFG,
                        ip_adapter_image=master if face_locked else None,
                        label=scene["filename"],
                    )
                    save_image(image, path)
                scene_paths.append(path)
                scene_metrics.append(
                    {"filename": scene["filename"], "seed": seed, "status": "ok"}
                )
            except Exception as exc:
                # One bad scene must not destroy the whole pilot.
                msg = f"{scene['filename']}: {type(exc).__name__}: {exc}"
                log(msg, "ERROR")
                log(traceback.format_exc(), "ERROR")
                metrics["errors"].append(msg)
                scene_metrics.append(
                    {"filename": scene["filename"], "seed": seed, "status": "error"}
                )
            finally:
                gc.collect()
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:
                    pass

        metrics["scenes"] = scene_metrics
        if len(scene_paths) != len(SCENES):
            raise RuntimeError(
                f"only {len(scene_paths)}/{len(SCENES)} scene variations succeeded"
            )

        # ---------------- Step C: contact sheet ---------------- #
        contact_path = output_dir / CONTACT_SHEET_FILENAME
        build_contact_sheet(
            master_path,
            scene_paths,
            ["01 · MASTER FACE", *[s["title"] for s in SCENES]],
            contact_path,
            timings,
        )

        from PIL import Image as _Image

        artifacts = []
        for path in [master_path, *scene_paths, contact_path]:
            if not path.exists():
                continue
            with _Image.open(path) as img:
                width, height = img.size
            artifacts.append(
                {
                    "filename": path.name,
                    "bytes": path.stat().st_size,
                    "width": width,
                    "height": height,
                }
            )
        metrics["artifacts"] = artifacts
        metrics["face_locked"] = bool(metrics.get("ip_adapter_active"))
        metrics["status"] = "success"
        log("PHASE 1 MICRO-PILOT 1 COMPLETE ✔")

    except Exception as exc:
        metrics["status"] = "error"
        metrics["errors"].append(f"{type(exc).__name__}: {exc}")
        log(f"FATAL: {type(exc).__name__}: {exc}", "ERROR")
        log(traceback.format_exc(), "ERROR")
    finally:
        metrics["finished_at"] = datetime.now(timezone.utc).isoformat()
        timings["total"] = round(time.perf_counter() - t_start, 3)
        log(f"total wall clock: {timings['total']}s")
        try:
            (output_dir / METRICS_FILENAME).write_text(
                json.dumps(metrics, indent=2, default=str), encoding="utf-8"
            )
        except Exception as exc:  # pragma: no cover
            log(f"could not write {METRICS_FILENAME}: {exc}", "ERROR")
        tee.close()
        sys.stdout = tee._stream  # type: ignore[attr-defined]

    return 0 if metrics.get("status") == "success" else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log("interrupted by user", "WARN")
        sys.exit(130)
