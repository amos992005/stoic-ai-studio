#!/usr/bin/env python3
"""
Luca Moretti — Process 1 of 2: identity-locked image carousel on cuda:0.

Architecture (deliberately different from the previous phases):

  * **InstantID** for identity, not an inswapper face swap. InstantID conditions
    the UNet directly (IP-Adapter cross-attention + IdentityNet ControlNet), so
    there is no post-hoc face transplant and therefore no seam, no second face
    model, and no img2img "adetailer" pass afterwards.
  * **Compel** for prompt embeddings so the 77-token CLIP window is not a
    constraint. Prompts are chunked and padded, with truncation disabled.
  * **antelopev2** (InstantID's own 512-d face encoder) via onnxruntime with
    CUDAExecutionProvider when the runtime offers one.

Hard prohibitions honoured here: no `pipe.to("cpu")`, no inswapper_128, no
SVD-XT, no SDXL img2img Adetailer after face generation.

Run: python generate_images.py
"""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/kaggle/working"))
if not OUTPUT_DIR.is_dir():
    OUTPUT_DIR = REPO_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BUILD_LOG = OUTPUT_DIR / "build.log"
METRICS_PATH = OUTPUT_DIR / "luca_metrics.json"

DEVICE = "cuda:0"
BASE_MODEL = "SG161222/RealVisXL_V5.0"
INSTANTID_REPO = "InstantX/InstantID"
INSTANTID_ADAPTER = "ip-adapter.bin"
INSTANTID_CONTROLNET = "ControlNetModel"
FACE_ANALYSIS_ROOT = Path(os.environ.get("FACE_MODEL_ROOT", "/kaggle/working/face_models"))

MASTER_FILENAME = "01_luca_master.png"
SCENE_FILES = {
    "yacht": "02_luca_yacht.png",
    "porsche": "03_luca_porsche.png",
    "party": "04_luca_party.png",
}

MASTER_SIZE = (1024, 1024)
SCENE_SIZE = (832, 1024)

BASE_SEED = int(os.environ.get("BASE_SEED", "42"))
MASTER_STEPS = 30
MASTER_CFG = 3.8
SCENE_STEPS = 30
SCENE_CFG = 4.0

#: The identity conditioning strengths the spec fixes for the scenes.
SCENE_IP_ADAPTER_SCALE = 0.80
SCENE_CONTROLNET_SCALE = 0.55

#: Prompt fragments. Fitted inside 77 tokens, but Compel is configured with
#: truncation disabled and chunking enabled, so longer prompts still work.
FACE_SPEC = (
    "27-year-old man, young Brad Pitt, dirty blonde tousled sun-kissed hair, "
    "piercing blue-green eyes, sharp jawline, light 3-day stubble, confident smirk"
)
LENS_SPEC = (
    "24mm iPhone 16 Pro candid photo, dirty blonde tousled hair, light 3-day "
    "stubble, natural sunlight, visible skin pores, slight film grain"
)
NEGATIVE_PROMPT = (
    "sunglasses, shades, eyeglasses, hat, mutated hands, extra fingers, fused "
    "fingers, deformed hands, bad anatomy, disfigured, plastic skin, airbrushed, "
    "CGI, 3d render, cartoon, illustration, painting, low quality, blurry, "
    "watermark, text, logo, twins, duplicate person, cloned face, child"
)


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #


class Tee:
    """Mirror stdout into a shared build log (both processes append)."""

    def __init__(self, path: Path, stream):
        self._stream = stream
        self._file = open(path, "a", encoding="utf-8", buffering=1)
        self.lock_path = path.with_suffix(".lock")

    def write(self, data: str) -> int:
        try:
            self._stream.write(data)
        except Exception:
            pass
        try:
            import fcntl

            with open(self.lock_path, "a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                self._file.write(data)
                fcntl.flock(lock, fcntl.LOCK_UN)
        except Exception:
            try:
                self._file.write(data)
            except Exception:
                pass
        return len(data)

    def flush(self) -> None:
        for target in (self._stream, self._file):
            try:
                target.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        return False

    def close(self) -> None:
        try:
            self._file.close()
        except Exception:
            pass


def ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(message: str, level: str = "INFO", tag: str = "img") -> None:
    prefix = f"[{ts()}] [{level:<5}] [{tag}]"
    print(f"{prefix} {message}", flush=True)


class timed:
    """Context manager that records wall-clock time per stage."""

    def __init__(self, label: str, timings: Dict[str, float], tag: str = "img"):
        self.label, self.timings, self.tag = label, timings, tag

    def __enter__(self):
        log(f"▶ start: {self.label}", tag=self.tag)
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        elapsed = time.perf_counter() - self.start
        key = f"{self.tag}/{self.label}"
        self.timings[key] = round(elapsed, 3)
        log(f"✔ done:  {self.label} ({elapsed:.1f}s)", tag=self.tag)
        return False


# --------------------------------------------------------------------------- #
# Dependencies
# --------------------------------------------------------------------------- #

#: Wheels only, and pinned where a floating version would break a run.
#:
#:  * insightface 2.1 is the only release with a pure-python wheel; 0.7.3 and
#:    everything older build a Cython .pyx, which fails on Kaggle (no compiler).
#:    It declares `opencv-python` as a hard dependency, and that GUI build
#:    imports libGL — absent on Kaggle — so insightface is installed with
#:    --no-deps and opencv-python-headless is installed afterwards to satisfy
#:    `import cv2`. Installing insightface normally also lets pip downgrade
#:    numpy out from under torch and diffusers.
#:  * onnxruntime-gpu 1.32.x is built against CUDA 13, but a Kaggle T4 runtime
#:    ships the CUDA 12 stack; importing it then fails at
#:    `onnxruntime.capi._pybind_state`. 1.22.0 is the newest release with a
#:    cp313 wheel built against CUDA 12.
#:  * onnx has no cp313 wheel of its own, only cp312-abi3 — which is what
#:    makes it importable on Kaggle's 3.13. Pinned so the resolver does not
#:    wander onto a source build.
IMAGE_DEPS = [
    "compel>=2.0.2",
    "onnx==1.23.2",              # cp312-abi3 wheel: installable on 3.13
    "onnxruntime-gpu==1.22.0",   # CUDA 12 build; 1.32 needs CUDA 13
    "opencv-python-headless",
    "einops",
]
#: insightface's own runtime deps, minus the GUI opencv it would otherwise drag in.
INSIGHTFACE_DEPS = ["onnx>=1.13", "scikit-image", "requests", "scipy", "tqdm"]

#: module name -> the spec that provides it
IMAGE_PROBE = {
    "compel": "compel>=2.0.2",
    "onnx": "onnx==1.23.2",
    "onnxruntime": "onnxruntime-gpu==1.22.0",
    "cv2": "opencv-python-headless",
    "einops": "einops",
}


def _pip(*args: str) -> int:
    proc = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                           "--no-cache-dir", *args], check=False,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        log(f"pip install {' '.join(args)} failed: "
            f"{(proc.stderr or proc.stdout)[-600:]}", "ERROR")
    return proc.returncode


def ensure_image_deps(timings: Dict[str, float]) -> None:
    """
    Install only what is missing, in an order that cannot break the runtime.

    Ordering is load-bearing: insightface --no-deps, then the headless opencv,
    because insightface as a wheel would otherwise install the libGL-bound
    opencv-python and shadow the headless build.
    """
    import importlib

    with timed("pip_install_deps", timings):
        missing = []
        for module, spec in IMAGE_PROBE.items():
            try:
                importlib.import_module(module)
            except Exception as exc:
                log(f"{module}: {type(exc).__name__}", "WARN")
                missing.append(spec)

        try:
            importlib.import_module("insightface")
        except Exception as exc:
            log(f"insightface: {type(exc).__name__}", "WARN")
            missing.append("insightface==2.1")

        if not missing:
            log("all image dependencies already present")
            return

        log(f"installing: {missing}")
        plain = [m for m in missing if not m.startswith("insightface")]
        if plain and _pip(*plain) != 0:
            raise RuntimeError("image dependency install failed — see above")

        if any(m.startswith("insightface") for m in missing):
            if _pip("--no-deps", "insightface==2.1") != 0:
                raise RuntimeError("insightface install failed — see above")
            # its wheel dependencies, minus the opencv variant we cannot import
            _pip(*INSIGHTFACE_DEPS)
            _pip("--force-reinstall", "--no-deps", "opencv-python-headless")

        for module in ("cv2", "insightface", "onnxruntime", "compel"):
            importlib.import_module(module)
        log("image dependencies import cleanly")


def face_providers() -> List[str]:
    """
    CUDA when the runtime actually exposes it, else CPU.

    Kaggle's onnxruntime build reports providers with a "-gpu" suffix, so an
    exact-membership test silently falls through to the CPU.
    """
    try:
        import onnxruntime as ort

        available = list(ort.get_available_providers())
    except Exception as exc:
        log(f"onnxruntime unavailable: {exc}", "WARN")
        return ["CPUExecutionProvider"]
    wanted = [p for p in available if "CUDA" in p][:1] + ["CPUExecutionProvider"]
    log(f"onnxruntime providers available={available} → using={wanted}")
    return wanted


# --------------------------------------------------------------------------- #
# Identity
# --------------------------------------------------------------------------- #


def load_face_app():
    """antelopev2 — InstantID's 512-d face encoder, prepared at 640px."""
    from insightface.app import FaceAnalysis

    root = FACE_ANALYSIS_ROOT
    root.mkdir(parents=True, exist_ok=True)
    app = FaceAnalysis(name="antelopev2", root=str(root),
                       providers=face_providers())
    app.prepare(ctx_id=0, det_size=(640, 640))
    log("antelopev2 ready")
    return app


def detect_faces(app, image_rgb) -> List[Any]:
    import cv2
    import numpy as np

    if not isinstance(image_rgb, np.ndarray):
        image_rgb = np.asarray(image_rgb)
    return app.get(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def pick_identity_face(faces: List[Any]) -> Tuple[Optional[Any], Optional[int]]:
    """
    The male face and its index.

    Returns the index explicitly and never recovers it later: insightface `Face`
    objects hold numpy arrays, so `list.index(face)` raises "truth value of an
    array with more than one element is ambiguous".
    """
    if not faces:
        return None, None
    indexed = list(enumerate(faces))
    males = [(i, f) for i, f in indexed if getattr(f, "sex", None) == "M"]
    if not males:
        log("no face scored male — using the largest face instead", "WARN")
        males = indexed
    # largest by area, tie-broken by x position
    idx, face = max(
        males,
        key=lambda p: float(
            (p[1].bbox[2] - p[1].bbox[0]) * (p[1].bbox[3] - p[1].bbox[1])
        ),
    )
    return face, idx


# --------------------------------------------------------------------------- #
# Pipelines
# --------------------------------------------------------------------------- #


def load_instantid_pipeline(timings: Dict[str, float]):
    """
    SDXL + InstantID as a diffusers *community* pipeline.

    InstantID is not in diffusers core, so it is loaded from the installed
    diffusers package's examples/community directory when present, and from a
    pinned copy of the file otherwise. Same source either way: the file ships in
    the diffusers wheel under `examples/community/` only in some builds, so we
    fetch it from the hub copy that matches the installed diffusers version.
    """
    import torch
    from diffusers.models import ControlNetModel

    # InstantID is a diffusers *community* pipeline, not part of core, so it is
    # materialised by the bootstrapper rather than assumed to be importable.
    from fetch_instantid import ensure_pipeline

    sys.path.insert(0, str(ensure_pipeline().parent))
    from instantid_pipeline import StableDiffusionXLInstantIDPipeline

    with timed("load_controlnet", timings):
        controlnet = ControlNetModel.from_pretrained(
            INSTANTID_REPO, subfolder=INSTANTID_CONTROLNET, torch_dtype=torch.float16
        )
        log("IdentityNet ControlNet loaded")

    with timed("load_sdxl", timings):
        pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
            BASE_MODEL,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
        log(f"{BASE_MODEL} loaded in float16")

    with timed("load_identity_adapter", timings):
        from huggingface_hub import hf_hub_download

        adapter = hf_hub_download(INSTANTID_REPO, INSTANTID_ADAPTER)
        pipe.load_ip_adapter_instantid(adapter)
        log(f"ip-adapter.bin loaded ({os.path.getsize(adapter) / 1e6:.0f} MB)")

    with timed("configure_pipeline", timings):
        pipe.to(DEVICE)                 # no CPU offload, no .to("cpu") anywhere
        # The community InstantID pipeline inherits from
        # StableDiffusionXLControlNetPipeline but does NOT mix in
        # DiffusionPipeline, so the enable_vae_slicing() helper does not exist —
        # calling it aborts the run after ~2 minutes of model loading (run
        # 45947923120). The flag it sets is just `vae.enable_slicing()`, so set
        # it on the VAE directly, and fall back to `use_slicing = True`.
        slicing = "off"
        vae = getattr(pipe, "vae", None)
        for attempt in ("enable_slicing",):
            if vae is not None and hasattr(vae, attempt):
                getattr(vae, attempt)()
                slicing = f"vae.{attempt}()"
                break
        if slicing == "off":
            if vae is not None:
                vae.use_slicing = True
                slicing = "vae.use_slicing=True"
            else:
                log("no VAE on the pipeline — VAE slicing not enabled", "WARN")
        try:
            pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass
        log(f"pipeline on {DEVICE} · vae slicing: {slicing}")

    return pipe


def build_compel(pipe):
    """
    CompelForSDXL, which already encodes with truncation disabled.

    Verified against compel 2.5.1: `CompelForSDXL.__init__(pipe, manager, device)`
    takes no truncation flag — it constructs both encoders with
    `truncate_long_prompts=False` internally, so the 77-token ceiling that
    silently dropped half of every prompt in earlier phases does not apply.
    """
    from compel import CompelForSDXL

    compel = CompelForSDXL(pipe)
    log("compel ready: CompelForSDXL with truncate_long_prompts=False on both "
        "encoders — prompts are chunked, not truncated at 77 tokens")
    return compel


def encode_prompts(compel, prompt: str, negative_prompt: str):
    """
    Encode positive and negative prompts into diffusers kwargs.

    CompelForSDXL returns a LabelledConditioning holding the concatenated
    (bi-encoder) embeds plus the pooled embeds; SDXL requires all four tensors,
    and positive/negative are already length-matched by compelling internals.
    """
    conditioning = compel(prompt, negative_prompt=negative_prompt)
    return {
        "prompt_embeds": conditioning.embeds,
        "pooled_prompt_embeds": conditioning.pooled_embeds,
        "negative_prompt_embeds": conditioning.negative_embeds,
        "negative_pooled_prompt_embeds": conditioning.negative_pooled_embeds,
    }


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #


def set_scheduler(pipe, timings: Dict[str, float]) -> None:
    """DPMSolverMultistep with Karras sigmas, from the model's own config."""
    from diffusers import DPMSolverMultistepScheduler

    with timed("configure_scheduler", timings):
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas=True,
            algorithm_type="dpmsolver++",
            solver_order=2,
        )
        log(f"scheduler: {type(pipe.scheduler).__name__} "
            f"(karras={pipe.scheduler.config.use_karras_sigmas})")


def generate_master(pipe, compel, timings: Dict[str, float]):
    """The clean master portrait: no hands near the face, no accessories."""
    import torch

    prompt = (
        f"{LENS_SPEC}, close-up portrait of a charismatic {FACE_SPEC}, "
        "wearing an unbuttoned white linen shirt on a luxury yacht deck in "
        "Monaco harbour, natural Mediterranean afternoon light, unretouched, "
        "photorealistic, sharp focus on the eyes, no hands in frame near the face"
    )
    with timed("master_portrait", timings):
        embeddings = encode_prompts(compel, prompt, NEGATIVE_PROMPT)
        image = pipe(
            **embeddings,
            width=MASTER_SIZE[0],
            height=MASTER_SIZE[1],
            num_inference_steps=MASTER_STEPS,
            guidance_scale=MASTER_CFG,
            generator=torch.Generator(device=DEVICE).manual_seed(BASE_SEED),
        ).images[0]
        path = OUTPUT_DIR / MASTER_FILENAME
        image.save(path, format="PNG", optimize=True, compress_level=9)
        log(f"wrote {path.name} ({path.stat().st_size / 1024:.0f} KB) · "
            f"steps={MASTER_STEPS} cfg={MASTER_CFG}")
    return image, prompt


def generate_scene(pipe, compel, face_emb, face_kps, prompt: str, seed: int,
                   filename: str, timings: Dict[str, float]):
    """One carousel scene, identity-conditioned from the master face."""
    import torch

    with timed(f"scene_{Path(filename).stem}", timings):
        embeddings = encode_prompts(compel, prompt, NEGATIVE_PROMPT)
        # Identity strength lives on the attention processors, not in __call__.
        # Passing `ip_adapter_scale=` as a kwarg would be swallowed by **kwargs
        # and silently ignored, so set it explicitly.
        pipe.set_ip_adapter_scale(SCENE_IP_ADAPTER_SCALE)
        image = pipe(
            **embeddings,
            image_embeds=face_emb,
            image=face_kps,
            width=SCENE_SIZE[0],
            height=SCENE_SIZE[1],
            num_inference_steps=SCENE_STEPS,
            guidance_scale=SCENE_CFG,
            controlnet_conditioning_scale=SCENE_CONTROLNET_SCALE,
            generator=torch.Generator(device=DEVICE).manual_seed(seed),
        ).images[0]
        path = OUTPUT_DIR / filename
        image.save(path, format="PNG", optimize=True, compress_level=9)
        log(f"wrote {path.name} ({path.stat().st_size / 1024:.0f} KB) · "
            f"ip_adapter={SCENE_IP_ADAPTER_SCALE} controlnet={SCENE_CONTROLNET_SCALE}")
    return image


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #


def cosine(a, b) -> Optional[float]:
    import numpy as np

    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size != b.size or not a.any() or not b.any():
        return None
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def verify_identity(app, master_face, image, label: str) -> Dict[str, Any]:
    """
    Compare the generated face against the master face.

    InstantID should keep this well above the 0.45 threshold that a post-hoc
    swap used to need; anything lower means the adapter lost the identity.
    """
    faces = detect_faces(app, image)
    face, idx = pick_identity_face(faces)
    result: Dict[str, Any] = {"faces": len(faces), "male_index": idx}
    if face is None or master_face is None:
        result["cosine"] = None
        return result
    similarity = cosine(master_face.normed_embedding, face.normed_embedding)
    result["cosine"] = None if similarity is None else round(similarity, 4)
    verdict = "MATCH" if (similarity or 0) >= 0.45 else "DRIFT"
    result["verdict"] = verdict
    log(f"identity {label}: cosine={similarity:.4f} ({verdict})")
    return result


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #



def merge_metrics(section: str, metrics: Dict[str, object], path: Path) -> None:
    """
    Fold this worker's half into the shared luca_metrics.json, safely.

    Both workers finish at unpredictable times and target the same file, so a
    naive read-modify-write loses whichever half lands first. The sidecar is
    written unconditionally first (it is the source of truth for post-mortems),
    then the merge runs under an exclusive file lock and replaces the file
    atomically, so a reader never sees a half-written document.
    """
    import json as _json

    sidecar = path.parent / f"luca_metrics_{section}.json"
    try:
        sidecar.write_text(_json.dumps(metrics, indent=2, default=str),
                           encoding="utf-8")
    except Exception as exc:
        log(f"could not write {sidecar.name}: {exc}", "WARN")

    lock = path.parent / "luca_metrics.lock"
    try:
        with open(lock, "a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                merged = {}
                if path.is_file():
                    try:
                        merged = _json.loads(path.read_text(encoding="utf-8"))
                    except Exception:
                        merged = {}
                merged[section] = metrics
                merged.setdefault("artifacts", [])
                merged["artifacts"] = sorted(
                    set(merged["artifacts"]) | set(metrics.get("artifacts") or []))
                tmp = path.with_suffix(".json.tmp")
                tmp.write_text(_json.dumps(merged, indent=2, default=str),
                               encoding="utf-8")
                os.replace(tmp, path)      # atomic: no torn reads
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except Exception as exc:
        log(f"could not merge metrics: {exc}", "WARN")

def main() -> int:
    started = time.perf_counter()
    telemetry = Tee(BUILD_LOG, sys.stdout)
    sys.stdout = telemetry  # type: ignore[assignment]

    timings: Dict[str, float] = {}
    metrics: Dict[str, Any] = {
        "process": "images",
        "device": DEVICE,
        "status": "running",
        "started_at": ts(),
        "base_model": BASE_MODEL,
        "identity_method": "InstantID (ip-adapter.bin + IdentityNet ControlNet)",
        "face_encoder": "insightface antelopev2",
        "prompt_encoder": "compel CompelForSDXL (truncation disabled)",
        "prohibitions_honoured": {
            "pipe.to_cpu": False,
            "inswapper_128": False,
            "svd_xt": False,
            "sdxl_img2img_adetailer": False,
        },
        "timings_seconds": timings,
        "errors": [],
        "artifacts": [],
        "identity": {},
    }
    exit_code = 0

    try:
        ensure_image_deps(timings)

        import diffusers
        import torch

        metrics["environment"] = {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "diffusers": diffusers.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        log(f"environment: {json.dumps(metrics['environment'])}")
        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA device — enable a Kaggle GPU accelerator")

        torch.cuda.set_device(0)
        pipe = load_instantid_pipeline(timings)
        set_scheduler(pipe, timings)
        compel = build_compel(pipe)

        # ---- master portrait -------------------------------------------------
        master, master_prompt = generate_master(pipe, compel, timings)
        metrics["artifacts"].append(MASTER_FILENAME)
        metrics["master_prompt_tokens"] = len(master_prompt.split())

        # ---- identity from the master ---------------------------------------
        app = load_face_app()
        master_faces = detect_faces(app, master)
        master_face, master_idx = pick_identity_face(master_faces)
        if master_face is None:
            raise RuntimeError("no face detected in the master portrait")
        from instantid_pipeline import draw_kps as instantid_draw_kps

        face_emb = master_face.normed_embedding
        # upstream's own stick-figure renderer: 5 keypoints, limbSeq [[0,2],[1,2],
        # [3,2],[4,2]], stickwidth 4 — identical to InstantID's reference path
        face_kps = instantid_draw_kps(master, master_face.kps)
        metrics["master_face"] = {
            "faces": len(master_faces),
            "male_index": master_idx,
            "bbox": [int(v) for v in master_face.bbox],
            "sex": getattr(master_face, "sex", None),
        }
        log(f"master identity locked: bbox={metrics['master_face']['bbox']}")

        # ---- scene carousel --------------------------------------------------
        scenes = [
            (
                "yacht",
                2,
                "standing at the teak railing of a luxury yacht in Monaco harbour "
                "at golden hour, one hand resting on the railing and the other "
                "holding a champagne flute, both hands visible with five natural "
                "fingers, relaxed confident smile, no hands near the face",
            ),
            (
                "porsche",
                3,
                "leaning against the driver door of a parked white Porsche 911 "
                "GT3 RS on a cliffside road above the Amalfi coast, one hand on "
                "the steering wheel through the open window and the other "
                "relaxed at his side, both hands visible with five natural "
                "fingers, sunglasses not worn, bright Mediterranean daylight",
            ),
            (
                "party",
                4,
                "at an exclusive Ibiza rooftop party at night, direct on-camera "
                "flash, standing beside a smiling young blonde woman in a black "
                "sequin dress on the left, he is on the right in a black silk "
                "shirt holding a whiskey glass, both hands visible with five "
                "natural fingers, night bokeh background",
            ),
        ]

        for name, index, scene_clause in scenes:
            filename = SCENE_FILES[name]
            prompt = (
                f"{LENS_SPEC}, medium shot of {FACE_SPEC}, {scene_clause}, "
                "unretouched, photorealistic"
            )
            image = generate_scene(pipe, compel, face_emb, face_kps, prompt,
                                   BASE_SEED + index, filename, timings)
            metrics["artifacts"].append(filename)
            metrics["identity"][filename] = verify_identity(
                app, master_face, image, filename)

        log("image pipeline complete — no face swap, no adetailer, no CPU offload")

    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        log(f"FATAL — {detail}", "ERROR")
        log(traceback.format_exc(), "ERROR")
        metrics["status"] = "failed"
        metrics["errors"].append({"stage": "fatal", "error": detail})
        exit_code = 1
    finally:
        metrics["total_seconds"] = round(time.perf_counter() - started, 1)
        metrics["finished_at"] = ts()
        if metrics["status"] == "running":
            metrics["status"] = "success" if not metrics["errors"] else "success_with_errors"
        # The two processes share luca_metrics.json, so write our half separately
        # and let the orchestrator merge. Writing the merged file here would race.
        merge_metrics("images", metrics, METRICS_PATH)

        log("=" * 74)
        log(f"image pipeline status: {metrics['status']} · {metrics['total_seconds']}s")
        for name in metrics["artifacts"]:
            path = OUTPUT_DIR / name
            if path.is_file():
                log(f"  ✓ {name} ({path.stat().st_size / 1024:.0f} KB)")
            else:
                log(f"  ✗ {name} MISSING", "ERROR")
        log("=" * 74)
        sys.stdout = telemetry._stream  # type: ignore[assignment]
        telemetry.close()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
