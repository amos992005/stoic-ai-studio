#!/usr/bin/env python3
"""
stoic-ai-studio · PHASE 3 · "LUCA MORETTI" flagship character
=============================================================

One Kaggle T4 run produces the complete, postable asset pack for a new
virtual creator: a lifestyle carousel, a clean synthetic voice line, and a
*real neural* image-to-video clip.

PART A — THE CAROUSEL
  01_luca_master.png     1024x1024  RealVisXL V5.0, fp16, DPM++ 2M Karras
  02..05                832x1024    vertical slides, face-locked to the master
                                     with IP-Adapter-Plus-Face @ 0.65

  Two-person slides (yacht / VIP party) are the dangerous ones: a face
  reference with no spatial constraint leaks Luca's identity onto the woman
  standing next to him ("face bleeding").  We solve it properly, at the
  attention level, with diffusers' *masked* IP-Adapter:

      cross_attention_kwargs={"ip_adapter_masks": [mask]}

  `mask` is `[1, num_ip_images, H, W]`, 1.0 over the region where Luca stands
  and 0.0 over the rest, with a short ramp so there is no visible seam.
  Verified against diffusers 0.42's `IPAdapterAttnProcessor2_0`: the mask is
  indexed per reference image, downsampled with bicubic interpolation to the
  attention query grid, and multiplied into the IP-Adapter contribution only.

PART B — VOICE + NEURAL VIDEO
  luca_voice.mp3    kokoro-onnx, `am_adam`, speed 1.0.
                    Audio processing is *linear peak normalisation only*.
                    No WSOLA time-stretching, no biquad/parametric EQ.
  luca_reel_clip.mp4 stabilityai/stable-video-diffusion-img2vid-xt (fully
                    public, no auth token) generates 25 genuine neural frames
                    from a carousel slide; the sequence is ping-pong looped to
                    exactly the natural length of the voice track and muxed
                    with ffmpeg.

  There is no procedural pixel-slice animation anywhere in this file. Every
  moving pixel comes out of Stable Video Diffusion.

Design rules honoured here
  * $0: Kaggle T4 does the compute, GitHub Actions only orchestrates.
  * <10 min wall clock on a T4; every heavy step is timed and retried.
  * Deterministic: fixed seeds, private generators, no shuffling.
  * Version drift never crashes the run -- explicit fallbacks everywhere.
"""

from __future__ import annotations

import gc
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

#: Photoreal SDXL checkpoint (public, ungated).
MODEL_ID = os.environ.get("SDXL_MODEL_ID", "SG161222/RealVisXL_V5.0")

#: CLIP vision tower required by the Face ID IP-Adapter.
CLIP_VISION_ID = os.environ.get(
    "CLIP_VISION_ID", "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
)

#: IP-Adapter repository / weights ("Face ID Plus" -- purpose-built for faces).
IP_ADAPTER_REPO = os.environ.get("IP_ADAPTER_REPO", "h94/IP-Adapter")
IP_ADAPTER_SUBFOLDER = "sdxl_models"
IP_ADAPTER_WEIGHT = "ip-adapter-plus-face_sdxl_vit-h.safetensors"
IP_ADAPTER_SCALE = float(os.environ.get("IP_ADAPTER_SCALE", "0.65"))

#: Real neural image-to-video. 100% public, native in diffusers, runs on a T4.
SVD_MODEL_ID = os.environ.get("SVD_MODEL_ID", "stabilityai/stable-video-diffusion-img2vid-xt")
SVD_NUM_FRAMES = int(os.environ.get("SVD_NUM_FRAMES", "25"))
SVD_STEPS = int(os.environ.get("SVD_STEPS", "25"))
SVD_MOTION_BUCKET = int(os.environ.get("SVD_MOTION_BUCKET", "127"))
SVD_NOISE_AUG = float(os.environ.get("SVD_NOISE_AUG", "0.02"))
SVD_FPS = int(os.environ.get("SVD_FPS", "25"))
#: Which carousel slide seeds the video (04 = Porsche GT3 RS, 02 = yacht).
SVD_SOURCE_SLIDE = os.environ.get("SVD_SOURCE_SLIDE", "04")

#: If the IP-Adapter cannot be loaded, fail loudly by default.
ALLOW_IP_ADAPTER_FALLBACK = os.environ.get("ALLOW_IP_ADAPTER_FALLBACK", "0") in {"1", "true", "yes"}
#: If the masked IP-Adapter call fails at runtime, retry that slide unmasked.
ALLOW_UNMASKED_RETRY = os.environ.get("ALLOW_UNMASKED_RETRY", "1") in {"1", "true", "yes"}

#: Golden reference.
BASE_SEED = int(os.environ.get("BASE_SEED", "777"))
MASTER_STEPS = int(os.environ.get("MASTER_STEPS", "30"))
MASTER_CFG = float(os.environ.get("MASTER_CFG", "5.5"))
MASTER_SIZE = (1024, 1024)

#: Carousel slides (vertical, Instagram 4:5).
SLIDE_STEPS = int(os.environ.get("SLIDE_STEPS", "30"))
SLIDE_CFG = float(os.environ.get("SLIDE_CFG", "5.5"))
SLIDE_SIZE = (832, 1024)
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "92"))

#: Voice.
SCRIPT_LINE = "They told me to slow down and play it safe. Look where that got them."
VOICE_NAME = os.environ.get("VOICE_NAME", "am_adam")
SPEECH_SPEED = float(os.environ.get("SPEECH_SPEED", "1.0"))
AUDIO_SR_TARGET = 24000
PEAK_TARGET = 0.85  # linear peak normalisation only

KOKORO_MODEL_URLS = [
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
    "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro-v1.0.onnx",
]
KOKORO_VOICES_URLS = [
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
    "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices-v1.0.bin",
]

MASTER_FILENAME = "01_luca_master.png"
CAROUSEL_SHEET = "luca_carousel_sheet.png"
VIDEO_SHEET = "luca_video_sheet.png"
VOICE_MP3 = "luca_voice.mp3"
VOICE_WAV = "luca_voice.wav"
REEL_MP4 = "luca_reel_clip.mp4"
METRICS_FILENAME = "luca_metrics.json"
BUILD_LOG = "build.log"

#: Shared identity, repeated verbatim so only the *scene* changes between panels.
LUCA = (
    "a 27-year-old extremely handsome man with sun-bleached tousled dirty-blonde "
    "hair, a sharp jawline, light 3-day stubble and piercing blue-green eyes, "
    "lean athletic swimmer's physique"
)

#: Deliberately photographic, never "AI render".
CAMERA = (
    "candid smartphone photograph, real skin texture with visible pores and "
    "imperfections, natural available light, slight motion blur, true-to-life "
    "colour, unretouched, authentic Instagram lifestyle photo"
)

NEGATIVE_PROMPT = (
    "cartoon, anime, illustration, painting, 3d render, cgi, doll, plastic skin, "
    "airbrushed, waxy, over-smoothed, beauty filter, oversaturated, hdr glow, "
    "blurry, low quality, jpeg artifacts, deformed, disfigured, mutated hands, "
    "extra fingers, extra limbs, bad anatomy, bad proportions, cross-eyed, "
    "asymmetric eyes, watermark, signature, text, logo, frame, border, "
    "teenager, child, old man, "
    "two men, two men together, male couple, duplicate person, cloned face"
)

MASTER_PROMPT = (
    f"Candid 35mm photo of {LUCA}, looking like a young Brad Pitt, with a subtle "
    "charismatic smirk, golden hour Mediterranean sunlight on a luxury terrace, "
    f"realistic skin pores, natural lighting, {CAMERA}"
)

#: `mask` selects the IP-Adapter spatial attention region:
#:   "left" -> identity applied only to the left part of the frame (Luca stands
#:             there, the woman stands right) -- this is the anti-bleed control.
#:   "full" -> identity applied everywhere (single-subject slides).
SLIDES: List[Dict[str, Any]] = [
    {
        "index": 2,
        "filename": "02_monaco_yacht_date.png",
        "title": "02 · MONACO · YACHT",
        "mask": "left",
        "people": 2,
        "prompt": (
            # Ordered description: "a woman ... standing next to a man ...".
            # Putting the *woman* first is what makes her the anchor subject;
            # with the man first SDXL collapses the second slot back into a man,
            # and the single male identity reference on the left reinforced it.
            "Candid iPhone 16 Pro 24mm photo of a woman and a man together on the "
            "sun deck of a luxury Riva yacht in Monaco at sunset. In the RIGHT "
            "HALF of the frame a stunning young brunette woman in a bikini is "
            "laughing naturally, long dark hair, feminine figure, looking away "
            "from the camera; she is standing next to her date. In the LEFT HALF "
            "of the frame a shirtless athletic 27-year-old man holds a champagne "
            f"flute with a confident smirk, looking at the camera ({LUCA}). One "
            "man and one woman, two different people, side by side. Bright ocean "
            "horizon, Monaco skyline behind them, harbour lights beginning to "
            f"glow, salt spray on the lens, {CAMERA}"
        ),
        "negative": (
            "two men, two women, male couple, men only, no woman, two boys, "
            "duplicate man, cloned face, twins, same face twice"
        ),
    },
    {
        "index": 3,
        "filename": "03_vip_party_flash.png",
        "title": "03 · IBIZA · VIP",
        "mask": "left",
        "people": 2,
        "prompt": (
            "Candid 35mm direct on-camera flash photo of a woman and a man "
            "together at an Ibiza rooftop VIP party at night. In the RIGHT HALF "
            "of the frame a gorgeous blonde woman in an evening dress holds a "
            "cocktail and smiles, feminine figure, long blonde hair, standing "
            "beside him; she is his date. In the LEFT HALF of the frame a "
            "27-year-old man in an unbuttoned black silk shirt smirks at the "
            f"camera ({LUCA}). One man and one woman, two different people, side "
            "by side. Harsh flash falloff, dark background with bokeh club "
            f"lights, slightly blown highlights, authentic night-flash aesthetic, "
            f"{CAMERA}"
        ),
        "negative": (
            "two men, two women, male couple, men only, no woman, two boys, "
            "duplicate man, cloned face, twins, same face twice"
        ),
    },
    {
        "index": 4,
        "filename": "04_porsche_gt3_pov.png",
        "title": "04 · AMALFI · GT3",
        "mask": "full",
        "prompt": (
            "Candid iPhone photo taken from the passenger seat of a Porsche 911 "
            "GT3 RS on the Amalfi coast cliff road. A 27-year-old dirty-blonde "
            f"man driving, sunglasses pushed up on his forehead, one hand on the "
            f"alcantara steering wheel, carefree grin straight at the camera "
            f"({LUCA}). Bright Mediterranean daylight, sea and guardrail through "
            f"the windscreen, motion in the background, {CAMERA}"
        ),
    },
    {
        "index": 5,
        "filename": "05_extreme_action.png",
        "title": "05 · EXTREME",
        "mask": "full",
        "prompt": (
            "Raw GoPro action shot of a 27-year-old athletic dirty-blonde man "
            f"leaning out of an open helicopter doorway above a turquoise "
            f"coastline, wind tearing through his hair, adrenaline grin ({LUCA}). "
            "Water droplets on the lens, sun flare, helicopter skid and rotor "
            "wash visible, distant cliffs and boats below, genuine action-camera "
            f"sharpness, {CAMERA}"
        ),
    },
]


# --------------------------------------------------------------------------- #
# Logging / timing
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

    On Kaggle we write to the *working directory root* (`/kaggle/working`) so
    that `kaggle kernels output ... -p ./outputs` lands files at
    `outputs/<file>` rather than `outputs/outputs/<file>`.
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
    info: Dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            info["gpu_total_vram_gb"] = [
                round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 2)
                for i in range(torch.cuda.device_count())
            ]
    except Exception as exc:  # pragma: no cover
        info["torch_error"] = f"{type(exc).__name__}: {exc}"
    for mod in ("diffusers", "transformers", "accelerate", "safetensors", "PIL", "onnxruntime"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = "unavailable"
    return info


def release_memory() -> None:
    """Best-effort VRAM/RAM reclaim between pipeline swaps."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Dependency bootstrap (wheels only -- never compile from source)
# --------------------------------------------------------------------------- #


def ensure_python_deps(timings: Dict[str, float]) -> None:
    required = {
        "kokoro_onnx": "kokoro-onnx",
        "soundfile": "soundfile",
        "imageio_ffmpeg": "imageio-ffmpeg",
        "onnxruntime": "onnxruntime-gpu",
    }
    missing: List[str] = []
    for module, package in required.items():
        try:
            __import__(module)
        except Exception:
            missing.append(package)

    if not missing:
        log("python deps already present: " + ", ".join(required.values()))
    else:
        log(f"installing missing deps: {missing}", "WARN")
        with timed("pip_install_deps", timings):
            cmd = [
                sys.executable, "-m", "pip", "install", "--quiet", "--no-input",
                "--disable-pip-version-check", "--only-binary=:all:", *missing,
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                log(proc.stderr[-3000:], "ERROR")
                proc = subprocess.run(cmd[:-1], capture_output=True, text=True)
                if proc.returncode != 0:
                    log(proc.stderr[-3000:], "ERROR")
                    raise RuntimeError(f"could not install {missing}")
        for module in required:
            try:
                __import__(module)
            except Exception as exc:
                raise RuntimeError(f"{module} still unavailable after install: {exc}") from exc

    # diffusers must know about IP-Adapter masks *and* Stable Video Diffusion.
    import diffusers

    log(f"diffusers {diffusers.__version__} detected")
    try:
        # Capability probe only -- the real imports happen inside the builders.
        assert diffusers.StableVideoDiffusionPipeline is not None
        assert diffusers.image_processor.IPAdapterMaskProcessor is not None
        return
    except Exception as exc:
        log(f"diffusers build lacks SVD/IP-Adapter-mask support ({exc}) -- upgrading", "WARN")

    with timed("pip_install_diffusers", timings):
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "--no-input",
             "--disable-pip-version-check", "--only-binary=:all:", "diffusers>=0.31.0"],
            capture_output=True,
            text=True,
        )
    import importlib

    importlib.reload(diffusers)
    # Re-probe after the upgrade; these shadow the names above on purpose.
    from diffusers import StableVideoDiffusionPipeline as _SVD  # noqa: F401
    from diffusers.image_processor import IPAdapterMaskProcessor as _MaskProc  # noqa: F401

    del _SVD, _MaskProc
    log(f"diffusers upgraded to {diffusers.__version__} ({'SVD + mask API ready'})")


# --------------------------------------------------------------------------- #
# ffmpeg helpers
# --------------------------------------------------------------------------- #


def ffmpeg_binary() -> str:
    """Prefer the system ffmpeg (libmp3lame + libx264), then imageio's static build."""
    system = shutil.which("ffmpeg")
    if system:
        return system
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def has_encoder(ffmpeg: str, codec: str) -> bool:
    try:
        proc = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True)
        return codec in proc.stdout
    except Exception:
        return False


def download_first(urls: Sequence[str], dest: Path, timeout: int = 300) -> Path:
    """Download the first working URL to `dest` (used for Kokoro weights)."""
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    errors: List[str] = []
    for url in urls:
        try:
            log(f"downloading {url.split('/')[-1]} ← {url.split('/')[2]}")
            req = urllib.request.Request(url, headers={"User-Agent": "stoic-ai-studio/3.0"})
            with urllib.request.urlopen(req, timeout=timeout) as response, open(dest, "wb") as handle:
                shutil.copyfileobj(response, handle)
            size_mb = dest.stat().st_size / 1024**2
            if size_mb < 1:
                raise RuntimeError(f"only {size_mb:.1f} MB downloaded -- truncated")
            log(f"  ✓ {dest.name} ({size_mb:.1f} MB)")
            return dest
        except Exception as exc:
            errors.append(f"{url} → {type(exc).__name__}: {exc}")
            log(f"  ✗ {errors[-1]}", "WARN")
    raise RuntimeError("every mirror failed:\n  " + "\n  ".join(errors))


# --------------------------------------------------------------------------- #
# Part A — SDXL pipeline
# --------------------------------------------------------------------------- #


def _first_available(owner, names: List[str], label: str) -> Optional[str]:
    """
    Call the first API that exists on this object and return its name.

    diffusers relocates these knobs between releases (0.42 dropped
    `pipe.enable_vae_slicing()` for `pipe.vae.enable_slicing()`), so we never
    hard-code a single signature for an *optional* VRAM optimisation.
    """
    for name in names:
        parts = name.split(".")
        target = owner
        try:
            for part in parts[:-1]:
                target = getattr(target, part)
            fn = getattr(target, parts[-1], None)
        except AttributeError:
            continue
        if not callable(fn):
            continue
        try:
            fn()
            log(f"{label}: applied {name}()")
            return name
        except Exception as exc:
            log(f"{label}: {name}() failed ({type(exc).__name__}: {exc})", "WARN")
    log(f"{label}: no supported API on this diffusers build (skipped)", "WARN")
    return None


def _load_pipeline(cls, model_id: str, label: str):
    """Load a diffusers pipeline in fp16, tolerating the dtype/torch_dtype rename."""
    import torch

    attempts: List[Dict[str, Any]] = [
        {"variant": "fp16", "dtype": torch.float16, "use_safetensors": True},
        {"variant": "fp16", "torch_dtype": torch.float16, "use_safetensors": True},
        {"dtype": torch.float16, "use_safetensors": True},
        {"torch_dtype": torch.float16, "use_safetensors": True},
        {"dtype": torch.float16},
    ]
    last_error: Optional[Exception] = None
    for kwargs in attempts:
        dtype_key = "dtype" if "dtype" in kwargs else "torch_dtype"
        variant = "fp16 variant" if "variant" in kwargs else "no variant"
        try:
            log(f"{label}: loading {model_id} ({dtype_key}=float16, {variant})")
            return cls.from_pretrained(model_id, **kwargs)
        except Exception as exc:
            last_error = exc
            log(f"{label}: attempt failed ({type(exc).__name__}: {exc})", "WARN")
    raise RuntimeError(f"could not load {model_id}: {last_error}") from last_error


def build_pipeline(timings: Dict[str, float]):
    """Load RealVisXL V5.0 in fp16 on cuda:0 with a Karras DPM-Solver++ scheduler."""
    import torch
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This kernel must run with a GPU "
            "(Kaggle: Settings → Accelerator → GPU T4 x2)."
        )

    device = "cuda:0"
    log(f"device={device} · {torch.cuda.get_device_name(0)}")

    with timed("load_sdxl_pipeline", timings):
        pipe = _load_pipeline(StableDiffusionXLPipeline, MODEL_ID, "sdxl")

    with timed("configure_scheduler_and_vram", timings):
        try:
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++"
            )
            log("scheduler: DPM-Solver++ 2M with Karras sigmas")
        except Exception as exc:
            log(f"scheduler swap failed, keeping default: {exc}", "WARN")

        try:
            pipe.set_progress_bar_config(disable=False, desc="diffusion")
        except Exception:
            pass

        pipe.to(device)
        log(f"pipeline moved to {device} (fp16 weights)")

        _first_available(pipe, ["vae.enable_slicing", "enable_vae_slicing"], "vae slicing")
        _first_available(pipe, ["vae.enable_tiling", "enable_vae_tiling"], "vae tiling")
        _first_available(pipe, ["set_use_resolution_binning"], "resolution binning")

        # Attention slicing is deliberately NOT enabled: it installs a
        # SlicedAttnProcessor that diffusers cannot re-instantiate while
        # converting the UNet for IP-Adapter, and it breaks the masked
        # attention path we rely on for anti-bleed.
        log("attention slicing left disabled (incompatible with IP-Adapter masks)")

        if getattr(pipe, "vae", None) is not None:
            pipe.vae.to(device=device, dtype=torch.float16)
            log("vae kept in float16 (matches unet dtype)")

    return pipe, device


def attach_face_adapter(pipe, device: str, timings: Dict[str, float]) -> bool:
    """Load the CLIP image encoder + IP-Adapter Face weights and set scale 0.65."""
    import torch
    from transformers import CLIPVisionModelWithProjection

    log(f"attaching IP-Adapter Face ({IP_ADAPTER_WEIGHT}) @ scale {IP_ADAPTER_SCALE}")
    try:
        with timed("load_clip_vision", timings):
            image_encoder = None
            last_error: Optional[Exception] = None
            for kwargs in (
                {"dtype": torch.float16, "use_safetensors": True},
                {"torch_dtype": torch.float16, "use_safetensors": True},
                {"dtype": torch.float16},
                {},
            ):
                try:
                    image_encoder = CLIPVisionModelWithProjection.from_pretrained(CLIP_VISION_ID, **kwargs)
                    break
                except Exception as exc:
                    last_error = exc
                    log(f"clip-vision load attempt failed ({type(exc).__name__}: {exc})", "WARN")
            if image_encoder is None:
                raise RuntimeError(f"could not load {CLIP_VISION_ID}: {last_error}") from last_error
            image_encoder = image_encoder.to(device, dtype=torch.float16)
            image_encoder.eval()
            pipe.image_encoder = image_encoder
            log("clip vision encoder ready (ViT-H, fp16)")

        with timed("load_ip_adapter", timings):
            try:
                pipe.load_ip_adapter(
                    IP_ADAPTER_REPO, subfolder=IP_ADAPTER_SUBFOLDER, weight_name=IP_ADAPTER_WEIGHT
                )
            except TypeError as exc:
                if "SlicedAttnProcessor" not in str(exc):
                    raise
                log("attention slicing blocks IP-Adapter conversion; resetting processors", "WARN")
                _first_available(pipe, ["unset_attention_slicing"], "attention slicing off")
                try:
                    from diffusers.models.attention_processor import AttnProcessor2_0

                    if getattr(pipe, "unet", None) is not None:
                        pipe.unet.set_attn_processor(AttnProcessor2_0())
                except Exception as inner:
                    log(f"could not reset attention processors: {inner}", "WARN")
                pipe.load_ip_adapter(
                    IP_ADAPTER_REPO, subfolder=IP_ADAPTER_SUBFOLDER, weight_name=IP_ADAPTER_WEIGHT
                )

            if hasattr(pipe, "set_ip_adapter_scale"):
                pipe.set_ip_adapter_scale(IP_ADAPTER_SCALE)
            else:
                setter = getattr(pipe, "_set_ip_adapter_scale", None)
                if setter is None:
                    raise RuntimeError("this diffusers build exposes no IP-Adapter scale setter")
                setter(IP_ADAPTER_SCALE)
            log(f"ip-adapter scale set to {IP_ADAPTER_SCALE}")

        _describe_ip_processor(pipe)
        log("IP-Adapter Face ready — character consistency is locked.")
        return True

    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        log(f"IP-Adapter setup FAILED — {detail}", "ERROR")
        log(traceback.format_exc(), "ERROR")
        if not ALLOW_IP_ADAPTER_FALLBACK:
            raise RuntimeError(
                "Face-lock adapter is mandatory for this run "
                "(set ALLOW_IP_ADAPTER_FALLBACK=1 to degrade gracefully)."
            ) from exc
        log("ALLOW_IP_ADAPTER_FALLBACK=1 → continuing WITHOUT face lock.", "WARN")
        try:
            pipe.set_ip_adapter_scale(0.0)
        except Exception:
            pass
        return False
    finally:
        release_memory()


def _describe_ip_processor(pipe) -> None:
    """Log which attention processor carries the IP-Adapter (mask capability probe)."""
    try:
        unet = getattr(pipe, "unet", None)
        names = set()
        supports_masks = False
        for name, proc in unet.attn_processors.items():
            names.add(type(proc).__name__)
            if "ip_adapter_masks" in getattr(proc.__call__, "__code__", type("x", (), {"co_varnames": ()})).co_varnames:
                supports_masks = True
        log(f"unet attention processors: {sorted(names)}")
        log(f"masked IP-Adapter (ip_adapter_masks) supported: {supports_masks}")
    except Exception as exc:
        log(f"processor description skipped: {exc}", "WARN")


# --------------------------------------------------------------------------- #
# Anti-bleed: spatial attention mask
# --------------------------------------------------------------------------- #


def build_ip_mask(kind: Optional[str], width: int, height: int):
    """
    Build the `[1, 1, H, W]` spatial mask that confines the face reference.

    "left": 1.0 over the left part of the frame (where Luca stands) fading to
            0.0 by ~48% of the width, so the woman on the right keeps her own
            face. The ramp is kept soft: the attention-mask downsampler
            interpolates bicubically, so a hard edge would show up as a seam.

            The ramp matters more than it looks. Run 45893591367 ramped 44%→60%
            and the male identity still reached x≈50%, so the second subject on
            the right was pushed towards a *man* -- SDXL resolves "the other
            person" as male when it inherits half an identity. Tightening to
            30%→48% keeps the reference contribution clear of the right half.

    "full" / None: no mask (single-subject slides).
    """
    import torch

    if kind in (None, "full"):
        return None

    if kind != "left":
        raise ValueError(f"unknown mask kind {kind!r}")

    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    ramp_start, ramp_end = 0.30, 0.48
    weights = np.clip((ramp_end - x) / (ramp_end - ramp_start), 0.0, 1.0)
    weights = weights * weights * (3.0 - 2.0 * weights)  # smoothstep
    column = weights.reshape(1, width).repeat(height, axis=0)

    mask = torch.from_numpy(column.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    log(
        f"ip mask 'left': {tuple(mask.shape)} · coverage {float(column.mean()):.1%} "
        f"(1.0 → x<{ramp_start:.0%}, 0.0 → x>{ramp_end:.0%})"
    )
    return mask


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #


def _run_pipeline(pipe, kwargs: Dict[str, Any], label: str):
    """One inference pass, with the classic fp16 VAE-dtype recovery."""
    import torch

    def _run():
        with torch.inference_mode():
            return pipe(**kwargs)

    try:
        return _run()
    except RuntimeError as exc:
        if "should be the same" not in str(exc):
            raise
        log(f"{label}: dtype mismatch at decode ({exc}); aligning VAE dtype", "WARN")
        vae, unet = getattr(pipe, "vae", None), getattr(pipe, "unet", None)
        if vae is None or unet is None:
            raise
        vae.to(device=unet.device, dtype=unet.dtype)
        release_memory()
        return _run()


def generate_image(
    pipe,
    device: str,
    *,
    prompt: str,
    width: int,
    height: int,
    seed: int,
    steps: int,
    cfg: float,
    ip_adapter_image: Optional[Any] = None,
    mask_kind: Optional[str] = None,
    negative_prompt: Optional[str] = None,
    label: str = "image",
) -> Tuple[Image.Image, Dict[str, Any]]:
    """Run one diffusion pass, optionally with a spatially masked face reference."""
    import torch

    info: Dict[str, Any] = {
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "size": [width, height],
        "ip_adapter_image": ip_adapter_image is not None,
        "ip_mask": mask_kind,
        "masked_ip_adapter": False,
        "unmasked_retry": False,
    }

    kwargs: Dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": negative_prompt or NEGATIVE_PROMPT,
        "width": width,
        "height": height,
        "num_inference_steps": steps,
        "guidance_scale": cfg,
        "generator": torch.Generator(device=device).manual_seed(seed),
    }
    if ip_adapter_image is not None:
        kwargs["ip_adapter_image"] = ip_adapter_image

    mask = build_ip_mask(mask_kind, width, height) if ip_adapter_image is not None else None

    log(f"sampling {label}: {width}x{height} · seed={seed} · steps={steps} · cfg={cfg}"
        + (f" · mask={mask_kind}" if mask is not None else ""))

    result = None
    if mask is not None:
        masked_kwargs = dict(kwargs)
        masked_kwargs["cross_attention_kwargs"] = {"ip_adapter_masks": [mask]}
        try:
            result = _run_pipeline(pipe, masked_kwargs, label)
            info["masked_ip_adapter"] = True
            log(f"{label}: masked IP-Adapter applied (anti-bleed active)")
        except Exception as exc:
            if not ALLOW_UNMASKED_RETRY:
                raise
            log(f"{label}: masked IP-Adapter failed ({type(exc).__name__}: {exc})", "ERROR")
            log(traceback.format_exc(), "ERROR")
            log(f"{label}: retrying WITHOUT the spatial mask — identity may bleed", "WARN")
            release_memory()
            info["unmasked_retry"] = True

    if result is None:
        result = _run_pipeline(pipe, kwargs, label)

    if not result or not getattr(result, "images", None):
        raise RuntimeError(f"Pipeline returned no images for {label}")

    try:
        allocated = torch.cuda.max_memory_allocated(device) / 1024**3
        log(f"{label}: peak VRAM {allocated:.2f} GB")
        info["peak_vram_gb"] = round(float(allocated), 2)
    except Exception:
        pass

    image = result.images[0]
    if image.size != (width, height):
        log(f"{label}: unexpected size {image.size}, expected {(width, height)}", "WARN")
    return image, info


def save_png(image: Image.Image, path: Path) -> None:
    image.save(path, format="PNG", optimize=True)
    log(f"wrote {path.name} ({path.stat().st_size / 1024:.0f} KB)")


def save_jpeg(image: Image.Image, path: Path, quality: int = JPEG_QUALITY) -> None:
    image.convert("RGB").save(path, format="JPEG", quality=quality, optimize=True, progressive=True)
    log(f"wrote {path.name} ({path.stat().st_size / 1024:.0f} KB, q={quality})")


# --------------------------------------------------------------------------- #
# Contact sheets
# --------------------------------------------------------------------------- #


def _font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def _fit(dfont, text: str, max_width: int) -> None:
    try:
        length = dfont.getlength(text)
    except Exception:
        return
    if length > max_width:
        log(f"sheet label may overflow: {text!r}", "WARN")


def build_carousel_sheet(items: List[Tuple[str, Image.Image]], path: Path) -> None:
    """Inspection grid: master on the left, then the four vertical slides."""
    pad, label_h, header_h = 24, 46, 78
    thumb_h = 560
    thumbs = []
    for title, image in items:
        scale = thumb_h / image.height
        thumbs.append((title, image.resize((max(1, int(image.width * scale)), thumb_h), Image.LANCZOS)))

    total_w = pad + sum(t.width + pad for _, t in thumbs)
    total_h = header_h + thumb_h + label_h + pad
    sheet = Image.new("RGB", (total_w, total_h), (16, 16, 18))
    draw = ImageDraw.Draw(sheet)

    draw.text((pad, 20), "LUCA MORETTI · Monaco & Ibiza Weekend · carousel",
              font=_font(30, bold=True), fill=(240, 240, 245))
    draw.text((pad, 52), "RealVisXL V5.0 · IP-Adapter-Plus-Face 0.65 · masked attention (anti-bleed)",
              font=_font(17), fill=(150, 152, 160))

    x = pad
    for title, thumb in thumbs:
        sheet.paste(thumb, (x, header_h))
        draw.rectangle([x, header_h, x + thumb.width - 1, header_h + thumb.height - 1],
                       outline=(58, 58, 64))
        draw.text((x + 2, header_h + thumb.height + 12), title, font=_font(19, bold=True),
                  fill=(225, 226, 232))
        x += thumb.width + pad

    sheet.save(path, format="PNG", optimize=True)
    log(f"wrote {path.name} ({path.stat().st_size / 1024:.0f} KB) · {sheet.size[0]}x{sheet.size[1]}")


def build_video_sheet(frames: List[Image.Image], path: Path, subtitle: str) -> None:
    """Horizontal strip of evenly spaced video frames."""
    pad, header_h, label_h = 14, 72, 34
    count = min(8, len(frames))
    picks = [frames[round(i * (len(frames) - 1) / max(1, count - 1))] for i in range(count)]
    thumb_h = 420
    thumbs = []
    for image in picks:
        scale = thumb_h / image.height
        thumbs.append(image.resize((max(1, int(image.width * scale)), thumb_h), Image.LANCZOS))

    total_w = pad + sum(t.width + pad for t in thumbs)
    sheet = Image.new("RGB", (total_w, header_h + thumb_h + label_h), (16, 16, 18))
    draw = ImageDraw.Draw(sheet)
    draw.text((pad, 18), "LUCA · neural image-to-video (Stable Video Diffusion XT)",
              font=_font(27, bold=True), fill=(240, 240, 245))
    draw.text((pad, 48), subtitle, font=_font(16), fill=(150, 152, 160))

    x = pad
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, (x, header_h))
        draw.rectangle([x, header_h, x + thumb.width - 1, header_h + thumb.height - 1],
                       outline=(58, 58, 64))
        draw.text((x + 2, header_h + thumb.height + 8), f"frame {i + 1}/{count}",
                  font=_font(16), fill=(200, 202, 210))
        x += thumb.width + pad

    sheet.save(path, format="PNG", optimize=True)
    log(f"wrote {path.name} ({path.stat().st_size / 1024:.0f} KB) · {sheet.size[0]}x{sheet.size[1]}")


# --------------------------------------------------------------------------- #
# Part B.1 — Voice (linear peak normalisation only)
# --------------------------------------------------------------------------- #


def synth_voice(output_dir: Path, timings: Dict[str, float]) -> Dict[str, Any]:
    """
    Kokoro -> wav -> mp3.

    The ONLY processing applied is a linear peak normalisation. No time
    stretching (the video is fitted to the audio, never the reverse), no
    biquad/parametric EQ, no dynamic compression.
    """
    import soundfile as sf
    from kokoro_onnx import Kokoro

    cache_dir = Path(os.environ.get("MODEL_CACHE_DIR", tempfile.gettempdir())) / "stoic_kokoro"
    cache_dir.mkdir(parents=True, exist_ok=True)

    with timed("download_kokoro_assets", timings):
        model_path = cache_dir / "kokoro-v1.0.onnx"
        voices_path = cache_dir / "voices-v1.0.bin"
        if not model_path.exists():
            download_first(KOKORO_MODEL_URLS, model_path)
        if not voices_path.exists():
            download_first(KOKORO_VOICES_URLS, voices_path)

    with timed("load_kokoro", timings):
        try:
            import onnxruntime as ort

            providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                         if p in ort.get_available_providers()]
            if providers and providers[0] == "CUDAExecutionProvider":
                session = ort.InferenceSession(str(model_path), providers=providers)
                kokoro = Kokoro.from_session(session, str(voices_path))
                log(f"kokoro providers: {session.get_providers()}")
            else:
                kokoro = Kokoro(str(model_path), str(voices_path))
                log("kokoro running on CPU (no CUDA provider available)")
        except Exception as exc:
            log(f"provider-pinned session failed ({exc}); using default loader", "WARN")
            kokoro = Kokoro(str(model_path), str(voices_path))

        available = set(kokoro.get_voices())
        log(f"kokoro voices: total={len(available)}")
        if VOICE_NAME not in available:
            fallbacks = [v for v in ("am_michael", "bm_lewis", "am_adam") if v in available]
            if not fallbacks:
                raise RuntimeError(f"voice {VOICE_NAME!r} unavailable and no fallback found")
            log(f"voice {VOICE_NAME!r} missing → using {fallbacks[0]!r}", "WARN")

    voice = VOICE_NAME if VOICE_NAME in available else fallbacks[0]
    # A voice *name* is passed through untouched: kokoro indexes the full
    # style matrix with the phoneme count (voice[len(tokens) - 1]). Slicing a
    # style tensor to (1, 256) is what produced `Invalid rank for input: style`.
    style_shape = None
    try:
        style_shape = tuple(np.asarray(kokoro.get_voice_style(voice)).shape)
        log(f"voice {voice!r} style tensor {style_shape} (passed by name, never sliced)")
    except Exception as exc:
        log(f"style probe skipped: {exc}", "WARN")

    with timed("synthesize_speech", timings):
        audio, sr = kokoro.create(SCRIPT_LINE, voice=voice, speed=SPEECH_SPEED, lang="en-us")
        audio = np.asarray(audio, dtype=np.float32).squeeze()
        if audio.size == 0:
            raise RuntimeError("kokoro returned empty audio")
        if sr != AUDIO_SR_TARGET:
            raise RuntimeError(
                f"kokoro returned {sr} Hz, expected {AUDIO_SR_TARGET} Hz — refusing to "
                "resample (any resampler would alter the signal we are trying to keep clean)"
            )
        log(f"raw speech: {audio.size / sr:.3f}s @ {sr} Hz, peak {np.max(np.abs(audio)):.4f}")

    peak_in = float(np.max(np.abs(audio)))
    if peak_in < 1e-5:
        raise RuntimeError("synthesised audio is silent — the TTS model produced no signal")

    with timed("normalize_peak", timings):
        # The entire signal chain: one scalar multiply. Nothing else.
        mastered = (audio / peak_in * PEAK_TARGET).astype(np.float32)
        mastered = np.clip(mastered, -1.0, 1.0)
        log(f"peak-normalised ×{PEAK_TARGET / peak_in:.3f} → "
            f"{20 * math.log10(max(float(np.max(np.abs(mastered))), 1e-10)):.2f} dBFS "
            f"({mastered.size / sr:.3f}s)")

    wav_path = output_dir / VOICE_WAV
    mp3_path = output_dir / VOICE_MP3
    with timed("write_audio", timings):
        sf.write(wav_path, mastered, sr, subtype="PCM_16")
        ffmpeg = ffmpeg_binary()
        cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(wav_path),
               "-c:a", "libmp3lame", "-b:a", "192k", str(mp3_path)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not mp3_path.exists():
            raise RuntimeError(f"mp3 encode failed: {proc.stderr.strip()[-400:]}")
        log(f"wrote {mp3_path.name} ({mp3_path.stat().st_size / 1024:.0f} KB)")

    return {
        "text": SCRIPT_LINE,
        "voice": voice,
        "voice_style_tensor_shape": list(style_shape) if style_shape else None,
        "speed": SPEECH_SPEED,
        "sample_rate": int(sr),
        "duration_s": round(float(mastered.size / sr), 3),
        "peak_dbfs": round(float(20 * math.log10(max(float(np.max(np.abs(mastered))), 1e-10))), 2),
        "processing": "linear peak normalisation only",
        "wsola_time_stretch": False,
        "biquad_or_parametric_eq": False,
        "audio": mastered,
    }


# --------------------------------------------------------------------------- #
# Part B.2 — Real neural video (Stable Video Diffusion XT)
# --------------------------------------------------------------------------- #


def build_video_pipeline(timings: Dict[str, float]):
    """
    Load SVD-XT in fp16 and offload it so a 16 GB T4 can hold all 25 frames.

    Verified against diffusers 0.42: `StableVideoDiffusionPipeline` has no
    `enable_vae_slicing()`, but its VAE (`AutoencoderKLTemporalDecoder`) does
    expose `enable_slicing()`/`enable_tiling()`.
    """
    import torch
    from diffusers import StableVideoDiffusionPipeline

    with timed("load_svd_pipeline", timings):
        pipe = _load_pipeline(StableVideoDiffusionPipeline, SVD_MODEL_ID, "svd")

    with timed("configure_svd_vram", timings):
        _first_available(pipe, ["vae.enable_slicing", "enable_vae_slicing"], "svd vae slicing")
        _first_available(pipe, ["vae.enable_tiling", "enable_vae_tiling"], "svd vae tiling")
        try:
            pipe.enable_model_cpu_offload(device="cuda:0")
            log("svd: model CPU offload enabled (unet streams from host RAM)")
        except Exception as exc:
            log(f"svd: cpu offload unavailable ({exc}); moving fully to GPU", "WARN")
            pipe.to("cuda:0", dtype=torch.float16)
        try:
            pipe.set_progress_bar_config(disable=False, desc="svd")
        except Exception:
            pass

    return pipe


def generate_neural_frames(pipe, image: Image.Image, timings: Dict[str, float]) -> Tuple[List[Image.Image], Dict[str, Any]]:
    """
    Run SVD with a VRAM-recovery ladder.

    Vertical 576x1024 is the native SVD field of view rotated for Reels; if a
    T4 cannot hold a given (frames, chunk) combination the next rung is tried
    instead of failing the run.
    """
    import torch

    width, height = 576, 1024
    ladder: List[Dict[str, int]] = [
        {"num_frames": SVD_NUM_FRAMES, "decode_chunk_size": 4},
        {"num_frames": SVD_NUM_FRAMES, "decode_chunk_size": 2},
        {"num_frames": max(14, SVD_NUM_FRAMES // 2), "decode_chunk_size": 2},
    ]
    last_error: Optional[Exception] = None

    for attempt, rung in enumerate(ladder, start=1):
        frames = rung["num_frames"]
        chunk = rung["decode_chunk_size"]
        label = f"svd_generate_f{frames}_c{chunk}"
        log(f"svd attempt {attempt}/{len(ladder)}: {frames} frames, decode_chunk_size={chunk}")
        try:
            with timed(label, timings):
                result = pipe(
                    image,
                    height=height,
                    width=width,
                    num_frames=frames,
                    num_inference_steps=SVD_STEPS,
                    motion_bucket_id=SVD_MOTION_BUCKET,
                    noise_aug_strength=SVD_NOISE_AUG,
                    decode_chunk_size=chunk,
                    generator=torch.Generator(device="cuda").manual_seed(BASE_SEED + 100),
                )
                out = result.frames[0]
            if not out:
                raise RuntimeError("SVD returned zero frames")
            log(f"svd: {len(out)} neural frames at {out[0].size[0]}x{out[0].size[1]}")
            return list(out), {
                "backend": "stable-video-diffusion-img2vid-xt",
                "num_frames": len(out),
                "num_inference_steps": SVD_STEPS,
                "motion_bucket_id": SVD_MOTION_BUCKET,
                "noise_aug_strength": SVD_NOISE_AUG,
                "decode_chunk_size": chunk,
                "native_size": [out[0].size[0], out[0].size[1]],
                "source_fps": 7,
                "attempt": attempt,
            }
        except Exception as exc:
            last_error = exc
            log(f"svd attempt {attempt} failed ({type(exc).__name__}: {exc})", "ERROR")
            release_memory()

    raise RuntimeError(f"Stable Video Diffusion failed on every rung: {last_error}") from last_error


def retime_to_duration(
    frames: List[Image.Image],
    source_fps: int,
    target_fps: int,
    seconds: float,
) -> Tuple[List[Image.Image], Dict[str, Any]]:
    """
    Fit the neural clip to the audio duration **without** changing its speed.

    Naively replaying SVD's 7 fps output at 25 fps would play Luca 3.5x too
    fast. Instead we:

      1. build a ping-pong cycle (forward, then backward) so the sequence can
         be repeated with no seam -- the frame before the turnaround is
         adjacent to the frame it reverses into;
      2. sample that cycle at the true source rate, i.e. position
         `i / target_fps * source_fps` source frames in, and cross-fade
         between the two neighbouring neural frames.

    Step 2 is *temporal upsampling* of existing neural frames: every pixel in
    the output is a blend of two frames Stable Video Diffusion actually
    generated. Nothing is drawn, warped or synthesised procedurally.
    """
    import torch

    if not frames:
        raise RuntimeError("no frames to retime")

    target_count = max(1, int(round(seconds * target_fps)))

    if len(frames) == 1:
        cycle = list(frames)
    else:
        cycle = list(frames) + list(frames[-2:0:-1])   # ping-pong, no repeat at the join

    cycle_len = len(cycle)
    # Total neural motion available in the cycle, in seconds.
    cycle_seconds = cycle_len / float(source_fps)

    out: List[Image.Image] = []
    for i in range(target_count):
        position = (i / float(target_fps)) * float(source_fps)   # index in source frames
        if cycle_seconds > 0:
            position = position % cycle_len
        low = int(math.floor(position)) % cycle_len
        high = (low + 1) % cycle_len
        frac = position - math.floor(position)

        if frac < 1e-3 or cycle_len == 1:
            out.append(cycle[low])
            continue

        a = torch.from_numpy(np.asarray(cycle[low], dtype=np.float32)).permute(2, 0, 1)[None]
        b = torch.from_numpy(np.asarray(cycle[high], dtype=np.float32)).permute(2, 0, 1)[None]
        blended = (a * (1.0 - frac) + b * frac).clamp(0, 255).round().to(torch.uint8)
        out.append(Image.fromarray(blended[0].permute(1, 2, 0).numpy()))

    info = {
        "source_fps": source_fps,
        "fps": target_fps,
        "ping_pong_cycle": cycle_len,
        "looped_frames": len(out),
        "duration_s": round(len(out) / float(target_fps), 3),
        "loop_mode": "ping-pong",
        "retime": f"cross-fade temporal upsampling {source_fps}→{target_fps} fps (speed preserved)",
    }
    log(
        f"retime: {len(frames)} neural frames → ping-pong cycle {cycle_len} @ {source_fps} fps "
        f"→ {len(out)} frames @ {target_fps} fps ({len(out) / target_fps:.3f}s, speed preserved)"
    )
    return out, info



def encode_video(frames: List[Image.Image], audio: np.ndarray, sr: int, out_path: Path,
                 timings: Dict[str, float], fps: int = SVD_FPS) -> None:
    """Pipe raw RGB into ffmpeg -> libx264 yuv420p + AAC, trimmed to the audio."""
    import soundfile as sf

    if not frames:
        raise RuntimeError("encode_video got no frames")
    width, height = frames[0].size
    tmp_wav = out_path.with_suffix(".tmp.wav")
    sf.write(tmp_wav, audio, sr, subtype="PCM_16")

    ffmpeg = ffmpeg_binary()
    video_codec = ["-c:v", "libx264", "-preset", "medium", "-crf", "19",
                   "-x264-params", "keyint=50:min-keyint=25"]
    if not has_encoder(ffmpeg, "libx264"):
        log("libx264 missing — falling back to mpeg4", "WARN")
        video_codec = ["-c:v", "mpeg4", "-q:v", "2"]

    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
        "-r", str(fps), "-i", "pipe:0",
        "-i", str(tmp_wav),
        *video_codec,
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-c:a", "aac", "-b:a", "160k", "-ar", str(sr), "-ac", "1",
        "-shortest", str(out_path),
    ]
    with timed("encode_video", timings):
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for frame in frames:
                process.stdin.write(np.asarray(frame.convert("RGB"), dtype=np.uint8).tobytes())
        except Exception as exc:
            log(f"frame pipe failed: {exc}", "ERROR")
        finally:
            # MUST always close stdin: an open pipe leaves ffmpeg waiting for
            # more frames forever and the kernel would burn its time budget.
            try:
                process.stdin.close()
            except Exception:
                pass
        stderr = process.stderr.read().decode("utf-8", "replace") if process.stderr else ""
        returncode = process.wait()
    tmp_wav.unlink(missing_ok=True)
    if returncode != 0 or not out_path.exists():
        raise RuntimeError(f"ffmpeg failed ({returncode}): {stderr[-1200:]}")
    log(f"encoded {out_path.name} ({out_path.stat().st_size / 1024:.0f} KB)")


def probe_video(path: Path) -> Dict[str, Any]:
    """
    Describe the deliverable so metrics.json reflects the *real* file.

    imageio-ffmpeg ships no `ffprobe` sibling, so we parse `ffmpeg -i`'s
    banner instead of depending on a tool that may not exist.
    """
    ffmpeg = ffmpeg_binary()
    try:
        proc = subprocess.run([ffmpeg, "-hide_banner", "-i", str(path)],
                              capture_output=True, text=True)
        banner = proc.stderr or ""
        info: Dict[str, Any] = {"size_bytes": path.stat().st_size}
        for line in banner.splitlines():
            line = line.strip()
            if line.startswith("Duration:"):
                info["duration"] = line.split("Duration:")[1].split(",")[0].strip()
            elif "Video:" in line:
                parts = [p.strip() for p in line.split("Video:")[1].split(",")]
                info["video_codec"] = parts[0]
                info["pix_fmt"] = parts[1] if len(parts) > 1 else None
                for part in parts:
                    if "x" in part and part.split("x")[0].isdigit():
                        info["resolution"] = part.split()[0]
                        break
                for part in parts:
                    if part.endswith("fps"):
                        info["fps"] = part.split()[0]
                        break
            elif "Audio:" in line:
                parts = [p.strip() for p in line.split("Audio:")[1].split(",")]
                info["audio_codec"] = parts[0]
                for part in parts:
                    if "Hz" in part:
                        info["sample_rate"] = part.split()[0]
                        break
                info["audio_layout"] = "mono" if "mono" in line else "stereo"
        return info
    except Exception as exc:
        return {"probe_error": f"{type(exc).__name__}: {exc}"}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> int:
    t_start = time.perf_counter()
    output_dir = resolve_output_dir()
    tee = Tee(output_dir / BUILD_LOG, sys.stdout)
    sys.stdout = tee  # type: ignore[assignment]

    metrics: Dict[str, Any] = {
        "phase": "phase-3-luca-moretti",
        "character": "Luca Moretti · 27 · young-millionaire playboy",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "config": {
            "model_id": MODEL_ID,
            "clip_vision_id": CLIP_VISION_ID,
            "ip_adapter_weight": IP_ADAPTER_WEIGHT,
            "ip_adapter_scale": IP_ADAPTER_SCALE,
            "anti_bleed": "spatial attention masking (cross_attention_kwargs.ip_adapter_masks)",
            "base_seed": BASE_SEED,
            "master": {"size": list(MASTER_SIZE), "steps": MASTER_STEPS, "cfg": MASTER_CFG},
            "slides": {"size": list(SLIDE_SIZE), "steps": SLIDE_STEPS, "cfg": SLIDE_CFG,
                       "jpeg_quality": JPEG_QUALITY},
            "svd": {"model_id": SVD_MODEL_ID, "num_frames": SVD_NUM_FRAMES,
                    "steps": SVD_STEPS, "motion_bucket_id": SVD_MOTION_BUCKET,
                    "fps": SVD_FPS, "source_slide": SVD_SOURCE_SLIDE},
            "voice": {"engine": "kokoro-onnx", "voice": VOICE_NAME, "speed": SPEECH_SPEED},
            "dtype": "float16",
            "device": "cuda:0",
            "procedural_animation": False,
            "wsola_time_stretch": False,
            "custom_biquad_filters": False,
        },
        "artifacts": [],
        "timings_seconds": {},
        "slides": [],
        "errors": [],
    }
    timings: Dict[str, float] = metrics["timings_seconds"]

    log("=" * 78)
    log("stoic-ai-studio · PHASE 3 · LUCA MORETTI")
    log(f"output dir: {output_dir}")
    log("=" * 78)

    exit_code = 0
    try:
        ensure_python_deps(timings)
        metrics["environment"] = env_fingerprint()
        log(f"environment: {json.dumps(metrics['environment'], default=str)}")

        # ---------------- Part A.1: identity master ---------------- #
        pipe, device = build_pipeline(timings)

        master_path = output_dir / MASTER_FILENAME
        with timed("generate_master", timings):
            master, master_info = generate_image(
                pipe, device,
                prompt=MASTER_PROMPT,
                width=MASTER_SIZE[0], height=MASTER_SIZE[1],
                seed=BASE_SEED, steps=MASTER_STEPS, cfg=MASTER_CFG,
                label="01_luca_master",
            )
            save_png(master, master_path)
            save_jpeg(master, output_dir / "luca_slide_01.jpg")
        metrics["slides"].append({"filename": MASTER_FILENAME, "role": "identity master",
                                  "status": "ok", **master_info})
        metrics["artifacts"] += [MASTER_FILENAME, "luca_slide_01.jpg"]

        # ---------------- Part A.2: face-locked carousel ---------------- #
        face_locked = attach_face_adapter(pipe, device, timings)
        metrics["ip_adapter_active"] = face_locked
        if not face_locked:
            log("running WITHOUT identity lock — slides will not be consistent", "WARN")

        slide_images: List[Tuple[str, Image.Image]] = []
        by_index: Dict[int, Image.Image] = {}

        for slide in SLIDES:
            seed = BASE_SEED + slide["index"]
            png_path = output_dir / slide["filename"]
            jpg_path = output_dir / f"luca_slide_{slide['index']:02d}.jpg"
            record: Dict[str, Any] = {
                "index": slide["index"],
                "filename": slide["filename"],
                "jpeg": jpg_path.name,
                "title": slide["title"],
                "mask": slide["mask"],
                "expected_people": slide.get("people", 1),
                "status": "pending",
            }
            try:
                with timed(f"generate_slide_{slide['index']:02d}", timings):
                    image, info = generate_image(
                        pipe, device,
                        prompt=slide["prompt"],
                        width=SLIDE_SIZE[0], height=SLIDE_SIZE[1],
                        seed=seed, steps=SLIDE_STEPS, cfg=SLIDE_CFG,
                        ip_adapter_image=master if face_locked else None,
                        mask_kind=slide["mask"],
                        negative_prompt=(
                            NEGATIVE_PROMPT + ", " + slide["negative"]
                            if slide.get("negative")
                            else None
                        ),
                        label=slide["filename"],
                    )
                    save_png(image, png_path)
                    save_jpeg(image, jpg_path)
                record.update(info)
                record["status"] = "ok"
                record["jpeg_kb"] = round(jpg_path.stat().st_size / 1024, 1)
                slide_images.append((slide["title"], image))
                by_index[slide["index"]] = image
                metrics["artifacts"] += [slide["filename"], jpg_path.name]
            except Exception as exc:
                detail = f"{type(exc).__name__}: {exc}"
                log(f"slide {slide['filename']} FAILED — {detail}", "ERROR")
                log(traceback.format_exc(), "ERROR")
                record["status"] = "error"
                record["error"] = detail
                metrics["errors"].append({"filename": slide["filename"], "error": detail})
            metrics["slides"].append(record)

        if not slide_images:
            raise RuntimeError("every carousel slide failed — nothing to assemble")

        sheet_items = [("01 · MASTER (identity reference)", master)] + slide_images
        with timed("build_carousel_sheet", timings):
            sheet_path = output_dir / CAROUSEL_SHEET
            build_carousel_sheet(sheet_items, sheet_path)
            metrics["artifacts"].append(CAROUSEL_SHEET)

        # ---------------- Part B.1: voice ---------------- #
        voice = synth_voice(output_dir, timings)
        audio = voice.pop("audio")
        metrics["voice"] = voice
        metrics["artifacts"] += [VOICE_MP3, VOICE_WAV]
        target_seconds = voice["duration_s"]

        # ---------------- Part B.2: real neural video ---------------- #
        source = by_index.get(int(SVD_SOURCE_SLIDE)) or slide_images[-1][1]
        source_name = SLIDES[int(SVD_SOURCE_SLIDE) - 2]["filename"] if int(SVD_SOURCE_SLIDE) - 2 < len(SLIDES) else "master"
        log(f"video source: slide {SVD_SOURCE_SLIDE} ({source_name}) {source.size[0]}x{source.size[1]}")

        # Free the SDXL pipeline before SVD: two fp16 UNets will not fit in 16 GB.
        log("releasing SDXL pipeline to make room for Stable Video Diffusion")
        try:
            pipe.to("cpu")
        except Exception:
            pass
        del pipe
        release_memory()

        svd_pipe = build_video_pipeline(timings)
        neural_frames, video_info = generate_neural_frames(svd_pipe, source, timings)

        looped, loop_info = retime_to_duration(
            neural_frames, video_info["source_fps"], SVD_FPS, target_seconds
        )
        video_info.update(loop_info)
        video_info["audio_duration_s"] = target_seconds
        video_info["motion_source"] = "100% Stable Video Diffusion (no procedural animation)"

        mp4_path = output_dir / REEL_MP4
        encode_video(looped, audio, voice["sample_rate"], mp4_path, timings, fps=SVD_FPS)
        metrics["artifacts"].append(REEL_MP4)

        with timed("build_video_sheet", timings):
            build_video_sheet(
                looped, output_dir / VIDEO_SHEET,
                f"{len(neural_frames)} SVD frames · ping-pong loop → {len(looped)} frames "
                f"@ {SVD_FPS} fps · {video_info['duration_s']}s audio-matched",
            )
        metrics["artifacts"].append(VIDEO_SHEET)

        with timed("probe_outputs", timings):
            metrics["video"] = {**video_info, "probe": probe_video(mp4_path)}

        try:
            svd_pipe.to("cpu")
        except Exception:
            pass
        del svd_pipe
        release_memory()

        metrics["status"] = "success"

    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        log(f"FATAL — {detail}", "ERROR")
        log(traceback.format_exc(), "ERROR")
        metrics["status"] = "failed"
        metrics["errors"].append({"stage": "fatal", "error": detail})
        exit_code = 1
    finally:
        metrics["total_seconds"] = round(time.perf_counter() - t_start, 1)
        metrics["finished_at"] = datetime.now(timezone.utc).isoformat()
        metrics["artifacts"] = sorted(set(metrics["artifacts"]) | {METRICS_FILENAME, BUILD_LOG})

        metrics_path = output_dir / METRICS_FILENAME
        with open(metrics_path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, default=str)

        log("=" * 78)
        log(f"status: {metrics['status']} · total {metrics['total_seconds']}s")
        for name in metrics["artifacts"]:
            candidate = output_dir / name
            if candidate.exists():
                log(f"  ✓ {name} ({candidate.stat().st_size / 1024:.0f} KB)")
        log(f"artifacts written to {output_dir}")
        log("=" * 78)

        sys.stdout = tee._stream  # type: ignore[assignment]
        tee.close()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
