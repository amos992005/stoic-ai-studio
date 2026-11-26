#!/usr/bin/env python3
"""
stoic-ai-studio · PHASE 3.1 · "LUCA MORETTI" — full architectural upgrade
=========================================================================

One Kaggle run (2x Tesla T4) produces the complete postable pack.

  cuda:0  ── Photo carousel + 3-stage biometric face lock + adetailer
  cuda:1  ── LivePortrait talking avatar + SVD b-roll  (runs in parallel)

--------------------------------------------------------------------------------
WHY THE PREVIOUS RUN DEFORMED HANDS
--------------------------------------------------------------------------------
Attaching `IP-Adapter-Plus-Face` -- which is fed a *head-only* portrait -- during
a full-scene txt2img corrupts the UNet's spatial body/hand layout: the face
reference drags the whole composition towards a head-and-shoulders framing and
the hands are the first thing to collapse. So identity locking moved *after*
generation, into a 3-stage pipeline:

  Stage 1  txt2img with ZERO IP-Adapter. `DPM++ SDE Karras` (SDE, not 2M: the
           stochastic step is what gives real pores and optical texture).
           Explicit biomechanical hand descriptions; hands are never hidden.
  Stage 2  insightface `buffalo_l` detects every face, picks the MALE one
           (gender + rightmost tie-break) and `inswapper_128` transfers Luca's
           bone structure onto it. The woman's face is never touched.
  Stage 3  the swapped face is upscaled to 1024x1024 through SDXL img2img with
           IP-Adapter-Plus-Face at 0.60 / strength 0.27, which preserves the
           swapped bone structure while rebuilding genuine high-res skin, then
           is Gaussian-feather-blended back into the slide.

--------------------------------------------------------------------------------
WHY THE PREVIOUS VIDEO LOOKED LIKE A VIBRATING PHOTO
--------------------------------------------------------------------------------
SVD-XT is an image-to-video model: it adds small camera/motion drift but has no
idea what a mouth is, so a "talking" clip came out as a closed-mouth vibration.
Shot 1 therefore uses `KwaiVGI/LivePortrait`, a dedicated portrait-animation
model: real 3D head rotation from a driving video, blinking, and lips retargeted
through its `retarget_lip` MLP from our own audio envelope.

Shots 2-3 are SVD lifestyle b-roll, played strictly forward-only.

Design rules honoured here
  * $0: Kaggle T4s do the compute, GitHub Actions only orchestrates.
  * Wheels only -- never compile from source on Kaggle's Python 3.13.
  * No procedural pixel-slice animation. No WSOLA. No biquad/EQ on the voice.
  * Deterministic: fixed seeds, private generators.
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
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

MODEL_ID = os.environ.get("SDXL_MODEL_ID", "SG161222/RealVisXL_V5.0")
CLIP_VISION_ID = os.environ.get("CLIP_VISION_ID", "laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
IP_ADAPTER_REPO = os.environ.get("IP_ADAPTER_REPO", "h94/IP-Adapter")
IP_ADAPTER_SUBFOLDER = "sdxl_models"
IP_ADAPTER_WEIGHT = "ip-adapter-plus-face_sdxl_vit-h.safetensors"

#: Stage 1 — pure scene generation (no IP-Adapter attached).
SCENE_STEPS = int(os.environ.get("SCENE_STEPS", "32"))
SCENE_CFG = float(os.environ.get("SCENE_CFG", "5.0"))
MASTER_SIZE = (1024, 1024)
SLIDE_SIZE = (832, 1024)
BASE_SEED = int(os.environ.get("BASE_SEED", "42"))

#: Stage 3 — high-res face adetailer.
ADETAILER_SCALE = 0.60
ADETAILER_STRENGTH = 0.27
ADETAILER_STEPS = int(os.environ.get("ADETAILER_STEPS", "22"))
ADETAILER_SIZE = 1024
FACE_CROP_MULT = 2.0

#: Stage 2 — biometric swap.
FACE_MODEL_ROOT = Path(os.environ.get("FACE_MODEL_ROOT", "/tmp/stoic_face_models"))
INSWAPPER_REPOS: List[Tuple[str, str]] = [
    ("ezioruan/inswapper_128.onnx", "inswapper_128.onnx"),
    ("CountFloyd/deepfake", "inswapper_128.onnx"),
    ("ashleykleynhans/inswapper", "inswapper_128.onnx"),
    ("thebiglaskowski/inswapper_128.onnx", "inswapper_128.onnx"),
]

#: Video.
REEL_FPS = 25
HOOK_FRAMES = int(os.environ.get("HOOK_FRAMES", "60"))     # 2.4 s LivePortrait
BROLL_FRAMES = int(os.environ.get("BROLL_FRAMES", "30"))   # 1.2 s per cutaway
BROLL_SIZE = (576, 1024)
SVD_STEPS = int(os.environ.get("SVD_STEPS", "25"))
SVD_MOTION_BUCKET = int(os.environ.get("SVD_MOTION_BUCKET", "127"))
SVD_NOISE_AUG = 0.02

#: Voice.
TTS_VOICE = os.environ.get("TTS_VOICE", "en-US-AndrewMultilingualNeural")
TTS_RATE = os.environ.get("TTS_RATE", "+3%")
TTS_PITCH = os.environ.get("TTS_PITCH", "-2Hz")
SCRIPT_LINE = (
    "Morning in Monaco, sunset in Ibiza... honestly, the hardest part of today "
    "was choosing the view."
)
AUDIO_SR = 24000
PEAK_TARGET = 0.89
VOICE_GAIN = 0.86
BED_GAIN = 0.30
BPM = 120

JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "92"))

MASTER_FILENAME = "01_luca_master.png"
CAROUSEL_SHEET = "luca_carousel_sheet.png"
VIDEO_SHEET = "luca_video_sheet.png"
VOICE_MP3 = "luca_voice.mp3"
REEL_MP4 = "luca_reel_clip.mp4"

#: Wall-clock budgets. The GitHub job allows 40 min and Kaggle kills a run at
#: 12 h, but the binding constraint is that a *finished* artifact must exist
#: early: the carousel is published first and the reel is strictly optional.
KERNEL_BUDGET_S = int(os.environ.get("KERNEL_BUDGET_S", str(30 * 60)))
CAROUSEL_BUDGET_S = int(os.environ.get("CAROUSEL_BUDGET_S", str(17 * 60)))
REEL_BUDGET_S = int(os.environ.get("REEL_BUDGET_S", str(15 * 60)))
REEL_MIN_BUDGET_S = int(os.environ.get("REEL_MIN_BUDGET_S", str(8 * 60)))
REEL_AWAIT_SLIDES_S = int(os.environ.get("REEL_AWAIT_SLIDES_S", "900"))
#: LivePortrait (module load + 60 frames) and one SVD shot, in seconds.
LIVEPORTRAIT_BUDGET_S = int(os.environ.get("LIVEPORTRAIT_BUDGET_S", str(10 * 60)))
SVD_SHOT_BUDGET_S = int(os.environ.get("SVD_SHOT_BUDGET_S", str(3 * 60)))
METRICS_FILENAME = "luca_metrics.json"
BUILD_LOG = "build.log"

#: Luca, stated once. Kept short here — the scenes carry the biomechanics.
LUCA = (
    "27-year-old handsome man, young Brad Pitt Ocean's Eleven aesthetic, "
    "sun-bleached dirty-blonde tousled hair, piercing blue-green eyes, sharp "
    "jawline, light 3-day stubble"
)

PHOTOREAL = (
    "candid iPhone 16 Pro photo, authentic smartphone sensor look, visible skin "
    "pores and fine skin texture, true-to-life colour, unretouched, natural "
    "available light, realistic anatomy"
)

#: Global negative — note "no sunglasses" is enforced here and in every prompt.
NEGATIVE_PROMPT = (
    # hands / anatomy (the whole point of Stage 1)
    "mutated hands, extra fingers, six fingers, fused fingers, deformed hands, "
    "floating limbs, merged body, extra background heads between subjects, "
    "bad anatomy, bad proportions, disfigured, malformed limbs, "
    # forbidden accessories
    "sunglasses, shades, eyeglasses, "
    # render artefacts
    "CGI, 3d render, plastic skin, airbrushed, oversaturated orange skin, "
    "waxy, doll, cartoon, anime, illustration, painting, low quality, blurry, "
    "jpeg artifacts, watermark, signature, text, logo, frame, border, "
    # identity confusion
    "twins, two men, duplicate person, cloned face, teenage boy, child, old man"
)

SLIDES: List[Dict[str, Any]] = [
    {
        "index": 1,
        "filename": MASTER_FILENAME,
        "title": "01 · MASTER",
        "size": MASTER_SIZE,
        "seed": 42,
        "role": "identity master",
        "people": 1,
        "needs_swap": False,
        "prompt": (
            f"Candid raw iPhone 16 Pro 24mm portrait of a charismatic {LUCA}, "
            "confident subtle smirk, wearing an unbuttoned white linen shirt on "
            "a luxury yacht deck in Monaco harbour, natural Mediterranean "
            "afternoon sunlight, visible skin pores, unretouched, "
            f"{PHOTOREAL}"
        ),
    },
    {
        "index": 2,
        "filename": "02_monaco_yacht_date.png",
        "title": "02 · MONACO · YACHT",
        "size": SLIDE_SIZE,
        "seed": 42 + 2,
        "people": 2,
        "needs_swap": True,
        "prompt": (
            "Candid iPhone 16 Pro photo on a luxury teak yacht sunpad in Monaco "
            "at golden hour, medium two-shot of a gorgeous young brunette woman "
            "in a stylish bikini on the left laughing naturally, and a handsome "
            "athletic 27-year-old man with tousled dirty-blonde hair and light "
            "stubble in an open white linen shirt on the right, his right arm "
            "resting naturally on the teak backrest and his left hand holding a "
            "chilled champagne flute by the stem with five natural fingers, "
            "looking at the camera with a confident smirk, clear blue-green "
            f"eyes, no sunglasses, warm sunset light, {PHOTOREAL}"
        ),
    },
    {
        "index": 3,
        "filename": "03_vip_party_flash.png",
        "title": "03 · IBIZA · VIP",
        "size": SLIDE_SIZE,
        "seed": 42 + 3,
        "people": 2,
        "needs_swap": True,
        "prompt": (
            "Authentic 35mm direct on-camera flash night party photo at an "
            "exclusive Ibiza rooftop VIP table, medium two-shot of a stunning "
            "young blonde woman in a sleek black sequin dress on the left "
            "smiling, and a handsome 27-year-old man with tousled dirty-blonde "
            "hair and light stubble in a fitted black silk shirt unbuttoned at "
            "the collar on the right, one hand resting naturally on the VIP "
            "table holding a crystal rocks glass with ice and amber whiskey "
            "with five natural fingers, looking at the camera with a confident "
            "smirk, clear eyes, no sunglasses, dark night sky bokeh background "
            f"with no extra faces between them, {PHOTOREAL}"
        ),
    },
    {
        "index": 4,
        "filename": "04_porsche_gt3_pov.png",
        "title": "04 · AMALFI · GT3",
        "size": SLIDE_SIZE,
        "seed": 42 + 4,
        "people": 1,
        "needs_swap": True,
        "prompt": (
            "Candid iPhone 16 Pro daylight photo at a coastal cliffside overlook "
            "on the Amalfi coast, medium three-quarter shot of a handsome "
            "athletic 27-year-old man with windblown dirty-blonde hair, sharp "
            "jawline, light stubble, and piercing blue-green eyes, wearing a "
            "fitted white polo shirt and tailored chinos, standing outside "
            "beside the driver door of a parked white Porsche 911 GT3 RS, one "
            "hand resting flat and naturally with five fingers on the car's "
            "sleek front fender and the other hand relaxed by his pocket, "
            "looking directly at the camera with a confident smirk, no "
            f"sunglasses, Mediterranean sea cliffs in background, {PHOTOREAL}"
        ),
    },
    {
        "index": 5,
        "filename": "05_extreme_action.png",
        "title": "05 · EXTREME",
        "size": SLIDE_SIZE,
        "seed": 42 + 5,
        "people": 1,
        "needs_swap": True,
        "prompt": (
            "Candid iPhone 16 Pro action lifestyle photo in Ibiza, medium shot "
            "of an athletic 27-year-old man with wet tousled dirty-blonde hair, "
            "sharp jawline, light stubble, and piercing blue-green eyes sitting "
            "astride a matte-black Yamaha Waverunner jet-ski in crystal-clear "
            "turquoise Mediterranean water, both hands gripping the jet-ski "
            "handlebars naturally with five fingers, water droplets glistening "
            "on his sun-kissed muscular forearms and shirtless torso, looking at "
            "the camera with an effortless smile, no sunglasses, bright natural "
            f"daylight, {PHOTOREAL}"
        ),
    },
]

#: LivePortrait weights: (repo, filename, repo_type, local filename)
LIVEPORTRAIT_WEIGHTS: List[Tuple[str, str]] = [
    ("appearance_feature_extractor.pth", "liveportrait/base_models/appearance_feature_extractor.pth"),
    ("motion_extractor.pth", "liveportrait/base_models/motion_extractor.pth"),
    ("warping_module.pth", "liveportrait/base_models/warping_module.pth"),
    ("spade_generator.pth", "liveportrait/base_models/spade_generator.pth"),
    ("stitching_retargeting_module.pth", "liveportrait/retargeting_models/stitching_retargeting_module.pth"),
]


# --------------------------------------------------------------------------- #
# Logging / timing
# --------------------------------------------------------------------------- #


def log(message: str, level: str = "INFO", tag: str = "") -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    prefix = f"[{stamp}] [{level:<5}]"
    if tag:
        prefix += f" [{tag}]"
    print(f"{prefix} {message}", flush=True)


@contextmanager
def timed(label: str, timings: Dict[str, float], tag: str = ""):
    log(f"▶ start: {label}", tag=tag)
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        key = f"{tag}/{label}" if tag else label
        timings[key] = round(elapsed, 3)
        log(f"✔ done:  {label} ({elapsed:.1f}s)", tag=tag)


def resolve_output_dir() -> Path:
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
    """Mirror stdout into the build log. Line-buffered so a crash keeps context."""

    def __init__(self, path: Path, stream):
        self._stream = stream
        self._file = open(path, "a", encoding="utf-8", buffering=1)

    def write(self, data: str) -> int:
        try:
            self._stream.write(data)
        except Exception:
            pass
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

    def isatty(self) -> bool:  # pragma: no cover
        return False

    def close(self) -> None:
        self.flush()
        try:
            self._file.close()
        except Exception:
            pass


def release_memory(device: Optional[str] = None) -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            if device:
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
            else:
                torch.cuda.empty_cache()
    except Exception:
        pass


def env_fingerprint() -> Dict[str, Any]:
    info: Dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception as exc:
        info["torch_error"] = f"{type(exc).__name__}: {exc}"
    for mod in ("diffusers", "transformers", "accelerate", "safetensors", "PIL",
                "onnxruntime", "insightface", "cv2", "edge_tts"):
        try:
            info[mod] = getattr(__import__(mod), "__version__", "unknown")
        except Exception:
            info[mod] = "unavailable"
    return info


# --------------------------------------------------------------------------- #
# ffmpeg
# --------------------------------------------------------------------------- #


def ffmpeg_binary() -> str:
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


# --------------------------------------------------------------------------- #
# Dependency bootstrap (wheels only)
# --------------------------------------------------------------------------- #


def _pip_install(packages: List[str], timings: Dict[str, float], label: str) -> bool:
    with timed(label, timings):
        for extra in (["--only-binary=:all:"], []):
            cmd = [sys.executable, "-m", "pip", "install", "--quiet", "--no-input",
                   "--disable-pip-version-check", *extra, *packages]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode == 0:
                log(f"installed: {' '.join(packages)}", tag="deps")
                return True
            log(f"pip attempt failed: {proc.stderr.strip()[-600:]}", "WARN", tag="deps")
    return False


def ensure_python_deps(timings: Dict[str, float]) -> None:
    """
    Install everything the kernel needs, in wheels-only order.

    insightface>=2.1 ships a pure-python wheel (verified: no .so/.pyx inside), so
    it installs on Kaggle's Python 3.13 without a compiler. Older pins ship a
    Cython .pyx and would try to build -- hence the floor.
    """
    required = {
        "soundfile": "soundfile",
        "imageio_ffmpeg": "imageio-ffmpeg",
        "cv2": "opencv-python-headless",
        "onnxruntime": "onnxruntime-gpu",
        "edge_tts": "edge-tts",
        "insightface": "insightface>=2.1",
    }
    missing = [pkg for mod, pkg in required.items() if not _importable(mod)]
    if missing:
        log(f"installing: {missing}", tag="deps")
        _pip_install(missing, timings, "pip_install_deps")
    else:
        log("python deps already present", tag="deps")

    # onnxruntime-gpu installs as module `onnxruntime`; if only CPU is present and
    # the GPU package is missing, try once (never fatal — CPU fallback is fine).
    if not _importable("onnxruntime"):
        _pip_install(["onnxruntime"], timings, "pip_install_ort_cpu")

    for mod, pkg in required.items():
        if not _importable(mod):
            log(f"{pkg} still unavailable — the kernel will fall back", "WARN", tag="deps")

    import diffusers

    log(f"diffusers {diffusers.__version__}", tag="deps")
    try:
        assert diffusers.StableVideoDiffusionPipeline is not None
        assert diffusers.image_processor.IPAdapterMaskProcessor is not None
    except Exception:
        _pip_install(["diffusers>=0.31.0"], timings, "pip_install_diffusers")
        import importlib

        importlib.reload(diffusers)


def _importable(module: str) -> bool:
    try:
        __import__(module)
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Stage 1 — scene generation (SDXL, DPM++ SDE Karras, NO IP-Adapter)
# --------------------------------------------------------------------------- #


def build_scene_pipeline(device: str, timings: Dict[str, float], tag: str = "gpu0"):
    """
    RealVisXL V5.0 in fp16 with DPM++ **SDE** Karras.

    `DPMSolverSinglestepScheduler` + `algorithm_type="sde-dpmsolver++"` is the
    stochastic variant: it re-injects noise each step, which is what yields
    believable skin pores and optical texture instead of the waxy look the
    deterministic 2M sampler produced.
    """
    import torch
    from diffusers import DPMSolverSinglestepScheduler, StableDiffusionXLPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available — enable a Kaggle GPU accelerator")

    with timed("load_sdxl", timings, tag):
        pipe = None
        last_error: Optional[Exception] = None
        for kwargs in (
            {"variant": "fp16", "torch_dtype": torch.float16, "use_safetensors": True},
            {"torch_dtype": torch.float16, "use_safetensors": True},
            {"torch_dtype": torch.float16},
        ):
            try:
                log(f"loading {MODEL_ID} ({list(kwargs)})", tag=tag)
                pipe = StableDiffusionXLPipeline.from_pretrained(MODEL_ID, **kwargs)
                break
            except Exception as exc:
                last_error = exc
                log(f"load attempt failed: {type(exc).__name__}: {exc}", "WARN", tag=tag)
        if pipe is None:
            raise RuntimeError(f"could not load {MODEL_ID}: {last_error}") from last_error

    with timed("configure_pipeline", timings, tag):
        try:
            pipe.scheduler = DPMSolverSinglestepScheduler.from_config(
                pipe.scheduler.config,
                use_karras_sigmas=True,
                algorithm_type="sde-dpmsolver++",
            )
            log("scheduler: DPM++ SDE Karras (singlestep, sde-dpmsolver++)", tag=tag)
        except Exception as exc:
            log(f"scheduler swap failed, keeping default: {exc}", "WARN", tag=tag)

        pipe.to(device)
        for name in ("vae.enable_slicing", "enable_vae_slicing", "vae.enable_tiling"):
            try:
                target = pipe
                for part in name.split(".")[:-1]:
                    target = getattr(target, part)
                getattr(target, name.split(".")[-1])()
                log(f"applied {name}()", tag=tag)
                break
            except Exception:
                continue

        # Attention slicing stays OFF: it swaps in SlicedAttnProcessor, which
        # diffusers cannot re-instantiate when converting the UNet for IP-Adapter
        # in Stage 3.
        log("attention slicing disabled (needed by Stage-3 IP-Adapter)", tag=tag)

        if getattr(pipe, "vae", None) is not None:
            pipe.vae.to(device=device, dtype=torch.float16)
    return pipe


def generate_scene(
    pipe,
    device: str,
    *,
    prompt: str,
    width: int,
    height: int,
    seed: int,
    label: str,
    steps: int = SCENE_STEPS,
    cfg: float = SCENE_CFG,
    negative: Optional[str] = None,
) -> Image.Image:
    import torch

    kwargs: Dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": negative or NEGATIVE_PROMPT,
        "width": width,
        "height": height,
        "num_inference_steps": steps,
        "guidance_scale": cfg,
        "generator": torch.Generator(device=device).manual_seed(seed),
    }
    log(f"txt2img {label}: {width}x{height} seed={seed} steps={steps} cfg={cfg} "
        f"ip_adapter=OFF", tag="gpu0")

    def _run():
        with torch.inference_mode():
            return pipe(**kwargs)

    try:
        result = _run()
    except RuntimeError as exc:
        if "should be the same" not in str(exc):
            raise
        log(f"dtype mismatch at decode ({exc}); aligning VAE", "WARN", tag="gpu0")
        pipe.vae.to(device=pipe.unet.device, dtype=pipe.unet.dtype)
        release_memory(device)
        result = _run()

    if not result or not getattr(result, "images", None):
        raise RuntimeError(f"pipeline returned no images for {label}")
    return result.images[0]


# --------------------------------------------------------------------------- #
# Stage 2 — biometric face detection + inswapper_128
# --------------------------------------------------------------------------- #


def _hf_token() -> Optional[str]:
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(key):
            return os.environ[key]
    return None


def stage_face_models(timings: Dict[str, float]) -> Dict[str, Path]:
    """
    Put the insightface packages and inswapper where both GPUs can find them.

    insightface resolves `root/models/<name>` and downloads from its own CDN.
    We download here, once, into a shared root with a short retry ladder, and
    then pass that directory explicitly (FaceAnalysis uses a directory verbatim
    when the path exists) so neither worker needs network access.
    """
    import urllib.request

    FACE_MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    def _download(url: str, dest: Path) -> bool:
        if dest.exists() and dest.stat().st_size > 1024:
            return True
        tmp = dest.with_suffix(dest.suffix + ".part")
        req = urllib.request.Request(url, headers={"User-Agent": "stoic-ai-studio/3.1"})
        try:
            with urllib.request.urlopen(req, timeout=300) as resp, open(tmp, "wb") as fh:
                shutil.copyfileobj(resp, fh)
            tmp.replace(dest)
            return dest.stat().st_size > 1024
        except Exception as exc:
            log(f"  ✗ {url.split('/')[-1]}: {type(exc).__name__}: {exc}", "WARN", tag="face")
            tmp.unlink(missing_ok=True)
            return False

    out: Dict[str, Path] = {}

    # ---- buffalo_l (detection + genderage; both release asset layouts) ----
    with timed("fetch_buffalo_l", timings, "face"):
        pkg_dir = FACE_MODEL_ROOT / "models" / "buffalo_l"
        pkg_dir.mkdir(parents=True, exist_ok=True)

        if not (pkg_dir / "det_10g.onnx").exists() or not (pkg_dir / "genderage.onnx").exists():
            for asset in ("buffalo_l.zip", "buffalo_l"):
                url = (f"https://github.com/deepinsight/insightface/releases/download/"
                       f"v0.7/{asset}")
                if _download(url, FACE_MODEL_ROOT / "_buffalo_l.zip"):
                    import zipfile

                    try:
                        with zipfile.ZipFile(FACE_MODEL_ROOT / "_buffalo_l.zip") as zf:
                            zf.extractall(pkg_dir)
                        for f in pkg_dir.glob("*.onnx"):
                            log(f"  ✓ {f.name} ({f.stat().st_size / 1e6:.1f} MB)", tag="face")
                        break
                    except Exception as exc:
                        log(f"  zip extract failed: {exc}", "WARN", tag="face")
                else:
                    log(f"  release asset {asset} unavailable", "WARN", tag="face")

            # Mirror fallback (single files, no zip).
            if not (pkg_dir / "det_10g.onnx").exists():
                mirror = "https://huggingface.co/immich-app/buffalo_l/resolve/main"
                for fname in ("det_10g.onnx", "genderage.onnx", "2d106det.onnx",
                              "w600k_r50.onnx", "1k3d68.onnx"):
                    _download(f"{mirror}/{fname}", pkg_dir / fname)

        out["buffalo_l"] = pkg_dir
        present = sorted(p.name for p in pkg_dir.glob("*.onnx"))
        log(f"buffalo_l models: {present}", tag="face")
        if "det_10g.onnx" not in present:
            raise RuntimeError(
                "buffalo_l detection model (det_10g.onnx) could not be downloaded from "
                "the insightface release assets or the HF mirror"
            )

    # ---- inswapper_128 ----
    with timed("fetch_inswapper", timings, "face"):
        swapper_path = FACE_MODEL_ROOT / "inswapper_128.onnx"
        if not (swapper_path.exists() and swapper_path.stat().st_size > 10_000_000):
            got = False
            try:
                from huggingface_hub import hf_hub_download
            except Exception:
                hf_hub_download = None  # type: ignore

            if hf_hub_download is not None:
                for repo, fname in INSWAPPER_REPOS:
                    for token in ({}, {"token": _hf_token()}) if _hf_token() else ({},):
                        try:
                            path = hf_hub_download(repo_id=repo, filename=fname, **token)
                            shutil.copyfile(path, swapper_path)
                            log(f"  ✓ inswapper from {repo} "
                                f"({swapper_path.stat().st_size / 1e6:.1f} MB)", tag="face")
                            got = True
                            break
                        except Exception as exc:
                            log(f"  ✗ {repo}: {type(exc).__name__}: "
                                f"{str(exc).splitlines()[0][:100]}", "WARN", tag="face")
                    if got:
                        break
            if not got:
                for repo, fname in INSWAPPER_REPOS:
                    if _download(f"https://huggingface.co/{repo}/resolve/main/{fname}",
                                 swapper_path):
                        log(f"  ✓ inswapper via direct URL ({repo})", tag="face")
                        got = True
                        break
            if not got:
                raise RuntimeError(
                    "inswapper_128.onnx could not be obtained from any of "
                    f"{[r for r, _ in INSWAPPER_REPOS]}"
                )
        out["inswapper"] = swapper_path

    return out


class FaceEngine:
    """
    insightface analysis + inswapper, wrapped for our two jobs:

      * `face_locks()` -> 512-d embeddings, so identity can be *measured*
      * `swap_male()`  -> replace Luca's face and nothing else

    Male selection: buffalo_l's genderage head gives `face.sex`. In a two-person
    slide the woman is never modified, so a face that scores female is excluded
    outright; remaining ties break rightmost (where the prompts put Luca).
    """

    def __init__(self, models: Dict[str, Path], device_id: int, timings: Dict[str, float],
                 tag: str = "gpu0", use_cuda: bool = True):
        self.models = models
        self.device_id = device_id
        self.tag = tag
        self.timings = timings
        self.use_cuda = use_cuda
        self.app = None
        self.swapper = None

    def _providers(self) -> List[str]:
        """
        Prefer CUDA, with a graceful CPU fallback.

        Kaggle's `onnxruntime` build has no CUDA provider at all and reports its
        providers with a "-gpu" suffix (`CPUExecutionProvider-gpu`), so an exact
        membership test against "CPUExecutionProvider" silently failed and we
        never asked for CUDA even when onnxruntime-gpu was installed.
        """
        try:
            import onnxruntime as ort

            available = list(ort.get_available_providers())
        except Exception as exc:
            log(f"onnxruntime unavailable: {exc}", "WARN", tag=self.tag)
            return ["CPUExecutionProvider"]

        wanted: List[str] = []
        if self.use_cuda and any("CUDA" in p for p in available):
            wanted.append(next(p for p in available if "CUDA" in p))
        wanted.append("CPUExecutionProvider")
        log(f"onnxruntime providers: available={available} → using={wanted}", tag=self.tag)
        return wanted

    def load(self) -> None:
        from insightface.app import FaceAnalysis

        with timed("load_insightface", self.timings, self.tag):
            providers = self._providers()
            log(f"providers: {providers}", tag=self.tag)
            self.app = FaceAnalysis(
                name=str(self.models["buffalo_l"]),
                providers=providers,
                allowed_modules=["detection", "genderage", "recognition"],
            )
            self.app.prepare(ctx_id=self.device_id, det_size=(640, 640), det_thresh=0.35)

        # NOTE: the swapper is instantiated lazily in swap_male() via
        # insightface.model_zoo.get_model — never as a bare InferenceSession.
        with timed("check_inswapper", self.timings, self.tag):
            size_mb = self.models["inswapper"].stat().st_size / 1024 ** 2
            log(f"inswapper_128.onnx present ({size_mb:.0f} MB)", tag=self.tag)

    # -- detection ---------------------------------------------------------- #

    def detect(self, image_rgb: np.ndarray) -> List[Any]:
        import cv2

        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        return self.app.get(bgr)

    def pick_male(self, faces: List[Any]) -> Tuple[Optional[Any], Optional[int]]:
        """
        The male face + its index, preferring the rightmost candidate.

        Returns (face, index) explicitly: `list.index(face)` compares numpy
        arrays inside Face and raises "truth value of an array is ambiguous".
        """
        if not faces:
            return None, None
        indexed = list(enumerate(faces))
        males = [(i, f) for i, f in indexed if getattr(f, "sex", None) == "M"]
        if not males:
            # genderage is unsure on stylised faces — fall back to the rightmost
            # face, which the prompts always place Luca on.
            log("no face scored male; falling back to rightmost face", "WARN", tag=self.tag)
            males = indexed
        idx, face = max(males, key=lambda pair: pair[1].bbox[0] + pair[1].bbox[2])
        return face, idx

    @staticmethod
    def face_box(face, width: int, height: int, mult: float = FACE_CROP_MULT
                 ) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = [float(v) for v in face.bbox[:4]]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        half = max(x2 - x1, y2 - y1) * mult / 2.0
        return (int(max(0, cx - half)), int(max(0, cy - half)),
                int(min(width, cx + half)), int(min(height, cy + half)))

    # -- swap --------------------------------------------------------------- #

    def swap_male(
        self,
        scene_rgb: np.ndarray,
        source_face: Any,
        faces: List[Any],
    ) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]], Dict[str, Any]]:
        """
        Transfer `source_face` onto the detected male face only.

        Returns the swapped image, the male bbox (for the Stage-3 crop) and a
        record of what was touched.
        """
        from insightface.model_zoo import get_model

        height, width = scene_rgb.shape[:2]
        info: Dict[str, Any] = {
            "faces_detected": len(faces),
            "genders": [getattr(f, "sex", "?") for f in faces],
            "male_index": None,
            "swapped": False,
        }
        male, male_index = self.pick_male(faces)
        info["male_index"] = male_index
        if male is None:
            log("no face detected — swap skipped", "WARN", tag=self.tag)
            return scene_rgb, None, info

        bbox = self.face_box(male, width, height)
        info["male_bbox"] = list(bbox)

        # ALWAYS go through insightface's INSwapper wrapper: it owns the FFHQ
        # alignment templates and calls `session.run(...)`. A raw
        # onnxruntime.InferenceSession has no `.get()`, which is exactly how the
        # first attempt lost every swap.
        if self.swapper is None:
            self.swapper = get_model(str(self.models["inswapper"]),
                                     providers=self._providers())
            log(f"inswapper model: {type(self.swapper).__name__}", tag=self.tag)

        out = np.asarray(self.swapper.get(scene_rgb.copy(), male, source_face,
                                         paste_back=True))
        info["swapped"] = True
        log(f"inswapper → male idx={male_index} bbox={bbox} genders={info['genders']}",
            tag=self.tag)
        return out, bbox, info

    # -- verification ------------------------------------------------------- #

    def embedding(self, image_rgb: np.ndarray) -> Optional[np.ndarray]:
        faces = self.detect(image_rgb)
        if not faces:
            return None
        face, _ = self.pick_male(faces)
        face = face if face is not None else faces[0]
        emb = getattr(face, "normed_embedding", None)
        return None if emb is None else np.asarray(emb, dtype=np.float32)


def cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None:
        return None
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return None
    return round(float(np.dot(a, b) / (na * nb)), 4)


# --------------------------------------------------------------------------- #
# Stage 3 — 1024px face adetailer (SDXL img2img + IP-Adapter-Plus-Face)
# --------------------------------------------------------------------------- #


def attach_adetailer_adapter(pipe, device: str, timings: Dict[str, float],
                             tag: str = "gpu0") -> bool:
    import torch
    from transformers import CLIPVisionModelWithProjection

    log(f"attaching IP-Adapter-Plus-Face @ {ADETAILER_SCALE}", tag=tag)
    try:
        with timed("load_clip_vision", timings, tag):
            encoder = None
            for kwargs in ({"dtype": torch.float16, "use_safetensors": True},
                           {"torch_dtype": torch.float16, "use_safetensors": True},
                           {"dtype": torch.float16}, {}):
                try:
                    encoder = CLIPVisionModelWithProjection.from_pretrained(CLIP_VISION_ID, **kwargs)
                    break
                except Exception as exc:
                    log(f"clip-vision attempt failed: {type(exc).__name__}", "WARN", tag=tag)
            if encoder is None:
                raise RuntimeError(f"could not load {CLIP_VISION_ID}")
            pipe.image_encoder = encoder.to(device, dtype=torch.float16).eval()

        with timed("load_ip_adapter", timings, tag):
            try:
                pipe.load_ip_adapter(IP_ADAPTER_REPO, subfolder=IP_ADAPTER_SUBFOLDER,
                                     weight_name=IP_ADAPTER_WEIGHT)
            except TypeError as exc:
                if "SlicedAttnProcessor" not in str(exc):
                    raise
                log("attention slicing blocks IP-Adapter; resetting processors", "WARN", tag=tag)
                try:
                    from diffusers.models.attention_processor import AttnProcessor2_0

                    pipe.unset_attention_slicing()
                    pipe.unet.set_attn_processor(AttnProcessor2_0())
                except Exception as inner:
                    log(f"processor reset failed: {inner}", "WARN", tag=tag)
                pipe.load_ip_adapter(IP_ADAPTER_REPO, subfolder=IP_ADAPTER_SUBFOLDER,
                                     weight_name=IP_ADAPTER_WEIGHT)
            pipe.set_ip_adapter_scale(ADETAILER_SCALE)
        log("adetailer adapter ready", tag=tag)
        return True
    except Exception as exc:
        log(f"adetailer adapter FAILED — {type(exc).__name__}: {exc}", "ERROR", tag=tag)
        log(traceback.format_exc(), "ERROR", tag=tag)
        return False


def build_img2img(pipe, timings: Dict[str, float], tag: str = "gpu0"):
    """Reuse the txt2img components as an img2img pipeline (no second load)."""
    from diffusers import StableDiffusionXLImg2ImgPipeline

    with timed("build_img2img", timings, tag):
        img2img = StableDiffusionXLImg2ImgPipeline(**pipe.components)
        log("img2img pipeline built from shared components", tag=tag)
    return img2img


def adetailer_face(
    img2img,
    slide_rgb: np.ndarray,
    bbox: Tuple[int, int, int, int],
    master_img: Image.Image,
    device: str,
    seed: int,
    tag: str = "gpu0",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Refine the swapped face at 1024x1024 and feather it back in.

    `strength=0.27` is the load-bearing number: low enough that the biometric
    bone structure from inswapper survives, high enough that SDXL rebuilds real
    pores, stubble and catchlights at four times the swap resolution.
    """
    import torch

    base = Image.fromarray(slide_rgb)
    width, height = base.size
    x1, y1, x2, y2 = bbox
    if x2 - x1 < 48 or y2 - y1 < 48:
        log("face crop too small for adetailer — skipped", "WARN", tag=tag)
        return slide_rgb, {"adetailer": False, "reason": "crop too small"}

    crop = base.crop((x1, y1, x2, y2)).resize((ADETAILER_SIZE, ADETAILER_SIZE), Image.LANCZOS)

    try:
        with torch.inference_mode():
            result = img2img(
                prompt=(
                    f"Extreme close-up portrait of {LUCA}, confident subtle smirk, "
                    "razor-sharp realistic skin pores, individual stubble hairs, "
                    "crisp eye catchlights, natural skin tone, photorealistic, "
                    "unretouched iPhone 16 Pro portrait"
                ),
                negative_prompt=NEGATIVE_PROMPT,
                image=crop,
                strength=ADETAILER_STRENGTH,
                num_inference_steps=ADETAILER_STEPS,
                guidance_scale=SCENE_CFG,
                ip_adapter_image=master_img,
                generator=torch.Generator(device=device).manual_seed(seed + 1000),
            )
    except Exception as exc:
        log(f"adetailer sample FAILED ({type(exc).__name__}: {exc})", "ERROR", tag=tag)
        log(traceback.format_exc(), "ERROR", tag=tag)
        return slide_rgb, {"adetailer": False, "reason": f"{type(exc).__name__}"}

    refined = result.images[0].resize((x2 - x1, y2 - y1), Image.LANCZOS)

    # Elliptical Gaussian feather: fully opaque in the middle of the face, fading
    # to nothing at the crop edge, so there is no visible seam on skin or hair.
    cw, ch = x2 - x1, y2 - y1
    sigma = max(4.0, cw * 0.12)
    yy, xx = np.mgrid[0:ch, 0:cw].astype(np.float32)
    cx, cy = (cw - 1) / 2.0, (ch - 1) / 2.0
    # normalised elliptical radius, squashed slightly on x (faces are taller)
    r = np.sqrt(((xx - cx) / (cw * 0.46)) ** 2 + ((yy - cy) / (ch * 0.46)) ** 2)
    mask = np.clip(1.0 - (r ** 2), 0.0, 1.0)
    mask = np.exp(-((r ** 2.4)) * 2.2).astype(np.float32)
    mask = np.asarray(Image.fromarray((mask * 255).astype(np.uint8))
                      .filter(ImageFilter.GaussianBlur(sigma * 0.16)), dtype=np.float32) / 255.0
    mask3 = mask[..., None]

    base_arr = np.asarray(base, dtype=np.float32).copy()
    region = base_arr[y1:y2, x1:x2]
    region[...] = np.asarray(refined, dtype=np.float32) * mask3 + region * (1.0 - mask3)
    base_arr[y1:y2, x1:x2] = region

    return np.clip(base_arr, 0, 255).astype(np.uint8), {
        "adetailer": True,
        "strength": ADETAILER_STRENGTH,
        "ip_adapter_scale": ADETAILER_SCALE,
        "crop": [x1, y1, x2, y2],
        "blend": f"elliptical gaussian feather, sigma≈{sigma:.1f}px",
    }


def optical_polish(image: Image.Image, seed: int = 0) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Make hands, skin and background share one sensor signature.

    Two steps only: pull 5% out of the orange/yellow cast the SDXL checkpoint
    likes, then add a whisper of photographic grain (sigma 1.3) so the
    upscaled face does not sit on a cleaner background than everything else.
    """
    arr = np.asarray(image, dtype=np.float32)
    # blue lift / warm pull, 5% toward neutral
    arr[..., 2] = np.clip(arr[..., 2] * 1.05, 0, 255)
    arr[..., 0] = np.clip(arr[..., 0] * 0.985, 0, 255)
    arr[..., 1] = np.clip(arr[..., 1] * 0.99, 0, 255)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.3, arr.shape[:2])[..., None]
    arr = arr + noise
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)), {
        "colour": "warm cast reduced 5% (blue ×1.05)",
        "grain_sigma": 1.3,
    }


# --------------------------------------------------------------------------- #
# Sheets
# --------------------------------------------------------------------------- #


def _font(size: int, bold: bool = False):
    names = ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
             "LiberationSans-Bold.ttf" if bold else "LiberationSans-Regular.ttf")
    for name in names:
        for root in ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/liberation"):
            try:
                return ImageFont.truetype(f"{root}/{name}", size)
            except Exception:
                continue
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def build_carousel_sheet(items: List[Tuple[str, Image.Image]], path: Path) -> None:
    pad, label_h, header_h, thumb_h = 24, 46, 82, 560
    thumbs = []
    for title, img in items:
        scale = thumb_h / img.height
        thumbs.append((title, img.resize((max(1, int(img.width * scale)), thumb_h), Image.LANCZOS)))
    total_w = pad + sum(t.width + pad for _, t in thumbs)
    sheet = Image.new("RGB", (total_w, header_h + thumb_h + label_h + pad), (16, 16, 18))
    draw = ImageDraw.Draw(sheet)
    draw.text((pad, 18), "LUCA MORETTI · Monaco & Ibiza Weekend",
              font=_font(30, bold=True), fill=(240, 240, 245))
    draw.text((pad, 52), "RealVisXL V5.0 · DPM++ SDE Karras · inswapper_128 · "
                         "1024px adetailer @ 0.27",
              font=_font(17), fill=(150, 152, 160))
    x = pad
    for title, thumb in thumbs:
        sheet.paste(thumb, (x, header_h))
        draw.rectangle([x, header_h, x + thumb.width - 1, header_h + thumb.height - 1],
                       outline=(58, 58, 64))
        draw.text((x + 2, header_h + thumb.height + 12), title,
                  font=_font(19, bold=True), fill=(225, 226, 232))
        x += thumb.width + pad
    sheet.save(path, format="PNG", optimize=True)
    log(f"wrote {path.name} ({path.stat().st_size / 1024:.0f} KB)")


def build_video_sheet(frames: List[Image.Image], path: Path, subtitle: str,
                      boundaries: Optional[List[int]] = None) -> None:
    pad, header_h, label_h, thumb_h = 14, 74, 34, 420
    count = min(8, len(frames))
    picks = [round(i * (len(frames) - 1) / max(1, count - 1)) for i in range(count)]
    thumbs = []
    for idx in picks:
        img = frames[idx]
        scale = thumb_h / img.height
        thumbs.append((idx, img.resize((max(1, int(img.width * scale)), thumb_h), Image.LANCZOS)))
    total_w = pad + sum(t.width + pad for _, t in thumbs)
    sheet = Image.new("RGB", (total_w, header_h + thumb_h + label_h), (16, 16, 18))
    draw = ImageDraw.Draw(sheet)
    draw.text((pad, 16), "LUCA · hybrid reel (LivePortrait hook + SVD b-roll)",
              font=_font(27, bold=True), fill=(240, 240, 245))
    draw.text((pad, 48), subtitle, font=_font(16), fill=(150, 152, 160))
    x = pad
    for idx, thumb in thumbs:
        sheet.paste(thumb, (x, header_h))
        draw.rectangle([x, header_h, x + thumb.width - 1, header_h + thumb.height - 1],
                       outline=(58, 58, 64))
        shot = "hook" if boundaries and idx < boundaries[0] else (
            "b-roll 1" if boundaries and idx < boundaries[1] else "b-roll 2")
        draw.text((x + 2, header_h + thumb.height + 8), f"f{idx + 1} · {shot}",
                  font=_font(15), fill=(200, 202, 210))
        x += thumb.width + pad
    sheet.save(path, format="PNG", optimize=True)
    log(f"wrote {path.name} ({path.stat().st_size / 1024:.0f} KB)")


# --------------------------------------------------------------------------- #
# Audio: edge-tts voice + synthesised 120 BPM bed
# --------------------------------------------------------------------------- #


def synth_voice(output_dir: Path, timings: Dict[str, float]) -> Dict[str, Any]:
    """
    edge-tts -> wav. The voice path is: decode, peak-normalise, mix. Nothing else.

    No time-stretch, no EQ, no compressor. The bed is generated separately and
    mixed underneath; the voice itself is never spectrally altered.
    """
    import asyncio

    import soundfile as sf

    ffmpeg = ffmpeg_binary()
    raw_mp3 = output_dir / "_voice_raw.mp3"
    wav_path = output_dir / "luca_voice.wav"
    mp3_path = output_dir / VOICE_MP3

    with timed("edge_tts", timings, "av"):
        import edge_tts

        async def _run():
            communicate = edge_tts.Communicate(SCRIPT_LINE, TTS_VOICE,
                                               rate=TTS_RATE, pitch=TTS_PITCH)
            await communicate.save(str(raw_mp3))

        asyncio.run(_run())
        if not raw_mp3.exists() or raw_mp3.stat().st_size < 2000:
            raise RuntimeError("edge-tts produced no audio (network blocked?)")
        log(f"edge-tts ok: {raw_mp3.stat().st_size / 1024:.0f} KB "
            f"voice={TTS_VOICE} rate={TTS_RATE} pitch={TTS_PITCH}", tag="av")

    with timed("decode_voice", timings, "av"):
        proc = subprocess.run(
            [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(raw_mp3),
             "-ac", "1", "-ar", str(AUDIO_SR), "-f", "wav", str(wav_path)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"voice decode failed: {proc.stderr[-500:]}")
        audio, sr = sf.read(wav_path, dtype="float32")
        audio = np.asarray(audio, dtype=np.float32).squeeze()
        raw_mp3.unlink(missing_ok=True)

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak < 1e-4:
        raise RuntimeError("decoded voice is silent")
    # The ONLY processing on the voice.
    voice = (audio / peak * PEAK_TARGET).astype(np.float32)
    log(f"voice: {voice.size / sr:.3f}s @ {sr} Hz, peak-normalised "
        f"{20 * math.log10(max(peak, 1e-9)):.1f} → "
        f"{20 * math.log10(PEAK_TARGET):.2f} dBFS", tag="av")

    sf.write(wav_path, voice, sr, subtype="PCM_16")
    proc = subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                           "-i", str(wav_path), "-c:a", "libmp3lame", "-b:a", "192k",
                           str(mp3_path)], capture_output=True, text=True)
    if proc.returncode != 0:
        log(f"mp3 encode failed: {proc.stderr[-300:]}", "WARN", tag="av")

    return {
        "engine": "edge-tts",
        "voice": TTS_VOICE,
        "rate": TTS_RATE,
        "pitch": TTS_PITCH,
        "text": SCRIPT_LINE,
        "sample_rate": int(sr),
        "duration_s": round(float(voice.size / sr), 3),
        "peak_dbfs": round(float(20 * math.log10(PEAK_TARGET)), 2),
        "processing": "decode + linear peak normalisation (voice untouched otherwise)",
        "wsola_time_stretch": False,
        "biquad_or_parametric_eq": False,
        "samples": voice,
    }


def speech_envelope(voice: np.ndarray, sr: int, n_frames: int, fps: int = REEL_FPS
                    ) -> Dict[str, np.ndarray]:
    """
    Frame-synchronous acoustic articulation curve.

    `mouth_open` is the RMS of the 300-3400 Hz vocal band (measured with an FFT
    band mask, so it tracks articulation rather than bass rumble), normalised to
    its own 98th percentile and smoothed. Also returns broadband RMS.
    """
    frame_len = int(round(sr / fps))
    band_lo, band_hi = 300.0, 3400.0

    mouth = np.zeros(n_frames, dtype=np.float32)
    energy = np.zeros(n_frames, dtype=np.float32)
    window = np.hanning(frame_len).astype(np.float32)
    freqs = np.fft.rfftfreq(frame_len, d=1.0 / sr)
    band = (freqs >= band_lo) & (freqs <= band_hi)

    for i in range(n_frames):
        start = i * frame_len
        chunk = voice[start:start + frame_len]
        if chunk.size < frame_len:
            chunk = np.pad(chunk, (0, frame_len - chunk.size))
        spectrum = np.abs(np.fft.rfft(chunk * window))
        mouth[i] = float(np.sqrt(np.mean(spectrum[band] ** 2)))
        energy[i] = float(np.sqrt(np.mean(chunk ** 2)))

    def _norm(x: np.ndarray) -> np.ndarray:
        hi = float(np.percentile(x, 98)) if x.size else 0.0
        if hi <= 1e-9:
            return np.zeros_like(x)
        return np.clip(x / hi, 0.0, 1.0)

    curve = _norm(mouth)
    # gentle attack/release smoothing so the jaw does not chatter
    smoothed = np.zeros_like(curve)
    attack, release = 0.55, 0.28
    for i, value in enumerate(curve):
        prev = smoothed[i - 1] if i else 0.0
        coeff = attack if value > prev else release
        smoothed[i] = prev + coeff * (value - prev)

    return {"mouth_open": smoothed.astype(np.float32), "rms": _norm(energy).astype(np.float32)}


def _spectral_noise(n: int, sr: int, lo: float, hi: float, rng: np.random.Generator,
                    tilt: float = 0.0) -> np.ndarray:
    """Band-limited noise via an FFT mask (no biquad anywhere)."""
    noise = rng.standard_normal(n)
    spec = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    mask = ((freqs >= lo) & (freqs <= hi)).astype(np.float32)
    if tilt:
        mask *= np.clip((np.maximum(freqs, 1e-6) / max(hi, 1e-6)) ** tilt, 0.0, 1.0)
    out = np.fft.irfft(spec * mask, n=n)
    peak = np.max(np.abs(out)) if out.size else 0.0
    return out / peak if peak > 0 else out


def build_music_bed(n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    """
    Stereo 120 BPM Ibiza deep-house groove + a soft Mediterranean breeze.

    Synthesised additively from oscillators and FFT-shaped noise, then mixed.
    Nothing here touches the voice; it is a separate signal.
    """
    beat = 60.0 / BPM
    t = np.arange(n, dtype=np.float32) / sr
    left = np.zeros(n, dtype=np.float32)
    right = np.zeros(n, dtype=np.float32)

    def place(signal: np.ndarray, at_sec: float, gain: float, pan: float = 0.0) -> None:
        start = int(at_sec * sr)
        if start >= n:
            return
        end = min(n, start + signal.size)
        seg = signal[: end - start] * gain
        left[start:end] += seg * (1.0 - max(0.0, pan))
        right[start:end] += seg * (1.0 + min(0.0, pan))

    # --- kick: 4 on the floor, pitch-swept sine ---
    kick_len = int(0.28 * sr)
    kt = np.arange(kick_len) / sr
    sweep = 150.0 * np.exp(-kt * 22.0) + 46.0
    kick = np.sin(2 * np.pi * np.cumsum(sweep) / sr) * np.exp(-kt * 9.0)
    for b in range(int(n / sr / beat) + 1):
        place(kick.astype(np.float32), b * beat, 0.85)

    # --- sub bass: root movement across a 4-bar loop (Am - F - C - G) ---
    roots = [55.0, 43.65, 65.41, 49.0]
    bar = 4 * beat
    for bar_i in range(int(n / sr / bar) + 1):
        root = roots[bar_i % len(roots)]
        seg_len = int(bar * sr)
        if seg_len <= 0:
            continue
        st = np.arange(seg_len) / sr
        env = np.exp(-((st % beat) * 2.4))
        sub = np.sin(2 * np.pi * root * st) * env * 0.5
        place(sub.astype(np.float32), bar_i * bar, 0.36)

    # --- chord stabs on the offbeat, minor 7ths ---
    for bar_i in range(int(n / sr / bar) + 1):
        root = roots[bar_i % len(roots)]
        for step in range(8):
            at = bar_i * bar + step * beat / 2.0 + beat / 4.0
            stab_len = int(0.16 * sr)
            st = np.arange(stab_len) / sr
            env = np.exp(-st * 26.0)
            tone = np.zeros(stab_len, dtype=np.float32)
            for mult in (2.0, 2.378, 2.997, 3.564):   # root, m3, 5th, m7
                tone += np.sin(2 * np.pi * root * mult * st).astype(np.float32)
            place((tone * env * 0.10).astype(np.float32), at, 0.34,
                  pan=0.55 if step % 2 else -0.55)

    # --- hats: 8ths, bright band-limited noise, alternating across the field ---
    hat_len = int(0.05 * sr)
    hat_env = np.exp(-np.arange(hat_len) / sr * 70.0)
    for step in range(int(n / sr / (beat / 2)) + 1):
        hat = _spectral_noise(hat_len, sr, 6000, 11000, rng) * hat_env * 0.46
        start = int(step * beat / 2.0 * sr)
        if start >= n:
            break
        end = min(n, start + hat_len)
        seg = hat[: end - start]
        # hard-panned: odd 8ths right, even 8ths left -> real stereo width
        if step % 2:
            right[start:end] += seg
        else:
            left[start:end] += seg

    # --- wide detuned pad: hard left / hard right detune is what actually
    # produces stereo width in this genre (the kick and sub are meant to be mono)
    bar_sec = bar
    for bar_i in range(int(n / bar_sec) + 2):
        root = roots[bar_i % len(roots)]
        seg_len = int(bar_sec * sr)
        start = int(bar_i * bar_sec * sr)
        if start >= n or seg_len <= 0:
            continue
        end = min(n, start + seg_len)
        seg_len = end - start
        st = np.arange(seg_len) / sr
        env = np.minimum(1.0, st / 0.35) * np.minimum(1.0, (seg_len / sr - st) / 0.35 + 0.15)
        env = np.clip(env, 0.0, 1.0).astype(np.float32)
        for ch, detune in ((left, 1.0035), (right, 0.9965)):
            voice = np.zeros(seg_len, dtype=np.float32)
            for mult in (2.0, 2.378, 2.997, 4.0):
                voice += np.sin(2 * np.pi * root * mult * detune * st).astype(np.float32)
            ch[start:end] += (voice * env * 0.055).astype(np.float32)

    # --- Mediterranean yacht breeze: two DECORRELATED noise fields, slowly
    # panned against each other, which is what actually widens the image ---
    lfo = (0.5 + 0.5 * np.sin(2 * np.pi * 0.09 * t + 0.7)).astype(np.float32)
    pan_lfo = (0.5 + 0.5 * np.sin(2 * np.pi * 0.05 * t)).astype(np.float32)
    breeze_l = (_spectral_noise(n, sr, 220, 2400, rng, tilt=-1.1) * lfo * 0.135).astype(np.float32)
    breeze_r = (_spectral_noise(n, sr, 220, 2400, rng, tilt=-1.1) * lfo * 0.135).astype(np.float32)
    left += breeze_l * (0.45 + 0.55 * (1.0 - pan_lfo))
    right += breeze_r * (0.45 + 0.55 * pan_lfo)

    stereo = np.stack([left, right], axis=1)
    peak = float(np.max(np.abs(stereo))) if stereo.size else 0.0
    if peak > 0:
        stereo = stereo / peak * 0.72
    return stereo.astype(np.float32)


def build_reel_audio(voice: np.ndarray, sr: int, frames: int,
                     timings: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Voice over the bed, length-locked to the video.

    The bed ducks under speech using the voice's own RMS as the sidechain —
    that is gain automation, not spectral processing, so the voice stays
    bit-for-bit the edge-tts take.
    """
    n = int(round(frames / REEL_FPS * sr))
    rng = np.random.default_rng(7)
    bed = build_music_bed(n, sr, rng)

    v = voice[:n]
    if v.size < n:
        v = np.pad(v, (0, n - v.size))

    # sidechain: track the voice's short-term level, map to a gentle duck
    frame_len = int(sr / 50)
    env = np.zeros(n, dtype=np.float32)
    for i in range(0, n, frame_len):
        chunk = v[i:i + frame_len]
        env[i:i + chunk.size] = float(np.sqrt(np.mean(chunk ** 2)))
    smooth = np.convolve(env, np.ones(frame_len) / frame_len, mode="same")
    hi = float(np.percentile(smooth, 95)) if smooth.size else 0.0
    duck = 1.0 - 0.55 * np.clip(smooth / hi, 0.0, 1.0) if hi > 1e-6 else np.ones(n, np.float32)

    mixed = bed * (BED_GAIN * duck[:, None]) + np.stack([v, v], axis=1) * VOICE_GAIN
    peak = float(np.max(np.abs(mixed)))
    if peak > 1e-6:
        # normalise to just under full scale: honest headroom without leaving the
        # reel 5 dB quieter than everything else on the feed
        mixed *= 0.95 / peak
    log(f"reel audio: {n / sr:.3f}s @ {sr} Hz, bed {BED_GAIN:.2f} ducked by voice, "
        f"peak {20 * math.log10(max(float(np.max(np.abs(mixed))), 1e-9)):.2f} dBFS", tag="av")
    return mixed.astype(np.float32), {
        "bpm": BPM,
        "bed": "synthesised 120 BPM deep-house groove + Mediterranean breeze",
        "bed_gain": BED_GAIN,
        "voice_gain": VOICE_GAIN,
        "duck": "gain automation from voice RMS (no spectral processing)",
        "duration_s": round(n / sr, 3),
    }


# --------------------------------------------------------------------------- #
# LivePortrait
# --------------------------------------------------------------------------- #

LIVEPORTRAIT_REPO = "https://github.com/KwaiVGI/LivePortrait.git"
LIVEPORTRAIT_DIR = Path(os.environ.get("LIVEPORTRAIT_DIR", "/tmp/LivePortrait"))


def fetch_liveportrait(timings: Dict[str, float]) -> Path:
    """Clone the repo (code only) and stage the five pure-PyTorch checkpoints."""
    if not (LIVEPORTRAIT_DIR / "src" / "live_portrait_wrapper.py").exists():
        with timed("clone_liveportrait", timings, "gpu1"):
            subprocess.run(["git", "clone", "--depth", "1", "-q", LIVEPORTRAIT_REPO,
                            str(LIVEPORTRAIT_DIR)], check=False, capture_output=True)
        if not (LIVEPORTRAIT_DIR / "src").exists():
            raise RuntimeError("LivePortrait clone failed")

    weights_root = LIVEPORTRAIT_DIR / "pretrained_weights"
    with timed("fetch_liveportrait_weights", timings, "gpu1"):
        from huggingface_hub import hf_hub_download

        candidates = [("KwaiVGI/LivePortrait", "model"), ("KwaiVGI/LivePortrait", "space"),
                      ("camenduru/LivePortrait", "model")]
        for _, rel in LIVEPORTRAIT_WEIGHTS:
            dest = weights_root / rel
            if dest.exists() and dest.stat().st_size > 1_000_000:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            got = False
            for repo, kind in candidates:
                try:
                    path = hf_hub_download(repo_id=repo, filename=rel, repo_type=kind)
                    shutil.copy(path, dest)
                    log(f"  ✓ {rel} ← {repo} ({kind})", tag="gpu1")
                    got = True
                    break
                except Exception as exc:
                    log(f"  ✗ {repo} {rel}: {str(exc).splitlines()[0][:80]}", "WARN", tag="gpu1")
            if not got:
                raise RuntimeError(f"could not fetch LivePortrait weight {rel}")

    # landmark.onnx is needed by the cropper for the 106-point lip landmarks.
    landmark = weights_root / "liveportrait" / "landmark.onnx"
    if not (landmark.exists() and landmark.stat().st_size > 100_000):
        landmark.parent.mkdir(parents=True, exist_ok=True)
        for repo, kind in candidates:
            try:
                path = hf_hub_download(repo_id=repo, filename="liveportrait/landmark.onnx",
                                       repo_type=kind)
                shutil.copy(path, landmark)
                log(f"  ✓ landmark.onnx ← {repo}", tag="gpu1")
                break
            except Exception as exc:
                log(f"  ✗ landmark.onnx {repo}: {str(exc).splitlines()[0][:80]}",
                    "WARN", tag="gpu1")
        if not landmark.exists():
            raise RuntimeError("landmark.onnx unavailable — lip retargeting needs it")

    return LIVEPORTRAIT_DIR


class LivePortraitDriver:
    """
    Thin, CLI-free driver over LivePortrait's own modules.

    Everything below mirrors `LivePortraitPipeline.execute()` but with two
    deliberate differences for our use case:

      * the head pose / eyes come from a real driving video's motion template;
      * the LIP is driven from our audio envelope via `retarget_lip`, so Luca's
        mouth articulates in sync with the edge-tts take instead of mouthing
        whatever the driving actor happened to say.
    """

    def __init__(self, repo: Path, device_id: int, face_root: Path, timings: Dict[str, float]):
        self.repo = repo
        self.device_id = device_id
        self.face_root = face_root
        self.timings = timings
        self.wrapper = None
        self.cropper = None
        self.crop_cfg = None

    def load(self) -> None:
        import sys as _sys

        if str(self.repo) not in _sys.path:
            _sys.path.insert(0, str(self.repo))

        import yaml

        from src.config.crop_config import CropConfig
        from src.config.inference_config import InferenceConfig
        from src.live_portrait_wrapper import LivePortraitWrapper
        from src.utils.cropper import Cropper

        import torch

        visible = torch.cuda.device_count()
        if self.device_id >= visible:
            raise RuntimeError(
                f"LivePortrait asked for cuda:{self.device_id} but only {visible} "
                f"GPU(s) exist — clamping to cuda:0")
        self.device_id = min(self.device_id, max(0, visible - 1))
        log(f"LivePortrait device cuda:{self.device_id} "
            f"(visible={visible})", tag="gpu1")

        with timed("load_liveportrait", self.timings, "gpu1"):
            weights = self.repo / "pretrained_weights"
            cfg = InferenceConfig(
                device_id=self.device_id,
                flag_use_half_precision=True,
                flag_pasteback=True,
                flag_do_crop=True,
                flag_stitching=True,
                flag_relative_motion=True,
                flag_lip_retargeting=True,
                flag_eye_retargeting=False,
                animation_region="all",
                driving_option="pose-friendly",
                output_fps=REEL_FPS,
                models_config=str(self.repo / "src" / "config" / "models.yaml"),
                checkpoint_F=str(weights / "liveportrait/base_models/appearance_feature_extractor.pth"),
                checkpoint_M=str(weights / "liveportrait/base_models/motion_extractor.pth"),
                checkpoint_W=str(weights / "liveportrait/base_models/warping_module.pth"),
                checkpoint_G=str(weights / "liveportrait/base_models/spade_generator.pth"),
                checkpoint_S=str(weights / "liveportrait/retargeting_models/stitching_retargeting_module.pth"),
            )
            checkpoints = {
                "appearance_feature_extractor": cfg.checkpoint_F,
                "motion_extractor": cfg.checkpoint_M,
                "warping_module": cfg.checkpoint_W,
                "spade_generator": cfg.checkpoint_G,
                "stitching_retargeting_module": cfg.checkpoint_S,
            }
            for name, path in checkpoints.items():
                candidate = Path(str(path))
                if not candidate.is_file() or candidate.stat().st_size < 1024:
                    raise RuntimeError(f"{name} checkpoint missing/empty: {candidate}")
                log(f"checkpoint {name}: {candidate.stat().st_size / 1024 ** 2:.1f} MB",
                    tag="gpu1")
            # The stall in the previous run happened during module construction,
            # immediately after motion_extractor — the first step that commits
            # GPU memory. Log free VRAM so the next failure is unambiguous.
            try:
                free_b, total_b = torch.cuda.mem_get_info(self.device_id)
                log(f"VRAM before LivePortrait load: "
                    f"{free_b / 1024 ** 2:.0f} MB free / {total_b / 1024 ** 2:.0f} MB",
                    tag="gpu1")
            except Exception:
                pass
            log("constructing LivePortraitWrapper (F, M, W, G, stitching)…", tag="gpu1")
            self.wrapper = LivePortraitWrapper(inference_cfg=cfg)
            log(f"wrapper on {self.wrapper.device}", tag="gpu1")

            self.crop_cfg = CropConfig(device_id=self.device_id,
                                       insightface_root=str(self.face_root))
            self.cropper = Cropper(crop_cfg=self.crop_cfg, device_id=self.device_id)
            log("cropper ready (insightface + landmark.onnx)", tag="gpu1")

        # `load_model` uses raw torch.load; assert the yaml is the one we expect
        yaml.load(open(str(self.repo / "src" / "config" / "models.yaml")), Loader=yaml.SafeLoader)

    def animate(
        self,
        source_rgb: np.ndarray,
        driving_video: Path,
        n_frames: int,
        mouth_open: np.ndarray,
        mouth_range: Tuple[float, float] = (0.02, 0.62),
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        import cv2
        import torch

        from src.utils.camera import get_rotation_matrix
        from src.utils.crop import prepare_paste_back, paste_back
        from src.utils.video import load_video

        wrapper = self.wrapper
        device = wrapper.device
        info: Dict[str, Any] = {"frames": 0}

        # ---- source identity ----
        crop_info = self.cropper.crop_source_image(source_rgb, self.crop_cfg)
        if crop_info is None:
            raise RuntimeError("no face detected in the LivePortrait source image")
        source_lmk = crop_info["lmk_crop"]
        img_crop = crop_info["img_crop_256x256"]

        I_s = wrapper.prepare_source(img_crop)
        x_s_info = wrapper.get_kp_info(I_s)
        x_c_s = x_s_info["kp"]
        R_s = get_rotation_matrix(x_s_info["pitch"], x_s_info["yaw"], x_s_info["roll"])
        f_s = wrapper.extract_feature_3d(I_s)
        x_s = wrapper.transform_keypoint(x_s_info)

        # Pasteback mask: the config already loaded the 512x512 template for us.
        mask_ori = prepare_paste_back(
            wrapper.inference_cfg.mask_crop,
            crop_info["M_c2o"],
            dsize=(source_rgb.shape[1], source_rgb.shape[0]),
        )

        # ---- driving motion template (head pose, eyes, translation) ----
        driving = load_video(str(driving_video))
        if not driving:
            raise RuntimeError(f"could not read driving video {driving_video}")
        while len(driving) < n_frames:
            driving = driving + driving
        driving = driving[:n_frames]

        ret_d = self.cropper.crop_driving_video(driving)
        crops = [cv2.resize(f, (256, 256)) for f in ret_d["frame_crop_lst"]][:n_frames]
        lmk_crop = ret_d["lmk_crop_lst"][:n_frames]
        c_d_eyes_lst, c_d_lip_lst = wrapper.calc_ratio(lmk_crop)

        I_d = wrapper.prepare_videos(crops)
        motion = []
        for i in range(I_d.shape[0]):
            x_i_info = wrapper.get_kp_info(I_d[i:i + 1])
            motion.append({
                "R": get_rotation_matrix(x_i_info["pitch"], x_i_info["yaw"],
                                         x_i_info["roll"]).cpu().numpy().astype(np.float32),
                "exp": x_i_info["exp"].cpu().numpy().astype(np.float32),
                "t": x_i_info["t"].cpu().numpy().astype(np.float32),
                "scale": x_i_info["scale"].cpu().numpy().astype(np.float32),
            })
        log(f"driving template: {len(motion)} frames from {driving_video.name}", tag="gpu1")

        # ---- render ----
        # Composition of motion, mirroring LivePortrait's Algorithm 1:
        #   * head pose (R), translation (t) and scale come from the DRIVING video
        #     (that is the natural 3D turn and the blinks);
        #   * the expression delta comes from the driving video too, EXCEPT the
        #     mouth, which we replace with our own retarget_lip delta driven by
        #     the audio envelope;
        #   * `stitching` then fuses driving-head/source-expression so the face
        #     keeps Luca's identity.
        lo, hi = mouth_range
        R_d_0 = torch.from_numpy(motion[0]["R"]).to(device)
        exp_0 = torch.from_numpy(motion[0]["exp"]).to(device)
        t_d_0 = torch.from_numpy(motion[0]["t"]).to(device)
        scale_d_0 = torch.from_numpy(motion[0]["scale"]).to(device)
        frames: List[np.ndarray] = []
        lip_values: List[float] = []

        for i in range(n_frames):
            m = motion[min(i, len(motion) - 1)]
            R_d_i = torch.from_numpy(m["R"]).to(device)
            t_d_i = torch.from_numpy(m["t"]).to(device)
            exp_d_i = torch.from_numpy(m["exp"]).to(device)
            scale_d_i = torch.from_numpy(m["scale"]).to(device)

            # driving pose, re-anchored to the source frame of reference
            R_new = (R_d_i @ R_d_0.transpose(1, 2)) @ R_s
            t_new = x_s_info["t"] + (t_d_i - t_d_0)
            t_new[..., 2].fill_(0)
            scale_new = x_s_info["scale"] * (scale_d_i / scale_d_0)

            # expression: driving delta on top of the source expression
            delta_new = x_s_info["exp"] + (exp_d_i - exp_0)

            x_d_i_new = scale_new * (x_c_s @ R_new + delta_new) + t_new

            # ---- our mouth, from the audio envelope ----
            openness = float(np.clip(mouth_open[min(i, mouth_open.size - 1)], 0.0, 1.0))
            c_d_lip = lo + (hi - lo) * openness
            lip_values.append(c_d_lip)
            combined = wrapper.calc_combined_lip_ratio([c_d_lip], source_lmk)
            lip_delta = wrapper.retarget_lip(x_s, combined)

            # stitch driving-pose keypoints, then swap in our lip delta. The lip
            # delta is a keypoint offset, so it survives the stitching untouched.
            stitched = wrapper.stitching(x_s, x_d_i_new)
            x_final = stitched + lip_delta

            out = wrapper.warp_decode(f_s, x_s, x_final)
            crop_rgb = wrapper.parse_output(out)[0]
            pasted = paste_back(crop_rgb, crop_info["M_c2o"], source_rgb, mask_ori)
            frames.append(pasted)

        info.update({
            "frames": len(frames),
            "driving": driving_video.name,
            "lip_drive": "audio envelope → retarget_lip MLP",
            "lip_range": [lo, hi],
            "lip_mean": round(float(np.mean(lip_values)), 4),
            "head_pose": "3D rotation from driving video",
        })
        log(f"liveportrait: {len(frames)} frames, lip ratio "
            f"{min(lip_values):.2f}–{max(lip_values):.2f}", tag="gpu1")
        return frames, info


# --------------------------------------------------------------------------- #
# SVD b-roll (forward-only)
# --------------------------------------------------------------------------- #


def build_svd_pipeline(device: str, timings: Dict[str, float]):
    import torch
    from diffusers import StableVideoDiffusionPipeline

    with timed("load_svd", timings, "gpu1"):
        pipe = None
        last: Optional[Exception] = None
        for kwargs in (
            {"variant": "fp16", "torch_dtype": torch.float16},
            {"torch_dtype": torch.float16},
        ):
            try:
                pipe = StableVideoDiffusionPipeline.from_pretrained(
                    "stabilityai/stable-video-diffusion-img2vid-xt", **kwargs)
                break
            except Exception as exc:
                last = exc
                log(f"svd load attempt failed: {type(exc).__name__}: {exc}", "WARN", tag="gpu1")
        if pipe is None:
            raise RuntimeError(f"could not load SVD-XT: {last}") from last

    with timed("configure_svd", timings, "gpu1"):
        for name in ("vae.enable_slicing", "enable_vae_slicing"):
            try:
                target = pipe
                for part in name.split(".")[:-1]:
                    target = getattr(target, part)
                getattr(target, name.split(".")[-1])()
                log(f"applied svd {name}()", tag="gpu1")
                break
            except Exception:
                continue
        try:
            pipe.enable_model_cpu_offload(device=device)
            log("svd: model CPU offload", tag="gpu1")
        except Exception as exc:
            log(f"svd offload unavailable ({exc}); moving to {device}", "WARN", tag="gpu1")
            pipe.to(device, dtype=torch.float16)
    return pipe


def svd_broll(pipe, image: Image.Image, n_frames: int, seed: int,
              timings: Dict[str, float], label: str) -> Tuple[List[Image.Image], Dict[str, Any]]:
    """
    Forward-only b-roll.

    `fps=REEL_FPS` is passed to SVD so the model calibrates its motion per frame
    for 25 fps playback. That is what lets the clip be used strictly
    forward-only with no ping-pong and no cross-fade retiming: the frames are
    already at the right temporal density, so we simply play them in order.
    """
    import torch

    width, height = BROLL_SIZE
    last: Optional[Exception] = None
    for chunk in (4, 2):
        try:
            with timed(f"svd_{label}", timings, "gpu1"):
                result = pipe(
                    image,
                    height=height,
                    width=width,
                    num_frames=n_frames,
                    num_inference_steps=SVD_STEPS,
                    motion_bucket_id=SVD_MOTION_BUCKET,
                    noise_aug_strength=SVD_NOISE_AUG,
                    fps=REEL_FPS,
                    decode_chunk_size=chunk,
                    generator=torch.Generator(device="cuda").manual_seed(seed),
                )
                frames = list(result.frames[0])
            log(f"svd {label}: {len(frames)} frames {width}x{height} (forward-only)", tag="gpu1")
            return frames, {
                "backend": "stable-video-diffusion-img2vid-xt",
                "frames": len(frames),
                "fps_conditioning": REEL_FPS,
                "playback": "forward-only (no ping-pong, no cross-fade)",
                "decode_chunk_size": chunk,
            }
        except Exception as exc:
            last = exc
            log(f"svd {label} chunk={chunk} failed: {type(exc).__name__}: {exc}",
                "WARN", tag="gpu1")
            release_memory(_reel_device())
    raise RuntimeError(f"SVD failed for {label}: {last}") from last


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #


def encode_reel(frames: List[np.ndarray], audio: np.ndarray, sr: int,
                out_path: Path, timings: Dict[str, float]) -> None:
    import soundfile as sf

    if not frames:
        raise RuntimeError("encode_reel got no frames")
    height, width = frames[0].shape[:2]
    tmp_wav = out_path.with_suffix(".tmp.wav")
    sf.write(tmp_wav, audio, sr, subtype="PCM_16")

    ffmpeg = ffmpeg_binary()
    vcodec = ["-c:v", "libx264", "-preset", "medium", "-crf", "19",
              "-x264-params", "keyint=50:min-keyint=25"]
    if not has_encoder(ffmpeg, "libx264"):
        log("libx264 missing — mpeg4 fallback", "WARN")
        vcodec = ["-c:v", "mpeg4", "-q:v", "2"]

    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
           "-r", str(REEL_FPS), "-i", "pipe:0",
           "-i", str(tmp_wav),
           *vcodec, "-pix_fmt", "yuv420p", "-movflags", "+faststart",
           "-c:a", "aac", "-b:a", "192k", "-ar", str(sr), "-ac", "2",
           "-shortest", str(out_path)]

    with timed("encode_reel", timings, "av"):
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        try:
            for frame in frames:
                proc.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
        except Exception as exc:
            log(f"frame pipe failed: {exc}", "ERROR")
        finally:
            # Always close stdin: an open pipe makes ffmpeg wait for ever and the
            # kernel would burn its whole time budget.
            try:
                proc.stdin.close()
            except Exception:
                pass
        stderr = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
        rc = proc.wait()
    tmp_wav.unlink(missing_ok=True)
    if rc != 0 or not out_path.exists():
        raise RuntimeError(f"ffmpeg failed ({rc}): {stderr[-1200:]}")
    log(f"encoded {out_path.name} ({out_path.stat().st_size / 1024:.0f} KB)")


def probe_video(path: Path) -> Dict[str, Any]:
    ffmpeg = ffmpeg_binary()
    try:
        proc = subprocess.run([ffmpeg, "-hide_banner", "-i", str(path)],
                              capture_output=True, text=True)
        info: Dict[str, Any] = {"size_bytes": path.stat().st_size}
        for line in (proc.stderr or "").splitlines():
            line = line.strip()
            if line.startswith("Duration:"):
                info["duration"] = line.split("Duration:")[1].split(",")[0].strip()
            elif "Video:" in line:
                parts = [p.strip() for p in line.split("Video:")[1].split(",")]
                info["video_codec"] = parts[0]
                info["pix_fmt"] = parts[1] if len(parts) > 1 else None
                for p in parts:
                    if "x" in p and p.split("x")[0].isdigit():
                        info["resolution"] = p.split()[0]
                        break
                for p in parts:
                    if p.endswith("fps"):
                        info["fps"] = p.split()[0]
                        break
            elif "Audio:" in line:
                parts = [p.strip() for p in line.split("Audio:")[1].split(",")]
                info["audio_codec"] = parts[0]
                for p in parts:
                    if "Hz" in p:
                        info["sample_rate"] = p.split()[0]
                        break
                info["audio_layout"] = "stereo" if "stereo" in line else "mono"
        return info
    except Exception as exc:
        return {"probe_error": f"{type(exc).__name__}: {exc}"}


# --------------------------------------------------------------------------- #
# GPU 0 worker — carousel
# --------------------------------------------------------------------------- #


def save_slide(image: Image.Image, png: Path, jpg: Path) -> None:
    # `compress_level=9` keeps the 1024-1210 KB PNGs safely under the 5 MB
    # commit ceiling; the q92 JPEG next to it is the postable copy.
    image.save(png, format="PNG", optimize=True, compress_level=9)
    image.convert("RGB").save(jpg, format="JPEG", quality=JPEG_QUALITY, optimize=True,
                              progressive=True)
    log(f"wrote {png.name} ({png.stat().st_size/1024:.0f} KB) + "
        f"{jpg.name} ({jpg.stat().st_size/1024:.0f} KB)")


def run_carousel(output_dir: Path, timings: Dict[str, float],
                 models: Dict[str, Path], q: Any) -> Dict[str, Any]:
    """cuda:0 — scene generation, biometric swap, 1024px adetailer, slide pack."""
    tag = "gpu0"
    device = "cuda:0"
    import torch

    torch.cuda.set_device(0)
    result: Dict[str, Any] = {"slides": [], "artifacts": [], "errors": []}

    pipe = build_scene_pipeline(device, timings, tag)

    # ---- Stage 1: pure scenes ----
    scenes: Dict[int, Image.Image] = {}
    for slide in SLIDES:
        width, height = slide["size"]
        with timed(f"scene_{slide['index']:02d}", timings, tag):
            img = generate_scene(pipe, device, prompt=slide["prompt"], width=width,
                                 height=height, seed=slide["seed"], label=slide["filename"])
        scenes[slide["index"]] = img
        result["slides"].append({
            "index": slide["index"], "filename": slide["filename"], "title": slide["title"],
            "seed": slide["seed"], "size": [width, height],
            "expected_people": slide["people"], "stage1": "ok",
            "ip_adapter_during_txt2img": False,
        })

    master = scenes[1]
    # The master is a deliverable in its own right; write it before the slower
    # swap/adetailer stages so it can never be collateral damage.
    _publish_slide(output_dir, 1, SLIDES[0]["filename"], master, timings, tag)

    # ---- Stage 2 + 3 ----
    engine = FaceEngine(models, 0, timings, tag)
    engine.load()
    source_faces = engine.detect(np.asarray(master))
    source_face, _ = engine.pick_male(source_faces)
    if source_face is None:
        raise RuntimeError("no face found in the master slide — biometric lock impossible")
    master_emb = engine.embedding(np.asarray(master))
    log(f"master face locked: bbox={[int(v) for v in source_face.bbox]}", tag=tag)

    img2img = build_img2img(pipe, timings, tag)
    adapter_ok = attach_adetailer_adapter(pipe, device, timings, tag)
    result["adetailer_adapter"] = adapter_ok

    final_slides: Dict[int, Image.Image] = {1: master}
    identities: Dict[str, Optional[float]] = {}

    for slide in SLIDES:
        if not slide["needs_swap"]:
            continue
        idx = slide["index"]
        record = result["slides"][idx - 1]
        arr = np.asarray(scenes[idx])
        try:
            with timed(f"swap_{idx:02d}", timings, tag):
                faces = engine.detect(arr)
                swapped, bbox, swap_info = engine.swap_male(arr, source_face, faces)
            record["stage2"] = swap_info

            if bbox and adapter_ok:
                with timed(f"adetailer_{idx:02d}", timings, tag):
                    arr2, adetailer_info = adetailer_face(
                        img2img, swapped, bbox, master, device, slide["seed"], tag)
                record["stage3"] = adetailer_info
            else:
                arr2 = swapped
                record["stage3"] = {"adetailer": False,
                                    "reason": "adapter unavailable" if not adapter_ok
                                    else "no face"}

            polished, polish_info = optical_polish(Image.fromarray(arr2), slide["seed"])
            record["polish"] = polish_info

            embedding = engine.embedding(np.asarray(polished))
            similarity = cosine(master_emb, embedding)
            record["identity_cosine"] = similarity
            identities[slide["filename"]] = similarity
            if similarity is not None:
                verdict = "MATCH" if similarity >= 0.45 else "DRIFT"
                log(f"identity {slide['filename']}: cosine={similarity:.4f} ({verdict})", tag=tag)
            final_slides[idx] = polished
            _publish_slide(output_dir, idx, slide["filename"], polished, timings, tag)
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            log(f"slide {slide['filename']} post-process FAILED — {detail}", "ERROR", tag=tag)
            log(traceback.format_exc(), "ERROR", tag=tag)
            record["error"] = detail
            result["errors"].append({"filename": slide["filename"], "error": detail})
            final_slides[idx] = scenes[idx]

    result["identity_cosine"] = identities

    # ---- write the pack ----
    sheet_items: List[Tuple[str, Image.Image]] = []
    for slide in SLIDES:
        idx = slide["index"]
        image = final_slides[idx]
        png = output_dir / slide["filename"]
        jpg = output_dir / f"luca_slide_{idx:02d}.jpg"
        if not (png.is_file() and jpg.is_file()):
            _publish_slide(output_dir, idx, slide["filename"], image, timings, tag)
        result["artifacts"] += [slide["filename"], jpg.name]
        sheet_items.append((slide["title"], image))

    with timed("carousel_sheet", timings, tag):
        build_carousel_sheet(sheet_items, output_dir / CAROUSEL_SHEET)
    result["artifacts"].append(CAROUSEL_SHEET)

    try:
        pipe.to("cpu")
    except Exception:
        pass
    del pipe
    release_memory(device)

    # Hand the final slides to the video worker through the queue (queue, not a
    # shared file: it cannot be half-written).
    try:
        q.put({"slides": {i: np.asarray(im) for i, im in final_slides.items()}})
    except Exception as exc:
        log(f"could not publish slides to the reel worker: {exc}", "WARN", tag=tag)
    return result


# --------------------------------------------------------------------------- #
# GPU 1 worker — voice + reel
# --------------------------------------------------------------------------- #


def run_reel(output_dir: Path, timings: Dict[str, float], models: Dict[str, Path],
             q: Any, wait_seconds: float = REEL_AWAIT_SLIDES_S,
             preflight_only: bool = False, deadline_s: float = 0.0) -> Dict[str, Any]:
    """
    cuda:1 — edge-tts voice, LivePortrait hook, SVD b-roll, muxed reel.

    `preflight_only` exercises every expensive/risky path (LivePortrait weights +
    session, one real frame, one 2-frame SVD pass) and returns before the voice,
    so a single GPU can verify the whole reel stack. It also means the carousel
    is never blocked by a slow avatar pipeline.
    """
    tag = "gpu1"
    import torch

    # Run 45907486374 stalled *inside* LivePortrait's model construction because
    # we asked for cuda:1 on a node that only had one GPU. Never assume: the
    # reel takes the last visible device.
    dev_idx = 0
    try:
        visible = torch.cuda.device_count()
        dev_idx = 1 if visible > 1 else 0
        torch.cuda.set_device(dev_idx)
    except Exception as exc:
        log(f"could not set the reel device: {exc}", "WARN", tag=tag)
    device = f"cuda:{dev_idx}"
    log(f"reel using {device} (visible GPUs: {torch.cuda.device_count()})", tag=tag)
    result: Dict[str, Any] = {"artifacts": [], "errors": []}

    if preflight_only:
        return _preflight_reel(output_dir, timings, models)

    # ---- voice first: the video length is derived from it ----
    voice = synth_voice(output_dir, timings)
    samples = voice.pop("samples")
    result["voice"] = voice
    result["artifacts"] += [VOICE_MP3, "luca_voice.wav"]

    total_frames = HOOK_FRAMES + 2 * BROLL_FRAMES
    mouth = speech_envelope(samples, voice["sample_rate"], total_frames, REEL_FPS)
    result["envelope"] = {
        "frames": int(total_frames),
        "fps": REEL_FPS,
        "vocal_band_hz": [300, 3400],
        "mouth_open_mean": round(float(mouth["mouth_open"].mean()), 4),
        "mouth_open_max": round(float(mouth["mouth_open"].max()), 4),
        "active_fraction": round(float((mouth["mouth_open"] > 0.05).mean()), 3),
    }
    log(f"envelope: {total_frames} frames, "
        f"{(mouth['mouth_open'] > 0.05).mean():.0%} active", tag=tag)

    # ---- the face-locked slides ----
    # cuda:0 publishes them to disk as it finishes each one, so we read from
    # there: that keeps this worker alive even when the carousel is still busy
    # or was terminated by its budget.
    slides: Dict[int, np.ndarray] = _load_slides(output_dir)
    if not slides and wait_seconds > 0:
        with timed("await_carousel", timings, tag):
            try:
                payload = q.get(timeout=wait_seconds)
                slides = payload.get("slides") or {}
            except Exception as exc:
                log(f"carousel payload unavailable: {type(exc).__name__}", "WARN", tag=tag)
        slides = slides or _load_slides(output_dir)
    log(f"b-roll source slides available: {sorted(slides)}", tag=tag)

    frames_all: List[np.ndarray] = []
    boundaries: List[int] = []

    t_deadline = (time.monotonic() + deadline_s) if deadline_s > 0 else float("inf")

    def _has_time(need: float, label: str) -> bool:
        left = t_deadline - time.monotonic()
        if left < need:
            log(f"skipping {label}: {left:.0f}s left, needs ~{need:.0f}s",
                "WARN", tag=tag)
            return False
        return True

    # ---- Shot 1: LivePortrait talking avatar ----
    # Strictly best-effort: if the avatar pass fails or runs out of budget we
    # fall back to SVD-only shots so the reel still ships. It must never raise
    # past this block — one broken model cannot cost the whole run.
    hook_frames: List[np.ndarray] = []
    if not _has_time(LIVEPORTRAIT_BUDGET_S, "LivePortrait shot 1"):
        result["shot1"] = {"skipped": "insufficient budget"}
    else:
        try:
            with timed("shot1_liveportrait", timings, tag):
                repo = fetch_liveportrait(timings)
                driver = LivePortraitDriver(repo, dev_idx, models["buffalo_l"], timings)
                driver.load()

                master = Image.open(output_dir / MASTER_FILENAME).convert("RGB")
                # 576x1024 9:16 crop around the face, head high in frame
                src = _nine_sixteen_portrait(master, np.asarray(master), (576, 1024))
                driving = repo / "assets" / "examples" / "driving" / "d0.mp4"
                if not driving.exists():
                    raise RuntimeError(f"driving video missing: {driving}")

                hook_frames, hook_info = driver.animate(
                    src, driving, HOOK_FRAMES, mouth["mouth_open"][:HOOK_FRAMES])
                result["shot1"] = hook_info
                del driver
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            log(f"LivePortrait unavailable — falling back to SVD shots: {detail}",
                "ERROR", tag=tag)
            log(traceback.format_exc(), "ERROR", tag=tag)
            result["errors"].append({"shot": "liveportrait", "error": detail})
            result["shot1"] = {"ok": False, "error": detail}
            hook_frames = []
        finally:
            release_memory(device)

    if hook_frames:
        frames_all.extend(hook_frames)
        boundaries.append(len(frames_all))

    # ---- Shots 2-3: SVD b-roll, forward-only ----
    svd = build_svd_pipeline(device, timings)
    broll_specs = [("porsche", 4), ("yacht", 2)]
    result["broll"] = []
    for name, slide_idx in broll_specs:
        arr = slides.get(slide_idx)
        if arr is None:
            log(f"slide {slide_idx} unavailable — skipping b-roll {name}", "WARN", tag=tag)
            result["errors"].append({"shot": name, "error": f"slide {slide_idx} missing"})
            continue
        if not _has_time(SVD_SHOT_BUDGET_S, f"b-roll {name}"):
            result["errors"].append({"shot": name, "error": "out of budget"})
            continue
        image = Image.fromarray(arr).resize(BROLL_SIZE, Image.LANCZOS)
        frames, info = svd_broll(svd, image, BROLL_FRAMES, BASE_SEED + 40 + slide_idx,
                                 timings, name)
        frames_all.extend([np.asarray(f.convert("RGB")) for f in frames])
        boundaries.append(len(frames_all))
        info["source_slide"] = slide_idx
        result["broll"].append({"shot": name, **info})

    if not frames_all:
        try:
            svd.to("cpu")
        except Exception:
            pass
        raise RuntimeError("no reel frames were produced (avatar and b-roll both unavailable)")

    try:
        svd.to("cpu")
    except Exception:
        pass
    del svd
    release_memory(device)

    # LivePortrait emits 512-based crops pasted into a 576x1024 canvas; the SVD
    # shots are natively 576x1024. Conform every frame to one canvas.
    frames_all = [_conform(f, BROLL_SIZE) for f in frames_all]

    # ---- audio ----
    mixed, audio_info = build_reel_audio(samples, voice["sample_rate"], len(frames_all), timings)
    result["audio_mix"] = audio_info

    # ---- encode ----
    reel = output_dir / REEL_MP4
    encode_reel(frames_all, mixed, voice["sample_rate"], reel, timings)
    result["artifacts"].append(REEL_MP4)
    result["video"] = {
        "frames": len(frames_all),
        "fps": REEL_FPS,
        "duration_s": round(len(frames_all) / REEL_FPS, 3),
        "shot_boundaries": boundaries,
        "shots": ["liveportrait talking hook", "porsche b-roll", "yacht b-roll"],
        "ping_pong": False,
        "cross_fade_retime": False,
        "procedural_animation": False,
        "probe": probe_video(reel),
    }

    with timed("video_sheet", timings, tag):
        build_video_sheet([Image.fromarray(f) for f in frames_all],
                          output_dir / VIDEO_SHEET,
                          f"{len(frames_all)} frames @ {REEL_FPS} fps · "
                          f"{result['video']['duration_s']}s · LivePortrait + SVD forward-only",
                          boundaries)
    result["artifacts"].append(VIDEO_SHEET)
    return result


def _preflight_reel(output_dir: Path, timings: Dict[str, float],
                    models: Dict[str, Path]) -> Dict[str, Any]:
    """
    Exercise the risky parts of the reel pipeline without writing deliverables.

    This is the "never lose a carousel to a slow avatar model" guard: if
    LivePortrait or SVD cannot run, we find out from a bounded probe instead of
    from a 25-minute poll timeout that discards 5 finished slides.
    """
    import torch

    dev_idx = 1 if torch.cuda.device_count() > 1 else 0
    device = f"cuda:{dev_idx}"
    report: Dict[str, Any] = {"preflight": True, "device": device, "checks": {}}
    probe_image = Image.fromarray(
        (np.random.default_rng(3).random((1024, 768, 3)) * 255).astype(np.uint8))

    # --- LivePortrait: weights + session + one real frame ---
    try:
        with timed("preflight_liveportrait", timings, "gpu1"):
            repo = fetch_liveportrait(timings)
            driver = LivePortraitDriver(repo, dev_idx, models["buffalo_l"], timings)
            driver.load()
            src = _nine_sixteen_portrait(probe_image, np.asarray(probe_image), (576, 1024))
            driving = repo / "assets" / "examples" / "driving" / "d0.mp4"
            frames, info = driver.animate(src, driving, 2, np.array([0.1, 0.6], np.float32))
            report["checks"]["liveportrait"] = {"ok": True, "probe_frames": len(frames),
                                                **{k: v for k, v in info.items()
                                                   if k not in ("frames",)}}
            del driver
            release_memory(device)
    except Exception as exc:
        report["checks"]["liveportrait"] = {"ok": False,
                                            "error": f"{type(exc).__name__}: {exc}"}
        log(f"preflight LivePortrait FAILED: {type(exc).__name__}: {exc}", "ERROR", tag="gpu1")
        log(traceback.format_exc(), "ERROR", tag="gpu1")

    # --- SVD: two frames is enough to prove the pipeline runs ---
    try:
        with timed("preflight_svd", timings, "gpu1"):
            svd = build_svd_pipeline(device, timings)
            frames, info = svd_broll(svd, probe_image.resize(BROLL_SIZE), 2, BASE_SEED + 9,
                                     timings, "preflight")
            report["checks"]["svd"] = {"ok": True, "probe_frames": len(frames),
                                       "chunk": info.get("decode_chunk_size")}
            try:
                svd.to("cpu")
            except Exception:
                pass
            del svd
            release_memory(device)
    except Exception as exc:
        report["checks"]["svd"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        log(f"preflight SVD FAILED: {type(exc).__name__}: {exc}", "ERROR", tag="gpu1")

    ok = all(c.get("ok") for c in report["checks"].values())
    report["ok"] = ok
    log(f"preflight result: {json.dumps(report['checks'], default=str)}", tag="gpu1")
    return report


def _nine_sixteen_portrait(image: Image.Image, arr: np.ndarray,
                           size: Tuple[int, int]) -> np.ndarray:
    """
    Crop a 9:16 window around the face so LivePortrait gets a portrait-shaped
    source with the head in the upper third (its own cropper re-frames the face).
    """
    width, height = size
    aspect = width / height
    src_w, src_h = image.size
    target_w = int(min(src_w, src_h * aspect))
    target_h = int(target_w / aspect)
    if target_h > src_h:
        target_h = src_h
        target_w = int(target_h * aspect)
    # bias the window upward: heads sit high in a portrait frame
    left = (src_w - target_w) // 2
    top = int((src_h - target_h) * 0.30)
    box = (left, top, left + target_w, top + target_h)
    return np.asarray(image.crop(box).resize((width, height), Image.LANCZOS))


def _conform(frame: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    width, height = size
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return np.asarray(Image.fromarray(frame).resize((width, height), Image.LANCZOS))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> int:
    t_start = time.perf_counter()
    output_dir = resolve_output_dir()
    tee = Tee(output_dir / BUILD_LOG, sys.stdout)
    sys.stdout = tee  # type: ignore[assignment]

    metrics: Dict[str, Any] = {
        "phase": "phase-3.1-luca-moretti",
        "character": "Luca Moretti · 27 · young-millionaire playboy",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "config": {
            "stage1": {"model": MODEL_ID, "scheduler": "DPM++ SDE Karras",
                       "algorithm": "sde-dpmsolver++", "steps": SCENE_STEPS,
                       "cfg": SCENE_CFG, "ip_adapter": False},
            "stage2": {"detector": "insightface buffalo_l", "swapper": "inswapper_128.onnx",
                       "target": "male face only"},
            "stage3": {"pipeline": "SDXL img2img + IP-Adapter-Plus-Face",
                       "scale": ADETAILER_SCALE, "strength": ADETAILER_STRENGTH,
                       "size": ADETAILER_SIZE, "steps": ADETAILER_STEPS},
            "video": {"hook": "KwaiVGI/LivePortrait", "hook_frames": HOOK_FRAMES,
                      "broll": "stable-video-diffusion-img2vid-xt",
                      "broll_frames": BROLL_FRAMES, "fps": REEL_FPS,
                      "ping_pong": False, "cross_fade": False},
            "audio": {"engine": "edge-tts", "voice": TTS_VOICE,
                      "bed_bpm": BPM, "wsola": False, "biquad_eq": False},
            "parallelism": "cuda:0 carousel ‖ cuda:1 reel",
        },
        "artifacts": [],
        "timings_seconds": {},
        "errors": [],
    }
    timings: Dict[str, float] = metrics["timings_seconds"]

    log("=" * 78)
    log("stoic-ai-studio · PHASE 3.1 · LUCA MORETTI · dual-GPU upgrade")
    log(f"output dir: {output_dir}")
    log("=" * 78)

    exit_code = 0
    try:
        ensure_python_deps(timings)
        metrics["environment"] = env_fingerprint()
        log(f"environment: {json.dumps(metrics['environment'], default=str)}")

        models = stage_face_models(timings)
        metrics["face_models"] = {k: str(v) for k, v in models.items()}

        # ---------------- GPU scheduling ---------------- #
        import multiprocessing as mp

        import torch

        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        log(f"GPUs visible: {gpu_count}", tag="main")
        metrics["gpu_count"] = gpu_count
        if gpu_count < 1:
            raise RuntimeError("no CUDA device — enable a Kaggle GPU accelerator")

        ctx = mp.get_context("spawn")
        results: Dict[str, Any] = {}

        # Budgets are enforced by the parent with a hard join timeout, so no
        # worker can ever wedge the kernel. The ordering is deliberate: the
        # slides ARE the deliverable, so cuda:0 always runs first and the reel
        # only gets the time that is provably left over.
        if gpu_count >= 2:
            # Dedicated devices — genuine parallel, as designed.
            queue: Any = ctx.Queue()
            with timed("parallel_gpu_workers", timings):
                procs = {
                    "carousel": ctx.Process(target=_worker_carousel,
                                            args=(output_dir, timings, models, queue),
                                            name="gpu0-carousel"),
                    "reel": ctx.Process(target=_worker_reel,
                                        args=(output_dir, timings, models, queue,
                                              REEL_AWAIT_SLIDES_S, False,
                                              float(REEL_BUDGET_S * 1.5)),
                                        name="gpu1-reel"),
                }
                _supervise(procs, {"carousel": CAROUSEL_BUDGET_S,
                                   "reel": REEL_BUDGET_S * 2},
                           timings, metrics)
        else:
            # One GPU. Carousel FIRST (it is the real deliverable), then the reel
            # on whatever budget is left. Run 45907486374 spent 40 min here and
            # delivered nothing because the avatar pass ran before the slides.
            log("single GPU: carousel first, reel if the budget allows", tag="main")
            car_queue: Any = ctx.Queue()
            car = ctx.Process(target=_worker_carousel,
                              args=(output_dir, timings, models, car_queue),
                              name="gpu0-carousel")
            car.start(); t_car = time.monotonic(); car.join(CAROUSEL_BUDGET_S)
            if car.is_alive():
                log(f"carousel exceeded {CAROUSEL_BUDGET_S}s — terminating "
                    f"(slides already written stay on disk)", "ERROR", tag="main")
                car.terminate(); car.join(30)
                metrics["errors"].append({"worker": "carousel", "error": "timeout"})
            car_seconds = time.monotonic() - t_car
            metrics["timings_seconds"]["main/carousel_wall"] = round(car_seconds, 1)
            log(f"carousel worker done in {car_seconds:.0f}s "
                f"(exit {car.exitcode})", tag="main")

            remaining = KERNEL_BUDGET_S - (time.monotonic() - t_start)
            log(f"budget left for the reel: {remaining:.0f}s "
                f"(need ≥{REEL_MIN_BUDGET_S}s)", tag="main")
            if remaining < REEL_MIN_BUDGET_S:
                msg = (f"reel skipped: only {remaining:.0f}s left of the "
                       f"{KERNEL_BUDGET_S}s kernel budget")
                log(msg, "WARN", tag="main")
                metrics["reel_skipped"] = msg
            else:
                # slides are already on disk, so the reel does not need them
                # passed through shared memory at all.
                with timed("reel_worker_single_gpu", timings):
                    reel_proc = ctx.Process(
                        target=_worker_reel,
                        args=(output_dir, timings, models, _PrefilledQueue({}),
                              0.0, False, float(max(60.0, remaining - 120))),
                        name="gpu0-reel")
                    reel_proc.start(); t_reel = time.monotonic()
                    reel_proc.join(remaining - 90)  # keep time to package + exit
                    if reel_proc.is_alive():
                        log("reel exceeded its budget — terminating; the carousel "
                            "pack is already complete", "ERROR", tag="main")
                        reel_proc.terminate(); reel_proc.join(30)
                        metrics["errors"].append({"worker": "reel", "error": "timeout"})
                    log(f"reel worker finished in {time.monotonic()-t_reel:.0f}s "
                        f"(exit {reel_proc.exitcode})", tag="main")

        # Workers return their detail through the sidecar files they write;
        # worker-local metrics are merged below from those files.
        for name in ("carousel", "reel"):
            sidecar = output_dir / f"_metrics_{name}.json"
            if sidecar.is_file():
                try:
                    results[name] = json.loads(sidecar.read_text(encoding="utf-8"))
                    sidecar.unlink()
                except Exception as exc:
                    log(f"could not read {sidecar.name}: {exc}", "WARN")

        if results.get("carousel"):
            car = results["carousel"]
            metrics["slides"] = car.get("slides", [])
            metrics["identity_cosine"] = car.get("identity_cosine", {})
            metrics["adetailer_adapter"] = car.get("adetailer_adapter")
            metrics["artifacts"] += car.get("artifacts", [])
            metrics["errors"] += car.get("errors", [])

        if results.get("reel"):
            reel = results["reel"]
            metrics["voice"] = reel.get("voice", {})
            metrics["envelope"] = reel.get("envelope", {})
            metrics["shot1"] = reel.get("shot1", {})
            metrics["broll"] = reel.get("broll", [])
            metrics["audio_mix"] = reel.get("audio_mix", {})
            metrics["video"] = reel.get("video", {})
            metrics["artifacts"] += reel.get("artifacts", [])
            metrics["errors"] += reel.get("errors", [])

        # ---------------- verification ---------------- #
        # Two tiers. The carousel is the deliverable and must exist; the reel is
        # best-effort and its absence is reported, never fatal — otherwise one
        # slow avatar model makes the whole run worthless.
        core_required = [CAROUSEL_SHEET, METRICS_FILENAME]
        slides_required = [f"luca_slide_{i:02d}.jpg" for i in range(1, 6)]
        reel_required = [REEL_MP4, VOICE_MP3]

        missing = [n for n in core_required + slides_required
                   if not (output_dir / n).is_file()]
        metrics["missing_slides"] = missing
        if missing:
            raise RuntimeError(f"carousel artifacts missing: {missing}")

        missing_reel = [n for n in reel_required if not (output_dir / n).is_file()]
        metrics["missing_reel_artifacts"] = missing_reel
        if missing_reel:
            msg = f"reel artifacts missing: {missing_reel}"
            log(msg, "ERROR")
            metrics["errors"].append({"stage": "reel_artifacts", "error": msg})
            metrics.setdefault("reel_skipped", msg)

        verdicts = []
        for name, value in (metrics.get("identity_cosine") or {}).items():
            if value is None:
                verdicts.append(f"{name}:unmeasured")
            elif value >= 0.45:
                verdicts.append(f"{name}:MATCH")
            else:
                verdicts.append(f"{name}:DRIFT")
        metrics["identity_verdicts"] = verdicts
        log(f"identity verdicts: {verdicts}", tag="main")

        video = metrics.get("video") or {}
        active = (metrics.get("envelope") or {}).get("active_fraction")
        log(f"reel: {video.get('frames')} frames @ {video.get('fps')} fps "
            f"({video.get('duration_s')}s), shots={video.get('shot_boundaries')}, "
            f"mouth-active={active}", tag="main")

        if metrics["errors"]:
            metrics["status"] = "success_with_errors"
        elif metrics.get("reel_skipped"):
            metrics["status"] = "carousel_only"
        else:
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
        metrics["artifacts"] = sorted(set(metrics["artifacts"]) |
                                       {METRICS_FILENAME, BUILD_LOG})
        with open(output_dir / METRICS_FILENAME, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2, default=str)

        log("=" * 78)
        log(f"status: {metrics['status']} · total {metrics['total_seconds']}s")
        for name in metrics["artifacts"]:
            candidate = output_dir / name
            if candidate.exists():
                log(f"  ✓ {name} ({candidate.stat().st_size / 1024:.0f} KB)")
        log("=" * 78)
        sys.stdout = tee._stream  # type: ignore[assignment]
        tee.close()

    return exit_code


def _publish_slide(output_dir: Path, index: int, filename: str,
                   image: Image.Image, timings: Dict[str, float], tag: str) -> None:
    """
    Write a finished slide to disk immediately.

    Kaggle keeps /kaggle/working on disk even while the kernel is still running,
    so publishing as we go means an over-running avatar pass can never cost us
    slides we already finished (which is exactly what happened in run 45907486374).
    """
    with timed(f"write_{index:02d}", timings, tag):
        save_slide(image, output_dir / filename, output_dir / f"luca_slide_{index:02d}.jpg")


def _reel_device() -> str:
    """The device the reel should use: cuda:1 when present, else cuda:0."""
    try:
        import torch

        return f"cuda:{1 if torch.cuda.device_count() > 1 else 0}"
    except Exception:
        return "cuda:0"


def _load_slides(output_dir: Path) -> Dict[int, np.ndarray]:
    """
    Read the face-locked slides from disk (cuda:0 writes each one as it lands).

    Disk, not inter-process memory, because the reel must be able to work with
    whatever the carousel managed to publish before its budget ran out.
    """
    slides: Dict[int, np.ndarray] = {}
    for entry in SLIDES:
        path = output_dir / f"luca_slide_{entry['index']:02d}.jpg"
        if not path.is_file():
            path = output_dir / entry["filename"]
        if path.is_file():
            try:
                slides[entry["index"]] = np.asarray(Image.open(path).convert("RGB"))
            except Exception as exc:
                log(f"could not read slide {path.name}: {exc}", "WARN", tag="gpu1")
    return slides


def _supervise(procs: Dict[str, Any], budgets: Dict[str, float],
               timings: Dict[str, float], metrics: Dict[str, Any]) -> None:
    """Run workers with hard wall-clock limits; nothing is allowed to wedge."""
    deadlines = {name: time.monotonic() + budgets[name] for name in procs}
    for name, proc in procs.items():
        proc.start()
        log(f"started {name} worker ({proc.name})", tag="main")
    for name, proc in procs.items():
        wait = max(1.0, deadlines[name] - time.monotonic())
        proc.join(wait)
        if proc.is_alive():
            log(f"{name} worker exceeded {budgets[name]:.0f}s — terminating",
                "ERROR", tag="main")
            proc.terminate()
            proc.join(30)
            metrics["errors"].append({"worker": name, "error": "timeout"})
        log(f"{name} worker exited with code {proc.exitcode}", tag="main")
        if proc.exitcode not in (0, None) and not proc.is_alive():
            metrics["errors"].append({"worker": name, "exitcode": proc.exitcode})


def _json_safe(value: Any) -> Any:
    """
    Strip numpy arrays / Paths out of worker results.

    The sidecars only need metadata; dumping full frames made the files huge and
    was the sort of thing that silently fails inside a finally block.
    """
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()
                if not isinstance(v, np.ndarray)}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value if not isinstance(v, np.ndarray)]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class _PrefilledQueue:
    """Stand-in for mp.Queue when the slides are already in hand (single GPU)."""

    def __init__(self, payload: Dict[str, Any]):
        self._payload = payload

    def get(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        if not self._payload:
            raise RuntimeError("no carousel payload available")
        return self._payload

    def put(self, item: Any) -> None:  # pragma: no cover - symmetry only
        self._payload = item


# ---- worker entry points (spawn-safe: module level, no closures) ---------- #


def _worker_carousel(output_dir: Path, timings: Dict[str, float],
                     models: Dict[str, Path], queue: Any) -> None:
    try:
        result = run_carousel(output_dir, timings, models, queue)
    except Exception as exc:
        log(f"carousel worker crashed: {type(exc).__name__}: {exc}", "ERROR", tag="gpu0")
        log(traceback.format_exc(), "ERROR", tag="gpu0")
        # Never leave cuda:1 blocked on the queue.
        try:
            queue.put({"slides": {}})
        except Exception:
            pass
        result = {"errors": [f"{type(exc).__name__}: {exc}"],
                  "artifacts": ["build.log"]}
    try:
        (output_dir / "_metrics_carousel.json").write_text(
            json.dumps(_json_safe(result), default=str), encoding="utf-8")
    except Exception as exc:
        log(f"could not write carousel sidecar: {exc}", "WARN", tag="gpu0")
    raise SystemExit(0 if not result.get("errors") else 1)


def _worker_reel(output_dir: Path, timings: Dict[str, float],
                 models: Dict[str, Path], queue: Any) -> None:
    try:
        result = run_reel(output_dir, timings, models, queue)
    except Exception as exc:
        log(f"reel worker crashed: {type(exc).__name__}: {exc}", "ERROR", tag="gpu1")
        log(traceback.format_exc(), "ERROR", tag="gpu1")
        result = {"errors": [f"{type(exc).__name__}: {exc}"], "artifacts": ["build.log"]}
    try:
        (output_dir / "_metrics_reel.json").write_text(
            json.dumps(_json_safe(result), default=str), encoding="utf-8")
    except Exception as exc:
        log(f"could not write reel sidecar: {exc}", "WARN", tag="gpu1")
    raise SystemExit(0 if not result.get("errors") else 1)


if __name__ == "__main__":
    sys.exit(main())
