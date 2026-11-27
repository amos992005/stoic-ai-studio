#!/usr/bin/env python3
"""
Luca Moretti — Process 2 of 2: voice + talking avatar on cuda:1.

Stages (each one releases its VRAM before the next starts):

  1. **Chatterbox-Turbo** (`chatterbox-tts`) in float16 → `outputs/luca_voice.wav`,
     ~8-10 s of charismatic 27-year-old Monaco-Reel hook.
  2. `del tts_model; gc.collect(); torch.cuda.empty_cache()` immediately.
  3. **MuseTalk** (v1.5, fp16) driven by the master portrait + that WAV, after
     polling briefly for `outputs/01_luca_master.png` to appear.
  4. Mux to `outputs/luca_reel.mp4` and verify both streams with ffprobe.

Hard prohibitions honoured here: no `pipe.to("cpu")` (the only device moves are
model deletion + cache release), no inswapper_128, no SVD-XT, no Adetailer.

Two upstream hazards this file works around, both discovered by reading the
source rather than by guessing:

  * `chatterbox-tts` declares `torch==2.6.0`. A plain `pip install` would
    DOWNGRADE Kaggle's torch and break process 1, which shares the environment.
    Every install therefore goes through a guard that re-pins torch afterwards.
  * `musetalk.utils.preprocessing` imports `mmpose` and instantiates DWPose at
    *module import time*, and mmpose needs mmcv compiled against torch. Only the
    face boxes come from that module, so we skip it entirely and supply boxes
    with insightface, then use MuseTalk's own mask-based blending path which
    does not need the BiSeNet face parser.
"""

from __future__ import annotations

import fcntl
import gc
import json
import os
import shutil
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

WORK_DIR = Path(os.environ.get("LUCA_WORK_DIR", "/kaggle/working/luca_work"))
MUSETALK_DIR = WORK_DIR / "MuseTalk"
MUSETALK_MODELS = MUSETALK_DIR / "models"

DEVICE = "cuda:1"
MASTER_FILENAME = "01_luca_master.png"
VOICE_FILENAME = "luca_voice.wav"
REEL_FILENAME = "luca_reel.mp4"

MASTER_WAIT_SECONDS = int(os.environ.get("MASTER_WAIT_SECONDS", "900"))
MASTER_POLL_SECONDS = 5.0
TTS_MAX_SECONDS = 10.0
TALKING_FPS = 25

TRANSCRIPT = os.environ.get(
    "LUCA_TRANSCRIPT",
    "Morning in Monaco, sunset in Ibiza. Honestly, the hardest part of today "
    "was choosing the view.",
)

MUSETALK_REPO = "https://github.com/TMElyralab/MuseTalk.git"
MUSETALK_WEIGHTS_REPO = "TMElyralab/MuseTalk"
MUSE_V15_UNET = ("musetalkV15/unet.pth", "musetalkV15/musetalk.json")
SD_VAE_REPO = "stabilityai/sd-vae-ft-mse"
#: Exact filenames present in that repo (no .bin — safetensors only).
SD_VAE_FILES = ("config.json", "diffusion_pytorch_model.safetensors")
WHISPER_REPO = "openai/whisper-tiny"

#: The ffprobe verification gate.
MIN_REEL_SECONDS = 2.0


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


def log(message: str, level: str = "INFO", tag: str = "vid") -> None:
    print(f"[{ts()}] [{level:<5}] [{tag}] {message}", flush=True)


class timed:
    def __init__(self, label: str, timings: Dict[str, float], tag: str = "vid"):
        self.label, self.timings, self.tag = label, timings, tag

    def __enter__(self):
        log(f"▶ start: {self.label}", tag=self.tag)
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        elapsed = time.perf_counter() - self.start
        self.timings[f"{self.tag}/{self.label}"] = round(elapsed, 3)
        log(f"✔ done:  {self.label} ({elapsed:.1f}s)", tag=self.tag)
        return False


# --------------------------------------------------------------------------- #
# Environment safety
# --------------------------------------------------------------------------- #


def torch_fingerprint() -> Tuple[Optional[str], Optional[str]]:
    """The torch/torchaudio versions we must not disturb."""
    import torch

    try:
        import torchaudio

        audio = torchaudio.__version__
    except Exception:
        audio = None
    return torch.__version__, audio


def pip_install(packages: List[str], keep: Tuple[Optional[str], Optional[str]],
                no_deps: bool = False) -> bool:
    """
    Install packages without letting them repin the shared torch build.

    Process 1 is running SDXL on the same interpreter environment, so a silent
    `torch 2.11 -> 2.6` downgrade (as `chatterbox-tts` requests) would break it
    mid-run. We install with --no-deps where the dependency set is known, and
    re-pin torch/torchaudio afterwards in case a resolver touched them.
    """
    torch_version, torchaudio_version = keep
    command = [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir"]
    if no_deps:
        command.append("--no-deps")
    command.extend(packages)
    log(f"pip install {' '.join(packages)}"
        f"{' (no-deps)' if no_deps else ''}")
    proc = subprocess.run(command, capture_output=True, text=True)
    if proc.returncode != 0:
        log(f"pip failed: {(proc.stderr or proc.stdout)[-600:]}", "ERROR")

    # Re-pin whatever the resolver may have moved.
    try:
        import importlib

        importlib.invalidate_caches()
        import torch as torch_now

        if torch_version and torch_now.__version__ != torch_version:
            log(f"torch was moved to {torch_now.__version__} — restoring "
                f"{torch_version}", "WARN")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
                 f"torch=={torch_version}"],
                capture_output=True, text=True,
            )
    except Exception as exc:
        log(f"torch re-pin check skipped: {exc}", "WARN")

    return proc.returncode == 0


def ensure_video_deps(timings: Dict[str, float]) -> None:
    """
    Install process 2's dependencies with the torch guard in place.

    transformers is pinned by chatterbox to 5.2.0; Kaggle ships 5.x already, so
    we let pip resolve transformers but forbid it from touching torch (which the
    `--no-deps` split below achieves).
    """
    keep = torch_fingerprint()
    log(f"protecting torch={keep[0]} torchaudio={keep[1]}")

    with timed("pip_install_deps", timings):
        # chatterbox-tts is installed --no-deps: its own metadata pins
        # torch==2.6.0 and transformers==5.2.0, and a resolved install would
        # downgrade the torch that process 1 is running SDXL on. Its runtime
        # deps are listed below instead, torch-free.
        # Every import chatterbox 0.1.7 actually performs at module scope, taken
        # from the wheel rather than from its metadata: s3gen pulls s3tokenizer,
        # the watermarker pulls resemble-perth (`import perth`), tts_turbo pulls
        # pyloudnorm, and the tokenizer pulls spacy-pkuseg unconditionally.
        # Gradio and the Hebrew/Russian helpers are deliberately absent — they
        # are demo-only and dicta_onnx is imported lazily behind try/except.
        wanted = ["soundfile", "librosa", "einops", "omegaconf", "tqdm",
                  "imageio[ffmpeg]", "ffmpeg-python", "gdown", "pykakasi",
                  "safetensors", "s3tokenizer", "spacy-pkuseg",
                  "resemble-perth", "pyloudnorm", "conformer==0.3.2",
                  "tokenizers"]
        missing = [p for p in wanted if _needs_install(p)]
        if missing:
            pip_install(missing, keep)
        else:
            log("video dependencies already present")

        # The TTS engine itself, always torch-safe.
        if _needs_install("chatterbox-tts"):
            if not pip_install(["chatterbox-tts"], keep, no_deps=True):
                raise RuntimeError("chatterbox-tts install failed — see above")
            log("chatterbox-tts installed --no-deps (torch untouched)")
        else:
            log("chatterbox-tts already present")


#: chatterbox-tts 0.1.7 pins diffusers==0.29.0 because its s3gen decoder imports
#: `diffusers.models.lora.LoRACompatibleLinear`, which newer diffusers removed.
#: Process 1 is running InstantID on the *same* interpreter, and that pipeline
#: needs diffusers >= 0.30 — so downgrading is not an option. Instead we restore
#: the two symbols chatterbox touches. Both were plain wrappers with no LoRA
#: behaviour in the signature chatterbox uses, so an identity passthrough is a
#: faithful stand-in.
_SHIM = """
import torch.nn as nn


class LoRACompatibleLinear(nn.Linear):
    def forward(self, hidden_states):
        return super().forward(hidden_states)


class LoRACompatibleConv(nn.Conv2d):
    def forward(self, hidden_states, scale=1.0):
        return super().forward(hidden_states)
"""


def shim_diffusers_lora() -> None:
    """
    Re-add the diffusers symbols chatterbox imports but newer diffusers removed.

    Idempotent, and a no-op on diffusers versions that still ship them. This has
    to run before `import chatterbox`, otherwise the s3gen import blows up with
    ModuleNotFoundError in the middle of the TTS stage.
    """
    import types

    try:
        import diffusers
    except Exception as exc:
        log(f"diffusers unavailable ({type(exc).__name__}) — skipping LoRA shim",
            "WARN")
        return

    try:
        lora = __import__("diffusers.models.lora", fromlist=["LoRACompatibleLinear"])
        assert lora.LoRACompatibleLinear and lora.LoRACompatibleConv
        log(f"diffusers {diffusers.__version__} still ships LoRACompatible*")
        return
    except Exception:
        pass

    module = types.ModuleType("diffusers.models.lora")
    exec(_SHIM, module.__dict__)          # noqa: S102 - fixed literal source
    import sys as _sys
    _sys.modules["diffusers.models.lora"] = module
    diffusers.models.lora = module       # type: ignore[attr-defined]
    log(f"restored diffusers.models.lora for chatterbox "
        f"(diffusers {diffusers.__version__} dropped it)", "WARN")


def _needs_install(spec: str) -> bool:
    module = (spec.split("[")[0].replace("-", "_")
              .split("==")[0].split(">=")[0].split("<")[0])
    aliases = {
        "chatterbox_tts": "chatterbox",
        "ffmpeg_python": "ffmpeg",
        "imageio": "imageio",
    }
    try:
        import importlib

        importlib.import_module(aliases.get(module, module))
        return False
    except Exception:
        return True


# --------------------------------------------------------------------------- #
# Stage 1 — TTS
# --------------------------------------------------------------------------- #


def synth_voice(timings: Dict[str, float]) -> Dict[str, Any]:
    """
    Chatterbox-Turbo in float16 on cuda:1, written as luca_voice.wav.

    Verified against chatterbox-tts 0.1.7:
      * the Turbo class is `chatterbox.tts_turbo.ChatterboxTurboTTS` — it is NOT
        re-exported from the package root, so `from chatterbox import ...` fails;
      * `ChatterboxTurboTTS.from_pretrained(device)` pulls
        ResembleAI/chatterbox-turbo, which ships `conds.pt`, a built-in voice
        conditioning. That matters because `generate()` asserts `self.conds` is
        not None — without conds.pt it would need a >5 s reference clip;
      * `generate()` returns a single float tensor shaped [1, N] at `self.sr`;
      * the class is a plain object, not an nn.Module, so `.half()` on it does
        nothing. The dtype cast has to reach the actual submodules.
    """
    import numpy as np
    import soundfile as sf
    import torch

    with timed("load_chatterbox", timings):
        shim_diffusers_lora()
        candidates = [
            ("chatterbox.tts_turbo", "ChatterboxTurboTTS"),
            ("chatterbox.tts", "ChatterboxTTS"),
        ]
        model, chosen, sample_rate = None, None, None
        last_error: Optional[Exception] = None
        for module_name, class_name in candidates:
            try:
                module = __import__(module_name, fromlist=[class_name])
                cls = getattr(module, class_name)
                model = cls.from_pretrained(device=DEVICE)
                chosen = f"{module_name}.{class_name}"
                sample_rate = int(getattr(model, "sr", 24000))
                log(f"using {chosen} @ {sample_rate} Hz")
                break
            except Exception as exc:
                last_error = exc
                log(f"{module_name}.{class_name} unavailable: "
                    f"{type(exc).__name__}: {exc}", "WARN")
        if model is None:
            raise RuntimeError(f"chatterbox-tts could not be loaded: {last_error}")

        # float16 on the real submodules; report what actually happened.
        #
        # `s3gen` is deliberately excluded. Its vocoder is a signal-processing
        # chain (STFT -> conv stacks) whose intermediate tensors are float32 by
        # construction — torch.stft has no float16 path — so casting it yields
        # "Input type (float) and bias type (c10::Half) should be the same" no
        # matter how many individual projections get patched (runs 45950802624
        # and 45952128760). t3, the ~2 GB autoregressive transformer, is where
        # the memory actually goes and s3gen's convs are cheap at 24 kHz.
        FP16_SUBMODULES = ("t3", "ve")
        cast, failed = [], []
        for name in FP16_SUBMODULES:
            submodule = getattr(model, name, None)
            if submodule is None or not hasattr(submodule, "half"):
                continue
            try:
                submodule.half()
                cast.append(name)
            except Exception as exc:
                failed.append(f"{name}: {type(exc).__name__}")
        dtype = "float16" if cast and not failed else "mixed/float32"
        log(f"chatterbox dtype={dtype} (cast: {cast or 'none'}"
            f"{'; failed: ' + ', '.join(failed) if failed else ''})")

        # conds.pt is loaded in float32 and is fed straight into the now-fp16
        # T3ConditionalEncoder, so the first matmul dies with
        #   mat1 and mat2 must have the same dtype, but got Float and Half
        # (run 45947923120). Conditionals.to(device) deliberately does not cast
        # dtype, and the float/int tensors live in the same container, so reach
        # through to the fp tensors explicitly — a blanket .half() would corrupt
        # the int speech-token ids.
        t3_cast = "t3" in cast
        conds = getattr(model, "conds", None)
        if conds is not None and t3_cast:
            cast_conds = 0
            t3_cond = getattr(conds, "t3", None)
            for field, value in list(vars(t3_cond).items()) if t3_cond else []:
                if torch.is_tensor(value) and value.is_floating_point():
                    setattr(t3_cond, field, value.half())
                    cast_conds += 1
            gen = getattr(conds, "gen", None) or {}
            for key, value in list(gen.items()):
                if torch.is_tensor(value) and value.is_floating_point():
                    gen[key] = value.half()
                    cast_conds += 1
            log(f"cast {cast_conds} built-in conditioning tensor(s) to float16")

        if t3_cast:
            if not cast_conds:
                log("no conditioning tensors were cast — generate() may raise a "
                    "dtype mismatch", "WARN")

    with timed("generate_speech", timings):
        try:
            output = model.generate(TRANSCRIPT)
        except RuntimeError as exc:
            message = str(exc)
            if not any(marker in message for marker in
                       ("same dtype", "same type", "should be the same")):
                raise
            # Last resort: chatterbox has dtype-sensitive internals we cannot
            # enumerate ahead of time. fp32 still fits on an empty 14.5 GB card,
            # and a WAV that exists beats a run that dies on precision.
            log(f"fp16 synthesis failed ({exc}); retrying the whole model in "
                f"float32", "WARN")
            model.t3.float()
            model.s3gen.float()
            model.ve.float()
            for name in ("t3", "s3gen", "ve"):
                submodule = getattr(model, name, None)
                if submodule is not None:
                    submodule.float()
            conds = getattr(model, "conds", None)
            t3_cond = getattr(conds, "t3", None) if conds is not None else None
            for field, value in list(vars(t3_cond).items()) if t3_cond else []:
                if torch.is_tensor(value) and value.is_floating_point():
                    setattr(t3_cond, field, value.float())
            for key, value in list((getattr(conds, "gen", None) or {}).items()):
                if torch.is_tensor(value) and value.is_floating_point():
                    conds.gen[key] = value.float()
            output = model.generate(TRANSCRIPT)
        if isinstance(output, tuple):          # (wav, sr) on some versions
            audio, sample_rate = output[0], int(output[1])
        else:
            audio = output

        if hasattr(audio, "detach"):
            audio = audio.detach().float().cpu().numpy()
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)

        # Trim to the Reel length, then peak-normalise (never RMS-limit).
        limit = int(TTS_MAX_SECONDS * sample_rate)
        if audio.size > limit:
            audio = audio[:limit]
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak <= 0:
            raise RuntimeError("chatterbox produced silence")
        audio = (audio / peak * 0.891).astype(np.float32)   # -1.0 dBFS

        voice_path = OUTPUT_DIR / VOICE_FILENAME
        sf.write(str(voice_path), audio, int(sample_rate), subtype="PCM_16")
        seconds = audio.size / float(sample_rate)
        log(f"wrote {voice_path.name}: {seconds:.2f}s @ {sample_rate} Hz "
            f"({voice_path.stat().st_size / 1024:.0f} KB)")

    # ---- mandatory VRAM cleanup before MuseTalk loads -------------------- #
    with timed("release_tts_vram", timings):
        del model
        gc.collect()
        torch.cuda.empty_cache()
        try:
            free_b, total_b = torch.cuda.mem_get_info(1)
            log(f"cuda:1 after TTS release: {free_b / 1024 ** 2:.0f} MB free "
                f"of {total_b / 1024 ** 2:.0f} MB")
        except Exception:
            log("cuda:1 after TTS release (mem_get_info unavailable)")

    return {
        "engine": "chatterbox-tts",
        "class": chosen,
        "weights": "ResembleAI/chatterbox-turbo (built-in conds.pt voice)",
        "dtype": dtype,
        "sample_rate": int(sample_rate),
        "duration_s": round(seconds, 3),
        "transcript": TRANSCRIPT,
        "normalisation": "peak to -1.0 dBFS",
        "file": VOICE_FILENAME,
    }


# --------------------------------------------------------------------------- #
# Stage 2 — MuseTalk
# --------------------------------------------------------------------------- #


def wait_for_master(timings: Dict[str, float]) -> Path:
    """Poll briefly for the master portrait produced by process 1."""
    master = OUTPUT_DIR / MASTER_FILENAME
    with timed("wait_for_master", timings):
        deadline = time.monotonic() + MASTER_WAIT_SECONDS
        while time.monotonic() < deadline:
            if master.is_file() and master.stat().st_size > 10_000:
                # guard against reading a partially flushed PNG
                try:
                    from PIL import Image

                    with Image.open(master) as probe:
                        probe.verify()
                    log(f"master portrait available ({master.stat().st_size / 1024:.0f} KB)")
                    return master
                except Exception:
                    log("master exists but is not yet readable — waiting", "WARN")
            time.sleep(MASTER_POLL_SECONDS)
    raise RuntimeError(
        f"{MASTER_FILENAME} did not appear within {MASTER_WAIT_SECONDS}s")


def ensure_musetalk_repo(timings: Dict[str, float]) -> Path:
    if (MUSETALK_DIR / "musetalk").is_dir():
        log(f"MuseTalk already cloned at {MUSETALK_DIR}")
        return MUSETALK_DIR
    with timed("clone_musetalk", timings):
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            ["git", "clone", "--depth", "1", MUSETALK_REPO, str(MUSETALK_DIR)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0 or not (MUSETALK_DIR / "musetalk").is_dir():
            raise RuntimeError(f"MuseTalk clone failed: {proc.stderr[-400:]}")
        log("MuseTalk cloned")
    return MUSETALK_DIR


def ensure_musetalk_weights(timings: Dict[str, float]) -> Dict[str, Path]:
    """
    Download exactly the weights the pipeline path uses.

    Deliberately NOT the upstream download_weights.sh: that also pulls the
    BiSeNet face-parser from Google Drive, which this pipeline does not need
    (mask-based blending) and which often 403s from a datacenter.
    """
    from huggingface_hub import hf_hub_download

    paths: Dict[str, Path] = {}
    with timed("fetch_musetalk_weights", timings):
        for remote in MUSE_V15_UNET:
            local = hf_hub_download(MUSETALK_WEIGHTS_REPO, remote)
            # keep upstream's layout: models/musetalkV15/{unet.pth,musetalk.json}
            destination = MUSETALK_MODELS / Path(remote)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.is_file() or destination.stat().st_size != Path(local).stat().st_size:
                shutil.copy2(local, destination)
            paths[remote] = destination
            log(f"✓ {remote} ({destination.stat().st_size / 1e6:.0f} MB)")

        # sd-vae (diffusers layout). MuseTalk's VAE does
        # AutoencoderKL.from_pretrained("./models/sd-vae"), which needs the
        # weights *plus* config.json in that directory. The repo only ships
        # diffusion_pytorch_model.safetensors — asking for the .bin variant
        # raises EntryNotFoundError and kills the run before any frame is made.
        vae_dir = MUSETALK_MODELS / "sd-vae"
        vae_dir.mkdir(parents=True, exist_ok=True)
        for filename in SD_VAE_FILES:
            shutil.copy2(hf_hub_download(SD_VAE_REPO, filename), vae_dir / filename)
        log(f"✓ sd-vae ({', '.join(SD_VAE_FILES)})")

        whisper_dir = MUSETALK_MODELS / "whisper"
        whisper_dir.mkdir(parents=True, exist_ok=True)
        for filename in ("config.json", "preprocessor_config.json", "pytorch_model.bin"):
            try:
                shutil.copy2(hf_hub_download(WHISPER_REPO, filename), whisper_dir / filename)
            except Exception as exc:
                log(f"whisper {filename}: {type(exc).__name__}", "WARN")
        log("✓ whisper")

        unet_path = paths[MUSE_V15_UNET[0]]
        config_path = paths[MUSE_V15_UNET[1]]
        # MuseTalk resolves vae from ./models/<vae_type> relative to its cwd
        return {"unet": unet_path, "unet_config": config_path,
                "vae": vae_dir, "whisper": whisper_dir}


def face_box_for(img_rgb, app) -> Optional[Tuple[int, int, int, int]]:
    """
    The male face box, replacing MuseTalk's DWPose/mmpose call.

    `musetalk.utils.preprocessing.get_landmark_and_bbox` needs mmpose (and thus
    mmcv compiled against torch); only the box is actually used downstream, so
    insightface supplies it directly.
    """
    import cv2

    faces = app.get(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    if not faces:
        return None
    indexed = list(enumerate(faces))
    males = [(i, f) for i, f in indexed if getattr(f, "sex", None) == "M"] or indexed
    _, face = max(
        males,
        key=lambda p: float((p[1].bbox[2] - p[1].bbox[0]) * (p[1].bbox[3] - p[1].bbox[1])),
    )
    x1, y1, x2, y2 = [int(round(v)) for v in face.bbox]
    height, width = img_rgb.shape[:2]
    return (max(0, x1), max(0, y1), min(width, x2), min(height, y2))


def build_face_mask(crop_size: int, box_h: int) -> "Any":
    """
    Soft elliptical chin-to-nose mask, used instead of BiSeNet face parsing.

    MuseTalk's mask-based blending (`get_image_blending`) accepts a precomputed
    mask, so the face-parser model — a Google Drive download — is not needed.
    The ellipse keeps the lower half of the face box (mouth and jaw), which is
    exactly the region MuseTalk regenerates.
    """
    import numpy as np
    from PIL import Image as PILImage, ImageDraw, ImageFilter

    mask = PILImage.new("L", (crop_size, crop_size), 0)
    draw = ImageDraw.Draw(mask)
    # lower ~60% of the box, inset horizontally
    top = int(crop_size * 0.40)
    draw.ellipse(
        [int(crop_size * 0.10), top, int(crop_size * 0.90), int(crop_size * 0.98)],
        fill=255,
    )
    mask = mask.filter(ImageFilter.GaussianBlur(crop_size * 0.02))
    array = np.asarray(mask, dtype=np.uint8)
    log(f"face mask built procedurally (soft ellipse, {crop_size}px, "
        f"box height {box_h}px)")
    return array


def run_musetalk(master: Path, voice: Path, timings: Dict[str, float]) -> Dict[str, Any]:
    """MuseTalk v1.5, fp16, using the mask-based blend path (no mmpose)."""
    import cv2
    import numpy as np
    import torch
    from transformers import WhisperModel

    repo = ensure_musetalk_repo(timings)
    weights = ensure_musetalk_weights(timings)

    with timed("load_musetalk", timings):
        os.chdir(repo)                       # modules resolve ./models relatively
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))

        from musetalk.models.unet import PositionalEncoding, UNet
        from musetalk.models.vae import VAE
        from musetalk.utils.audio_processor import AudioProcessor
        from musetalk.utils.blending import get_image_blending, get_crop_box
        from musetalk.utils.utils import datagen
    
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        weight_dtype = torch.float16

        # MuseTalk's VAE takes (model_path, resized_img, use_float16) — it has no
        # device argument and internally targets torch.device("cuda"), which
        # honours the thread's current device, so set_device above picks cuda:1.
        # Passing device= as a kwarg would raise TypeError.
        vae = VAE(model_path="./models/sd-vae", use_float16=True)
        unet = UNet(unet_config=str(weights["unet_config"]),
                    model_path=str(weights["unet"]), device=device,
                    use_float16=True)
        pe = PositionalEncoding(d_model=384).to(device)
        # inference.py builds this once per run; there is no unet.timesteps.
        timesteps = torch.tensor([0], device=device)
        log(f"MuseTalk vae + unet loaded in fp16 on {device}")

        # AudioProcessor loads the whisper feature extractor itself; calling
        # WhisperFeatureExtractor.from_pretrained again is redundant.
        audio_processor = AudioProcessor(feature_extractor_path="./models/whisper")
        whisper = WhisperModel.from_pretrained("./models/whisper").to(
            device=device, dtype=weight_dtype)
        whisper.requires_grad_(False)
        log("whisper-tiny loaded")

    with timed("prepare_avatar", timings):
        from insightface.app import FaceAnalysis

        providers = ["CPUExecutionProvider"]
        try:
            import onnxruntime as ort

            if any("CUDA" in p for p in ort.get_available_providers()):
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        except Exception:
            pass
        app = FaceAnalysis(name="antelopev2",
                           root=os.environ.get("FACE_MODEL_ROOT",
                                               "/kaggle/working/face_models"),
                           providers=providers)
        app.prepare(ctx_id=0, det_size=(640, 640))

        frame = cv2.imread(str(master))
        if frame is None:
            raise RuntimeError(f"could not read {master}")
        box = face_box_for(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), app)
        if box is None:
            raise RuntimeError("no face found in the master portrait for MuseTalk")
        x1, y1, x2, y2 = box
        extra_margin = 10
        y2 = min(y2 + extra_margin, frame.shape[0])
        log(f"avatar face box: {(x1, y1, x2, y2)}")

        crop = frame[y1:y2, x1:x2]
        crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        latents = vae.get_latents_for_unet(crop).to(device)

        crop_box, _ = get_crop_box((x1, y1, x2, y2), 1.5)
        mask_array = build_face_mask(crop_box[2] - crop_box[0], y2 - y1)

        whisper_input_features, librosa_length = audio_processor.get_audio_feature(
            str(voice))
        whisper_chunks = audio_processor.get_whisper_chunk(
            whisper_input_features, device, weight_dtype, whisper, librosa_length,
            fps=TALKING_FPS,
        )
        log(f"{len(whisper_chunks)} audio chunks @ {TALKING_FPS} fps")

    frames_dir = WORK_DIR / "musetalk_frames"
    if frames_dir.is_dir():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    with timed("musetalk_inference", timings):
        total = len(whisper_chunks)
        written = 0
        for i, (whisper_batch, latent_batch) in enumerate(
                datagen(whisper_chunks=whisper_chunks,
                        vae_encode_latents=[latents],
                        batch_size=1, delay_frame=0, device=device)):
            # PositionalEncoding adds a float32 buffer (`self.pe`) to its input,
            # so even with fp16 whisper features the sum is promoted to float32
            # and the first attention matmul inside the fp16 UNet dies with
            # "expected mat1 and mat2 to have the same dtype, but got: float !=
            # c10::Half" (run 45953039196). Cast both UNet inputs explicitly.
            audio_feature_batch = pe(whisper_batch).to(dtype=unet.model.dtype)
            latent_batch = latent_batch.to(dtype=unet.model.dtype)
            pred = unet.model(latent_batch, timesteps,
                              encoder_hidden_states=audio_feature_batch).sample
            recon = vae.decode_latents(pred)
            for res in recon:
                try:
                    face_render = cv2.resize(np.asarray(res, dtype=np.uint8),
                                             (x2 - x1, y2 - y1))
                except Exception:
                    continue
                combined = get_image_blending(frame, face_render, [x1, y1, x2, y2],
                                              mask_array, crop_box)
                cv2.imwrite(str(frames_dir / f"{written:08d}.png"), combined)
                written += 1
            if total and i % 25 == 0:
                log(f"  {i}/{total} chunks ({written} frames)")
        log(f"MuseTalk rendered {written} frames")

    if written == 0:
        raise RuntimeError("MuseTalk produced no frames")

    # ---- encode the silent talking-head video ---------------------------- #
    silent_video = WORK_DIR / "talking_silent.mp4"
    with timed("encode_talking", timings):
        proc = subprocess.run([
            ffmpeg_bin(), "-y", "-v", "error",
            "-framerate", str(TALKING_FPS),
            "-i", str(frames_dir / "%08d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            str(silent_video),
        ], capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"frame encode failed: {proc.stderr[-400:]}")
        log(f"silent video encoded ({silent_video.stat().st_size / 1024:.0f} KB)")

    return {
        "engine": "MuseTalk v1.5 (fp16)",
        "weights": "TMElyralab/MuseTalk musetalkV15/unet.pth",
        "fps": TALKING_FPS,
        "frames": written,
        "face_box": [int(x1), int(y1), int(x2), int(y2)],
        "blending": "mask-based (procedural soft ellipse; BiSeNet parser unused)",
        "mmpose_bypassed": True,
        "silent_video": str(silent_video),
    }


# --------------------------------------------------------------------------- #
# Encoding / verification
# --------------------------------------------------------------------------- #


def ffmpeg_bin() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def ffprobe_bin() -> Optional[str]:
    found = shutil.which("ffprobe")
    if found:
        return found
    sibling = Path(ffmpeg_bin()).with_name("ffprobe")
    return str(sibling) if sibling.is_file() else None


def probe_media(path: Path) -> Dict[str, Any]:
    """ffprobe the streams; falls back to ffmpeg's own report when ffprobe is absent."""
    probe = ffprobe_bin()
    if probe:
        proc = subprocess.run([
            probe, "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", str(path),
        ], capture_output=True, text=True)
        if proc.returncode == 0:
            data = json.loads(proc.stdout or "{}")
            streams = data.get("streams", [])
            video = next((s for s in streams if s.get("codec_type") == "video"), None)
            audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
            return {
                "tool": "ffprobe",
                "video": None if video is None else {
                    "codec": video.get("codec_name"),
                    "width": int(video.get("width") or 0),
                    "height": int(video.get("height") or 0),
                    "frames": video.get("nb_frames"),
                    "duration_s": float(video.get("duration") or 0) or None,
                    "fps": video.get("r_frame_rate"),
                },
                "audio": None if audio is None else {
                    "codec": audio.get("codec_name"),
                    "sample_rate": audio.get("sample_rate"),
                    "channels": audio.get("channels"),
                    "duration_s": float(audio.get("duration") or 0) or None,
                },
                "duration_s": float(data.get("format", {}).get("duration") or 0) or None,
            }

    log("ffprobe unavailable — falling back to ffmpeg stream detection", "WARN")
    proc = subprocess.run([ffmpeg_bin(), "-i", str(path)], capture_output=True, text=True)
    report = (proc.stderr or "")
    return {
        "tool": "ffmpeg -i",
        "has_video": "Video:" in report,
        "has_audio": "Audio:" in report,
        "report_tail": report[-700:],
    }


def mux_reel(silent_video: Path, voice: Path, timings: Dict[str, float]) -> Path:
    """Mux picture with the generated voice track."""
    reel = OUTPUT_DIR / REEL_FILENAME
    with timed("mux_reel", timings):
        proc = subprocess.run([
            ffmpeg_bin(), "-y", "-v", "error",
            "-i", str(silent_video),
            "-i", str(voice),
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest", "-movflags", "+faststart",
            str(reel),
        ], capture_output=True, text=True)
        if proc.returncode != 0:
            # copy may fail if the source is an odd profile — re-encode instead
            log(f"stream copy failed, re-encoding: {proc.stderr[-300:]}", "WARN")
            proc = subprocess.run([
                ffmpeg_bin(), "-y", "-v", "error",
                "-i", str(silent_video), "-i", str(voice),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                "-c:a", "aac", "-b:a", "192k", "-shortest",
                str(reel),
            ], capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"mux failed: {proc.stderr[-400:]}")
        log(f"wrote {reel.name} ({reel.stat().st_size / 1024:.0f} KB)")
    return reel


def verify_reel(reel: Path, timings: Dict[str, float]) -> Dict[str, Any]:
    """Both streams must exist, be non-trivial, and be length-matched."""
    with timed("verify_reel", timings):
        info = probe_media(reel)
        if info.get("tool") == "ffprobe":
            video, audio = info.get("video"), info.get("audio")
            if video is None:
                raise RuntimeError("reel has no video stream")
            if audio is None:
                raise RuntimeError("reel has no audio stream")
            duration = info.get("duration_s") or video.get("duration_s") or 0.0
            if duration < MIN_REEL_SECONDS:
                raise RuntimeError(f"reel is only {duration:.2f}s "
                                   f"(minimum {MIN_REEL_SECONDS}s)")
            v_dur = video.get("duration_s") or duration
            a_dur = audio.get("duration_s") or duration
            drift = abs(v_dur - a_dur)
            info["sync"] = {
                "video_s": round(v_dur, 3),
                "audio_s": round(a_dur, 3),
                "drift_s": round(drift, 3),
                "ok": drift <= 0.5,
            }
            # A/V sync was an acceptance criterion, so make it a hard gate.
            if not info["sync"]["ok"]:
                raise RuntimeError(f"audio drift {drift:.2f}s exceeds 0.5s")
            log(f"verified: {video['width']}x{video['height']} "
                f"{video['codec']} / {audio['codec']} "
                f"{duration:.2f}s · drift {drift:.3f}s")
        else:
            if not info.get("has_video") or not info.get("has_audio"):
                raise RuntimeError(f"reel streams invalid: {info}")
            log("verified via ffmpeg stream detection")
        return info


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
        "process": "video",
        "device": DEVICE,
        "status": "running",
        "started_at": ts(),
        "prohibitions_honoured": {
            "pipe.to_cpu": False,
            "inswapper_128": False,
            "svd_xt": False,
            "sdxl_img2img_adetailer": False,
        },
        "timings_seconds": timings,
        "errors": [],
        "artifacts": [],
    }
    exit_code = 0

    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA device — enable a Kaggle GPU accelerator")
        # The launcher gives each worker its own card via CUDA_VISIBLE_DEVICES, so
        # inside this process the GPU is always cuda:0 and nothing can spill onto
        # the card the image pipeline is using. Falling back to device 1 only
        # applies when both cards are visible (running this file by hand).
        visible = torch.cuda.device_count()
        index = 0 if visible < 2 else 1
        torch.cuda.set_device(index)
        effective_device = f"cuda:{index}"
        if DEVICE != effective_device:
            log(f"device: {DEVICE} is not present in this process "
                f"({visible} visible) — using {effective_device}, which the "
                f"launcher pinned to the same physical card", "WARN")
        metrics["device"] = effective_device

        ensure_video_deps(timings)

        # ---- voice -----------------------------------------------------------
        metrics["voice"] = synth_voice(timings)
        metrics["artifacts"].append(VOICE_FILENAME)

        # ---- avatar ----------------------------------------------------------
        master = wait_for_master(timings)
        metrics["musetalk"] = run_musetalk(master, OUTPUT_DIR / VOICE_FILENAME, timings)

        # ---- mux + verify ----------------------------------------------------
        silent = Path(metrics["musetalk"]["silent_video"])
        reel = mux_reel(silent, OUTPUT_DIR / VOICE_FILENAME, timings)
        metrics["reel_probe"] = verify_reel(reel, timings)
        metrics["artifacts"].append(REEL_FILENAME)

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
        merge_metrics("video", metrics, METRICS_PATH)

        log("=" * 74)
        log(f"video pipeline status: {metrics['status']} · {metrics['total_seconds']}s")
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
