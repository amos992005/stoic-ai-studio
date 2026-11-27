"""
Luca Moretti — Process 2 of 2: voice + talking avatar on cuda:1.

    Chatterbox → luca_voice.wav → SadTalker → mux → luca_reel.mp4

Two constraints shape this file:

  * **The mux is mandatory, not cosmetic.** SadTalker writes whatever codec its
    own ffmpeg call happens to produce, and a phone that will not play the reel
    is a bad deliverable. Everything is re-encoded to H.264 High/yuv420p with AAC
    audio and `+faststart`, then *verified with ffprobe* rather than assumed.

  * **The workers run at the same time.** The picture depends on process 1's
    master portrait, so this one waits for that file and does the voice first —
    which is long enough that the two overlap almost completely.
"""

from __future__ import annotations

import gc
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from luca_common import (BUILD_LOG, METRICS_PATH, OUTPUT_DIR, Tee, log,
                         merge_metrics, missing_packages, pip_install, timed, ts)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

PINNED_DEVICE = "cuda:1"     # what the architecture says
DEVICE = "cuda:0"            # what this process may use; see resolve_device()

WORK_DIR = Path(os.environ.get("LUCA_WORK_DIR", "/kaggle/working/luca_work"))
SADTALKER_DIR = WORK_DIR / "SadTalker"

MASTER_FILENAME = "01_luca_master.png"
VOICE_FILENAME = "luca_voice.wav"
RAW_VIDEO_FILENAME = "luca_raw.mp4"
REEL_FILENAME = "luca_reel.mp4"

MASTER_WAIT_SECONDS = int(os.environ.get("MASTER_WAIT_SECONDS", "1800"))
MASTER_POLL_SECONDS = 5.0

#: SadTalker's output is 256px by default and 512px when asked; 512 is the
#: difference between "a talking head" and "a phone wallpaper".
SADTALKER_SIZE = int(os.environ.get("SADTALKER_SIZE", "512"))
#: gfpgan on: it repairs the softened mouth region SadTalker's renderer leaves.
SADTALKER_ENHANCER = os.environ.get("SADTALKER_ENHANCER", "gfpgan")

TRANSCRIPT = os.environ.get(
    "LUCA_TRANSCRIPT",
    "Morning in Monaco, sunset in Ibiza. Honestly, the hardest part of today "
    "was choosing the view.",
)

TTS_MAX_SECONDS = 12.0
MIN_REEL_SECONDS = 2.0

#: Chatterbox's runtime deps, taken from the wheel's module-scope imports rather
#: than its metadata (which pins torch==2.6 and would downgrade this runtime).
CHATTERBOX_DEPS = ["soundfile", "librosa", "einops", "omegaconf", "tqdm",
                   "imageio[ffmpeg]", "ffmpeg-python", "pykakasi", "safetensors",
                   "s3tokenizer", "spacy-pkuseg", "resemble-perth", "pyloudnorm",
                   "conformer==0.3.2", "tokenizers"]

#: SadTalker's runtime deps. Deliberately excludes basicsr: only the RealESRGAN
#: *background* upsampler needs it, `--enhancer gfpgan` does not, and basicsr's
#: setup.py cannot build on a current Kaggle image (it shells out to a CUDA
#: toolkit download at build time).
SADTALKER_DEPS = ["librosa", "face_alignment", "imageio", "imageio-ffmpeg",
                  "numba", "resampy", "pydub", "scipy", "kornia", "yacs",
                  "scikit-image", "facexlib", "av", "safetensors"]

#: Installed separately, and *lastly*, because their build is the fragile part.
#: ``basicsr``'s setup.py imports torch and torchvision at build time, so pip's
#: default build isolation cannot install it ("Getting requirements to build
#: wheel did not run successfully ... No available output" — run 45980036759).
#: ``gfpgan`` depends on basicsr, so both go in the same no-isolation call.
GFPGAN_DEPS = ["basicsr==1.4.2", "gfpgan==1.3.8"]
#: basicsr is imported by the SadTalker shim, not by gfpgan's own metadata.
GFPGAN_BUILD_DEPS = ["numpy", "opencv-python-headless", "torchvision", "scipy",
                     "tqdm", "yapf", "addict", "future", "lmdb", "pyyaml",
                     "requests", "scikit-image", "tqdm"]

SADTALKER_REPO = "https://github.com/OpenTalker/SadTalker.git"

#: Every SadTalker weight, all from GitHub releases (a host the Kaggle runner can
#: reach). The single `.safetensors` is what makes init_path() pick the
#: newer single-file model instead of the five legacy `.pth` files.
SADTALKER_WEIGHTS = [
    ("https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/"
     "SadTalker_V0.0.2_{size}.safetensors", "checkpoints"),
    ("https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/"
     "mapping_00229-model.pth.tar", "checkpoints"),
    ("https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/"
     "mapping_00109-model.pth.tar", "checkpoints"),
    # facexlib: the retinaface detector and the FAN landmark model
    ("https://github.com/xinntao/facexlib/releases/download/v0.1.0/"
     "detection_Resnet50_Final.pth", "gfpgan/weights"),
    ("https://github.com/xinntao/facexlib/releases/download/v0.1.0/"
     "alignment_WFLW_4HG.pth", "gfpgan/weights"),
    ("https://github.com/xinntao/facexlib/releases/download/v0.2.2/"
     "parsing_parsenet.pth", "gfpgan/weights"),
    # the enhancer itself
    ("https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/"
     "GFPGANv1.4.pth", "gfpgan/weights"),
]

#: face_alignment downloads s3fd/2DFAN4 into the torch hub cache on first use.
FACE_ALIGNMENT_CACHE = Path(os.environ.get(
    "TORCH_HOME", str(Path.home() / ".cache" / "torch")))


# --------------------------------------------------------------------------- #
# Compatibility shims
# --------------------------------------------------------------------------- #


def apply_legacy_shims() -> None:
    """
    The three module-level landmines in a 2023 codebase on a 2026 runtime.

    Each of these fails at *import* time, before any of SadTalker's own code
    runs, so they have to be fixed before the repo is imported rather than
    caught later:

      * ``torchvision.transforms.functional_tensor`` was removed in torchvision
        0.17; ``basicsr`` (which ``gfpgan`` imports) still imports it.
      * ``np.VisibleDeprecationWarning`` was removed in numpy 2.0.
      * the face-Alignment models are downloaded by ``face_alignment`` on
        first use, so the cache directory has to exist and be writable.
    """
    import numpy as np

    try:
        import torchvision.transforms.functional as functional

        sys.modules.setdefault(
            "torchvision.transforms.functional_tensor", functional)
    except Exception as exc:
        log(f"functional_tensor shim skipped: {exc}", "WARN")

    if not hasattr(np, "VisibleDeprecationWarning"):
        np.VisibleDeprecationWarning = DeprecationWarning
    for legacy, modern in (("float_", float), ("int_", int), ("bool_", bool),
                           ("object_", object), ("str_", str)):
        if not hasattr(np, legacy):
            setattr(np, legacy, modern)


# --------------------------------------------------------------------------- #
# Voice
# --------------------------------------------------------------------------- #


def resolve_device() -> str:
    """
    The GPU this process may use, and make it current.

    The launcher pins each worker to one card with CUDA_VISIBLE_DEVICES, so the
    card is cuda:0 inside this process. Choosing cuda:1 unconditionally raised
    "invalid device ordinal" the moment that isolation was introduced.
    """
    global DEVICE
    try:
        import torch
    except Exception:
        return DEVICE
    if not torch.cuda.is_available():
        return DEVICE
    visible = torch.cuda.device_count()
    index = 0 if visible < 2 else 1
    torch.cuda.set_device(index)
    DEVICE = f"cuda:{index}"
    return DEVICE


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
    Re-add the two symbols chatterbox imports that newer diffusers removed.

    chatterbox pins diffusers 0.29 for this reason. Downgrading would break
    process 1 (InstantID needs >= 0.30) so the symbols are restored instead;
    both were plain wrappers with no LoRA behaviour in the signature chatterbox
    uses.
    """
    import types

    try:
        import diffusers
    except Exception as exc:
        log(f"diffusers unavailable ({type(exc).__name__}) — skipping shim", "WARN")
        return
    try:
        __import__("diffusers.models.lora", fromlist=["LoRACompatibleLinear"])
        return
    except Exception:
        pass

    module = types.ModuleType("diffusers.models.lora")
    exec(_SHIM, module.__dict__)          # noqa: S102 - fixed literal source
    sys.modules["diffusers.models.lora"] = module
    diffusers.models.lora = module       # type: ignore[attr-defined]
    log(f"restored diffusers.models.lora for chatterbox "
        f"(diffusers {diffusers.__version__} dropped it)", "WARN")


def _cast_floats(container: Any, torch) -> int:
    """
    Halve every float tensor held by ``container``, by attribute *or* by key.

    Written defensively because the real ``conds.t3`` may be a dataclass, a
    namedtuple or a plain object with class-level attributes — ``vars()`` only
    sees the last of those three when the values are on the instance. Integer
    tensors (the speech token ids) share the same container and must not be
    touched, so the cast is guarded by ``is_floating_point``.
    """
    if container is None:
        return 0
    cast = 0
    items = {}
    if isinstance(container, dict):
        items = container
    else:
        names = set(getattr(container, "__dict__", {}) or {})
        names |= {name for name in dir(container) if not name.startswith("_")}
        items = {}
        for name in names:
            try:
                items[name] = getattr(container, name)
            except Exception:
                continue
    for field, value in list(items.items()):
        if torch.is_tensor(value) and value.is_floating_point():
            value = value.half()
            try:
                if isinstance(container, dict):
                    container[field] = value
                else:
                    setattr(container, field, value)
            except Exception:
                continue
            cast += 1
    return cast


def synth_voice(timings: Dict[str, float]) -> Dict[str, Any]:
    """
    Chatterbox-Turbo → luca_voice.wav.

    ``s3gen`` is kept in float32 deliberately. Its vocoder runs an STFT into
    conv stacks, and ``torch.stft`` has no float16 path, so casting it produces
    "Input type (float) and bias type (c10::Half) should be the same" no matter
    how many individual layers are patched. ``t3`` — the ~2 GB autoregressive
    transformer — is where the memory actually goes.
    """
    import numpy as np
    import soundfile as sf
    import torch

    device = resolve_device()

    with timed("load_chatterbox", timings):
        shim_diffusers_lora()
        candidates = [("chatterbox.tts_turbo", "ChatterboxTurboTTS"),
                      ("chatterbox.tts", "ChatterboxTTS")]
        model = chosen = None
        sample_rate = 24000
        last_error: Optional[Exception] = None
        for module_name, class_name in candidates:
            try:
                module = __import__(module_name, fromlist=[class_name])
                model = getattr(module, class_name).from_pretrained(device=device)
                chosen = f"{module_name}.{class_name}"
                sample_rate = int(getattr(model, "sr", 24000))
                log(f"using {chosen} @ {sample_rate} Hz on {device}")
                break
            except Exception as exc:
                last_error = exc
                log(f"{module_name}.{class_name} unavailable: "
                    f"{type(exc).__name__}: {exc}", "WARN")
        if model is None:
            raise RuntimeError(f"chatterbox-tts could not be loaded: {last_error}")

        cast, failed = [], []
        for name in ("t3", "ve"):
            submodule = getattr(model, name, None)
            if submodule is None or not hasattr(submodule, "half"):
                continue
            try:
                submodule.half()
                cast.append(name)
            except Exception as exc:
                failed.append(f"{name}: {type(exc).__name__}")
        dtype = "float16" if cast and not failed else "float32"
        log(f"chatterbox dtype={dtype} (cast: {cast or 'none'}; s3gen stays float32)")

        # conds.pt loads in float32 and feeds the now-fp16 t3. Reach through to
        # the float tensors only: a blanket .half() corrupts the int token ids
        # stored in the same container.
        conds = getattr(model, "conds", None)
        cast_conds = 0
        if conds is not None and "t3" in cast:
            cast_conds += _cast_floats(getattr(conds, "t3", None), torch)
            gen = getattr(conds, "gen", None)
            if isinstance(gen, dict):
                for key, value in list(gen.items()):
                    if torch.is_tensor(value) and value.is_floating_point():
                        gen[key] = value.half()
                        cast_conds += 1
        log(f"cast {cast_conds} conditioning tensor(s) with t3")

    with timed("generate_speech", timings):
        output = model.generate(TRANSCRIPT)
        if isinstance(output, tuple):
            audio, sample_rate = output[0], int(output[1])
        else:
            audio = output
        if hasattr(audio, "detach"):
            audio = audio.detach().float().cpu().numpy()
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)

        limit = int(TTS_MAX_SECONDS * sample_rate)
        if audio.size > limit:
            audio = audio[:limit]
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak <= 0:
            raise RuntimeError("chatterbox produced silence")
        audio = (audio / peak * 0.891).astype(np.float32)     # -1.0 dBFS

        voice = OUTPUT_DIR / VOICE_FILENAME
        sf.write(str(voice), audio, int(sample_rate), subtype="PCM_16")
        seconds = audio.size / float(sample_rate)
        log(f"wrote {VOICE_FILENAME}: {seconds:.2f}s @ {sample_rate} Hz "
            f"({voice.stat().st_size / 1024:.0f} KB)")

    with timed("release_tts_vram", timings):
        del model
        gc.collect()
        torch.cuda.empty_cache()
        try:
            free, total = torch.cuda.mem_get_info()
            log(f"after TTS release: {free / 1024 ** 2:.0f} MB free of "
                f"{total / 1024 ** 2:.0f} MB")
        except Exception:
            log("after TTS release (mem_get_info unavailable)")

    return {
        "engine": "chatterbox-tts",
        "class": chosen,
        "dtype": dtype,
        "s3gen_dtype": "float32 (STFT has no fp16 path)",
        "sample_rate": int(sample_rate),
        "duration_s": round(seconds, 3),
        "transcript": TRANSCRIPT,
        "file": VOICE_FILENAME,
    }


# --------------------------------------------------------------------------- #
# Avatar — SadTalker
# --------------------------------------------------------------------------- #


def wait_for_master(timings: Dict[str, float]) -> Path:
    """Poll for process 1's master portrait."""
    master = OUTPUT_DIR / MASTER_FILENAME
    with timed("wait_for_master", timings):
        deadline = time.monotonic() + MASTER_WAIT_SECONDS
        while time.monotonic() < deadline:
            if master.is_file():
                # Structural, not a byte-count guess: a partially written PNG is
                # exactly the case worth rejecting, and PIL's verify() catches
                # both a truncated file and a non-image. The dimension floor
                # rules out a placeholder that would animate into a postage
                # stamp.
                try:
                    from PIL import Image

                    with Image.open(master) as probe:
                        probe.verify()
                        width, height = probe.size
                    if min(width, height) >= 512:
                        log(f"master portrait available ({width}x{height}, "
                            f"{master.stat().st_size / 1024:.0f} KB)")
                        return master
                    log(f"master is only {width}x{height} — waiting for the "
                        f"real render", "WARN")
                except Exception:
                    log("master exists but is not readable yet — waiting", "WARN")
            time.sleep(MASTER_POLL_SECONDS)
    raise RuntimeError(f"{MASTER_FILENAME} did not appear within "
                       f"{MASTER_WAIT_SECONDS}s")


def run(*command: str, timeout: int = 3600) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True,
                          timeout=timeout)


def ensure_sadtalker(timings: Dict[str, float]) -> Path:
    """Clone the repo and fetch every weight it needs."""
    with timed("clone_sadtalker", timings):
        if (SADTALKER_DIR / "inference.py").is_file():
            log(f"SadTalker already cloned at {SADTALKER_DIR}")
        else:
            WORK_DIR.mkdir(parents=True, exist_ok=True)
            proc = run("git", "clone", "--depth", "1", SADTALKER_REPO,
                       str(SADTALKER_DIR))
            if proc.returncode != 0:
                raise RuntimeError(f"SadTalker clone failed: {proc.stderr[-500:]}")
            log("SadTalker cloned")

    with timed("fetch_sadtalker_weights", timings):
        for url_template, subdir in SADTALKER_WEIGHTS:
            # The safetensors name carries the render size. If only the other
            # size is present, init_path() still finds a *.safetensors and then
            # asks for exactly SadTalker_V0.0.2_<size>.safetensors, so a size
            # mismatch is a hard failure rather than a fallback.
            url = url_template.format(size=SADTALKER_SIZE)
            destination = SADTALKER_DIR / subdir / Path(url).name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.is_file() and destination.stat().st_size > 1024:
                log(f"  ✓ {subdir}/{destination.name} (cached)")
                continue
            proc = run("curl", "-sSL", "--retry", "3", "--max-time", "900",
                       "-o", str(destination), url)
            if proc.returncode != 0 or not destination.is_file():
                raise RuntimeError(
                    f"could not download {destination.name}: {proc.stderr[-300:]}")
            log(f"  ✓ {subdir}/{destination.name} "
                f"({destination.stat().st_size / 1e6:.0f} MB)")

    with timed("ffmpeg_on_path", timings):
        ensure_ffmpeg_on_path()

    with timed("sadtalker_face_models", timings):
        # face_alignment fetches these on first construction; doing it here means
        # a network failure is reported as a download problem, not a crash
        # halfway through the render.
        # Importing constructs the detector, which triggers the one-time weight
        # download. Doing it here means a network failure reads as a download
        # problem instead of a crash halfway through a render.
        __import__("face_alignment")
        log(f"face_alignment ready (cache {FACE_ALIGNMENT_CACHE})")
    return SADTALKER_DIR


def run_sadtalker(master: Path, voice: Path, timings: Dict[str, float]
                  ) -> Dict[str, Any]:
    """
    Render the talking head with SadTalker's own inference entry point.

    ``--still`` is deliberately NOT passed: that flag crops the result back to the
    source image's framing, which for a portrait means the head barely moves.
    Without it SadTalker animates the full 3D face region — head pose, blinks and
    expression — which is the whole point.

    ``--preprocess crop`` frames the face before animating; ``full`` would need
    the extra BFM fitting weights and is slower for a portrait crop.
    """
    output_dir = WORK_DIR / "sadtalker_out"
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable, "inference.py",
        "--driven_audio", str(voice),
        "--source_image", str(master),
        "--result_dir", str(output_dir),
        "--checkpoint_dir", str(SADTALKER_DIR / "checkpoints"),
        "--preprocess", "crop",
        "--size", str(SADTALKER_SIZE),
        "--batch_size", "2",
        "--expression_scale", "1.0",
        "--verbose",
    ]
    enhancer = SADTALKER_ENHANCER
    if enhancer:
        # Never pass --enhancer for a package that is not installed: SadTalker
        # imports gfpgan lazily inside the render loop and the whole run dies
        # there, minutes in, instead of degrading here, immediately.
        try:
            __import__("gfpgan")
        except Exception as exc:
            log(f"gfpgan unavailable ({type(exc).__name__}) — rendering "
                f"without the face enhancer", "WARN")
            enhancer = None
    if enhancer:
        command += ["--enhancer", enhancer]

    with timed("sadtalker_inference", timings):
        log(f"SadTalker: {' '.join(command[1:])}")
        proc = subprocess.run(command, cwd=str(SADTALKER_DIR),
                              capture_output=True, text=True, timeout=5400)
        tail = ((proc.stdout or "") + (proc.stderr or "")).strip().splitlines()
        for line in tail[-40:]:
            log(f"  [sadtalker] {line}", "INFO")
        if proc.returncode != 0:
            raise RuntimeError(f"SadTalker failed (exit {proc.returncode}): "
                               f"{(proc.stderr or proc.stdout)[-800:]}")

    # SadTalker writes <timestamp>/ for its intermediates and the finished
    # <timestamp>.mp4 beside it — plus <timestamp>_enhanced.mp4 when `--enhancer`
    # ran. The enhanced render is the one to mux: it is the same frames with the
    # mouth region repaired.
    candidates = list(output_dir.glob("*.mp4"))
    if not candidates:
        raise RuntimeError(f"SadTalker produced no mp4 in {output_dir}")
    enhanced = [p for p in candidates if p.stem.endswith("_enhanced")]
    pool = enhanced or candidates
    result = max(pool, key=lambda p: p.stat().st_mtime)

    # Record what the renderer actually consumed. SadTalker resizes every frame
    # to --size, so a 1024² portrait becomes a 512 px reel; logging the source
    # keeps that visible in the metrics instead of a surprise in the output.
    try:
        from PIL import Image as _PilImage

        with _PilImage.open(master) as probe:
            source_px = list(probe.size)
    except Exception:
        source_px = None
    log(f"avatar source {source_px} -> render {SADTALKER_SIZE}px wide")
    if not enhanced:
        log("SadTalker produced no *_enhanced.mp4 — using the raw render "
            f"({result.name})", "WARN")
    log(f"SadTalker wrote {result.name} ({result.stat().st_size / 1024:.0f} KB)")
    return {"engine": "SadTalker", "raw": str(result),
            "enhanced": bool(enhanced), "size": SADTALKER_SIZE,
            "preprocess": "crop", "enhancer": enhancer,
            "still": False, "fps": 25, "master_source_px": source_px}


# --------------------------------------------------------------------------- #
# Mux + verification
# --------------------------------------------------------------------------- #


def ensure_ffmpeg_on_path() -> str:
    """
    Put a real ``ffmpeg`` executable on PATH before SadTalker runs.

    SadTalker does not call ffmpeg through a library: it builds a shell string
    and runs it with ``os.system`` (``save_video_with_watermark`` in
    ``src/utils/videoio.py``). ``os.system`` reports failure only through its
    return code, which SadTalker ignores, so a missing binary surfaces later as
    ``FileNotFoundError`` from a ``shutil.move`` of a file that was never
    created. A Kaggle image has no system ffmpeg, so this is not hypothetical.

    ``imageio-ffmpeg`` ships a static build; its directory is prepended to PATH,
    with a ``ffmpeg`` symlink in case the bundled name is versioned.
    """
    found = shutil.which("ffmpeg")
    if found:
        log(f"system ffmpeg: {found}")
        return found

    import imageio_ffmpeg

    binary = Path(imageio_ffmpeg.get_ffmpeg_exe())
    shim_dir = WORK_DIR / "bin"
    shim_dir.mkdir(parents=True, exist_ok=True)
    shim = shim_dir / "ffmpeg"
    if not shim.is_file():
        try:
            shim.symlink_to(binary)
        except OSError:
            shutil.copy2(binary, shim)
            shim.chmod(0o755)
    os.environ["PATH"] = f"{shim_dir}{os.pathsep}{os.environ.get('PATH', '')}"
    os.environ["IMAGEIO_FFMPEG_EXE"] = str(binary)
    log(f"ffmpeg shimmed onto PATH from imageio-ffmpeg: {binary.name}")
    return str(binary)


def ffmpeg_bin() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        found = shutil.which("ffmpeg")
        if not found:
            raise RuntimeError("no ffmpeg on PATH and imageio-ffmpeg missing")
        return found


def ffprobe_bin() -> Optional[str]:
    found = shutil.which("ffprobe")
    if found:
        return found
    # imageio-ffmpeg ships ffmpeg only; look beside it in case a probe exists
    try:
        import imageio_ffmpeg

        sibling = Path(imageio_ffmpeg.get_ffmpeg_exe()).with_name("ffprobe")
        if sibling.is_file():
            return str(sibling)
    except Exception:
        pass
    return None


def mux_reel(raw_video: Path, voice: Path, timings: Dict[str, float]) -> Path:
    """
    The mandatory iPhone-compatible re-encode.

    Every parameter here exists because a phone needs it:

        -c:v libx264        H.264, the only codec every iOS and Android build
                            decodes in hardware
        -pix_fmt yuv420p    4:2:0 chroma; without it libx264 may pick yuv444p,
                            which QuickTime and Safari refuse to play
        -c:a aac            AAC-LC audio, not whatever SadTalker embedded
        -movflags +faststart  moves the moov atom to the front, so the file
                            starts playing before it is fully downloaded
        -shortest           the reel ends with the shorter of the two streams
                            instead of trailing silent video

    A single-stream copy is deliberately NOT used: SadTalker's own output is not
    guaranteed to be a compatible profile, and "it played on the machine that
    made it" is not the acceptance criterion.
    """
    reel = OUTPUT_DIR / REEL_FILENAME
    with timed("mux_reel", timings):
        command = [
            ffmpeg_bin(), "-y", "-v", "error",
            "-i", str(raw_video),
            "-i", str(voice),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-profile:v", "high",
            "-crf", "20", "-preset", "medium",
            "-c:a", "aac", "-b:a", "160k", "-ar", "44100", "-ac", "2",
            "-movflags", "+faststart",
            "-shortest",
            str(reel),
        ]
        proc = subprocess.run(command, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"mux failed: {proc.stderr[-600:]}")
        log(f"wrote {REEL_FILENAME} ({reel.stat().st_size / 1024:.0f} KB) · "
            f"h264/yuv420p + aac, faststart")
    return reel


def probe_media(path: Path) -> Dict[str, Any]:
    """ffprobe if present, else ffmpeg's stream table."""
    import json as _json

    probe = ffprobe_bin()
    if probe:
        proc = subprocess.run(
            [probe, "-v", "error", "-show_streams", "-show_format",
             "-of", "json", str(path)],
            capture_output=True, text=True)
        if proc.returncode == 0:
            try:
                data = _json.loads(proc.stdout)
            except Exception:
                data = {}
            streams = data.get("streams") or []
            video = next((s for s in streams
                          if s.get("codec_type") == "video"), None)
            audio = next((s for s in streams
                          if s.get("codec_type") == "audio"), None)
            return {
                "tool": "ffprobe",
                "video": None if not video else {
                    "codec": video.get("codec_name"),
                    "profile": video.get("profile"),
                    "pix_fmt": video.get("pix_fmt"),
                    "width": video.get("width"),
                    "height": video.get("height"),
                    "duration_s": float(video.get("duration") or 0.0),
                    "fps": video.get("r_frame_rate"),
                },
                "audio": None if not audio else {
                    "codec": audio.get("codec_name"),
                    "sample_rate": audio.get("sample_rate"),
                    "channels": audio.get("channels"),
                    "duration_s": float(audio.get("duration") or 0.0),
                },
                "duration_s": float((data.get("format") or {}).get("duration") or 0.0),
            }

    return _parse_ffmpeg_report(path)


def _parse_ffmpeg_report(path: Path) -> Dict[str, Any]:
    """
    Read ffmpeg's stream table when ffprobe is not available.

    Kaggle has ffmpeg (via imageio-ffmpeg) but not necessarily ffprobe, so the
    fallback has to verify the *same* contract, not a weaker one. ffmpeg's
    `-i` report is less structured but still states the codec and pixel format::

        Stream #0:0: Video: h264 (High), yuv420p(progressive), 1080x1920, 25 fps
        Stream #0:1: Audio: aac (LC), 44100 Hz, stereo, fltp, 160 kb/s

    Anything this cannot read is returned as ``None``, and the caller treats
    None as a failure rather than as "probably fine" — a reel we could not
    verify is not a reel we should ship.
    """
    report = subprocess.run([ffmpeg_bin(), "-hide_banner", "-i", str(path)],
                            capture_output=True, text=True).stderr
    video: Dict[str, Any] = {}
    audio: Dict[str, Any] = {}
    duration = 0.0

    for line in report.splitlines():
        text = line.strip()
        if text.startswith("Stream") and "Video:" in text:
            video["codec"] = text.split("Video:")[1].strip().split(" ")[0].split(",")[0]
            match = re.search(r"\b(yuv\w+|rgb\w+)(?:\(\w+\))?", text)
            video["pix_fmt"] = match.group(1) if match else None
            size = re.search(r"\b(\d{2,5})x(\d{2,5})\b", text)
            if size:
                video["width"], video["height"] = int(size.group(1)), int(size.group(2))
            fps = re.search(r"([\d.]+) fps", text)
            video["fps"] = fps.group(1) if fps else None
        elif text.startswith("Stream") and "Audio:" in text:
            audio["codec"] = text.split("Audio:")[1].strip().split(" ")[0].split(",")[0]
            rate = re.search(r"(\d{4,6}) Hz", text)
            audio["sample_rate"] = rate.group(1) if rate else None
        elif text.startswith("Duration:"):
            match = re.search(r"Duration: (\d+):(\d+):([\d.]+)", text)
            if match:
                hours, minutes, seconds = match.groups()
                duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)

    if not video and not audio:
        return {"tool": "ffmpeg", "video": None, "audio": None,
                "duration_s": duration,
                "report_tail": "\n".join(report.strip().splitlines()[-12:])}
    return {
        "tool": "ffmpeg",
        "video": video or None,
        "audio": audio or None,
        "duration_s": duration,
        "report_tail": "\n".join(report.strip().splitlines()[-12:]),
    }


def verify_reel(reel: Path, timings: Dict[str, float]) -> Dict[str, Any]:
    """
    The reel must be iPhone-playable, not merely present.

    Checks the exact contract the brief asks for — ``h264``, ``yuv420p``, ``aac``
    — plus a length-match between the streams, because a reel whose audio and
    picture disagree looks broken on a phone just as badly as a codec it cannot
    decode.
    """
    with timed("verify_reel", timings):
        info = probe_media(reel)
        if info.get("tool") != "ffprobe":
            log("ffprobe unavailable — verifying from ffmpeg's stream report",
                "WARN")

        video, audio = info.get("video"), info.get("audio")
        if video is None:
            raise RuntimeError("reel has no video stream")
        if audio is None:
            raise RuntimeError("reel has no audio stream")

        problems = []
        if video.get("codec") != "h264":
            problems.append(f"video codec is {video.get('codec')}, not h264")
        if video.get("pix_fmt") != "yuv420p":
            problems.append(f"pixel format is {video.get('pix_fmt')}, not yuv420p")
        if not video.get("width") or not video.get("height"):
            problems.append("video dimensions could not be read")
        if audio.get("codec") != "aac":
            problems.append(f"audio codec is {audio.get('codec')}, not aac")
        if problems:
            raise RuntimeError("reel is not iPhone-compatible: "
                               + "; ".join(problems))

        duration = info.get("duration_s") or video.get("duration_s") or 0.0
        if duration < MIN_REEL_SECONDS:
            raise RuntimeError(f"reel is only {duration:.2f}s "
                               f"(minimum {MIN_REEL_SECONDS}s)")
        v_dur = video.get("duration_s") or duration
        a_dur = audio.get("duration_s") or duration
        drift = abs(v_dur - a_dur)
        info["sync"] = {"video_s": round(v_dur, 3), "audio_s": round(a_dur, 3),
                        "drift_s": round(drift, 3), "ok": drift <= 0.5}
        if not info["sync"]["ok"]:
            raise RuntimeError(f"audio drift {drift:.2f}s exceeds 0.5s")

        info["iphone_compatible"] = True
        log(f"verified: {video['width']}x{video['height']} "
            f"{video['codec']}/{video['pix_fmt']} + {audio['codec']} · "
            f"{duration:.2f}s · drift {drift:.3f}s · iPhone-compatible ✓")
        return info


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def ensure_video_deps(timings: Dict[str, float]) -> None:
    with timed("pip_install_deps", timings):
        missing = missing_packages(CHATTERBOX_DEPS)
        if missing:
            pip_install(missing)
        if missing_packages(["chatterbox-tts"]):
            if not pip_install(["chatterbox-tts"], no_deps=True):
                raise RuntimeError("chatterbox-tts install failed — see above")
        log("chatterbox dependencies ready")

        sadtalker_missing = missing_packages(SADTALKER_DEPS)
        if sadtalker_missing:
            if not pip_install(sadtalker_missing):
                raise RuntimeError("SadTalker dependency install failed — see above")

        # gfpgan + basicsr: a source build whose metadata import needs the torch
        # stack we already have, so isolation has to be off.  --no-deps is not
        # used here: gfpgan's own runtime needs basicsr, facexlib and torch,
        # and the build deps above are cheap to resolve.
        if missing_packages(GFPGAN_DEPS):
            pip_install([d for d in GFPGAN_BUILD_DEPS
                         if missing_packages([d])],
                        no_build_isolation=True)
            if not pip_install(GFPGAN_DEPS, no_build_isolation=True):
                # A missing enhancer costs sharpness, not the reel: SadTalker
                # renders fine without --enhancer, so degrade instead of dying.
                log("gfpgan/basicsr could not be installed — the reel will be "
                    "rendered without the face enhancer", "WARN")
        log("SadTalker dependencies ready")


def main() -> int:
    started = time.perf_counter()
    telemetry = Tee(BUILD_LOG, sys.stdout)
    sys.stdout = telemetry  # type: ignore[assignment]

    timings: Dict[str, float] = {}
    metrics: Dict[str, Any] = {
        "process": "video",
        "device": PINNED_DEVICE,
        "status": "running",
        "started_at": ts(),
        "pipeline": "Chatterbox → SadTalker → H.264/AAC (iPhone-compatible)",
        "prohibitions_honoured": {
            "celebrity_name_in_prompts": False,
            "lip_sync_face_swap": False,
            "pipe.to_cpu": False,
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
        device = resolve_device()
        metrics["device"] = device
        log(f"device: {device} ({torch.cuda.device_count()} visible; "
            f"pinned device is {PINNED_DEVICE})")

        # The shims go first: they exist because `import basicsr` fails on this
        # torchvision/numpy, and the dependency probe below imports every
        # candidate package to decide what to install. Probing before shimming
        # makes an installed package look missing, and reinstalling it does not
        # help.
        apply_legacy_shims()
        ensure_video_deps(timings)

        # ---- voice ---------------------------------------------------------- #
        metrics["voice"] = synth_voice(timings)
        metrics["artifacts"].append(VOICE_FILENAME)

        # ---- avatar --------------------------------------------------------- #
        master = wait_for_master(timings)
        ensure_sadtalker(timings)
        avatar = run_sadtalker(master, OUTPUT_DIR / VOICE_FILENAME, timings)
        metrics["sadtalker"] = avatar

        raw = Path(avatar["raw"])
        archived = OUTPUT_DIR / RAW_VIDEO_FILENAME
        # keep the pre-mux file for post-mortems, but it is not a deliverable
        try:
            shutil.copy2(raw, archived)
        except Exception:
            pass

        # ---- mandatory iPhone-compatible mux -------------------------------- #
        reel = mux_reel(raw, OUTPUT_DIR / VOICE_FILENAME, timings)
        metrics["reel_probe"] = verify_reel(reel, timings)
        metrics["artifacts"].append(REEL_FILENAME)

        log("video pipeline complete — iPhone-compatible H.264/AAC reel")

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
        merge_metrics("video", metrics, METRICS_PATH, tag="vid")

        log("=" * 74)
        log(f"video pipeline status: {metrics['status']} · "
            f"{metrics['total_seconds']}s")
        for name in metrics["artifacts"]:
            path = OUTPUT_DIR / name
            log(f"  {'✓' if path.is_file() else '✗'} {name}"
                + (f" ({path.stat().st_size / 1024:.0f} KB)" if path.is_file() else ""))
        log("=" * 74)
        sys.stdout = telemetry._stream  # type: ignore[assignment]
        telemetry.close()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
