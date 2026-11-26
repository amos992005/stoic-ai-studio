#!/usr/bin/env python3
"""
stoic-ai-studio :: PHASE 2 — Micro-Pilot 2
"VOICE & TALKING AVATAR" — one-shot 5-6 s speaking video on a Kaggle T4.

Pipeline (all on one GPU box, ~3-4 min total)
--------------------------------------------
1. VOICE  : kokoro-onnx (ONNX Runtime, zero C++ compilation) with a blended
            style vector 0.6*bm_george + 0.4*am_michael at speed 0.92.
            Broadcast mastering: 80 Hz high-pass, warmth shelf, presence tilt,
            peak-normalise to -1.5 dBFS -> julian_voice.wav + .mp3
2. LIPSYNC: primary = Wav2Lip ONNX (introspected at runtime, so any export
            layout works). If the weights cannot be fetched or do not match,
            we fall back to a *viseme-driven* mouth synthesiser built from
            Kokoro's phoneme timings (lip-closure visemes + cavity darkening +
            feathered sharpening). The backend actually used is recorded in
            talking_metrics.json — no silent pretending.
3. MOTION : speech-envelope driven head micro-motion (yaw/pitch/roll/sway),
            breathing, natural blinks. 25 FPS.
4. ENCODE : ffmpeg (imageio-ffmpeg static build) -> libx264, yuv420p, AAC.
5. OUTPUT : talking_contact_sheet.png (frames + mouth zoom + waveform/lip curve)
            talking_metrics.json (incl. mouth-open vs audio-RMS correlation).

Why this shape
--------------
* Kaggle runs Python 3.13 + torch 2.11/cu130. Everything here is pure Python +
  ONNX Runtime (onnxruntime-gpu) + numpy + pillow, i.e. no `dlib`, `basicsr`,
  `spacy` or other source-compiling dependencies.
* Every optional stage degrades instead of aborting: a missing GPU, missing
  lip-sync weights or a missing ffmpeg encoder still produces a usable artifact.
"""

from __future__ import annotations

import gc
import json
import math
import os
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

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

SCRIPT_LINE = (
    "Most men are ruled by their impulses. "
    "The moment you master your silence, you master the room."
)
VOICE_BLEND = {"bm_george": 0.6, "am_michael": 0.4}
SPEECH_SPEED = 0.92
AUDIO_SR_TARGET = 24000
MASTER_PEAK_DBFS = -1.5

FPS = 25
MAX_SECONDS = 6.0          # pilot is a 5-6 s reel clip
OUT_W, OUT_H = 832, 1024   # portrait 4:5-ish crop of the 1024x1024 master

PORTRAIT_NAME = "julian_portrait_1024.png"
#: Kaggle dataset slug CI publishes the portrait under (files land in /kaggle/input).
PORTRAIT_DATASET = os.environ.get("PORTRAIT_DATASET_SLUG", "stoic-ai-studio-portrait")
VOICE_WAV = "julian_voice.wav"
VOICE_MP3 = "julian_voice.mp3"
VIDEO_MP4 = "pilot_talking.mp4"
CONTACT_SHEET = "talking_contact_sheet.png"
METRICS_JSON = "talking_metrics.json"
BUILD_LOG = "build.log"

#: Face geometry on the 1024x1024 master portrait (measured from the render).
EYE_L = (472, 436)
EYE_R = (576, 436)
#: Patch that contains upper lip -> chin, so the lower lip can travel down.
MOUTH_BOX = (426, 646, 558, 742)
#: The lip parting line inside MOUTH_BOX (darkest row = where the lips meet).
PARTING_Y = 686
#: Maximum lip separation in output pixels at full openness (1024-scale portrait).
MAX_LIP_DROP = 11

#: Wav2Lip ONNX candidates — tried in order, first hit wins. HF is reachable
#: from Kaggle (enable_internet) even though it is blocked in the authoring env.
WAV2LIP_REPOS: List[Tuple[str, str]] = [
    ("instant-high/wav2lip-onnx-96", "wav2lip.onnx"),
    ("Nekochu/Wav2Lip", "wav2lip.onnx"),
    ("camenduru/Wav2Lip", "wav2lip.onnx"),
    ("vinthony/wav2lip", "wav2lip_gan.onnx"),
]
ENABLE_NEURAL_LIPSYNC = os.environ.get("ENABLE_NEURAL_LIPSYNC", "1") in {"1", "true", "yes"}

#: Kokoro model assets (GitHub release + HuggingFace fallbacks).
KOKORO_MODEL_URLS = [
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
    "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro-v1.0.onnx",
]
KOKORO_VOICES_URLS = [
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
    "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices-v1.0.bin",
]


# --------------------------------------------------------------------------- #
# Logging / timing
# --------------------------------------------------------------------------- #


def log(message: str, level: str = "INFO") -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{stamp}] [{level:<5}] {message}", flush=True)


@contextmanager
def timed(label: str, timings: Dict[str, float]):
    log(f"▶ start: {label}")
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        timings[label] = round(elapsed, 3)
        log(f"✔ done:  {label} ({elapsed:.1f}s)")


class Tee:
    """Mirror stdout into build.log so crashes are debuggable from the repo."""

    def __init__(self, path: Path, stream):
        self._stream = stream
        self._file = open(path, "a", encoding="utf-8", buffering=1)

    def write(self, data: str) -> int:
        self._stream.write(data)
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


def resolve_portrait() -> Path:
    """The portrait is shipped next to this script by `kaggle kernels push`."""
    here = Path(__file__).resolve().parent
    candidates = [
        here / PORTRAIT_NAME,
        here / "assets" / PORTRAIT_NAME,
        Path("/kaggle/src") / PORTRAIT_NAME,          # sibling upload (best effort)
        Path.cwd() / PORTRAIT_NAME,
        Path.cwd() / "assets" / PORTRAIT_NAME,
        Path("/kaggle/input") / PORTRAIT_DATASET / PORTRAIT_NAME,
        Path("/kaggle/input") / "stoic-ai-studio-portrait" / PORTRAIT_NAME,
        Path("/kaggle/input") / "stoic-ai-studio" / PORTRAIT_NAME,
        here.parent / "assets" / "julian_master_face.png",
    ]
    # Any layout under /kaggle/input/*/ that happens to hold the portrait.
    if Path("/kaggle/input").is_dir():
        candidates.extend(sorted(Path("/kaggle/input").rglob(PORTRAIT_NAME)))
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "portrait not found. Attach the portrait as a Kaggle dataset source "
        f"(expected at /kaggle/input/{PORTRAIT_DATASET}/{PORTRAIT_NAME}). "
        "Searched: " + ", ".join(str(c) for c in candidates)
    )


# --------------------------------------------------------------------------- #
# Dependency bootstrap (never compile from source)
# --------------------------------------------------------------------------- #


def ensure_python_deps(timings: Dict[str, float]) -> None:
    """Install the pure-wheel dependency set if Kaggle's image lacks it."""
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
        return

    log(f"installing missing deps: {missing}", "WARN")
    with timed("pip_install_deps", timings):
        cmd = [
            sys.executable, "-m", "pip", "install", "--quiet", "--no-input",
            "--disable-pip-version-check", "--only-binary=:all:", *missing,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            log(proc.stderr[-3000:], "ERROR")
            # Retry without the binary-only flag as a last resort (still wheels
            # for every package we require, but tolerate a stray sdist).
            proc = subprocess.run(cmd[:-1], capture_output=True, text=True)
            if proc.returncode != 0:
                log(proc.stderr[-3000:], "ERROR")
                raise RuntimeError(f"could not install {missing}")
    for module in required:
        try:
            __import__(module)
        except Exception as exc:
            raise RuntimeError(f"{module} still unavailable after install: {exc}") from exc


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


# --------------------------------------------------------------------------- #
# Audio helpers (numpy only — no scipy needed)
# --------------------------------------------------------------------------- #


def biquad_highpass(signal: np.ndarray, sr: int, cutoff_hz: float, q: float = 0.707) -> np.ndarray:
    """RBJ cookbook 2nd-order high-pass, applied twice for a 24 dB/oct slope."""
    if signal.size == 0:
        return signal
    w0 = 2 * math.pi * cutoff_hz / sr
    alpha = math.sin(w0) / (2 * q)
    cos_w0 = math.cos(w0)
    b0 = (1 + cos_w0) / 2
    b1 = -(1 + cos_w0)
    b2 = (1 + cos_w0) / 2
    a0 = 1 + alpha
    a1 = -2 * cos_w0
    a2 = 1 - alpha
    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])

    out = signal.astype(np.float32).copy()
    for _ in range(2):
        out = _lfilter(b, a, out)
    return out


def _lfilter(b: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Minimal direct-form II transposed biquad (numpy loop; audio is short)."""
    y = np.empty_like(x)
    x1 = x2 = y1 = y2 = 0.0
    for i in range(x.size):
        xi = float(x[i])
        yi = b[0] * xi + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2
        x2, x1 = x1, xi
        y2, y1 = y1, yi
        y[i] = yi
    return y


def peaking_boost(signal: np.ndarray, sr: int, freq: float, gain_db: float, q: float = 0.9) -> np.ndarray:
    """Gentle RBJ peaking EQ — used for 'presence warmth' around 200 Hz."""
    if signal.size == 0 or abs(gain_db) < 0.01:
        return signal
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * math.pi * freq / sr
    alpha = math.sin(w0) / (2 * q)
    cos_w0 = math.cos(w0)
    b0 = 1 + alpha * A
    b1 = -2 * cos_w0
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * cos_w0
    a2 = 1 - alpha / A
    return _lfilter(np.array([b0, b1, b2]) / a0, np.array([1.0, a1 / a0, a2 / a0]),
                    signal.astype(np.float32))


def normalize_peak(signal: np.ndarray, target_dbfs: float) -> np.ndarray:
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    if peak <= 1e-9:
        return signal
    target = 10 ** (target_dbfs / 20.0)
    return (signal * (target / peak)).astype(np.float32)


def db(x: np.ndarray, floor: float = 1e-10) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(x), floor))


def rms_envelope(signal: np.ndarray, sr: int, fps: int = FPS) -> np.ndarray:
    """Per-video-frame RMS, normalised to 0..1 — the speech envelope."""
    hop = max(1, int(round(sr / fps)))
    frames = int(math.ceil(len(signal) / hop))
    out = np.zeros(frames, dtype=np.float32)
    for i in range(frames):
        chunk = signal[i * hop : (i + 1) * hop]
        out[i] = float(np.sqrt(np.mean(chunk**2))) if chunk.size else 0.0
    peak = out.max() if out.size else 0.0
    return out / peak if peak > 1e-9 else out


def smooth(values: np.ndarray, radius: int, passes: int = 2) -> np.ndarray:
    """Cheap gaussian-ish smoothing without scipy."""
    if values.size == 0 or radius <= 0:
        return values
    out = values.astype(np.float32)
    kernel = np.hanning(radius * 2 + 1)
    kernel /= kernel.sum()
    for _ in range(passes):
        out = np.convolve(out, kernel, mode="same")
    return out


# --------------------------------------------------------------------------- #
# 1. VOICE
# --------------------------------------------------------------------------- #


def download_first(urls: Sequence[str], dest: Path, timeout: int = 300) -> Path:
    """Download the first URL that works, resuming nothing (small files)."""
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    errors: List[str] = []
    for url in urls:
        try:
            log(f"downloading {url}")
            request = urllib.request.Request(url, headers={"User-Agent": "stoic-ai-studio/2"})
            with urllib.request.urlopen(request, timeout=timeout) as response, open(dest, "wb") as handle:
                shutil.copyfileobj(response, handle, length=1024 * 1024)
            size_mb = dest.stat().st_size / 1024**2
            if size_mb < 1:
                raise RuntimeError(f"file looks truncated ({size_mb:.2f} MB)")
            log(f"saved {dest.name} ({size_mb:.1f} MB)")
            return dest
        except Exception as exc:
            errors.append(f"{url} -> {type(exc).__name__}: {exc}")
            log(f"download failed: {errors[-1]}", "WARN")
    raise RuntimeError("all download sources failed:\n  " + "\n  ".join(errors))


def synthesize_voice(output_dir: Path, timings: Dict[str, float]) -> Dict[str, Any]:
    """Kokoro blend -> mastered wav + mp3. Returns metadata for metrics.json."""
    import soundfile as sf
    from kokoro_onnx import Kokoro

    # Anything inside output_dir is downloaded by `kaggle kernels output`, so the
    # ~330 MB of Kokoro weights must live elsewhere (tmp) or every run pays for
    # transferring them back to CI.
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
            # Use the GPU provider when onnxruntime-gpu is installed (T4): CPU
            # inference would otherwise dominate the 10-minute budget.
            import onnxruntime as ort
            providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                         if p in ort.get_available_providers()]
            if providers and providers[0] == "CUDAExecutionProvider":
                session = ort.InferenceSession(str(model_path), providers=providers)
                kokoro = Kokoro.from_session(session, str(voices_path))
                log(f"kokoro session providers: {session.get_providers()}")
            else:
                kokoro = Kokoro(str(model_path), str(voices_path))
                log("kokoro running on CPU (no CUDA provider installed)")
        except Exception as exc:
            log(f"provider-pinned session failed ({exc}); using default loader", "WARN")
            kokoro = Kokoro(str(model_path), str(voices_path))
        available = set(kokoro.get_voices())
        log(f"kokoro voices available: {sorted(available)[:12]}… total={len(available)}")

    # Blend style vectors: voices-v1.0.bin holds them as a 510-row matrix per voice.
    with timed("blend_voice_styles", timings):
        missing = [v for v in VOICE_BLEND if v not in available]
        if missing:
            raise RuntimeError(f"voice(s) {missing} not in voices-v1.0.bin")
        # voices-v1.0.bin ships TWO layouts depending on the export:
        #   * (510, 256)  -> one style row per phoneme count
        #   * (1, 256)    -> the packer already resolved a single row
        # The weighted sum must preserve whatever rank/shape the file has, or
        # onnxruntime rejects the tensor ("Invalid rank for input: style").
        blend: Optional[np.ndarray] = None
        reference_shape: tuple = ()
        for name, weight in VOICE_BLEND.items():
            style = np.asarray(kokoro.get_voice_style(name), dtype=np.float32)
            if reference_shape and style.shape != reference_shape:
                raise RuntimeError(
                    f"voice {name} has shape {style.shape}, expected {reference_shape}"
                )
            reference_shape = style.shape
            if not np.isfinite(style).all():
                raise RuntimeError(f"voice {name} contains non-finite values")
            blend = style * weight if blend is None else blend + style * weight
        assert blend is not None
        blend = np.ascontiguousarray(blend, dtype=np.float32)
        # Guard: the model expects (rows, 256)-ish; a 1-D tensor cannot be valid.
        if blend.ndim == 1:
            raise RuntimeError(f"blended style collapsed to rank 1: {blend.shape}")
        log(f"blended style matrix: {blend.shape} from "
            + ", ".join(f"{k}={v}" for k, v in VOICE_BLEND.items()))

    with timed("synthesize_speech", timings):
        audio, sr, phoneme_timings = kokoro.create_timed(
            SCRIPT_LINE, blend, speed=SPEECH_SPEED, lang="en-us"
        )
        audio = np.asarray(audio, dtype=np.float32).squeeze()
        if audio.size == 0:
            raise RuntimeError("kokoro returned empty audio")
        log(f"raw speech: {audio.size / sr:.2f}s @ {sr} Hz, {len(phoneme_timings)} phoneme timings")
        if sr != AUDIO_SR_TARGET:
            log(f"resampling {sr} -> {AUDIO_SR_TARGET} Hz", "WARN")
            audio = resample_linear(audio, sr, AUDIO_SR_TARGET)
            sr = AUDIO_SR_TARGET
            # Timings were computed against the original rate; scale to match.
            scale = AUDIO_SR_TARGET / float(sr)

    peak_in = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak_in < 1e-5:
        raise RuntimeError(
            "synthesised audio is silent (peak < -100 dBFS) — the TTS model or "
            "voice blend produced no signal"
        )
    log(f"raw peak: {float(db(audio).max()):.1f} dBFS")

    with timed("master_audio", timings):
        mastered = biquad_highpass(audio, sr, 80.0)          # 80 Hz high-pass
        mastered = peaking_boost(mastered, sr, 190.0, 2.2)   # warmth
        mastered = peaking_boost(mastered, sr, 4200.0, -1.2) # tame harshness
        mastered = normalize_peak(mastered, MASTER_PEAK_DBFS)
        mastered = np.clip(mastered, -1.0, 1.0)
        log(f"mastered: peak {float(db(mastered).max()):.2f} dBFS, "
            f"{mastered.size / sr:.2f}s")

    wav_path = output_dir / VOICE_WAV
    with timed("write_audio", timings):
        sf.write(wav_path, mastered, sr, subtype="PCM_16")
        ffmpeg = ffmpeg_binary()
        mp3_path = output_dir / VOICE_MP3
        cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(wav_path),
               "-c:a", "libmp3lame", "-b:a", "192k", str(mp3_path)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not mp3_path.exists():
            log(f"mp3 encode failed ({proc.stderr.strip()[-300:]}); trying aac/m4a", "WARN")
            cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(wav_path),
                   "-c:a", "aac", "-b:a", "192k", str(mp3_path.with_suffix(".m4a"))]
            subprocess.run(cmd, capture_output=True, text=True)
        log(f"wrote {wav_path.name} and {mp3_path.name}")

    return {
        "text": SCRIPT_LINE,
        "voice_blend": VOICE_BLEND,
        "speed": SPEECH_SPEED,
        "sample_rate": int(sr),
        "duration_s": round(float(mastered.size / sr), 3),
        "peak_dbfs": round(float(db(mastered).max()), 2),
        "phoneme_timings": [
            {"phoneme": t.phoneme, "start": round(float(t.start), 4), "end": round(float(t.end), 4)}
            for t in (phoneme_timings or [])
        ],
        "audio": mastered,
    }


def resample_linear(signal: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out or signal.size == 0:
        return signal
    n_out = int(round(signal.size * sr_out / sr_in))
    x_old = np.linspace(0, 1, signal.size, endpoint=False)
    x_new = np.linspace(0, 1, n_out, endpoint=False)
    return np.interp(x_new, x_old, signal).astype(np.float32)


# --------------------------------------------------------------------------- #
# Visemes -> mouth-openness / width drivers
# --------------------------------------------------------------------------- #

#: IPA character -> (openness 0..1, width -1..1). Kokoro timings report IPA.
_VISEMES: Dict[str, Tuple[float, float]] = {
    # bilabials: lips pressed together, then release
    "m": (0.05, 0.0), "b": (0.10, 0.0), "p": (0.10, 0.0),
    # labiodentals: slight upper-teeth/lip contact
    "f": (0.14, 0.15), "v": (0.16, 0.15),
    # rounded vowels / w
    "o": (0.55, -0.85), "ɔ": (0.62, -0.8), "u": (0.35, -0.95), "ʊ": (0.34, -0.85),
    "w": (0.22, -0.7), "ou": (0.5, -0.8),
    # open vowels
    "a": (0.95, 0.1), "ɑ": (0.95, 0.05), "æ": (0.85, 0.55), "ʌ": (0.7, 0.25),
    "ɐ": (0.7, 0.2), "ə": (0.42, 0.15), "ɜ": (0.55, 0.2), "ɚ": (0.5, 0.2),
    # mid vowels
    "e": (0.55, 0.7), "ɛ": (0.75, 0.6), "ɪ": (0.4, 0.6), "i": (0.34, 0.8),
    # alveolar / velar / others: mouth slightly open, neutral
    "t": (0.18, 0.1), "d": (0.2, 0.1), "n": (0.16, 0.05), "l": (0.3, 0.1),
    "r": (0.3, -0.15), "ɹ": (0.3, -0.15), "s": (0.12, 0.35), "z": (0.14, 0.35),
    "ʃ": (0.22, -0.35), "ʒ": (0.22, -0.35), "tʃ": (0.24, -0.3), "dʒ": (0.24, -0.3),
    "k": (0.22, 0.0), "g": (0.24, 0.0), "ŋ": (0.2, 0.0),
    "h": (0.3, 0.1), "θ": (0.18, 0.2), "ð": (0.2, 0.2),
    "j": (0.28, 0.7), "ʤ": (0.24, -0.3), "ʧ": (0.24, -0.3),
}
_NEUTRAL = (0.12, 0.0)


def viseme_track(
    phoneme_timings: Sequence[Dict[str, float]],
    duration: float,
    fps: int = FPS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build per-frame mouth-openness and mouth-width curves from phoneme timings.

    Falls back to a neutral curve (later blended with the audio envelope) when the
    model export does not report timings.
    """
    frames = max(1, int(math.ceil(duration * fps)))
    openness = np.full(frames, _NEUTRAL[0], dtype=np.float32)
    width = np.full(frames, _NEUTRAL[1], dtype=np.float32)
    if not phoneme_timings:
        return openness, width

    for item in phoneme_timings:
        phoneme = str(item.get("phoneme", "")).strip().lower()
        if not phoneme:
            continue
        # Kokoro may hand back a character or a short cluster; take the longest match.
        key = next((k for k in sorted(_VISEMES, key=len, reverse=True) if k in phoneme), None)
        target_open, target_width = _VISEMES.get(key, _NEUTRAL) if key else _NEUTRAL
        start = max(0, int(math.floor(float(item.get("start", 0.0)) * fps)))
        end = min(frames, int(math.ceil(float(item.get("end", 0.0)) * fps)) + 1)
        if end <= start:
            continue
        span = end - start
        ramp = max(1, span // 3)
        for i in range(start, end):
            # ease in / hold / ease out so consecutive phonemes interpolate
            offset = i - start
            if offset < ramp:
                t = (offset + 1) / (ramp + 1)
            elif offset > span - ramp:
                t = max(0.15, (span - offset) / (ramp + 1))
            else:
                t = 1.0
            openness[i] += (target_open - openness[i]) * t
            width[i] += (target_width - width[i]) * t

    return openness, width


# --------------------------------------------------------------------------- #
# 2. Neural lip-sync (Wav2Lip ONNX) — optional, introspected, never fatal
# --------------------------------------------------------------------------- #


def _mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """Triangular mel filterbank (HTK mel scale), no librosa needed."""
    def hz_to_mel(f: float) -> float:
        return 2595.0 * math.log10(1.0 + f / 700.0)

    def mel_to_hz(m: float) -> float:
        return 700.0 * (10 ** (m / 2595.0) - 1.0)

    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    hz = mel_to_hz(mels)
    bank = np.zeros((n_mels, freqs.size), dtype=np.float32)
    for i in range(n_mels):
        lower, center, upper = hz[i], hz[i + 1], hz[i + 2]
        for j, f in enumerate(freqs):
            if lower <= f <= center and center > lower:
                bank[i, j] = (f - lower) / (center - lower)
            elif center < f <= upper and upper > center:
                bank[i, j] = (upper - f) / (upper - center)
    return bank


def audio_to_wav2lip_mel(audio: np.ndarray, sr: int) -> np.ndarray:
    """STFT -> mel -> log/dB normalisation, Wav2Lip style. Returns (80, T) float32."""
    n_fft, hop, n_mels, fmin, fmax = 800, 200, 80, 55.0, 7600.0
    target_sr = 16000
    if sr != target_sr:
        audio = resample_linear(audio, sr, target_sr)
    work = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])  # pre-emphasis
    if work.size < n_fft:
        work = np.pad(work, (0, n_fft - work.size))
    frames = 1 + (work.size - n_fft) // hop
    window = np.hanning(n_fft + 1)[:-1]
    stft = np.empty((n_fft // 2 + 1, frames), dtype=np.complex64)
    for i in range(frames):
        segment = work[i * hop : i * hop + n_fft] * window
        stft[:, i] = np.fft.rfft(segment)
    magnitudes = np.abs(stft)
    bank = _mel_filterbank(target_sr, n_fft, n_mels, fmin, fmax)
    mel = bank @ magnitudes
    mel = np.log(np.maximum(1e-5, mel))
    # reference preprocessing: min_level_db=-100, ref_level_db=20 -> [0, 1]
    mel = np.clip((mel + 100.0) / 100.0, 0.0, 1.0)
    return mel.astype(np.float32)


def try_neural_lipsync(
    portrait_rgb: np.ndarray,
    audio: np.ndarray,
    sr: int,
    n_frames: int,
    timings: Dict[str, float],
) -> Optional[Dict[str, Any]]:
    """
    Attempt Wav2Lip ONNX. Returns {"frames": list[(crop, box)], "model": name} or
    None. Any failure (download, shape mismatch, CUDA) returns None so the caller
    silently falls back to the viseme synthesiser.
    """
    if not ENABLE_NEURAL_LIPSYNC:
        log("neural lip-sync disabled by ENABLE_NEURAL_LIPSYNC=0")
        return None

    try:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        log(f"neural lip-sync unavailable ({exc})", "WARN")
        return None

    model_path: Optional[Path] = None
    model_name = ""
    for repo, filename in WAV2LIP_REPOS:
        try:
            log(f"probing lip-sync model {repo}/{filename}")
            model_path = Path(hf_hub_download(repo_id=repo, filename=filename))
            model_name = f"{repo}/{filename}"
            break
        except Exception as exc:
            log(f"  miss ({type(exc).__name__}: {str(exc)[:120]})", "WARN")
    if model_path is None:
        log("no Wav2Lip ONNX weights reachable — using viseme synthesiser", "WARN")
        return None

    try:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        available = ort.get_available_providers()
        providers = [p for p in providers if p in available] or ["CPUExecutionProvider"]
        session = ort.InferenceSession(str(model_path), providers=providers)
        log(f"wav2lip session on {session.get_providers()}")

        inputs = {i.name: i for i in session.get_inputs()}
        outputs = session.get_outputs()
        log(f"  inputs: {[(i.name, i.shape) for i in inputs.values()]}")
        log(f"  outputs: {[(o.name, o.shape) for o in outputs]}")

        mel = audio_to_wav2lip_mel(audio, sr)  # (80, T)
        face_bgr = portrait_rgb[:, :, ::-1].astype(np.float32) / 255.0
        box = MOUTH_BOX
        pad = 56
        x0, y0 = max(0, box[0] - pad), max(0, box[1] - pad * 2)
        x1, y1 = min(face_bgr.shape[1], box[2] + pad), min(face_bgr.shape[0], box[3] + pad)
        face_crop = face_bgr[y0:y1, x0:x1]
        face_96 = np.asarray(
            _resize_hwc(face_crop, 96), dtype=np.float32
        )

        mel_per_frame = mel.shape[1] / max(1, n_frames)
        crops: List[np.ndarray] = []
        for index in range(n_frames):
            centre = int(index * mel_per_frame)
            start = max(0, centre - 8)
            chunk = mel[:, start : start + 16]
            if chunk.shape[1] < 16:
                chunk = np.pad(chunk, ((0, 0), (0, 16 - chunk.shape[1])), mode="edge")
            feed: Dict[str, np.ndarray] = {}
            for name, spec in inputs.items():
                shape = [d if isinstance(d, int) and d > 0 else None for d in spec.shape]
                if len(spec.shape) == 4 and "img" not in name:      # mel chunk (N,1,80,16)
                    feed[name] = chunk[None, None, :, :].astype(np.float32)
                elif len(spec.shape) == 4:                          # face (N,3,96,96)
                    feed[name] = face_96[None].transpose(0, 3, 1, 2).astype(np.float32)
                elif len(spec.shape) == 2:                          # flat mel
                    feed[name] = chunk.reshape(1, -1).astype(np.float32)
                else:
                    fill = [1] + [1 for _ in spec.shape[1:]]
                    feed[name] = np.zeros(fill, dtype=np.float32)
            pred = session.run(None, feed)[0]
            pred = np.asarray(pred)
            if pred.ndim == 4:
                pred = pred[0]
            if pred.shape[0] == 6:      # image + mask
                pred = pred[3:]
            pred = np.transpose(pred, (1, 2, 0))
            if pred.max() <= 1.01:
                pred = pred * 255.0
            crops.append(np.clip(pred, 0, 255).astype(np.uint8))

        log(f"neural lip-sync produced {len(crops)} mouth crops via {model_name}")
        return {"crops": crops, "box": (x0, y0, x1, y1), "model": model_name}
    except Exception as exc:
        log(f"neural lip-sync failed at inference ({type(exc).__name__}: {exc})", "WARN")
        log(traceback.format_exc()[-1200:], "DEBUG")
        return None
    finally:
        gc.collect()


def _resize_hwc(image: np.ndarray, size: int) -> np.ndarray:
    from PIL import Image

    pil = Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8))
    return np.asarray(pil.resize((size, size), Image.BILINEAR), dtype=np.uint8)


# --------------------------------------------------------------------------- #
# 3. Frame synthesis: motion, blinks, mouth
# --------------------------------------------------------------------------- #


def feather_mask(shape: Tuple[int, int], box: Tuple[int, int, int, int], feather: int) -> np.ndarray:
    """Soft elliptical-ish mask over `box`, 1 inside, 0 outside, feathered edges."""
    height, width = shape
    x0, y0, x1, y1 = box
    mask = np.zeros((height, width), dtype=np.float32)
    ys = np.arange(y0, min(y1, height))
    xs = np.arange(x0, min(x1, width))
    if ys.size == 0 or xs.size == 0:
        return mask
    cy, cx = (y0 + y1) / 2.0, (x0 + x1) / 2.0
    ry, rx = max(1.0, (y1 - y0) / 2.0), max(1.0, (x1 - x0) / 2.0)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    dist = np.sqrt(((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2)
    inner = max(0.0, 1.0 - feather / max(rx, ry))
    alpha = np.clip((1.0 - dist) / max(1e-3, 1.0 - inner), 0.0, 1.0)
    alpha = alpha * alpha * (3 - 2 * alpha)  # smoothstep
    mask[y0 : ys.max() + 1, x0 : xs.max() + 1] = alpha
    return mask


def open_mouth_patch(
    region: np.ndarray, amount: float, parting_local: int, max_drop: int = MAX_LIP_DROP
) -> np.ndarray:
    """
    Open the mouth inside a lip patch.

    Stretching the patch vertically smears the lips into matching bands (looks
    like a cheap morph). Instead we translate the *lower lip + chin* band down
    and paint the revealed gap as a shadowed oral cavity, which is what a real
    opening looks like from this angle.
    """
    if amount <= 0.04:
        return region
    height, width = region.shape[:2]
    drop = int(round(max_drop * float(np.clip(amount, 0.0, 1.0))))
    drop = max(0, min(drop, height - parting_local - 4))
    if drop < 1:
        return region

    out = region.copy()
    # 1) slide the lower lip / chin band down (top edge stays put)
    out[parting_local + drop : height] = region[parting_local : height - drop]

    # 2) the revealed gap becomes a dark cavity, feathered at the lip corners
    ys = np.arange(drop)
    vertical = np.sin(np.pi * (ys + 0.5) / drop)[:, None, None]      # 0 -> 1 -> 0
    xs = np.linspace(-1.0, 1.0, width)[None, :, None]
    horizontal = np.clip(1.15 - 1.05 * np.abs(xs) ** 1.7, 0.0, 1.0)  # fades at corners
    alpha = np.clip(vertical * horizontal * (0.62 + 0.33 * amount), 0.0, 0.95)
    cavity = np.array([30, 11, 13], dtype=np.float32)
    gap = out[parting_local : parting_local + drop].astype(np.float32)
    out[parting_local : parting_local + drop] = (gap * (1 - alpha) + cavity * alpha).astype(np.uint8)

    # 3) shadow under the upper lip + a lit top edge on the lower lip
    shadow = out[parting_local : parting_local + 1].astype(np.float32) * 0.82
    out[parting_local : parting_local + 1] = np.clip(shadow, 0, 255).astype(np.uint8)
    lip_top = out[parting_local + drop : parting_local + drop + 2].astype(np.float32)
    out[parting_local + drop : parting_local + drop + 2] = np.clip(lip_top * 1.06 + 3, 0, 255).astype(np.uint8)
    return out


def apply_head_motion(
    frame: np.ndarray, yaw: float, pitch: float, roll: float, sway: float
) -> np.ndarray:
    """
    Cheap pseudo-3D head motion: horizontal shear (yaw), vertical scale (pitch),
    rotation (roll) and translation (sway), then a slight zoom so no border shows.
    """
    from PIL import Image

    h, w = frame.shape[:2]
    image = Image.fromarray(frame)
    zoom = 1.06
    # yaw -> horizontal shear; pitch -> vertical shear; roll -> rotation
    a = math.cos(roll)
    b = math.sin(roll)
    cx, cy = w / 2.0, h / 2.0
    # affine: x' = a*x + b*y + tx ; y' = c*x + d*y + ty  (inverse mapping)
    # Motion is deliberately measured in PIXELS: a misplaced multiplier here
    # (e.g. pitch * h) makes the head slide by hundreds of pixels and pulls the
    # mouth out of its tracking box.
    sway_px = sway * w * 0.6          # ~0-3 px lateral drift
    pitch_px = pitch * 9.0            # ~0-6 px vertical bob
    data = (
        a / zoom, b / zoom, (cx - a * cx / zoom - b * cy / zoom) + sway_px,
        -b / zoom * 1.0, a / zoom, (cy + b * cx / zoom - a * cy / zoom) + pitch_px,
    )
    sheared = image.transform((w, h), Image.AFFINE, data, Image.BICUBIC)
    # yaw: horizontal shear about the vertical centre -> reads as a 3D turn
    if abs(yaw) > 1e-4:
        shear = yaw * 0.035
        sheared = sheared.transform(
            (w, h), Image.AFFINE, (1, shear, -shear * cy, 0, 1, 0), Image.BICUBIC
        )
    return np.asarray(sheared)


def apply_blink(frame: np.ndarray, amount: float) -> np.ndarray:
    """Close the eyes by painting a lid-coloured ellipse over each eye."""
    if amount <= 0.01:
        return frame
    from PIL import Image, ImageDraw

    image = Image.fromarray(frame)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for (ex, ey) in (EYE_L, EYE_R):
        # sample the brow/skin colour just above the eye so the lid matches skin
        sample = frame[max(0, ey - 26) : ey - 18, ex - 8 : ex + 8]
        colour = tuple(int(v) for v in (sample.reshape(-1, 3).mean(axis=0) if sample.size else (70, 55, 45)))
        rx, ry = 26, int(15 * amount) + 2
        draw.ellipse((ex - rx, ey - ry, ex + rx, ey + ry), fill=colour + (int(215 * min(1.0, amount)),))
    composited = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    return np.asarray(composited)


def sharpen_region(frame: np.ndarray, box: Tuple[int, int, int, int], mask: np.ndarray) -> np.ndarray:
    """Feathered unsharp-mask over the mouth region (keeps the face razor-sharp)."""
    from PIL import Image, ImageFilter

    x0, y0, x1, y1 = box
    patch = Image.fromarray(frame[y0:y1, x0:x1]).filter(
        ImageFilter.UnsharpMask(radius=2.2, percent=85, threshold=2)
    )
    sharp = frame.copy()
    sharp[y0:y1, x0:x1] = np.asarray(patch)
    alpha = mask[..., None]
    return (sharp * alpha + frame * (1.0 - alpha)).astype(np.uint8)


# --------------------------------------------------------------------------- #
# 4. Video render
# --------------------------------------------------------------------------- #


def prepare_base(portrait: np.ndarray) -> np.ndarray:
    """Crop the square master to a portrait framing for shorts."""
    from PIL import Image

    height, width = portrait.shape[:2]
    crop_w = int(round(height * OUT_W / OUT_H))
    x0 = max(0, (width - crop_w) // 2)
    crop = portrait[:, x0 : x0 + crop_w]
    return np.asarray(Image.fromarray(crop).resize((OUT_W, OUT_H), Image.LANCZOS))


def scaled_face_geometry(portrait_size: int, base_shape: Tuple[int, int]) -> Dict[str, Any]:
    """
    Map face landmarks from the square master onto the cropped output frame.

    `prepare_base()` centre-crops horizontally and then scales to OUT_W x OUT_H, so
    x = (x - crop_offset) * OUT_W / crop_w and y = y * OUT_H / 1024. Treating the
    crop as a plain translation would misplace the mouth by the crop offset.
    """
    height, width = base_shape
    crop_w = int(round(portrait_size * OUT_W / OUT_H))
    crop_w = min(portrait_size, max(1, crop_w))
    offset_x = (portrait_size - crop_w) / 2.0
    sx = width / float(crop_w)
    sy = height / float(portrait_size)

    def map_x(x: float) -> int:
        return int(round((x - offset_x) * sx))

    def map_y(y: float) -> int:
        return int(round(y * sy))

    box = (map_x(MOUTH_BOX[0]), map_y(MOUTH_BOX[1]), map_x(MOUTH_BOX[2]), map_y(MOUTH_BOX[3]))
    return {
        "mouth_box": box,
        "parting_local": max(1, map_y(PARTING_Y) - box[1]),
        "eye_l": (map_x(EYE_L[0]), map_y(EYE_L[1])),
        "eye_r": (map_x(EYE_R[0]), map_y(EYE_R[1])),
    }


def blink_curve(n_frames: int, fps: int) -> np.ndarray:
    """Deterministic natural blink schedule (2-4 s apart, ~160 ms closure)."""
    curve = np.zeros(n_frames, dtype=np.float32)
    rng = np.random.default_rng(7)
    t = 1.4
    while True:
        start = int(t * fps)
        if start >= n_frames:
            break
        width = max(2, int(0.16 * fps))
        for i in range(width):
            idx = start + i
            if idx >= n_frames:
                break
            curve[idx] = max(curve[idx], math.sin(math.pi * (i + 0.5) / width) ** 1.5)
        t += float(rng.uniform(2.4, 4.2))
    return curve


def render_video(
    portrait: np.ndarray,
    audio: np.ndarray,
    sr: int,
    phoneme_timings: Sequence[Dict[str, float]],
    neural: Optional[Dict[str, Any]],
    timings: Dict[str, float],
) -> Dict[str, Any]:
    from PIL import Image

    duration = min(MAX_SECONDS, audio.size / sr)
    n_frames = max(1, int(round(duration * FPS)))
    log(f"rendering {n_frames} frames @ {FPS} fps ({duration:.2f}s)")

    base = prepare_base(portrait)
    geometry = scaled_face_geometry(portrait.shape[0], base.shape[:2])
    mouth_box = geometry["mouth_box"]
    log(f"output {OUT_W}x{OUT_H}, mouth box {mouth_box}")

    # --- drivers -----------------------------------------------------------
    env = rms_envelope(audio, sr, FPS)[:n_frames]
    if env.size < n_frames:
        env = np.pad(env, (0, n_frames - env.size))
    env_smooth = smooth(env, 2, passes=2)
    open_curve, width_curve = viseme_track(phoneme_timings, duration, FPS)
    if open_curve.size < n_frames:
        pad = n_frames - open_curve.size
        open_curve = np.pad(open_curve, (0, pad), constant_values=_NEUTRAL[0])
        width_curve = np.pad(width_curve, (0, pad), constant_values=_NEUTRAL[1])

    # Viseme drives shape; the acoustic envelope guarantees the lips track energy
    # even when timings are missing (and keeps audio/mouth in lock-step).
    openness = np.clip(0.70 * open_curve + 0.75 * env_smooth, 0.0, 1.0)
    openness = smooth(openness, 1, passes=1)
    viseme_weight = 1.0 if phoneme_timings else 0.0
    if not phoneme_timings:
        openness = np.clip(open_curve * 0.0 + 0.95 * env_smooth, 0.0, 1.0)
        log("no phoneme timings — mouth driven by the acoustic envelope only", "WARN")

    blinks = blink_curve(n_frames, FPS)
    lowpass = smooth(env, 6, passes=2)
    fade = np.clip(np.minimum(np.arange(n_frames) / (0.5 * FPS), (n_frames - np.arange(n_frames)) / (0.5 * FPS)), 0, 1)
    breath = np.sin(2 * math.pi * np.arange(n_frames) / (FPS * 4.2))

    feather = max(10, int((mouth_box[3] - mouth_box[1]) * 0.55))
    mask = feather_mask(base.shape[:2], mouth_box, feather)
    sharp_mask = feather_mask(base.shape[:2],
                              (mouth_box[0] - 18, mouth_box[1] - 14, mouth_box[2] + 18, mouth_box[3] + 14),
                              feather)

    frames: List[np.ndarray] = []
    mouth_crops: List[np.ndarray] = []
    rng = np.random.default_rng(11)

    with timed("render_frames", timings):
        for index in range(n_frames):
            frame = base.copy()

            # 1) mouth: neural crop (if any) else viseme warp + cavity shading
            open_amount = float(openness[index])
            if neural is not None:
                nx0, ny0, nx1, ny1 = neural["box"]
                crop = neural["crops"][index % len(neural["crops"])]
                patch = Image.fromarray(crop).resize(
                    (mouth_box[2] - mouth_box[0], mouth_box[3] - mouth_box[1]), Image.LANCZOS
                )
                region = frame[mouth_box[1] : mouth_box[3], mouth_box[0] : mouth_box[2]]
                blended = (np.asarray(patch).astype(np.float32) * mask[mouth_box[1] : mouth_box[3], mouth_box[0] : mouth_box[2]][..., None]
                           + region.astype(np.float32) * (1 - mask[mouth_box[1] : mouth_box[3], mouth_box[0] : mouth_box[2]][..., None]))
                frame[mouth_box[1] : mouth_box[3], mouth_box[0] : mouth_box[2]] = blended.astype(np.uint8)
            else:
                sub = frame[mouth_box[1] : mouth_box[3], mouth_box[0] : mouth_box[2]]
                opened = open_mouth_patch(sub, open_amount, geometry["parting_local"])
                sub_mask = mask[mouth_box[1] : mouth_box[3], mouth_box[0] : mouth_box[2]][..., None]
                frame[mouth_box[1] : mouth_box[3], mouth_box[0] : mouth_box[2]] = (
                    opened.astype(np.float32) * sub_mask + sub.astype(np.float32) * (1 - sub_mask)
                ).astype(np.uint8)

            frame = sharpen_region(frame, mouth_box, sharp_mask)

            # 2) blinks
            frame = apply_blink(frame, float(blinks[index]) * float(fade[index]))

            # 3) head micro-motion: speech envelope + breathing + fades
            yaw = 0.55 * float(lowpass[index]) + 0.12 * breath[index]
            pitch = 0.30 * float(lowpass[index]) * (1 + 0.4 * float(rng.normal()))
            roll = 0.010 * float(lowpass[index]) * math.sin(index / 11.0)
            sway = 0.004 * float(lowpass[index])
            frame = apply_head_motion(frame, yaw * fade[index], pitch * fade[index],
                                      roll * fade[index], sway * fade[index])

            frames.append(frame)
            mx0, my0, mx1, my1 = mouth_box
            pad = 8
            mouth_crops.append(
                frame[max(0, my0 - pad) : my1, max(0, mx0 - pad) : min(OUT_W, mx1 + pad)].copy()
            )

    return {
        "frames": frames,
        "mouth_crops": mouth_crops,
        "duration": duration,
        "n_frames": n_frames,
        "openness": openness,
        "envelope": env_smooth,
        "mouth_box": mouth_box,
        "viseme_weight": viseme_weight,
    }


def encode_video(frames: List[np.ndarray], audio: np.ndarray, sr: int, out_path: Path,
                 timings: Dict[str, float]) -> None:
    """Pipe raw RGB into ffmpeg -> libx264 yuv420p + AAC."""
    ffmpeg = ffmpeg_binary()
    h, w = frames[0].shape[:2]
    tmp_wav = out_path.with_suffix(".tmp.wav")
    import soundfile as sf

    sf.write(tmp_wav, audio, sr, subtype="PCM_16")

    video_codec = ["-c:v", "libx264", "-preset", "medium", "-crf", "19"]
    if not has_encoder(ffmpeg, "libx264"):
        log("libx264 missing — falling back to mpeg4", "WARN")
        video_codec = ["-c:v", "mpeg4", "-q:v", "2"]

    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "pipe:0",
        "-i", str(tmp_wav),
        *video_codec,
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-c:a", "aac", "-b:a", "160k", "-shortest",
        str(out_path),
    ]
    with timed("encode_video", timings):
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for frame in frames:
                process.stdin.write(frame.astype(np.uint8).tobytes())
            process.stdin.close()
        except Exception as exc:
            log(f"frame pipe failed: {exc}", "ERROR")
        stderr = process.stderr.read().decode("utf-8", "replace") if process.stderr else ""
        returncode = process.wait()
    tmp_wav.unlink(missing_ok=True)
    if returncode != 0 or not out_path.exists():
        raise RuntimeError(f"ffmpeg failed ({returncode}): {stderr[-1200:]}")
    log(f"encoded {out_path.name} ({out_path.stat().st_size / 1024:.0f} KB)")


# --------------------------------------------------------------------------- #
# 5. Contact sheet & metrics
# --------------------------------------------------------------------------- #


def build_contact_sheet(
    frames: List[np.ndarray],
    mouth_crops: List[np.ndarray],
    audio: np.ndarray,
    sr: int,
    openness: np.ndarray,
    out_path: Path,
    timings: Dict[str, float],
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    with timed("build_contact_sheet", timings):
        n_frames = len(frames)
        picks = [int(round(i * (n_frames - 1) / 7)) for i in range(8)]
        thumb_w, thumb_h = 260, 320
        gutter, margin, header_h = 14, 24, 84

        def load_font(size: int, bold: bool = False):
            for name in (
                "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
                "LiberationSans-Bold.ttf" if bold else "LiberationSans-Regular.ttf",
            ):
                try:
                    return ImageFont.truetype(name, size)
                except Exception:
                    continue
            try:
                return ImageFont.load_default(size=size)
            except Exception:
                return ImageFont.load_default()

        title_font = load_font(30, bold=True)
        sub_font = load_font(18)
        label_font = load_font(16, bold=True)

        mouth_h = 92
        wave_h = 240
        width = margin * 2 + thumb_w * 8 + gutter * 7
        height = margin + header_h + 30 + thumb_h + 26 + mouth_h + 30 + wave_h + margin

        canvas = Image.new("RGB", (width, height), (12, 12, 14))
        draw = ImageDraw.Draw(canvas)
        draw.text((margin, 18), "JULIAN · VOICE & TALKING AVATAR", fill=(240, 240, 246), font=title_font)
        draw.text((margin, 54),
                  f"PHASE 2 · MICRO-PILOT 2 · {len(frames)} frames @ {FPS} fps · "
                  f"{len(frames) / FPS:.2f}s · kokoro-onnx + viseme lip-sync",
                  fill=(150, 152, 162), font=sub_font)

        y_frames = margin + header_h
        draw.text((margin, y_frames - 4), "KEYFRAMES", fill=(196, 106, 62), font=label_font)
        y_img = y_frames + 24

        for slot, frame_index in enumerate(picks):
            x = margin + slot * (thumb_w + gutter)
            thumb = Image.fromarray(frames[frame_index]).resize((thumb_w, thumb_h), Image.LANCZOS)
            canvas.paste(thumb, (x, y_img))
            draw.rectangle([x, y_img, x + thumb_w - 1, y_img + thumb_h - 1], outline=(70, 70, 78), width=2)
            stamp = f"{frame_index / FPS:5.2f}s"
            draw.rectangle([x, y_img, x + 76, y_img + 24], fill=(24, 24, 30))
            draw.text((x + 7, y_img + 4), stamp, fill=(245, 245, 250), font=label_font)

        y_mouth = y_img + thumb_h + 24
        draw.text((margin, y_mouth - 4), "MOUTH ZOOM · lip progression", fill=(196, 106, 62), font=label_font)
        y_mouth += 24
        for slot, frame_index in enumerate(picks):
            x = margin + slot * (thumb_w + gutter)
            crop = Image.fromarray(mouth_crops[frame_index])
            scale = mouth_h / crop.height
            crop = crop.resize((int(crop.width * scale), mouth_h), Image.LANCZOS)
            canvas.paste(crop, (x, y_mouth))
            draw.rectangle([x, y_mouth, x + crop.width - 1, y_mouth + mouth_h - 1],
                           outline=(196, 106, 62), width=1)

        y_wave = y_mouth + mouth_h + 28
        draw.text((margin, y_wave - 6), "AUDIO WAVEFORM · lip-openness curve", fill=(196, 106, 62), font=label_font)
        y_wave += 20
        wave_w = width - margin * 2
        wave_canvas = Image.new("RGB", (wave_w, wave_h), (18, 18, 22))
        wdraw = ImageDraw.Draw(wave_canvas)
        wdraw.rectangle([0, 0, wave_w - 1, wave_h - 1], outline=(60, 60, 68))

        mid = wave_h / 2
        step = max(1, audio.size // wave_w)
        peaks = np.array([np.max(np.abs(audio[i * step : (i + 1) * step])) if audio[i * step : (i + 1) * step].size else 0.0
                          for i in range(wave_w)])
        norm = peaks.max() or 1.0
        for i, peak in enumerate(peaks):
            amp = (peak / norm) * (mid - 8)
            wdraw.line([(i, mid - amp), (i, mid + amp)], fill=(90, 120, 160))

        # lip-openness curve (0..1 -> bottom to top)
        for i in range(wave_w):
            idx = min(len(openness) - 1, int(i / wave_w * len(openness)))
            y = wave_h - 10 - float(openness[idx]) * (wave_h - 20)
            if i:
                wdraw.line([(i - 1, prev_y), (i, y)], fill=(196, 106, 62), width=3)
            prev_y = y
        canvas.paste(wave_canvas, (margin, y_wave))

        canvas.save(out_path, format="PNG", optimize=True)
        log(f"saved {out_path.name} ({canvas.width}x{canvas.height}, "
            f"{out_path.stat().st_size / 1024:.0f} KB)")


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    n = min(a.size, b.size)
    if n < 3:
        return 0.0
    a, b = a[:n].astype(np.float64), b[:n].astype(np.float64)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    started = time.perf_counter()
    output_dir = resolve_output_dir()
    tee = Tee(output_dir / BUILD_LOG, sys.stdout)
    sys.stdout = tee  # type: ignore[assignment]

    timings: Dict[str, float] = {}
    metrics: Dict[str, Any] = {
        "phase": "phase-2-micro-pilot-2",
        "pilot": "voice-and-talking-avatar",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "config": {
            "script": SCRIPT_LINE,
            "voice_blend": VOICE_BLEND,
            "speed": SPEECH_SPEED,
            "fps": FPS,
            "resolution": [OUT_W, OUT_H],
            "max_seconds": MAX_SECONDS,
            "tts": "kokoro-onnx (onnxruntime)",
        },
        "timings_seconds": timings,
        "errors": [],
    }

    log("=" * 78)
    log("stoic-ai-studio · VOICE & TALKING AVATAR · Micro-Pilot 2")
    log(f"output dir: {output_dir}")
    log("=" * 78)

    try:
        ensure_python_deps(timings)
        from PIL import Image

        with timed("load_portrait", timings):
            portrait_path = resolve_portrait()
            portrait = np.asarray(Image.open(portrait_path).convert("RGB"))
            log(f"portrait: {portrait_path} {portrait.shape}")
            # ship a copy so the workflow can pull the exact input back for review
            shutil.copyfile(portrait_path, output_dir / PORTRAIT_NAME)

        voice = synthesize_voice(output_dir, timings)
        audio, sr = voice["audio"], voice["sample_rate"]
        metrics["voice"] = {k: v for k, v in voice.items() if k != "audio"}

        duration = min(MAX_SECONDS, audio.size / sr)
        n_frames = max(1, int(round(duration * FPS)))

        neural = try_neural_lipsync(portrait, audio[: int(duration * sr)], sr, n_frames, timings)
        metrics["lipsync_backend"] = "wav2lip-onnx" if neural else "viseme-synthesiser"
        metrics["lipsync_model"] = neural["model"] if neural else None

        if neural is None:
            metrics["lipsync_fallback_reason"] = (
                "neural weights unavailable or failed validation — used the "
                "viseme/envelope synthesiser (probe details in build.log)"
            )

        render = render_video(
            portrait, audio[: int(duration * sr)], sr,
            voice["phoneme_timings"], neural, timings,
        )

        video_path = output_dir / VIDEO_MP4
        encode_video(render["frames"], audio[: int(duration * sr)], sr, video_path, timings)

        build_contact_sheet(
            render["frames"], render["mouth_crops"], audio[: int(duration * sr)], sr,
            render["openness"], output_dir / CONTACT_SHEET, timings,
        )

        correlation = pearson(render["openness"], render["envelope"])
        metrics.update({
            "duration_s": round(render["duration"], 3),
            "fps": FPS,
            "frame_count": render["n_frames"],
            "resolution": [OUT_W, OUT_H],
            "mouth_open_vs_audio_rms_correlation": round(correlation, 4),
            "realistic_motion": {
                "head_micro_motion": "speech-envelope driven yaw/pitch/roll + sway",
                "breathing": "4.2 s sine on pitch",
                "blinks": int(np.sum(np.diff((blink_curve(render['n_frames'], FPS) > 0.2).astype(int)) == 1)),
                "mouth_sharpening": "feathered unsharp mask on mouth region",
            },
            "files": {
                name: {
                    "bytes": (output_dir / name).stat().st_size,
                    "mb": round((output_dir / name).stat().st_size / 1024**2, 2),
                }
                for name in (VOICE_WAV, VOICE_MP3, VIDEO_MP4, CONTACT_SHEET)
                if (output_dir / name).exists()
            },
            "status": "success",
        })
        log(f"lip-openness vs audio-RMS correlation: {correlation:.3f}")
        log("MICRO-PILOT 2 COMPLETE ✔")

    except Exception as exc:
        metrics["status"] = "error"
        metrics["errors"].append(f"{type(exc).__name__}: {exc}")
        log(f"FATAL: {type(exc).__name__}: {exc}", "ERROR")
        log(traceback.format_exc(), "ERROR")
    finally:
        metrics["finished_at"] = datetime.now(timezone.utc).isoformat()
        timings["total"] = round(time.perf_counter() - started, 3)
        log(f"total wall clock: {timings['total']}s")
        try:
            (output_dir / METRICS_JSON).write_text(
                json.dumps(metrics, indent=2, default=str), encoding="utf-8"
            )
        except Exception as exc:  # pragma: no cover
            log(f"could not write {METRICS_JSON}: {exc}", "ERROR")
        tee.close()
        sys.stdout = tee._stream  # type: ignore[attr-defined]

    return 0 if metrics.get("status") == "success" else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log("interrupted", "WARN")
        sys.exit(130)
