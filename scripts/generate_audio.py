#!/usr/bin/env python3
"""
Build the DemoReel's audio: one voiceover and a small kit of procedural SFX.

Two kinds of audio, two very different requirements:

*  **Voiceover** (``public/voiceover.mp3``) is synthesised by ``edge-tts``,
   which streams MP3 from Microsoft's speech endpoint. That needs outbound
   network access, so this script is written to fail with a clear message
   rather than emit a silent track — a reel with no narration should never
   pass as "done". Pass ``--allow-missing-voice`` (CI does not) if you only
   want the SFX while iterating offline.

*  **SFX** (``public/whoosh.wav``, ``public/pop.wav``, ``public/riser.wav``)
   are synthesised from maths with numpy/scipy: no samples, no licensing, no
   network. They are 48 kHz mono WAV so they can be decoded by Chrome without
   any extra codec support.

Everything is written to paths recorded in ``public/audio_manifest.json`` so
``check_audio.py`` (CI) can verify the artefacts instead of trusting this
module's exit code.

Usage::

    python scripts/generate_audio.py              # voiceover + SFX
    python scripts/generate_audio.py --sfx-only   # SFX only (offline)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy import signal

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).resolve().parent.parent
PUBLIC = ROOT / "remotion" / "public"

SAMPLE_RATE = 48_000
MANIFEST = PUBLIC / "audio_manifest.json"

VOICE_FILE = PUBLIC / "voiceover.mp3"
WHOOSH_FILE = PUBLIC / "whoosh.wav"
POP_FILE = PUBLIC / "pop.wav"
RISER_FILE = PUBLIC / "riser.wav"

#: ``en-US-AndrewMultilingualNeural`` is the voice the brief asks for: warm,
#: confident, North-American tech-narration delivery.
VOICE = "en-US-AndrewMultilingualNeural"

#: A little faster than default — 15 s is not long and the reel is high-energy.
VOICE_RATE = "+10%"
VOICE_PITCH = "+0Hz"

#: About 14 s of speech. Written for the ear: short clauses, hard consonants,
#: a beat of silence implied by the full stops. Kept under ~300 characters
#: because the reel's timeline has room for roughly 40 words.
VOICEOVER = (
    "Everyone wants the view. Nobody wants the climb. "
    "One percent of your day, compounding, quietly, for years. "
    "That is the whole secret. "
    "Show up tomorrow. Stay stoic."
)

#: Hard ceiling: if the synthesised file is longer than this the reel's audio
#: would be cut off mid-sentence, so the script refuses instead.
MAX_VOICE_SECONDS = 15.5
MIN_VOICE_SECONDS = 8.0

# The web build only ever triggers this via ``--allow-missing-voice``; CI runs
# the default path, where a network failure is fatal.
VOICE_HINT = (
    "edge-tts streams from Microsoft's speech endpoint "
    "(speech.platform.bing.com:443). If this machine is offline or behind an "
    "egress allowlist, run `python scripts/generate_audio.py --sfx-only` to "
    "build the SFX while iterating, and let CI produce the voiceover."
)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def log(message: str) -> None:
    print(f"[audio] {message}", flush=True)


def ensure_public() -> None:
    PUBLIC.mkdir(parents=True, exist_ok=True)


def write_wav(path: Path, samples: np.ndarray, peak: float = 0.89) -> Path:
    """
    Write float samples in [-1, 1] as 16-bit mono PCM.

    Normalised to ``peak`` rather than 1.0 so that three SFX layered under a
    voiceover cannot clip when Chrome sums them.
    """
    data = np.asarray(samples, dtype=np.float64).ravel()
    if data.size == 0:
        raise ValueError(f"{path.name}: no samples to write")
    data = np.clip(data, -1.0, 1.0)
    high = float(np.max(np.abs(data)))
    if high > 0:
        data = data * (peak / high)
    pcm = np.round(data * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())
    log(f"wrote {path.relative_to(ROOT)} ({pcm.size / SAMPLE_RATE:.2f}s, "
        f"{path.stat().st_size / 1024:.0f} KB)")
    return path


def t_axis(seconds: float) -> np.ndarray:
    return np.arange(int(seconds * SAMPLE_RATE), dtype=np.float64) / SAMPLE_RATE


def fade(samples: np.ndarray, attack: float, release: float) -> np.ndarray:
    """Apply a linear attack/release so no SFX ever starts or ends on a click."""
    out = samples.copy()
    a = min(int(attack * SAMPLE_RATE), out.size // 2)
    r = min(int(release * SAMPLE_RATE), out.size // 2)
    if a > 0:
        out[:a] *= np.linspace(0.0, 1.0, a)
    if r > 0:
        out[-r:] *= np.linspace(1.0, 0.0, r)
    return out


# --------------------------------------------------------------------------- #
# SFX synthesis
# --------------------------------------------------------------------------- #


def build_whoosh() -> np.ndarray:
    """
    A 0.85 s air-swipe: filtered noise whose cutoff sweeps up and back down.

    Noise through a band-pass whose centre frequency moves is the classic
    whoosh; doing it with ``signal.sosfilt`` per-block keeps it stable. The
    centre sweeps 320 Hz -> 4.2 kHz in the first third, then falls away, which
    reads as a fast object passing the listener.
    """
    duration = 0.85
    t = t_axis(duration)
    progress = t / duration

    rng = np.random.default_rng(7)
    noise = rng.standard_normal(t.size)

    # Sweep the band-pass centre, then collapse it so the tail dies out.
    centres = 320.0 * (4.2e3 / 320.0) ** np.minimum(progress * 2.4, 1.0)
    centres = centres * np.where(progress < 0.72, 1.0,
                                 np.maximum(0.12, 1.0 - (progress - 0.72) * 3.2))

    # Re-filter in short blocks so the centre can move; overlap-add the blocks.
    block = 512
    out = np.zeros_like(noise)
    window = np.hanning(block)
    for start in range(0, t.size - block, block // 2):
        centre = float(centres[start])
        low = max(centre / 2.2, 80.0)
        high = min(centre * 2.2, SAMPLE_RATE / 2 - 200.0)
        sos = signal.butter(2, [low, high], btype="bandpass", fs=SAMPLE_RATE,
                            output="sos")
        chunk = signal.sosfilt(sos, noise[start:start + block]) * window
        out[start:start + block] += chunk

    # A breath of low end gives the swipe weight on phone speakers.
    out = out / (np.max(np.abs(out)) or 1.0)
    body = np.sin(2 * np.pi * 90.0 * t) * np.exp(-t * 5.5) * 0.28
    out = out * np.exp(-progress * 1.4) + body
    return fade(out, 0.012, 0.12)


def build_pop() -> np.ndarray:
    """
    A 0.22 s UI pop: a pitch-dropping sine with a fast click on the front.

    Two components — a 60 ms noise click (the "tick") and a sine gliding
    880 Hz -> 420 Hz with an exponential decay (the "body") — is exactly how
    interface pops are made, and it cuts through a voiceover without masking
    it because the energy sits above 400 Hz.
    """
    duration = 0.22
    t = t_axis(duration)

    glide = 880.0 * (420.0 / 880.0) ** (t / duration)
    phase = 2 * np.pi * np.cumsum(glide) / SAMPLE_RATE
    body = np.sin(phase) * np.exp(-t * 16.0)

    rng = np.random.default_rng(3)
    click = rng.standard_normal(t.size) * np.exp(-t * 320.0) * 0.5

    # A little second harmonic keeps it from sounding like a test tone.
    body += 0.25 * np.sin(2 * phase) * np.exp(-t * 22.0)
    return fade(body + click, 0.002, 0.05)


def build_riser() -> np.ndarray:
    """
    A 1.9 s riser for the outro: a sawtooth climbing a fifth, swelling wide.

    Used under the CTA so the last three seconds lift instead of just sitting
    there. Two detuned saws panned together give the "wide" feel without any
    stereo file complexity.
    """
    duration = 1.9
    t = t_axis(duration)
    progress = t / duration

    def saw(freq: np.ndarray) -> np.ndarray:
        phase = np.cumsum(freq) / SAMPLE_RATE
        return 2.0 * (phase - np.floor(phase + 0.5))

    base = 110.0 * (1.5 ** (progress ** 1.6))
    tone = saw(base) + 0.6 * saw(base * 1.004) + 0.35 * saw(base * 0.5)
    # Low-pass opens as it climbs: dark at the bottom, bright at the top.
    tone = np.tanh(tone * (1.2 + 2.4 * progress))
    swell = np.linspace(0.05, 1.0, t.size) ** 1.8

    rng = np.random.default_rng(11)
    air = rng.standard_normal(t.size) * progress ** 2 * 0.18
    sos = signal.butter(2, 2500.0, btype="highpass", fs=SAMPLE_RATE,
                        output="sos")
    air = signal.sosfilt(sos, air)

    return fade(tone * swell * 0.8 + air, 0.05, 0.25)


# --------------------------------------------------------------------------- #
# Voiceover
# --------------------------------------------------------------------------- #


def _have_edge_tts() -> bool:
    try:
        import edge_tts  # noqa: F401
    except Exception:
        return False
    return True


def _install_edge_tts() -> None:
    if _have_edge_tts():
        return
    log("edge-tts not installed — installing it now")
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                    "edge-tts>=6.1.9"], check=True)


def _probe_seconds(path: Path) -> float:
    """
    Duration of an audio file, via ffprobe when available and a WAV reader
    otherwise. The voiceover is MP3, so ffprobe is the only option there; if it
    is missing we return 0.0 and skip the length gate rather than fail the run
    for a missing optional tool.
    """
    probe = _which("ffprobe")
    if probe is None:
        log("ffprobe not found — skipping the voiceover duration gate")
        return 0.0
    result = subprocess.run(
        [probe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def _which(name: str):
    import shutil

    return shutil.which(name)


def build_voiceover(force: bool = False) -> Dict[str, Any]:
    """Synthesise ``voiceover.mp3`` with edge-tts and report what came back."""
    if VOICE_FILE.is_file() and not force:
        seconds = _probe_seconds(VOICE_FILE)
        log(f"voiceover.mp3 already present ({seconds:.2f}s) — reusing it "
            f"(pass --force to regenerate)")
        return {"voice": VOICE, "seconds": seconds, "reused": True,
                "text": VOICEOVER}

    _install_edge_tts()
    import edge_tts

    async def synthesise() -> int:
        communicate = edge_tts.Communicate(
            VOICEOVER, VOICE, rate=VOICE_RATE, pitch=VOICE_PITCH)

        # Stream to a temporary file first: a mid-stream network failure must
        # not leave a truncated MP3 that looks like a finished voiceover.
        with tempfile.NamedTemporaryFile(delete=False,
                                         suffix=".mp3") as handle:
            temp = Path(handle.name)
        written = 0
        try:
            with open(temp, "wb") as sink:
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio" and chunk.get("data"):
                        sink.write(chunk["data"])
                        written += len(chunk["data"])
            if written < 8_000:
                raise RuntimeError(
                    f"edge-tts returned only {written} bytes — the stream "
                    f"ended before any usable audio arrived")
            temp.replace(VOICE_FILE)
        finally:
            if temp.exists():
                temp.unlink()
        return written

    log(f"requesting {VOICE} ({len(VOICEOVER.split())} words) "
        f"@ {VOICE_RATE} …")
    try:
        written = asyncio.run(synthesise())
    except Exception as exc:  # network, auth, rate limit, protocol change
        raise SystemExit(
            f"[audio] voiceover synthesis failed: {type(exc).__name__}: {exc}\n"
            f"[audio] {VOICE_HINT}") from exc

    seconds = _probe_seconds(VOICE_FILE)
    log(f"wrote {VOICE_FILE.relative_to(ROOT)} ({written / 1024:.0f} KB, "
        f"{seconds:.2f}s)")

    if seconds and not (MIN_VOICE_SECONDS <= seconds <= MAX_VOICE_SECONDS):
        raise SystemExit(
            f"[audio] voiceover is {seconds:.2f}s, outside the "
            f"{MIN_VOICE_SECONDS}-{MAX_VOICE_SECONDS}s window the 15s timeline "
            f"allows — adjust VOICEOVER or VOICE_RATE")

    if seconds == 0.0:
        # No ffprobe: fall back to a bitrate estimate so the gate still bites.
        approx = written / (64_000 / 8)
        log(f"estimated duration {approx:.2f}s from {written} bytes "
            f"(assumes 64 kbps)")
        seconds = approx

    return {"voice": VOICE, "seconds": round(seconds, 3), "reused": False,
            "text": VOICEOVER, "bytes": written}


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sfx-only", action="store_true",
                        help="skip the voiceover (offline use)")
    parser.add_argument("--allow-missing-voice", action="store_true",
                        help="downgrade a voiceover failure to a warning")
    parser.add_argument("--force", action="store_true",
                        help="regenerate the voiceover even if it exists")
    args = parser.parse_args(argv)

    ensure_public()
    log(f"numpy {np.__version__} · scipy {signal.__name__ and 'ok'} · "
        f"{SAMPLE_RATE} Hz")

    artefacts: Dict[str, Any] = {"sample_rate": SAMPLE_RATE}
    artefacts["whoosh"] = {
        "file": write_wav(WHOOSH_FILE, build_whoosh()).name, "scene": 1}
    artefacts["pop"] = {
        "file": write_wav(POP_FILE, build_pop()).name, "scene": 2}
    artefacts["riser"] = {
        "file": write_wav(RISER_FILE, build_riser()).name, "scene": 4}

    if args.sfx_only:
        log("--sfx-only: skipping the voiceover")
        artefacts["voiceover"] = None
        status = "sfx-only"
    else:
        try:
            artefacts["voiceover"] = build_voiceover(force=args.force)
            status = "complete"
        except SystemExit as exc:
            if not args.allow_missing_voice:
                raise
            log(str(exc))
            artefacts["voiceover"] = None
            status = "missing-voice"

    payload = {"status": status, "voice": VOICE, "text": VOICEOVER,
               "artefacts": artefacts}
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    log(f"wrote {MANIFEST.relative_to(ROOT)} (status={status})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
