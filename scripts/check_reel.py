#!/usr/bin/env python3
"""
Gate the rendered DemoReel before it is published.

CI runs this after ``npx remotion render``. The point is that nothing is taken
on trust: the file's container, codec, pixel format, audio stream, duration,
dimensions and loudness are all measured with ffprobe, and a reel that fails
any one of them fails the job instead of being committed.

Two modes:

*  ``--audio-only`` checks just the generated audio (used before rendering, so
   a missing voiceover fails in seconds rather than after a 10-minute render).
*  default checks the finished ``outputs/demo_reel.mp4``.

Exit codes: 0 pass, 1 a check failed, 2 a required tool is missing.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs" / "demo_reel.mp4"
PUBLIC = ROOT / "remotion" / "public"
MANIFEST = PUBLIC / "audio_manifest.json"

WIDTH, HEIGHT = 1080, 1920
FPS = 30
FRAMES = 450
EXPECTED_SECONDS = FRAMES / FPS  # 15.0
#: Containers round; a 15.0 s reel often probes as 14.97 or 15.03.
DURATION_TOLERANCE = 0.35
#: The voiceover is ~14 s inside a 15 s reel; quieter than this means the
#: music bed is missing or the mux dropped a channel.
MIN_MEAN_VOLUME_DB = -38.0
MIN_AUDIO_STREAM_SECONDS = 13.5


class Checker:
    def __init__(self, verbose: bool = True) -> None:
        self.failures: List[str] = []
        self.verbose = verbose

    def want(self, condition: bool, message: str) -> bool:
        if condition:
            if self.verbose:
                print(f"  ok   {message}")
        else:
            self.failures.append(message)
            print(f"  FAIL {message}", file=sys.stderr)
        return condition

    def report(self) -> int:
        if self.failures:
            print(f"\n{len(self.failures)} check(s) FAILED", file=sys.stderr)
            for failure in self.failures:
                print(f"  - {failure}", file=sys.stderr)
            return 1
        print("\nreel checks passed")
        return 0


def which(name: str) -> str | None:
    return shutil.which(name)


def ffprobe_json(path: Path) -> Dict[str, Any]:
    result = subprocess.run(
        [which("ffprobe") or "ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def mean_volume_db(path: Path) -> float | None:
    """
    Integrated mean volume via ffmpeg's volumedetect.

    Used instead of ``ffprobe`` (which reports no loudness) and instead of
    ``ebur128`` (which needs an extra filter build); volumedetect is in every
    standard ffmpeg, including the one the ubuntu-latest image ships.
    """
    ffmpeg = which("ffmpeg")
    if ffmpeg is None:
        return None
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-nostats", "-i", str(path),
         "-map", "0:a:0", "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True)
    for line in result.stderr.splitlines():
        if "mean_volume:" in line:
            try:
                return float(line.split("mean_volume:")[1].split("dB")[0].strip())
            except ValueError:
                return None
    return None


# --------------------------------------------------------------------------- #
# The two modes
# --------------------------------------------------------------------------- #


def check_audio(check: Checker) -> None:
    """The audio that the reel is built from, before any rendering happens."""
    print("audio artefacts")
    check.want(MANIFEST.is_file(), f"{MANIFEST.relative_to(ROOT)} exists")
    manifest: Dict[str, Any] = {}
    if MANIFEST.is_file():
        manifest = json.loads(MANIFEST.read_text())

    artefact = manifest.get("artefacts", {}) or {}
    voice = artefact.get("voiceover")
    check.want(bool(voice), "the manifest records a voiceover")
    voice_file = PUBLIC / "voiceover.mp3"
    check.want(voice_file.is_file(), "public/voiceover.mp3 exists")
    if voice_file.is_file():
        size = voice_file.stat().st_size
        check.want(size > 40_000, f"voiceover.mp3 is {size / 1024:.0f} KB (>40 KB)")
        with open(voice_file, "rb") as handle:
            head = handle.read(3)
        # "ID3" (tagged) or 0xFF 0xFB (bare MPEG frame). Anything else is not
        # an MP3 and Chrome would fail to decode it at render time.
        check.want(head == b"ID3" or head[:2] == b"\xff\xfb",
                   "voiceover.mp3 starts with a valid MPEG header")

    for key in ("whoosh", "pop", "riser"):
        entry = artefact.get(key) or {}
        name = entry.get("file")
        check.want(bool(name), f"the manifest records {key}")
        if not name:
            continue
        path = PUBLIC / name
        check.want(path.is_file(), f"public/{name} exists")
        if path.is_file():
            check.want(path.stat().st_size > 10_000,
                       f"{name} is {path.stat().st_size / 1024:.0f} KB")
            with open(path, "rb") as handle:
                riff = handle.read(12)
            check.want(riff[:4] == b"RIFF" and riff[8:12] == b"WAVE",
                       f"{name} is a RIFF/WAVE file")

    # The voiceover must fit the timeline and must not be a silent placeholder.
    if voice_file.is_file() and which("ffprobe"):
        info = ffprobe_json(voice_file)
        seconds = float(info["format"]["duration"])
        check.want(8.0 <= seconds <= 15.5,
                   f"voiceover.mp3 is {seconds:.2f}s (8.0–15.5s window)")
        volume = mean_volume_db(voice_file)
        if volume is not None:
            check.want(volume > -32.0,
                       f"voiceover is audible (mean {volume:.1f} dB > -32 dB)")


def check_video(check: Checker) -> None:
    print("rendered reel")
    if which("ffprobe") is None:
        print("error: ffprobe not found", file=sys.stderr)
        raise SystemExit(2)

    if not check.want(OUTPUT.is_file(), f"{OUTPUT.relative_to(ROOT)} exists"):
        return
    size_mb = OUTPUT.stat().st_size / 1e6
    check.want(1.0 < size_mb < 60.0, f"file size {size_mb:.1f} MB (1–60 MB)")

    info = ffprobe_json(OUTPUT)
    streams = info.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    if check.want(video is not None, "has a video stream") and video:
        check.want(video.get("codec_name") == "h264",
                   f"video codec is h264 (got {video.get('codec_name')})")
        check.want(video.get("pix_fmt") == "yuv420p",
                   f"pixel format is yuv420p (got {video.get('pix_fmt')})")
        check.want(video.get("width") == WIDTH and video.get("height") == HEIGHT,
                   f"resolution is {WIDTH}x{HEIGHT} "
                   f"(got {video.get('width')}x{video.get('height')})")
        check.want(video.get("profile") in ("High", "Constrained Baseline",
                                            "Main", "Baseline"),
                   f"h264 profile is {video.get('profile')}")

        rate = video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/1"
        try:
            num, _, den = rate.partition("/")
            fps = float(num) / float(den or 1)
        except (ZeroDivisionError, ValueError):
            fps = 0.0
        check.want(abs(fps - FPS) < 0.6,
                   f"frame rate is {fps:.2f} fps (target {FPS})")

        frames = video.get("nb_frames")
        if frames and frames.isdigit():
            check.want(abs(int(frames) - FRAMES) <= 4,
                       f"{frames} frames (target {FRAMES})")

    if check.want(audio is not None, "has an audio stream") and audio:
        check.want(audio.get("codec_name") == "aac",
                   f"audio codec is aac (got {audio.get('codec_name')})")

    duration = float(info.get("format", {}).get("duration", 0.0) or 0.0)
    check.want(abs(duration - EXPECTED_SECONDS) <= DURATION_TOLERANCE,
               f"duration is {duration:.2f}s (target {EXPECTED_SECONDS:.2f}s "
               f"+/-{DURATION_TOLERANCE}s)")

    if audio:
        audio_seconds = float(audio.get("duration", 0.0) or 0.0)
        check.want(audio_seconds >= MIN_AUDIO_STREAM_SECONDS,
                   f"audio runs {audio_seconds:.2f}s (>= {MIN_AUDIO_STREAM_SECONDS}s)")

    volume = mean_volume_db(OUTPUT)
    if volume is not None:
        check.want(volume > MIN_MEAN_VOLUME_DB,
                   f"reel is not silent (mean {volume:.1f} dB > {MIN_MEAN_VOLUME_DB} dB)")
    else:
        print("  note ffmpeg unavailable — skipped the loudness check")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-only", action="store_true",
                        help="check the generated audio only")
    args = parser.parse_args(argv)

    check = Checker()
    if args.audio_only:
        check_audio(check)
    else:
        check_audio(check)
        check_video(check)
    return check.report()


if __name__ == "__main__":
    sys.exit(main())
