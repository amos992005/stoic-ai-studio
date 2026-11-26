#!/usr/bin/env python3
"""
stoic-ai-studio :: Phase 3 orchestrator — "Luca Moretti" flagship character.

Reuses the hardened credential / metadata / push / poll / download / validate
logic from `scripts/run_kaggle_job.py` and swaps in the Luca kernel, artifact
manifest and metrics expectations.

The kernel needs **no auxiliary dataset** (everything is pulled from the
HuggingFace hub inside Kaggle), so this driver is a thin manifest swap — no
dataset publish step.

Every knob is env-overridable so the same code can be dry-run locally:

    KERNEL_SLUG                       default stoic-luca-studio
    KAGGLE_POLL_INTERVAL_SECONDS / KAGGLE_POLL_TIMEOUT_SECONDS
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------- #
# Phase 3 manifest
# --------------------------------------------------------------------------- #

KERNEL_SLUG = "stoic-luca-studio"
KERNEL_TITLE = KERNEL_SLUG
KERNEL_DIR = REPO_ROOT / "kaggle_kernel_luca"
CODE_FILE = "luca_studio.py"

#: Success = a postable carousel, a real voice line, and a real neural clip.
REQUIRED_ARTIFACTS: List[str] = [
    "luca_carousel_sheet.png",
    "luca_voice.mp3",
    "luca_reel_clip.mp4",
    "luca_metrics.json",
]
#: Nice-to-have (warn only) — individual slides are also delivered as JPEGs.
OPTIONAL_ARTIFACTS: List[str] = [
    "01_luca_master.png",
    "02_monaco_yacht_date.png",
    "03_vip_party_flash.png",
    "04_porsche_gt3_pov.png",
    "05_extreme_action.png",
    "luca_slide_02.jpg",
    "luca_slide_03.jpg",
    "luca_slide_04.jpg",
    "luca_slide_05.jpg",
    "luca_video_sheet.png",
    "luca_voice.wav",
]
#: Everything the workflow may commit back (5 MB per-file ceiling applies).
COMMITTABLE_ARTIFACTS: List[str] = [
    "outputs/luca_carousel_sheet.png",
    "outputs/luca_video_sheet.png",
    "outputs/luca_reel_clip.mp4",
    "outputs/luca_voice.mp3",
    "outputs/luca_metrics.json",
    "outputs/build.log",
    "outputs/01_luca_master.png",
    "outputs/luca_slide_01.jpg",
    "outputs/luca_slide_02.jpg",
    "outputs/luca_slide_03.jpg",
    "outputs/luca_slide_04.jpg",
    "outputs/luca_slide_05.jpg",
]
#: The dual-GPU run is ~15 min; on a single T4 the avatar work runs *after* the
#: carousel, so allow a longer poll budget than the Phase-3 default.
POLL_TIMEOUT_SECONDS = int(os.environ.get("KAGGLE_POLL_TIMEOUT_SECONDS", str(50 * 60)))
#: Runtime guard for the warning only — the work is genuinely longer now.
MAX_RUNTIME_SECONDS = 1800


def load_orchestrator():
    """Import the Phase 1 orchestrator so every phase shares one code path."""
    path = REPO_ROOT / "scripts" / "run_kaggle_job.py"
    spec = importlib.util.spec_from_file_location("run_kaggle_job", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_kaggle_job"] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    orchestrator = load_orchestrator()

    orchestrator.KERNEL_SLUG = KERNEL_SLUG
    orchestrator.KERNEL_TITLE = KERNEL_TITLE
    orchestrator.KERNEL_DIR = KERNEL_DIR
    orchestrator.CODE_FILE = CODE_FILE
    orchestrator.EXPECTED_ARTIFACTS = list(REQUIRED_ARTIFACTS) + list(OPTIONAL_ARTIFACTS)
    orchestrator.REQUIRED_ARTIFACTS = list(REQUIRED_ARTIFACTS)
    orchestrator.OPTIONAL_ARTIFACTS = list(OPTIONAL_ARTIFACTS)
    orchestrator.COMMITTABLE_ARTIFACTS = list(COMMITTABLE_ARTIFACTS)
    # No dataset_sources: Phase 3 downloads every weight from the public hub.
    orchestrator.DEFAULT_DATASET_SOURCES = []
    orchestrator.POLL_TIMEOUT_SECONDS = POLL_TIMEOUT_SECONDS

    orchestrator.log("=" * 78)
    orchestrator.log("PHASE 3 · LUCA MORETTI · CAROUSEL + VOICE + NEURAL VIDEO")
    orchestrator.log(f"kernel: {KERNEL_DIR.relative_to(REPO_ROOT)}/{CODE_FILE}")
    orchestrator.log("=" * 78)

    if not (KERNEL_DIR / CODE_FILE).is_file():
        orchestrator.annotate_error(f"kernel source missing: {KERNEL_DIR / CODE_FILE}")
        return 2

    exit_code = orchestrator.main()
    if exit_code == 0:
        summarize(orchestrator)
    return exit_code


def summarize(orchestrator) -> None:
    """Print a compact success summary from luca_metrics.json."""
    import json

    metrics_path = orchestrator.OUTPUT_DIR / "luca_metrics.json"
    if not metrics_path.is_file():
        orchestrator.log("luca_metrics.json missing from outputs", "WARN")
        return
    try:
        data: Dict[str, Any] = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception as exc:
        orchestrator.log(f"could not read luca_metrics.json: {exc}", "WARN")
        return

    total = float(data.get("total_seconds") or 0.0)
    voice = data.get("voice") or {}
    video = data.get("video") or {}
    slides = data.get("slides") or []
    envelope = data.get("envelope") or {}

    ok = [s for s in slides if not s.get("error")]
    swapped = [s for s in slides if (s.get("stage2") or {}).get("swapped")]
    refined = [s for s in slides if (s.get("stage3") or {}).get("adetailer")]
    orchestrator.log(
        f"carousel: {len(ok)}/{len(slides)} slides · {len(swapped)} biometric swaps · "
        f"{len(refined)} 1024px adetailer passes"
    )
    for slide in slides:
        stage2 = slide.get("stage2") or {}
        stage3 = slide.get("stage3") or {}
        orchestrator.log(
            f"  {slide.get('filename')}: people={slide.get('expected_people')} "
            f"genders={stage2.get('genders')} swapped={stage2.get('swapped')} "
            f"adetailer={stage3.get('adetailer')} "
            f"identity={slide.get('identity_cosine')}"
        )
    verdicts = data.get("identity_verdicts") or []
    orchestrator.log(f"identity verdicts (cosine ≥ 0.45 = MATCH): {verdicts}")
    orchestrator.log(
        f"voice: {voice.get('duration_s')}s @ {voice.get('sample_rate')} Hz · "
        f"{voice.get('engine')} {voice.get('voice')} · peak {voice.get('peak_dbfs')} dBFS"
    )
    mix = data.get("audio_mix") or {}
    orchestrator.log(
        f"reel audio: {mix.get('duration_s')}s · {mix.get('bpm')} BPM bed · {mix.get('bed')}"
    )
    orchestrator.log(
        f"video: {video.get('frames')} frames @ {video.get('fps')} fps · "
        f"{video.get('duration_s')}s · shots={video.get('shots')} · "
        f"boundaries={video.get('shot_boundaries')} · "
        f"ping_pong={video.get('ping_pong')} cross_fade={video.get('cross_fade_retime')}"
    )
    orchestrator.log(
        f"mouth articulation: {envelope.get('active_fraction')} of frames active · "
        f"mean {envelope.get('mouth_open_mean')} · band {envelope.get('vocal_band_hz')} Hz"
    )
    orchestrator.log(
        f"banned techniques — procedural_animation={video.get('procedural_animation')} · "
        f"wsola={(data.get('voice') or {}).get('wsola_time_stretch')} · "
        f"biquad_eq={(data.get('voice') or {}).get('biquad_or_parametric_eq')}"
    )
    orchestrator.log(f"kernel total: {total:.0f}s")
    if total > MAX_RUNTIME_SECONDS:
        orchestrator.log(f"kernel used {total:.0f}s (> {MAX_RUNTIME_SECONDS}s budget)", "WARN")
    for error in data.get("errors") or []:
        orchestrator.log(f"kernel error: {error}", "WARN")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:  # pragma: no cover - last-resort guard
        import traceback

        traceback.print_exc()
        print(f"::error::luca orchestrator crashed: {type(exc).__name__}: {exc}", flush=True)
        sys.exit(1)
