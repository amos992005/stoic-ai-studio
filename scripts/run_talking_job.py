#!/usr/bin/env python3
"""
stoic-ai-studio :: Phase 2 orchestrator — "Voice & Talking Avatar" (one shot).

Reuses the hardened credential/push/poll/download logic from
`scripts/run_kaggle_job.py` and swaps in the talking-pilot kernel, artifact
manifest and metrics expectations.

Every knob is env-overridable so the same code can be dry-run locally:

    KERNEL_SLUG              default stoic-talking-pilot
    KERNEL_DIR               default kaggle_kernel_talking
    KAGGLE_POLL_INTERVAL_SECONDS / KAGGLE_POLL_TIMEOUT_SECONDS
    KAGGLE_DATASET_AUTODETECT=0 to skip the dataset existence probe
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------- #
# Phase 2 manifest
# --------------------------------------------------------------------------- #

KERNEL_SLUG = "stoic-talking-pilot"
KERNEL_TITLE = KERNEL_SLUG
KERNEL_DIR = REPO_ROOT / "kaggle_kernel_talking"
CODE_FILE = "talking_pilot.py"

#: `kernels push` uploads only CODE_FILE, so the portrait travels as a private
#: dataset that the kernel mounts at /kaggle/input/<slug>/.
PORTRAIT_SLUG = "stoic-ai-studio-portrait"
PORTRAIT_FILES = [KERNEL_DIR / "assets" / "julian_portrait_1024.png"]

#: A talking pilot is only useful with audio *and* video, so both are required.
EXPECTED_ARTIFACTS: List[str] = [
    "pilot_talking.mp4",
    "julian_voice.mp3",
    "talking_contact_sheet.png",
    "talking_metrics.json",
]
#: Nice-to-have artifacts (warn only).
OPTIONAL_ARTIFACTS: List[str] = [
    "julian_voice.wav",
    "portrait.png",
]
#: Files the workflow is allowed to commit back (5 MB per-file ceiling).
COMMITTABLE_ARTIFACTS: List[str] = [
    "outputs/pilot_talking.mp4",
    "outputs/julian_voice.mp3",
    "outputs/talking_contact_sheet.png",
    "outputs/talking_metrics.json",
    "outputs/build.log",
]
#: Expected runtime guard — the kernel must stay inside the 10-minute budget.
MAX_RUNTIME_SECONDS = 600


def load_orchestrator():
    """Import the Phase 1 orchestrator so both phases share one code path."""
    path = REPO_ROOT / "scripts" / "run_kaggle_job.py"
    spec = importlib.util.spec_from_file_location("run_kaggle_job", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_kaggle_job"] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    orchestrator = load_orchestrator()

    # Patch the Phase 1 manifest with the Phase 2 one, then run the exact same
    # push/poll/download/validate flow.
    orchestrator.KERNEL_SLUG = KERNEL_SLUG
    orchestrator.KERNEL_TITLE = KERNEL_TITLE
    orchestrator.KERNEL_DIR = KERNEL_DIR
    orchestrator.CODE_FILE = CODE_FILE
    orchestrator.EXPECTED_ARTIFACTS = list(EXPECTED_ARTIFACTS)
    # Success = video + metrics present (audio/sheet are validated too).
    orchestrator.REQUIRED_ARTIFACTS = ["pilot_talking.mp4", "talking_metrics.json"]
    orchestrator.DEFAULT_DATASET_SOURCES = [
        f"amos992005/{KERNEL_SLUG}",
        "amos992005/stoic-ai-studio",
    ]
    # Metrics/success criteria live in the orchestrator's report step.
    orchestrator.OPTIONAL_ARTIFACTS = list(OPTIONAL_ARTIFACTS)
    orchestrator.COMMITTABLE_ARTIFACTS = list(COMMITTABLE_ARTIFACTS)

    orchestrator.log("=" * 78)
    orchestrator.log("PHASE 2 · MICRO-PILOT 2 · VOICE & TALKING AVATAR")
    orchestrator.log(f"kernel: {orchestrator.KERNEL_DIR.relative_to(REPO_ROOT)}/{CODE_FILE}")
    orchestrator.log("=" * 78)

    # --- ensure the portrait is available to the kernel -------------------- #
    username = orchestrator.sanitize(os.environ.get("KAGGLE_USERNAME"))
    key = orchestrator.sanitize(os.environ.get("KAGGLE_KEY"))
    if not username or not key:
        return orchestrator.main()   # let main() emit the proper auth error

    env = orchestrator.env_with_credentials(username, key)
    try:
        orchestrator.write_kaggle_credentials(username, key)
    except Exception as exc:  # pragma: no cover
        orchestrator.log(f"could not persist credentials: {exc}", "WARN")

    dataset_slug = f"{username}/{PORTRAIT_SLUG}"
    missing_portrait = [str(f.relative_to(REPO_ROOT)) for f in PORTRAIT_FILES if not f.is_file()]
    if missing_portrait:
        orchestrator.annotate_error(f"portrait asset missing: {missing_portrait}")
        return 2

    orchestrator.log_group(f"publishing portrait dataset {dataset_slug}")
    published = orchestrator.publish_dataset(dataset_slug, PORTRAIT_FILES, env)
    orchestrator.log_group_end()
    if not published:
        orchestrator.log(
            "dataset publish failed — the kernel will fail to find its portrait",
            "WARN",
        )

    # Attach the dataset to the generated kernel metadata.
    original_build_metadata = orchestrator.build_metadata

    def build_metadata_with_portrait(user: str):
        meta = original_build_metadata(user)
        sources = list(meta.get("dataset_sources") or [])
        if dataset_slug not in sources:
            sources.insert(0, dataset_slug)
        meta["dataset_sources"] = sources
        return meta

    orchestrator.build_metadata = build_metadata_with_portrait

    exit_code = orchestrator.main()
    if exit_code == 0:
        summarize(orchestrator)
    return exit_code


def summarize(orchestrator) -> None:
    """Print a compact success summary from talking_metrics.json."""
    import json

    metrics_path = orchestrator.OUTPUT_DIR / "talking_metrics.json"
    if not metrics_path.is_file():
        orchestrator.log("talking_metrics.json missing from outputs", "WARN")
        return
    try:
        data: Dict[str, Any] = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception as exc:
        orchestrator.log(f"could not read talking_metrics.json: {exc}", "WARN")
        return

    total = float(data.get("timings_seconds", {}).get("total", 0.0))
    orchestrator.log(
        f"pilot: {data.get('frame_count')} frames @ {data.get('fps')} fps · "
        f"{data.get('duration_s')}s · lip/RMS corr "
        f"{data.get('mouth_open_vs_audio_rms_correlation')} · "
        f"backend {data.get('lipsync_backend')} · kernel {total:.0f}s"
    )
    if total > MAX_RUNTIME_SECONDS:
        orchestrator.log(
            f"kernel used {total:.0f}s (> {MAX_RUNTIME_SECONDS}s budget)", "WARN"
        )
    files = data.get("files", {})
    for name, info in files.items():
        orchestrator.log(f"  {name}: {info.get('mb')} MB")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:  # pragma: no cover - last-resort guard
        import traceback

        traceback.print_exc()
        print(f"::error::talking orchestrator crashed: {type(exc).__name__}: {exc}", flush=True)
        sys.exit(1)
