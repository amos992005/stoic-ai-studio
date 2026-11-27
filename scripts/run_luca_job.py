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

#: Success = the full v2 pack: the master portrait, three scenes, the voice
#: and the finished reel. Each is a separate deliverable from one run.
REQUIRED_ARTIFACTS: List[str] = [
    "01_luca_master.png",
    "02_luca_yacht.png",
    "03_luca_porsche.png",
    "04_luca_party.png",
    "luca_voice.wav",
    "luca_reel.mp4",
    "luca_metrics.json",
]
#: Warn only.
OPTIONAL_ARTIFACTS: List[str] = [
    "luca_metrics_images.json",
    "luca_metrics_video.json",
    "build.log",
]
#: Everything the workflow may commit back (5 MB per-file ceiling applies).
COMMITTABLE_ARTIFACTS: List[str] = [
    "outputs/01_luca_master.png",
    "outputs/02_luca_yacht.png",
    "outputs/03_luca_porsche.png",
    "outputs/04_luca_party.png",
    "outputs/luca_voice.wav",
    "outputs/luca_reel.mp4",
    "outputs/luca_metrics.json",
    "outputs/luca_metrics_images.json",
    "outputs/luca_metrics_video.json",
    "outputs/build.log",
]
#: Must exceed the kernel's own budget plus provisioning, the ~8 GB of weights
#: (SDXL + InstantID + Chatterbox + MuseTalk + whisper) and the artifact pull,
#: or we kill a kernel that is about to report success.
POLL_TIMEOUT_SECONDS = int(os.environ.get("KAGGLE_POLL_TIMEOUT_SECONDS", str(55 * 60)))
#: Runtime guard for the warning only.
MAX_RUNTIME_SECONDS = 2400


#: Kaggle only uploads the kernel *directory*, and the workers must be importable
#: siblings of luca_studio.py. They are staged in from scripts/luca_v2/ before
#: the push (and removed from the kernel dir again so git stays a single source
#: of truth — the copies are build artifacts, not sources).
WORKER_SOURCES = REPO_ROOT / "scripts" / "luca_v2"
STAGED_INTO_KERNEL = [
    "generate_images.py",
    "generate_video.py",
    "fetch_instantid.py",
]
#: Vendor data dirs/files copied verbatim next to the staged workers.
STAGED_TREES = ["_vendor"]


def stage_kernel_dir() -> None:
    """Copy the v2 workers into the kernel directory so the push ships them."""
    import shutil

    KERNEL_DIR.mkdir(parents=True, exist_ok=True)
    for name in STAGED_INTO_KERNEL:
        source = WORKER_SOURCES / name
        if not source.is_file():
            raise FileNotFoundError(f"worker script missing: {source}")
        shutil.copy2(source, KERNEL_DIR / name)
    for tree in STAGED_TREES:
        source = WORKER_SOURCES / tree
        if not source.is_dir():
            raise FileNotFoundError(f"worker asset dir missing: {source}")
        target = KERNEL_DIR / tree
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    # Kaggle uploads the directory verbatim; stale bytecode is dead weight.
    import shutil as _shutil
    pycache = KERNEL_DIR / "__pycache__"
    if pycache.is_dir():
        _shutil.rmtree(pycache)

    uploaded = [p.name for p in KERNEL_DIR.iterdir()
                if p.is_file() or p.is_dir()]
    print(f"kernel dir staged for upload: {sorted(uploaded)}")


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
    orchestrator.log("LUCA v2 · DUAL-GPU · MASTER + SCENES + VOICE + REEL")
    orchestrator.log(f"kernel: {KERNEL_DIR.relative_to(REPO_ROOT)}/{CODE_FILE}")
    orchestrator.log("=" * 78)

    try:
        stage_kernel_dir()
    except Exception as exc:
        orchestrator.annotate_error(f"could not stage the kernel directory: {exc}")
        return 2

    if not (KERNEL_DIR / CODE_FILE).is_file():
        orchestrator.annotate_error(f"kernel source missing: {KERNEL_DIR / CODE_FILE}")
        return 2

    exit_code = orchestrator.main()
    if exit_code == 0:
        summarize(orchestrator)
    return exit_code


def summarize(orchestrator) -> None:
    """Print a compact success summary from the two v2 metrics halves."""
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

    images = data.get("images") or {}
    video = data.get("video") or {}
    if images:
        orchestrator.log(
            f"images: {images.get('status')} in {images.get('total_seconds')}s "
            f"on {images.get('device')} · master face "
            f"{images.get('master_face')}"
        )
        for filename, result in (images.get("identity") or {}).items():
            orchestrator.log(
                f"  {filename}: cosine={result.get('cosine')} "
                f"verdict={result.get('verdict')} "
                f"repaired={result.get('repaired')}"
            )
        env = images.get("environment") or {}
        orchestrator.log(
            f"  prompt chunks: master={images.get('master_prompt_tokens')} words · "
            f"{env.get('base_model')} · {env.get('insightface_provider')}"
        )

    if video:
        voice = video.get("voice") or {}
        probe = video.get("reel_probe") or {}
        sync = probe.get("sync") or {}
        orchestrator.log(
            f"voice: {voice.get('duration_s')}s @ {voice.get('sample_rate')} Hz · "
            f"{voice.get('class')} {voice.get('dtype')} · "
            f"peak {voice.get('peak_dbfs')} dBFS · vram freed "
            f"{video.get('vram_after_tts_mb')} MB"
        )
        orchestrator.log(
            f"reel: {probe.get('video', {}).get('width')}x"
            f"{probe.get('video', {}).get('height')} "
            f"{probe.get('video', {}).get('codec')} / "
            f"{probe.get('audio', {}).get('codec')} · "
            f"{probe.get('format', {}).get('duration')}s · "
            f"drift {sync.get('drift_s')}s"
        )
        musetalk = video.get("musetalk") or {}
        orchestrator.log(
            f"musetalk: {musetalk.get('frames')} frames @ "
            f"{musetalk.get('fps')} fps"
        )

    artifacts = data.get("artifacts") or []
    orchestrator.log(f"artifacts reported by the workers: {artifacts}")
    missing = [a for a in REQUIRED_ARTIFACTS
               if not (orchestrator.OUTPUT_DIR / a).is_file()]
    if missing:
        orchestrator.log(f"REQUIRED artifacts missing: {missing}", "ERROR")
    else:
        orchestrator.log("all required artifacts present")


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
