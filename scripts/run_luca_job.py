#!/usr/bin/env python3
"""
stoic-ai-studio :: Luca v2 orchestrator — "Luca Moretti" flagship character.

Reuses the hardened credential / metadata / push / poll / download / validate
logic from `scripts/run_kaggle_job.py` and swaps in the Luca kernel, artifact
manifest and metrics expectations.

Model weights are pulled from the HuggingFace hub inside Kaggle, but the
*code* still needs shipping: `kaggle kernels push` uploads only the single
code file, so the two workers are published as a private dataset and mounted
at /kaggle/input/<slug>/.

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


#: `kaggle kernels push` uploads ONLY CODE_FILE — sibling files in the kernel
#: directory are silently dropped (confirmed in run 45947135155: the launcher
#: started, then died with "worker script not found"). The workers therefore
#: ship as a private dataset the kernel mounts at /kaggle/input/<slug>/.
WORKER_SOURCES = REPO_ROOT / "scripts" / "luca_v2"
WORKER_FILES = [
    WORKER_SOURCES / "generate_images.py",
    WORKER_SOURCES / "generate_video.py",
    WORKER_SOURCES / "fetch_instantid.py",
    WORKER_SOURCES / "_vendor" / "pipeline_stable_diffusion_xl_instantid.py",
]
#: Dataset slug suffix; the username is prefixed at publish time.
WORKER_DATASET_SLUG = os.environ.get("LUCA_WORKER_DATASET", "luca-v2-workers")


def publish_workers(orchestrator, username: str, env) -> str:
    """
    Publish/refresh the worker dataset and return the slug to mount.

    Filenames are preserved (publish_dataset copies by basename), so
    /kaggle/input/<slug>/generate_images.py is exactly what the launcher probes.
    """
    missing = [str(f.relative_to(REPO_ROOT)) for f in WORKER_FILES if not f.is_file()]
    if missing:
        raise FileNotFoundError(f"worker sources missing: {missing}")

    slug = f"{username}/{WORKER_DATASET_SLUG}"
    orchestrator.log_group(f"publishing worker dataset {slug}")
    try:
        published = orchestrator.publish_dataset(slug, WORKER_FILES, env)
    finally:
        orchestrator.log_group_end()
    if not published:
        # Not fatal on a refresh: the previous version is still mounted and
        # usable, and the push that follows is what actually matters.
        orchestrator.log(
            "worker dataset publish reported failure — the kernel may mount a "
            "stale (or no) copy of the workers", "WARN")
    return slug


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

    if not (KERNEL_DIR / CODE_FILE).is_file():
        orchestrator.annotate_error(f"kernel source missing: {KERNEL_DIR / CODE_FILE}")
        return 2

    # --- ship the workers, then mount them --------------------------------- #
    username = orchestrator.sanitize(os.environ.get("KAGGLE_USERNAME"))
    key = orchestrator.sanitize(os.environ.get("KAGGLE_KEY"))
    if not username or not key:
        return orchestrator.main()      # let main() emit the proper auth error

    env = orchestrator.env_with_credentials(username, key)
    try:
        orchestrator.write_kaggle_credentials(username, key)
    except Exception as exc:            # pragma: no cover
        orchestrator.log(f"could not persist credentials: {exc}", "WARN")

    try:
        worker_slug = publish_workers(orchestrator, username, env)
    except Exception as exc:
        orchestrator.annotate_error(f"could not publish the worker dataset: {exc}")
        return 2

    original_build_metadata = orchestrator.build_metadata

    def build_metadata_with_workers(user: str):
        meta = original_build_metadata(user)
        sources = list(meta.get("dataset_sources") or [])
        if worker_slug not in sources:
            sources.insert(0, worker_slug)
        meta["dataset_sources"] = sources
        return meta

    orchestrator.build_metadata = build_metadata_with_workers

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
