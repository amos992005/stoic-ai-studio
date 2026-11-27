"""
Shared plumbing for the two Luca workers.

Both processes run in the same interpreter environment at the same time, so
anything that touches the environment (pip) has to be coordinated, not just
written twice. Everything here is deliberately dependency-free: it is imported
before torch, diffusers or any network client exists.
"""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/kaggle/working"))
BUILD_LOG = OUTPUT_DIR / "build.log"
METRICS_PATH = OUTPUT_DIR / "luca_metrics.json"

#: Serialises package installs between the two processes.
PIP_LOCK = OUTPUT_DIR / ".pip.lock"


def ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Tee:
    """
    Mirror stdout into a shared build log.

    Both workers append to the same file and the kernel streams our stdout to
    Kaggle, so every line lands twice: once live, once greppable.
    """

    def __init__(self, path: Path, stream):
        self._stream = stream
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(path, "a", encoding="utf-8", buffering=1)
        self.lock_path = path.with_suffix(".lock")

    def write(self, data: str) -> int:
        try:
            self._stream.write(data)
        except Exception:
            pass
        try:
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


def log(message: str, level: str = "INFO", tag: str = "img") -> None:
    print(f"[{ts()}] [{level:<5}] [{tag}] {message}", flush=True)


class timed:
    """Context manager that records wall-clock time per stage."""

    def __init__(self, label: str, timings: Dict[str, float], tag: str = "img"):
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
# Environment
# --------------------------------------------------------------------------- #


def torch_fingerprint() -> Tuple[Optional[str], Optional[str]]:
    """The torch/torchaudio versions that must survive every pip call."""
    import torch

    try:
        import torchaudio

        audio = torchaudio.__version__
    except Exception:
        audio = None
    return torch.__version__, audio


def pip_install(packages: Iterable[str], *, no_deps: bool = False,
                no_build_isolation: bool = False, prepare_sdist=None,
                tag: str = "img") -> bool:
    """
    Install packages under a cross-process lock, never letting torch move.

    The two workers share one interpreter, so a concurrent `pip install` that
    rewrites a package the other process is importing is a real failure mode,
    and a resolver that "fixes" torch while the image worker is mid-sample is a
    worse one. The lock makes installs serial; the re-pin repairs the damage a
    dependency's metadata can still do.
    """
    packages = list(packages)
    command_prefix: List[str] = []
    torch_version, torchaudio_version = torch_fingerprint()
    command = [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir"]
    if no_deps:
        command.append("--no-deps")
    if no_build_isolation:
        # basicsr's setup.py expects torch and torchvision to be importable
        # already; in an isolated build environment they are not, and the wheel
        # fails with "No available output" before any compilation happens.
        command.append("--no-build-isolation")
    if prepare_sdist is not None:
        # Some 2022-era setup.py files do not survive a modern setuptools. There
        # is no pip flag that edits a build, so the sdist is fetched, patched
        # and installed from a local path instead.
        packages = prepare_sdist(packages)
    command.extend(packages)

    flags = "".join(flag for flag, on in ((" (no-deps)", no_deps),
                                          (" (no-build-isolation)",
                                           no_build_isolation),
                                          (" (patched sdist)",
                                           prepare_sdist is not None)) if on)
    log(f"pip install {' '.join(packages)}{flags}", tag=tag)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(PIP_LOCK, "a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            proc = subprocess.run(command, capture_output=True, text=True)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    if proc.returncode != 0:
        log(f"pip failed: {(proc.stderr or proc.stdout)[-700:]}", "ERROR", tag)
        return False

    # Repair a torch move, still under the lock.
    try:
        import importlib

        importlib.invalidate_caches()
        import torch as torch_now

        if torch_version and torch_now.__version__ != torch_version:
            log(f"torch moved to {torch_now.__version__} — restoring "
                f"{torch_version}", "WARN", tag)
            with open(PIP_LOCK, "a+") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                try:
                    subprocess.run(
                        [sys.executable, "-m", "pip", "install", "-q",
                         "--no-cache-dir", f"torch=={torch_version}"],
                        capture_output=True, text=True,
                    )
                finally:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    except Exception as exc:
        log(f"torch re-pin check skipped: {exc}", "WARN", tag)
    return True


def module_name(package_spec: str) -> str:
    """'imageio[ffmpeg]' -> 'imageio'; 'opencv-python-headless' -> 'cv2'."""
    base = (package_spec.split("[")[0]
            .split("==")[0].split(">=")[0].split("<")[0].split("~=")[0])
    return {"opencv_python_headless": "cv2",
            "chatterbox_tts": "chatterbox",
            "ffmpeg_python": "ffmpeg",
            "pillow": "PIL"}.get(base.replace("-", "_").lower(),
                                 base.replace("-", "_"))


def missing_packages(specs: Iterable[str]) -> List[str]:
    """Which of these are not importable yet."""
    import importlib

    missing = []
    for spec in specs:
        try:
            importlib.import_module(module_name(spec))
        except Exception:
            missing.append(spec)
    return missing


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def merge_metrics(section: str, metrics: Dict[str, object], path: Path,
                  tag: str = "img") -> None:
    """
    Fold this worker's half into the shared luca_metrics.json, safely.

    Both workers finish at unpredictable times and target the same file, so a
    naive read-modify-write loses whichever half lands first. The sidecar is
    written unconditionally first (it is the source of truth for post-mortems),
    then the merge runs under an exclusive lock and replaces the file
    atomically, so a reader never sees a half-written document.
    """
    sidecar = path.parent / f"luca_metrics_{section}.json"
    try:
        sidecar.write_text(json.dumps(metrics, indent=2, default=str),
                           encoding="utf-8")
    except Exception as exc:
        log(f"could not write {sidecar.name}: {exc}", "WARN", tag)

    lock = path.parent / "luca_metrics.lock"
    try:
        with open(lock, "a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                merged: Dict[str, Any] = {}
                if path.is_file():
                    try:
                        merged = json.loads(path.read_text(encoding="utf-8"))
                    except Exception:
                        merged = {}
                merged[section] = metrics
                merged.setdefault("artifacts", [])
                merged["artifacts"] = sorted(
                    set(merged["artifacts"]) | set(metrics.get("artifacts") or []))
                tmp = path.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(merged, indent=2, default=str),
                               encoding="utf-8")
                os.replace(tmp, path)      # atomic: no torn reads
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except Exception as exc:
        log(f"could not merge metrics: {exc}", "WARN", tag)


def resolve_device() -> str:
    """
    The GPU this process may use, and make it current.

    The launcher gives each worker its own card via CUDA_VISIBLE_DEVICES, so the
    card is always cuda:0 inside the process; the second branch only applies
    when both cards are visible (running a worker directly).
    """
    try:
        import torch
    except Exception:
        return "cuda:0"
    if not torch.cuda.is_available():
        return "cuda:0"
    index = 0 if torch.cuda.device_count() < 2 else 1
    torch.cuda.set_device(index)
    return f"cuda:{index}"
