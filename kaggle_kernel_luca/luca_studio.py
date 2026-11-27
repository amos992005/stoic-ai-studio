"""
Luca Moretti · v2 — the launcher Kaggle actually executes.

This file does as little as possible: it gets the two worker scripts onto disk
next to each other, refuses to start if a prohibited technique has crept back
in, bootstraps the InstantID community pipeline, and then runs

    python generate_images.py &     # cuda:0 · SDXL + InstantID
    python generate_video.py  &     # cuda:1 · Chatterbox -> MuseTalk -> mp4
    wait

Both workers stream their logs back here so a stalled run is diagnosable from
the kernel output alone.

Deliberately NOT done here: no model loading, no GPU work, no orchestration
logic beyond the two subprocesses. Everything model-specific lives in the
workers so it can be unit-checked without a GPU.
"""

from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
import sys
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/kaggle/working"))
WORKER_DIR = Path(os.environ.get("LUCA_WORKER_DIR", "/kaggle/working/luca_workers"))

#: Banned tokens, checked against both workers before any GPU work starts.
#: Matched against executable source only — see `_executable_source`.
BANNED = {
    'pipe.to("cpu")': "CPU offload is prohibited",
    "inswapper_128": "inswapper_128 is prohibited",
    "StableVideoDiffusion": "SVD-XT is prohibited",
    "stable-video-diffusion": "SVD-XT is prohibited",
    "StableDiffusionXLImg2ImgPipeline": "SDXL img2img Adetailer is prohibited",
}

#: Files that must be copied into WORKER_DIR for the subprocesses to run.
WORKER_FILES = ("generate_images.py", "generate_video.py", "fetch_instantid.py")

#: Lines that merely *report* the prohibition (a metrics dict, a log line) are
#: not offences. An offence is an executable call or an import.
ALLOWED_PATTERNS = (
    re.compile(r'^\s*"[^"]*"\s*:\s*False\s*,?\s*$'),        # "x": False,
    re.compile(r"^\s*'[^']*'\s*:\s*False\s*,?\s*$"),
)


def ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def banner(message: str, level: str = "INFO") -> None:
    print(f"[{ts()}] [{level:<5}] [launcher] {message}", flush=True)


def _executable_source(source: str) -> str:
    """
    Drop everything that is documentation or a report of the ban.

    A plain comment-stripping filter is not enough: both workers *state* the
    prohibitions in their module docstring and *report* them in a metrics dict
    (`"inswapper_128": False`), so a naive substring scan rejects the very
    files that comply. Docstrings are removed via the AST, comment-only lines
    by prefix, self-reporting lines by pattern.
    """
    try:
        tree: ast.AST | None = ast.parse(source)
    except SyntaxError:
        tree = None          # cannot reason about it; fall back to text only

    if tree is not None:
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node, clean=False)
                if doc:
                    docstrings.add(doc)
        for doc in docstrings:
            source = source.replace(doc, "")

    kept = []
    for line in source.splitlines():
        if line.strip().startswith("#"):
            continue
        if any(pattern.match(line) for pattern in ALLOWED_PATTERNS):
            continue
        kept.append(line)
    return "\n".join(kept)


def stage_workers() -> Path:
    """
    Copy the worker scripts somewhere importable by a subprocess.

    Kaggle runs this file from /kaggle/src/script.py, which is injected into
    *this* interpreter's sys.path but is not a real importable package for a
    child process. Copying next to the workers (and running from there) is what
    makes `python generate_images.py` resolve its siblings — including the
    vendored InstantID pipeline that `fetch_instantid.py` writes.
    """
    WORKER_DIR.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent

    def find(name: str) -> Path:
        for candidate in (here / name, here.parent / name,
                          here.parent / "scripts" / "luca_v2" / name,
                          Path("/kaggle/src") / name, Path(name)):
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"worker script not found: {name} (looked in {here} and redeployed "
            f"paths — the kernel must ship scripts/luca_v2/* next to this file)")

    for name in WORKER_FILES:
        source = find(name)
        destination = WORKER_DIR / name
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        banner(f"staged {name} ({destination.stat().st_size / 1024:.1f} KB)")

    # The vendored InstantID pipeline travels with the code so the bootstrapper
    # never has to reach the network mid-run.
    vendor = here / "_vendor"
    if not vendor.is_dir():
        vendor = here.parent / "scripts" / "luca_v2" / "_vendor"
    if vendor.is_dir():
        target = WORKER_DIR / "_vendor"
        target.mkdir(parents=True, exist_ok=True)
        for item in vendor.iterdir():
            if item.is_file():
                shutil.copy2(item, target / item.name)
        banner(f"staged _vendor/ ({len(list(target.iterdir()))} file(s))")
    else:
        banner("no _vendor/ directory — fetch_instantid.py will use the network",
               "WARN")

    return WORKER_DIR


def check_prohibitions(workdir: Path) -> None:
    """Fail fast if a banned construct reappeared in either worker."""
    offenders = []
    for name in ("generate_images.py", "generate_video.py"):
        code = _executable_source((workdir / name).read_text(encoding="utf-8"))
        for token, reason in BANNED.items():
            if token in code:
                offenders.append(f"{name}: {token} — {reason}")
    if offenders:
        raise SystemExit("prohibited constructs present:\n  " + "\n  ".join(offenders))
    banner("prohibition check passed: no pipe.to('cpu'), inswapper, SVD-XT or "
           "img2img adetailer in either worker")


def preflight_instantid(workdir: Path) -> None:
    """
    Make sure the InstantID community pipeline exists before the GPU work starts.

    Importing it lazily inside the worker would waste the SDXL load time before
    failing; this runs the bootstrapper outright.
    """
    proc = subprocess.run(
        [sys.executable, "fetch_instantid.py"],
        cwd=str(workdir), capture_output=True, text=True,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    for line in output.strip().splitlines():
        banner(line.strip())
    if proc.returncode != 0:
        raise SystemExit("InstantID pipeline bootstrap failed — see above")


def run_parallel(workdir: Path) -> Dict[str, int]:
    """
    The whole point: both pipelines live at the same time on separate GPUs.

        python generate_images.py & python generate_video.py & wait
    """
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["OUTPUT_DIR"] = str(OUTPUT_DIR)
    env.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
    # Keep fragmentation from turning a 14.5 GB card into a 12 GB one.
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    commands = {
        "images": [sys.executable, "generate_images.py"],
        "video": [sys.executable, "generate_video.py"],
    }
    banner("=" * 70)
    banner("launching both pipelines in parallel")
    banner(f"  images → cuda:0 : {' '.join(commands['images'])}")
    banner(f"  video  → cuda:1 : {' '.join(commands['video'])}")
    banner("=" * 70)

    procs = {}
    for name, command in commands.items():
        procs[name] = subprocess.Popen(
            command, cwd=str(workdir), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        banner(f"started {name} (pid {procs[name].pid})")

    # Stream both logs as they arrive so a stalled run is diagnosable.
    def pump(name: str, proc) -> None:
        prefix = f"[{name}] "
        for line in iter(proc.stdout.readline, ""):
            sys.stdout.write(prefix + line)
            sys.stdout.flush()

    threads = [threading.Thread(target=pump, args=(n, p), daemon=True)
               for n, p in procs.items()]
    for thread in threads:
        thread.start()

    exit_codes: Dict[str, int] = {}
    for name, proc in procs.items():
        exit_codes[name] = proc.wait()
        banner(f"{name} exited with code {exit_codes[name]}",
               "INFO" if exit_codes[name] == 0 else "ERROR")
    for thread in threads:
        thread.join(timeout=30)
    return exit_codes


def main() -> int:
    banner("=" * 70)
    banner("stoic-ai-studio · LUCA v2 · parallel dual-GPU architecture")
    banner("=" * 70)

    try:
        workdir = stage_workers()
        check_prohibitions(workdir)
        preflight_instantid(workdir)
        exit_codes = run_parallel(workdir)

        images_ok = exit_codes.get("images") == 0
        video_ok = exit_codes.get("video") == 0
        banner(f"summary: images={'ok' if images_ok else 'FAILED'} "
               f"video={'ok' if video_ok else 'FAILED'}")

        # Report what actually landed on disk.
        for name in ("01_luca_master.png", "02_luca_yacht.png",
                     "03_luca_porsche.png", "04_luca_party.png",
                     "luca_voice.wav", "luca_reel.mp4"):
            path = OUTPUT_DIR / name
            banner(f"  {'✓' if path.is_file() else '✗'} {name}"
                   + (f" ({path.stat().st_size / 1024:.0f} KB)"
                      if path.is_file() else ""))

        if not images_ok:
            banner("image pipeline failed — the carousel is the deliverable",
                   "ERROR")
            return 1
        if not video_ok:
            # The reel is the second half of one run, not a separate deliverable.
            banner("video pipeline failed — carousel artifacts are still valid",
                   "WARN")
            return 0
        return 0

    except SystemExit as exc:
        banner(str(exc), "ERROR")
        return 2
    except Exception as exc:
        banner(f"launcher crashed: {type(exc).__name__}: {exc}", "ERROR")
        for line in traceback.format_exc().splitlines():
            banner(line, "ERROR")
        return 2


if __name__ == "__main__":
    sys.exit(main())
