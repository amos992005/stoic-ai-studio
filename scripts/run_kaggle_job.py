#!/usr/bin/env python3
"""
stoic-ai-studio :: Kaggle job orchestrator (runs on a GitHub Actions CPU runner).

Responsibilities
----------------
1. Authenticate the `kaggle` CLI from KAGGLE_USERNAME / KAGGLE_KEY, tolerating
   copy-paste damage (stray quotes, whitespace, `KGAT_`-prefixed keys) — both via
   env vars *and* via a private `~/.kaggle/kaggle.json`, because some CLI
   versions/credential types only behave with one of the two.
2. Generate `kaggle_kernel/kernel-metadata.json` on the fly (never hard-coded).
3. Push the kernel, poll its status until it finishes (25 min budget), then pull
   the artifacts + log into `./outputs/`.
4. Always leave a readable `outputs/build.log` behind and set the process exit
   code so the workflow can fail loudly.

Exit codes
----------
0  kernel completed and outputs were downloaded
1  kernel errored / cancelled / timed out (log fetched when possible)
2  configuration or authentication failure (nothing was ever pushed)
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parent.parent
KERNEL_DIR = REPO_ROOT / "kaggle_kernel"
CODE_FILE = "character_forge.py"
OUTPUT_DIR = REPO_ROOT / "outputs"
BUILD_LOG = OUTPUT_DIR / "build.log"

KERNEL_SLUG = "stoic-character-forge"
#: Fallback dataset/source so a `kernel_type: script` kernel can locate its code
#: even when the sandbox does not seed the private dataset from the pushed file.
DEFAULT_DATASET_SOURCES = [
    f"amos992005/{KERNEL_SLUG}",
    "amos992005/stoic-ai-studio",
]
#: Kaggle warns unless slugify(title) == kernel slug; keep them identical.
KERNEL_TITLE = KERNEL_SLUG

#: 25 s polling, 25 min budget — overridable so the flow can be dry-run locally.
POLL_INTERVAL_SECONDS = max(1, int(os.environ.get("KAGGLE_POLL_INTERVAL_SECONDS", "25")))
POLL_TIMEOUT_SECONDS = max(10, int(os.environ.get("KAGGLE_POLL_TIMEOUT_SECONDS", str(25 * 60))))

#: Artifacts we care about (used for post-download validation + size guard).
EXPECTED_ARTIFACTS = [
    "contact_sheet.png",
    "01_master_face.png",
    "02_library_scene.png",
    "03_rainy_street.png",
    "04_podcast_studio.png",
    "metrics.json",
]
#: Artifacts larger than this are downloaded but never committed by CI.
MAX_COMMIT_BYTES = 5 * 1024 * 1024

#: Files whose presence defines success, and the files the workflow may commit.
#: Phase-specific orchestrators (e.g. run_talking_job.py) override these.
REQUIRED_ARTIFACTS: List[str] = ["contact_sheet.png", "metrics.json"]
OPTIONAL_ARTIFACTS: List[str] = []


# --------------------------------------------------------------------------- #
# Logging helpers (Actions-annotated, timestamped)
# --------------------------------------------------------------------------- #


def ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def log(message: str, level: str = "INFO") -> None:
    print(f"[{ts()}] [{level:<5}] {message}", flush=True)


def log_group(title: str) -> None:
    print(f"::group::{title}", flush=True)


def log_group_end() -> None:
    print("::endgroup::", flush=True)


def annotate_error(message: str) -> None:
    safe = message.replace("\n", " ")[:900]
    print(f"::error::{safe}", flush=True)


# --------------------------------------------------------------------------- #
# Credentials
# --------------------------------------------------------------------------- #

_QUOTES = "\"'`\u2018\u2019\u201c\u201d"


def sanitize(value: Optional[str]) -> str:
    """
    Strip accidental quotes/whitespace/newlines pasted along with a secret.

    Kaggle keys are typically 32 hex chars; newer ones are `KGAT_...` API tokens.
    We keep the value verbatim (minus wrapper damage) so both styles work.
    """
    if value is None:
        return ""
    cleaned = value.strip().strip(_QUOTES).strip()
    cleaned = cleaned.replace("\r", "").replace("\n", "").replace("\t", "")
    cleaned = cleaned.strip(_QUOTES).strip()
    if cleaned.lower().startswith("kaggle_key=") or cleaned.lower().startswith("key="):
        cleaned = cleaned.split("=", 1)[1].strip()
    if cleaned.lower().startswith("kaggle_username="):
        cleaned = cleaned.split("=", 1)[1].strip()
    # Drop zero-width / non-breaking characters that survive copy-paste.
    cleaned = cleaned.replace("\u200b", "").replace("\u00a0", "")
    return cleaned


def is_access_token(key: str) -> bool:
    """Kaggle's newer API tokens are prefixed `KGAT_`; legacy keys are 32 hex."""
    return key.strip().lower().startswith("kgat_")


def unpack_credentials(username: str, key: str) -> tuple:
    """
    Tolerate people pasting the whole `kaggle.json` (~ or {"username","key"})
    into a single secret; extract the real username/key pair.
    """
    blob = key if key.strip().startswith("{") else (
        username if username.strip().startswith("{") else ""
    )
    if blob:
        try:
            data = json.loads(blob)
            extracted_user = str(data.get("username", "") or "").strip()
            extracted_key = str(data.get("key", "") or "").strip()
            if extracted_user and extracted_key:
                log("parsed a kaggle.json blob out of the secret")
                return extracted_user, extracted_key
        except json.JSONDecodeError:
            log("secret looks like JSON but could not be parsed; using it verbatim", "WARN")
    return username, key


def write_kaggle_credentials(username: str, key: str) -> List[Path]:
    """
    Persist credentials for the CLI.

    * legacy key  -> `~/.kaggle/kaggle.json` ({"username", "key"}), mode 600
    * `KGAT_...`  -> `~/.kaggle/access_token` (bare token), mode 600

    CLI 2.x only accepts KGAT tokens through the access-token path; putting them
    in kaggle.json's "key" field makes auth fail with "Authentication required".
    Both files are 0600 and the directory 0700.
    """
    kaggle_dir = Path.home() / ".kaggle"
    if kaggle_dir.exists() and not kaggle_dir.is_dir():
        kaggle_dir.unlink()
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    try:
        kaggle_dir.chmod(0o700)
    except OSError:
        pass

    written: List[Path] = []

    def _write(path: Path, content: str) -> None:
        path.write_text(content, encoding="utf-8")
        try:
            path.chmod(0o600)
        except OSError as exc:
            log(f"chmod 600 failed on {path}: {exc}", "WARN")
        written.append(path)
        log(f"wrote {path} (mode {oct(path.stat().st_mode & 0o777)})")

    token = is_access_token(key)
    if token:
        _write(kaggle_dir / "access_token", key + "\n")
        # Keep username discoverable for the kernel slug without an API call.
        _write(kaggle_dir / "username", username + "\n")
    else:
        _write(kaggle_dir / "kaggle.json", json.dumps({"username": username, "key": key}, indent=2))
    return written


def env_with_credentials(username: str, key: str) -> Dict[str, str]:
    """
    Build a clean environment for the `kaggle` CLI.

    Exactly one auth style is exported so the CLI cannot fall back to a legacy
    lookup with a token that only works on the access-token path.
    """
    env = {k: v for k, v in os.environ.items() if k not in {"KAGGLE_KEY", "KAGGLE_API_TOKEN"}}
    if is_access_token(key):
        env["KAGGLE_API_TOKEN"] = key
        env["KAGGLE_USERNAME"] = username
        log("auth mode: access token (KAGGLE_API_TOKEN + ~/.kaggle/access_token)")
    else:
        env["KAGGLE_USERNAME"] = username
        env["KAGGLE_KEY"] = key
        log("auth mode: legacy API key (KAGGLE_USERNAME + KAGGLE_KEY)")
    env.setdefault("KAGGLE_CONFIG_DIR", str(Path.home() / ".kaggle"))
    env["PYTHONUNBUFFERED"] = "1"
    return env


# --------------------------------------------------------------------------- #
# Subprocess wrapper
# --------------------------------------------------------------------------- #


def run(
    args: List[str],
    *,
    env: Dict[str, str],
    check: bool = False,
    timeout: int = 600,
    stream: bool = True,
) -> subprocess.CompletedProcess:
    """Run a command, echo its output with a prefix, and never crash on failure."""
    log(f"$ {' '.join(args)}")
    try:
        proc = subprocess.run(
            args,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        log(f"command not found: {args[0]} ({exc})", "ERROR")
        return subprocess.CompletedProcess(args, 127, "", str(exc))
    except subprocess.TimeoutExpired:
        log(f"command timed out after {timeout}s: {' '.join(args)}", "ERROR")
        return subprocess.CompletedProcess(args, 124, "", "timeout")

    if stream:
        for line in (proc.stdout or "").splitlines():
            print(f"    | {line}", flush=True)
        for line in (proc.stderr or "").splitlines():
            print(f"    ! {line}", flush=True)

    if proc.returncode != 0:
        log(f"exit code {proc.returncode}: {' '.join(args)}", "WARN")
        if check:
            raise RuntimeError(
                f"command failed ({proc.returncode}): {' '.join(args)}\n"
                f"{(proc.stderr or proc.stdout or '').strip()[:2000]}"
            )
    return proc


# --------------------------------------------------------------------------- #
# Kernel metadata
# --------------------------------------------------------------------------- #


def build_metadata(username: str) -> Dict[str, Any]:
    """
    Compose kernel-metadata.json.

    `kernel_type: script` + `code_file` is what Kaggle expects for a plain .py
    kernel. When `code_file` is set, the CLI rejects an accompanying `"source"`
    key, so the dataset fallback is only emitted for notebook-style kernels.
    """
    metadata: Dict[str, Any] = {
        "id": f"{username}/{KERNEL_SLUG}",
        # Must slugify to exactly the kernel slug, otherwise the CLI warns and
        # the kernel can be created/updated under a surprising name.
        "title": KERNEL_TITLE,
        "code_file": CODE_FILE,
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }

    # Optional: if the user has published the repo as a Kaggle dataset, prefer
    # the real slug over our guess (env var wins, then first existing guess).
    explicit = sanitize(os.environ.get("KAGGLE_DATASET_SLUG"))
    if explicit:
        metadata["dataset_sources"] = [explicit]
    elif os.environ.get("KAGGLE_DATASET_AUTODETECT", "1") in {"1", "true", "yes"}:
        for candidate in DEFAULT_DATASET_SOURCES:
            if dataset_exists(candidate):
                metadata["dataset_sources"] = [candidate]
                break
    return metadata


def publish_dataset(
    slug: str,
    files: List[Path],
    env: Dict[str, str],
    timings: Optional[Dict[str, float]] = None,
    overwrite: bool = True,
) -> bool:
    """
    Publish (or refresh) a private Kaggle dataset from local files.

    `kaggle kernels push` only uploads the code file, so any auxiliary asset a
    kernel needs (e.g. the portrait PNG) must arrive as a dataset mount under
    /kaggle/input/<dataset-slug>/.
    """
    staging = Path(os.environ.get("DATASET_STAGING_DIR", str(REPO_ROOT / ".kaggle_dataset")))
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    for source in files:
        if not source.is_file():
            log(f"dataset file missing: {source}", "WARN")
            continue
        shutil.copyfile(source, staging / source.name)

    meta = {
        "title": slug.replace("-", " ")[:50],
        "id": slug,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (staging / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    exists = dataset_exists(slug)
    args = [
        "kaggle", "datasets", "version" if exists else "create",
        "-p", str(staging),
    ]
    if exists:
        args += ["-m", f"refresh {ts()}", "--dir-mode", "zip"]
        if overwrite:
            args.append("--quiet")
    else:
        args += ["--dir-mode", "zip"]
        if overwrite:
            args.append("--quiet")

    proc = run(args, env=env, timeout=900)
    if proc.returncode != 0:
        combined = f"{proc.stdout or ''}\n{proc.stderr or ''}"
        # A dataset cannot be updated while a kernel is still mounting it.
        if "409" in combined or "Conflict" in combined or "still being" in combined:
            log("python-sdk 409: the dataset is locked by a running kernel", "WARN")
            return dataset_exists(slug)
        log(f"dataset publish failed ({proc.returncode})", "ERROR")
        return False
    log(f"dataset published: {slug}")
    return True


def dataset_exists(slug: str) -> bool:
    """Cheap, non-fatal existence probe (kaggle datasets list -s <slug>)."""
    try:
        proc = subprocess.run(
            ["kaggle", "datasets", "list", "-s", slug, "--format", "csv"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0:
            return False
        for line in proc.stdout.splitlines()[1:]:
            first_cell = line.split(",")[0].strip().strip('"')
            if first_cell.lower() == slug.lower():
                return True
    except Exception as exc:  # pragma: no cover - network flake
        log(f"dataset probe failed for {slug}: {exc}", "DEBUG")
    return False


def write_metadata(metadata: Dict[str, Any]) -> Path:
    KERNEL_DIR.mkdir(parents=True, exist_ok=True)
    path = KERNEL_DIR / "kernel-metadata.json"
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    log(f"generated {path.relative_to(REPO_ROOT)}:")
    for line in path.read_text(encoding="utf-8").splitlines():
        print(f"    | {line}", flush=True)
    return path


# --------------------------------------------------------------------------- #
# Status parsing
# --------------------------------------------------------------------------- #

_STATUS_RE = re.compile(
    r"\b(cancelacknowledged|cancelrequested|canceled|cancelled|complete|"
    r"error|failed|running|queued|started|unknown)\b",
    re.IGNORECASE,
)

#: `kaggle kernels status` prints a protobuf enum; map it to something readable.
_STATUS_ALIASES = {
    "kernelworkerstatus.complete": "complete",
    "kernelworkerstatus.error": "error",
    "kernelworkerstatus.cancelacknowledged": "cancelacknowledged",
    "kernelworkerstatus.cancelrequested": "cancelrequested",
    "kernelworkerstatus.canceled": "canceled",
    "kernelworkerstatus.running": "running",
    "kernelworkerstatus.queued": "queued",
    "kernelworkerstatus.new": "queued",
}


def parse_status(raw: str) -> Optional[str]:
    """
    Extract a normalised status from `kaggle kernels status` output.

    The CLI prints things like:
        stoic/stoic-character-forge has status "KernelWorkerStatus.COMPLETE"
    and older builds used `"complete"` / `"KernelWorkerStatus.RUNNING"`.
    """
    if not raw:
        return None
    text = raw.strip()

    # 1) quoted payload, e.g. has status "KernelWorkerStatus.COMPLETE"
    quoted = re.search(r'status\s+"?([^"\n]+?)"?$', text, re.IGNORECASE | re.MULTILINE)
    if quoted:
        payload = quoted.group(1).strip().strip('"').lower()
        payload = _STATUS_ALIASES.get(payload, payload)
        if payload in _STATUS_ALIASES:
            return _STATUS_ALIASES[payload]
        match = _STATUS_RE.search(payload)
        if match:
            found = match.group(1).lower()
            return "canceled" if found == "cancelled" else found

    # 2) bare enum token anywhere in the text
    for token in re.findall(r"kernelworkerstatus\.(\w+)", text, re.IGNORECASE):
        normalized = token.lower().replace("_", "")
        mapped = _STATUS_ALIASES.get(f"kernelworkerstatus.{normalized}")
        if mapped:
            return mapped

    # 3) plain word anywhere in the text
    match = _STATUS_RE.search(text)
    if not match:
        return None
    found = match.group(1).lower()
    return "canceled" if found == "cancelled" else found


TERMINAL_OK = {"complete"}
TERMINAL_FAIL = {"error", "failed", "canceled", "cancelacknowledged", "cancelrequested"}


def normalize_status(status: Optional[str]) -> Optional[str]:
    if status is None:
        return None
    status = status.lower()
    if status == "cancelacknowledged":
        return "cancelAcknowledged"
    return status


def get_status(slug: str, env: Dict[str, str]) -> Optional[str]:
    proc = run(
        ["kaggle", "kernels", "status", slug],
        env=env,
        stream=False,
        timeout=180,
    )
    combined = f"{proc.stdout or ''}\n{proc.stderr or ''}".strip()
    raw = combined.splitlines()[-1] if combined else ""
    status = parse_status(combined)
    if proc.returncode != 0:
        log(f"status query exited {proc.returncode}: {raw[:300]}", "WARN")
    log(f"status → {status or 'unparseable'}"
        + (f"  (raw: {raw[:160]})" if not status else ""))
    return status


# --------------------------------------------------------------------------- #
# Output download
# --------------------------------------------------------------------------- #


def download_outputs(slug: str, env: Dict[str, str], dest: Path) -> bool:
    """`kaggle kernels output` into ./outputs (which is the kernel workdir)."""
    dest.mkdir(parents=True, exist_ok=True)
    proc = run(
        ["kaggle", "kernels", "output", slug, "-p", str(dest)],
        env=env,
        timeout=900,
    )
    if proc.returncode == 0:
        log(f"outputs downloaded into {dest}")
        return True
    log(f"output download failed (exit {proc.returncode})", "ERROR")
    return False


def flatten_outputs(root: Path) -> None:
    """
    Defensive: if the CLI nested artifacts (outputs/outputs/...), hoist them so
    `outputs/contact_sheet.png` is always the canonical path.
    """
    for nested in sorted(p for p in root.rglob("*") if p.is_dir()):
        for item in list(nested.iterdir()):
            if item.is_file():
                target = root / item.name
                if not target.exists():
                    shutil.move(str(item), str(target))
                    log(f"hoisted {item} → {target.name}")


def fetch_kernel_log_via_cli(slug: str, env: Dict[str, str]) -> bool:
    """
    `kaggle kernels logs <slug>` (CLI >= 2.x) returns the run log even when the
    kernel failed before writing anything to /kaggle/working.
    """
    proc = run(
        ["kaggle", "kernels", "logs", slug],
        env=env,
        stream=False,
        timeout=300,
    )
    content = (proc.stdout or "").strip()
    if proc.returncode != 0 or not content:
        log("`kaggle kernels logs` unavailable or empty (older CLI?)", "WARN")
        return False
    append_log_block("kaggle kernels logs", content)
    return True


def append_log_block(source: str, content: str) -> None:
    """Append a labelled log block to outputs/build.log."""
    header = f"\n===== {source} · fetched {ts()} =====\n"
    with open(BUILD_LOG, "a", encoding="utf-8") as handle:
        handle.write(header)
        handle.write(content)
        if not content.endswith("\n"):
            handle.write("\n")
    log(f"appended {source} → {BUILD_LOG.relative_to(REPO_ROOT)} ({len(content)} chars)")


def collect_build_log(root: Path) -> bool:
    """
    Kaggle returns the run log as `<slug>.log`; mirror it to outputs/build.log so
    the repo has a debuggable trace even when the kernel crashes.
    """
    # Never treat our own rotated build.log as "the kernel log" (that would
    # append the file to itself and then delete it).
    candidates = sorted(
        [
            p
            for p in root.rglob("*")
            if p.is_file()
            and p.suffix == ".log"
            and p.resolve() != BUILD_LOG.resolve()
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        log("no kernel log file found in outputs", "WARN")
        return False
    source = candidates[0]
    try:
        content = source.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        log(f"could not read kernel log {source}: {exc}", "WARN")
        return False

    append_log_block(f"kernel log file {source.name}", content)
    try:
        source.unlink()  # keep only the canonical build.log
    except OSError:
        pass
    return True


def report_artifacts(root: Path) -> Dict[str, Any]:
    """Summarise what landed in outputs/ and flag oversized files."""
    summary: Dict[str, Any] = {"downloaded": {}, "oversized": [], "missing": []}
    for name in EXPECTED_ARTIFACTS:
        path = root / name
        if path.exists():
            size = path.stat().st_size
            summary["downloaded"][name] = size
            if size > MAX_COMMIT_BYTES:
                summary["oversized"].append(name)
            log(f"artifact ✔ {name} ({size / 1024:.0f} KB)")
        else:
            summary["missing"].append(name)
            log(f"artifact ✘ {name} (missing)", "WARN")
    for name in OPTIONAL_ARTIFACTS:
        path = root / name
        if path.exists():
            summary["downloaded"][name] = path.stat().st_size
            log(f"artifact ○ {name} ({path.stat().st_size / 1024:.0f} KB, optional)")
    return summary


# --------------------------------------------------------------------------- #
# Main flow
# --------------------------------------------------------------------------- #


def rotate_build_log() -> None:
    """
    Start every run with a fresh, bounded build.log.

    GitHub Actions tees into this file and we also append the Kaggle kernel log,
    so without rotation it would grow forever across runs.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        previous = BUILD_LOG.stat().st_size if BUILD_LOG.exists() else 0
        BUILD_LOG.write_text(
            f"===== stoic-ai-studio · orchestrator run · {ts()} =====\n"
            f"===== (previous log was {previous} bytes and has been rotated) =====\n",
            encoding="utf-8",
        )
        log(f"rotated {BUILD_LOG.relative_to(REPO_ROOT)} (previous size: {previous} bytes)")
    except OSError as exc:
        log(f"could not rotate build.log: {exc}", "WARN")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rotate_build_log()
    log("=" * 78)
    log("stoic-ai-studio · Kaggle orchestrator · THE CHARACTER FORGE")
    log("=" * 78)

    username = sanitize(os.environ.get("KAGGLE_USERNAME"))
    key = sanitize(os.environ.get("KAGGLE_KEY"))

    if not username or not key:
        annotate_error(
            "KAGGLE_USERNAME and/or KAGGLE_KEY are missing. Add them as repository "
            "secrets (Settings → Secrets and variables → Actions)."
        )
        log("missing Kaggle credentials in environment", "ERROR")
        return 2

    username, key = unpack_credentials(username, key)
    if not username or not key:
        annotate_error(
            "Kaggle credentials are malformed after sanitising (empty username or key)."
        )
        return 2

    log(f"username: {username}")
    log(f"key: {key[:4]}…{key[-2:]} (len={len(key)})" if len(key) > 8 else "key: <short>")
    if is_access_token(key):
        log("detected KGAT_ API token — using the access-token auth path")

    env = env_with_credentials(username, key)
    try:
        write_kaggle_credentials(username, key)
    except Exception as exc:
        log(f"could not persist Kaggle credentials: {exc}", "WARN")

    log_group("kaggle CLI version")
    run(["kaggle", "--version"], env=env, timeout=120)
    log_group_end()

    slug = f"{username}/{KERNEL_SLUG}"
    code_path = KERNEL_DIR / CODE_FILE
    if not code_path.is_file():
        annotate_error(f"kernel code file not found: {code_path}")
        return 2

    try:
        metadata = build_metadata(username)
        write_metadata(metadata)
    except Exception as exc:
        annotate_error(f"could not build kernel-metadata.json: {exc}")
        log(f"metadata generation failed: {exc}", "ERROR")
        return 2

    # ---------------------------- push ---------------------------- #
    log_group(f"pushing kernel {slug}")
    push = run(
        ["kaggle", "kernels", "push", "-p", str(KERNEL_DIR)],
        env=env,
        timeout=600,
    )
    log_group_end()
    if push.returncode != 0:
        hint = (push.stderr or push.stdout or "").strip()
        annotate_error(f"kaggle kernels push failed (exit {push.returncode}): {hint[:600]}")
        if "401" in hint or "Unauthorized" in hint:
            log("hint: re-check KAGGLE_KEY (Kaggle → Settings → API → Create New Token)", "ERROR")
        return 2
    log("kernel push accepted — Kaggle is provisioning a T4.")

    # --------------------------- polling --------------------------- #
    deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
    last_status: Optional[str] = None
    status_history: List[Dict[str, str]] = []
    attempt = 0
    final_status: Optional[str] = None

    while True:
        attempt += 1
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            log(
                f"polling budget of {POLL_TIMEOUT_SECONDS // 60} min exhausted "
                f"after {attempt - 1} checks (last status: {last_status})",
                "ERROR",
            )
            final_status = (
                last_status if last_status in TERMINAL_OK | TERMINAL_FAIL else "timeout"
            )
            break

        status = normalize_status(get_status(slug, env))

        if status and status != last_status:
            status_history.append({"at": ts(), "status": status})
            log(f"★ status transition: {last_status or '(none)'} → {status}")
            last_status = status

        if status in TERMINAL_OK:
            final_status = status
            log("kernel reported COMPLETE")
            break
        if status in {s.lower() for s in TERMINAL_FAIL} | {"cancelacknowledged"}:
            final_status = status
            log(f"kernel ended with failure status: {status}", "ERROR")
            break

        log(
            f"still {status or 'unknown'} · {int(remaining)}s of budget left · "
            f"sleeping {POLL_INTERVAL_SECONDS}s"
        )
        time.sleep(min(POLL_INTERVAL_SECONDS, max(1, int(remaining))))

    log(f"status timeline: {json.dumps(status_history)}")

    # -------------------------- download -------------------------- #
    log_group("downloading kernel outputs")
    downloaded = download_outputs(slug, env, OUTPUT_DIR)
    log_group_end()

    if not downloaded:
        # A failed kernel still has a log; try a log-only probe so CI is debuggable.
        log("retrying output fetch once before giving up", "WARN")
        time.sleep(10)
        downloaded = download_outputs(slug, env, OUTPUT_DIR)

    flatten_outputs(OUTPUT_DIR)
    # The authoritative log comes from the CLI itself; the file inside the kernel
    # output is a fallback (it only exists if the script got far enough to write).
    got_log = fetch_kernel_log_via_cli(slug, env)
    file_log = collect_build_log(OUTPUT_DIR)
    summary = report_artifacts(OUTPUT_DIR)

    if not (got_log or file_log):
        with open(BUILD_LOG, "a", encoding="utf-8") as handle:
            handle.write(
                f"[{ts()}] no kernel log could be retrieved "
                f"(final status: {final_status})\n"
            )

    missing_required = [n for n in REQUIRED_ARTIFACTS if n not in summary["downloaded"]]
    success = final_status == "complete" and not missing_required
    if missing_required:
        log(f"required artifacts absent: {', '.join(missing_required)}", "ERROR")

    if summary["oversized"]:
        log(
            "these artifacts exceed the 5 MB commit budget and will stay untracked: "
            f"{', '.join(summary['oversized'])}",
            "WARN",
        )
    if summary["missing"]:
        log(f"missing artifacts: {', '.join(summary['missing'])}", "WARN")

    if success:
        primary = REQUIRED_ARTIFACTS[0]
        log(f"✔ artifacts for {KERNEL_SLUG} are ready in ./outputs")
        log(f"   {primary}: {summary['downloaded'][primary] / 1024:.0f} KB")
        return 0

    annotate_error(
        f"character forge did not produce a complete result "
        f"(status={final_status}, downloaded={downloaded}). See outputs/build.log."
    )
    log("✘ job failed — inspect outputs/build.log", "ERROR")
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log("interrupted", "WARN")
        sys.exit(130)
    except Exception as exc:  # last-resort guard: always explain the crash
        import traceback

        traceback.print_exc()
        annotate_error(f"orchestrator crashed: {type(exc).__name__}: {exc}")
        sys.exit(1)
