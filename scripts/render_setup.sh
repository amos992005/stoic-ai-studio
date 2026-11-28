#!/usr/bin/env bash
#
# Install the Remotion project's dependencies and the browser the headless
# render needs, so the render step itself touches no network.
#
# Remotion renders with "Chrome Headless Shell", a ~100 MB download from
# remotion.media that is NOT part of node_modules. Fetching it here (as part of
# setup) keeps the render reproducible and makes a browser download failure
# show up as a setup error rather than halfway through 450 frames.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTION_DIR="$REPO_ROOT/remotion"

echo "==> node $(node --version) / npm $(npm --version)"
echo "==> installing remotion dependencies"
cd "$REMOTION_DIR"
# `npm ci` needs a lockfile; fall back to install the first time round.
if [ -f package-lock.json ]; then
  npm ci --no-audit --no-fund
else
  npm install --no-audit --no-fund
fi

echo "==> downloading Chrome Headless Shell"
# The render is offline-safe once this succeeds; if it fails, every later
# render will try to download the browser itself and fail far less clearly.
npx remotion browser ensure --log=verbose 2>&1 | tail -5 || true
# Retried: remotion.media is a CDN and occasionally drops the first TLS
# handshake, which would otherwise fail a whole render job.
for attempt in 1 2 3; do
  if npx remotion browser ensure; then
    echo "==> browser ready"
    ls -la ~/.cache/remotion 2>/dev/null || true
    break
  fi
  echo "==> browser download failed (attempt $attempt/3), retrying in 10s"
  sleep 10
  if [ "$attempt" = "3" ]; then
    echo "error: could not download Chrome Headless Shell" >&2
    exit 1
  fi
done

echo "==> type checking"
npx tsc --noEmit

echo "==> setup complete"
