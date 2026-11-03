#!/usr/bin/env bash
# Download the simulation asset archive and unpack it into space_robotics_bench/.
# Usage: ASSETS_URL=<archive url> scripts/fetch_assets.sh   (or: scripts/fetch_assets.sh <local archive>)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SRC=${1:-}
if [ -z "$SRC" ]; then
  : "${ASSETS_URL:?set ASSETS_URL to the asset archive location (see README)}"
  SRC=$(mktemp --suffix=.tar.gz)
  curl -L --fail -o "$SRC" "$ASSETS_URL"
fi
tar -xzf "$SRC" -C "$ROOT/space_robotics_bench"
test -f "$ROOT/space_robotics_bench/assets/lunar_base/usd/scene/lunar_base/lunar_base.usda"
echo "assets unpacked into $ROOT/space_robotics_bench/assets"
