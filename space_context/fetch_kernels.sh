#!/usr/bin/env bash
# Download the SPICE kernels of space_context/kernels.tm from the NAIF generic kernels into space_context/kernels/
# and check them against space_context/kernels.md5 (the files the Earth-visibility profile was computed with).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
NAIF=https://naif.jpl.nasa.gov/pub/naif/generic_kernels
mkdir -p "$HERE/kernels"
for f in lsk/naif0012.tls pck/pck00011.tpc spk/planets/de440s.bsp pck/moon_pa_de440_200625.bpc fk/satellites/moon_de440_250416.tf; do
  name=$(basename "$f")
  [ -s "$HERE/kernels/$name" ] || curl -fsSL -o "$HERE/kernels/$name" "$NAIF/$f"
done
(cd "$HERE/kernels" && md5sum -c "$HERE/kernels.md5")
