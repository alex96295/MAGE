#!/usr/bin/env bash
set -e

# Output folder
OUTDIR="assets/language"
mkdir -p "$OUTDIR"

# URLs (direct HTTP links, not chrome-extension)
SV_URL="https://rfsoc.mit.edu/6S965/_static/F24/documentation/1800-2017.pdf"
GOTCHAS_URL="https://picture.iczhiku.com/resource/eetop/wyKErzSAliSTINNx.pdf"

# Filenames
SV_OUT="$OUTDIR/SystemVerilog-1800-2017.pdf"
GOTCHAS_OUT="$OUTDIR/101-SystemVerilog-Gotchas.pdf"

echo "Downloading IEEE 1800-2017 SystemVerilog manual..."
wget -O "$SV_OUT" "$SV_URL"

echo "Downloading 101 SystemVerilog Gotchas book..."
wget -O "$GOTCHAS_OUT" "$GOTCHAS_URL"

echo "All files downloaded to: $OUTDIR"
ls -lh "$OUTDIR"
