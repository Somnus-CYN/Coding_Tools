#!/usr/bin/env bash
# Simple wrapper around onnx2ncnn and ncnnoptimize to generate NCNN artifacts.
# Usage: ./convert_onnx_to_ncnn.sh ../python/config.yaml

set -euo pipefail

CFG=${1:-../python/config.yaml}
ONNX_PATH=$(python - <<'PY'
import yaml, sys
from pathlib import Path
cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
print(cfg['paths']['onnx'])
PY
)

PARAM_OUT=${2:-../release/models/yolo11n.param}
BIN_OUT=${3:-../release/models/yolo11n.bin}

mkdir -p "$(dirname "$PARAM_OUT")"

if ! command -v onnx2ncnn >/dev/null; then
  echo "onnx2ncnn not found. Please install NCNN tools and ensure onnx2ncnn is on PATH." >&2
  exit 1
fi

echo "Converting $ONNX_PATH -> $PARAM_OUT / $BIN_OUT"
onnx2ncnn "$ONNX_PATH" "$PARAM_OUT" "$BIN_OUT"

if command -v ncnnoptimize >/dev/null; then
  echo "Running ncnnoptimize (FP16 disabled by default for mobile accuracy)."
  TMP_PARAM=${PARAM_OUT}.tmp
  TMP_BIN=${BIN_OUT}.tmp
  ncnnoptimize "$PARAM_OUT" "$BIN_OUT" "$TMP_PARAM" "$TMP_BIN" 0
  mv "$TMP_PARAM" "$PARAM_OUT"
  mv "$TMP_BIN" "$BIN_OUT"
else
  echo "ncnnoptimize not found, skipping optimization."
fi

echo "Done. Generated files are under $(dirname "$PARAM_OUT")."
