#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
MODEL=${DPA4C_CONTEST_MODEL:?set DPA4C_CONTEST_MODEL}
STRUCTURE=${DPA4C_CONTEST_STRUCTURE:?set DPA4C_CONTEST_STRUCTURE}
OUTPUT=${DPA4C_TOLERANCE_OUTPUT:?set DPA4C_TOLERANCE_OUTPUT}
SOURCE_COMMIT=${DPA4C_SOURCE_COMMIT:?set DPA4C_SOURCE_COMMIT}
SOURCE_TREE=${DPA4C_SOURCE_TREE:?set DPA4C_SOURCE_TREE}

export PYTHONPATH="${DPA4C_DEEPMD_SOURCE:?set DPA4C_DEEPMD_SOURCE}${PYTHONPATH:+:$PYTHONPATH}"
export PPU_SDK=${PPU_SDK:?set PPU_SDK}
export CUDA_HOME=${CUDA_HOME:?set CUDA_HOME}
export CUDA_VISIBLE_DEVICES=0
export DP_COMPILE_INFER=0
export DP_TF32_INFER=0
export DP_AMP_INFER=0

mkdir -p "$OUTPUT"
exec python "$ROOT/tolerance_freeze.py" \
  --model "$MODEL" \
  --structure "$STRUCTURE" \
  --output "$OUTPUT" \
  --source-commit "$SOURCE_COMMIT" \
  --source-tree "$SOURCE_TREE" \
  2>&1 | tee "$OUTPUT/tolerance-freeze.log"
