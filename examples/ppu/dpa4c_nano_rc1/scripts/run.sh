#!/usr/bin/env bash
set -Eeuo pipefail
RC1_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
PYTHON="${DPA4C_CANDIDATE_PYTHON:?set DPA4C_CANDIDATE_PYTHON}"
BUILD_ROOT="${DPA4C_BUILD_ROOT:?set DPA4C_BUILD_ROOT outside the Git checkout}"
RESULT_ROOT="${DPA4C_RESULT_ROOT:?set DPA4C_RESULT_ROOT outside the Git checkout}"
MODEL="${DPA4C_MODEL:?set DPA4C_MODEL}"
STRUCTURE="${DPA4C_STRUCTURE:?set DPA4C_STRUCTURE}"
SO="$($PYTHON -s -c 'import json,os; print(json.load(open(os.environ["DPA4C_BUILD_ROOT"] + "/install.json"))["shared_object"]["path"])')"
mkdir -p "$RESULT_ROOT"
env -u PYTHONPATH -u PYTHONHOME -u CONDA_PREFIX -u CONDA_DEFAULT_ENV \
  PYTHONNOUSERSITE=1 DP_COMPILE_INFER=0 DP_TF32_INFER=0 DP_AMP_INFER=0 \
  "$PYTHON" -s "$RC1_DIR/runner/run_efs_demo.py" \
  --model "$MODEL" --structure "$STRUCTURE" --shared-object "$SO" \
  --tolerance-json "$RC1_DIR/protocol/public-smoke-tolerance.json" \
  --output "$RESULT_ROOT/smoke.json" --warmup 1 --measure 2
