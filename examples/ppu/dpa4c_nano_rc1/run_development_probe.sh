#!/usr/bin/env bash
set -euo pipefail

RC1_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(git -C "$RC1_DIR" rev-parse --show-toplevel)
WORK_ROOT=${DPA4C_RC1_WORK_ROOT:?set DPA4C_RC1_WORK_ROOT outside the Git checkout}
MODEL=${DPA4C_RC1_MODEL:?set DPA4C_RC1_MODEL to the verified DPA4C Nano checkpoint}
STRUCTURE=${DPA4C_RC1_STRUCTURE:?set DPA4C_RC1_STRUCTURE to the verified 1024-atom structure}
BUILD_DIR="$WORK_ROOT/build"
EVIDENCE_DIR="$WORK_ROOT/evidence"

case "$WORK_ROOT/" in
  "$REPO_ROOT"/*) echo "DPA4C_RC1_WORK_ROOT must be outside $REPO_ROOT" >&2; exit 2 ;;
esac

mkdir -p "$BUILD_DIR" "$EVIDENCE_DIR/logs"
export PYTHONPATH="$RC1_DIR:$REPO_ROOT"
export DP_COMPILE_INFER=0
export DP_TF32_INFER=0
export DP_AMP_INFER=0
export DPA4C_CONTEST_FORMAL=0

run_stage() {
  local stage_name=$1
  shift
  set +e
  "$@" 2>&1 | tee "$EVIDENCE_DIR/logs/${stage_name}.log"
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "stage=$stage_name status=FAIL rc=$rc" >&2
    exit "$rc"
  fi
}

run_stage preflight python "$RC1_DIR/tools/preflight.py" \
  --model "$MODEL" --structure "$STRUCTURE" --output "$EVIDENCE_DIR/preflight.json"
run_stage build python "$RC1_DIR/tools/build_runtime.py" \
  --source "$RC1_DIR/candidate/dpa4c_contest_ops.cu" \
  --build "$BUILD_DIR" --result "$EVIDENCE_DIR/build.json"

SO=$(python - "$EVIDENCE_DIR/build.json" <<'PY'
import json
import sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
objects = data.get("shared_objects", [])
if len(objects) != 1:
    raise SystemExit(f"expected one shared object, got {len(objects)}")
print(objects[0]["path"])
PY
)

run_stage dispatch python "$RC1_DIR/tools/runtime_probe.py" \
  --shared-object "$SO" --output "$EVIDENCE_DIR/dispatch.json"
run_stage efs python "$RC1_DIR/runner/run_efs_demo.py" \
  --model "$MODEL" --structure "$STRUCTURE" --shared-object "$SO" \
  --output "$EVIDENCE_DIR/efs-demo.json" --warmup 1 --measure 1

echo "PASS: development probe completed; evidence=$EVIDENCE_DIR"
