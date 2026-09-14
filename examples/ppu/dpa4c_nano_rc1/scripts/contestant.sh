#!/usr/bin/env bash
set -Eeuo pipefail

RC1_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
REPO_ROOT="$(git -C "$RC1_DIR" rev-parse --show-toplevel)"
COMMAND="${1:-}"
shift || true
MODEL=""
STRUCTURE=""
WORK_ROOT=""
BASE_COMMIT="14a71f13bb840c10d78465f75a00e3a31764a0fd"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --structure) STRUCTURE="$2"; shift 2 ;;
    --work-root) WORK_ROOT="$2"; shift 2 ;;
    --base-commit) BASE_COMMIT="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ -z "$COMMAND" || -z "$MODEL" || -z "$STRUCTURE" || -z "$WORK_ROOT" ]]; then
  echo "usage: $0 {bootstrap|build|check|benchmark|score|package|all} --model FILE --structure FILE --work-root DIR [--base-commit COMMIT]" >&2
  exit 2
fi
WORK_ROOT="$(mkdir -p "$WORK_ROOT" && cd "$WORK_ROOT" && pwd)"
case "$WORK_ROOT/" in "$REPO_ROOT"/*) echo "work root must be outside the Git checkout" >&2; exit 2;; esac
VENV="$WORK_ROOT/venv"
BUILD_ROOT="$WORK_ROOT/build"
RESULT_ROOT="$WORK_ROOT/results"
LOG_ROOT="$WORK_ROOT/logs"
mkdir -p "$LOG_ROOT" "$RESULT_ROOT"

export DPA4C_CANDIDATE_PYTHON="$VENV/bin/python"
export DPA4C_BUILD_ROOT="$BUILD_ROOT"
export DPA4C_RESULT_ROOT="$RESULT_ROOT"
export DPA4C_MODEL="$MODEL"
export DPA4C_STRUCTURE="$STRUCTURE"
export PPU_SDK="${PPU_SDK:-/usr/local/PPU_SDK}"
export CUDA_HOME="${CUDA_HOME:-$PPU_SDK/CUDA_SDK}"

bootstrap() {
  command -v git >/dev/null
  test -x "$CUDA_HOME/bin/nvcc"
  test -f "$MODEL"
  test -f "$STRUCTURE"
  test "$(sha256sum "$MODEL" | awk '{print $1}')" = "f894ac16adfb7f5030d4fe4e2849db6c608f9c7074badfafb4a50c9b9afed00a"
  test "$(sha256sum "$STRUCTURE" | awk '{print $1}')" = "137056e51cf63bd7dabf0508a29959218baf109da0c5e19a3765d11873c891e7"
  if [[ ! -x "$VENV/bin/python" ]]; then /opt/ac2/bin/python -m venv --system-site-packages "$VENV"; fi
  env -u PYTHONPATH -u PYTHONHOME -u CONDA_PREFIX -u CONDA_DEFAULT_ENV PYTHONNOUSERSITE=1 \
    "$VENV/bin/python" -s -c 'import deepmd, deepmd.lib, torch; assert torch.cuda.is_available(); print(deepmd.__file__); print(deepmd.lib.__path__); print(torch.cuda.get_device_name(0))'
}
build() { bootstrap; "$RC1_DIR/scripts/build.sh"; "$RC1_DIR/scripts/install.sh"; }
check() {
  test -f "$BUILD_ROOT/install.json"
  SO="$($VENV/bin/python -s -c 'import json,os; print(json.load(open(os.environ["DPA4C_BUILD_ROOT"] + "/install.json"))["shared_object"]["path"])')"
  env -u PYTHONPATH -u PYTHONHOME -u CONDA_PREFIX -u CONDA_DEFAULT_ENV PYTHONNOUSERSITE=1 \
    "$VENV/bin/python" -s "$RC1_DIR/tools/runtime_probe.py" --shared-object "$SO" --output "$RESULT_ROOT/check.json"
}
benchmark() { test -f "$BUILD_ROOT/install.json"; "$RC1_DIR/scripts/run.sh"; }
score() {
  "$VENV/bin/python" -s "$RC1_DIR/tools/score_smoke.py" --smoke "$RESULT_ROOT/smoke.json" --output "$RESULT_ROOT/result.json"
}
package() {
  "$VENV/bin/python" -s "$RC1_DIR/tools/package_submission.py" --repo "$REPO_ROOT" --base-commit "$BASE_COMMIT" \
    --result "$RESULT_ROOT/result.json" --smoke "$RESULT_ROOT/smoke.json" --output "$WORK_ROOT/submission"
}

case "$COMMAND" in
  bootstrap) bootstrap 2>&1 | tee "$LOG_ROOT/bootstrap.log" ;;
  build) build 2>&1 | tee "$LOG_ROOT/build-install.log" ;;
  check) check 2>&1 | tee "$LOG_ROOT/check.log" ;;
  benchmark) benchmark 2>&1 | tee "$LOG_ROOT/benchmark.log" ;;
  score) score 2>&1 | tee "$LOG_ROOT/score.log" ;;
  package) package 2>&1 | tee "$LOG_ROOT/package.log" ;;
  all)
    bootstrap 2>&1 | tee "$LOG_ROOT/bootstrap.log"
    { "$RC1_DIR/scripts/build.sh"; "$RC1_DIR/scripts/install.sh"; } 2>&1 | tee "$LOG_ROOT/build-install.log"
    check 2>&1 | tee "$LOG_ROOT/check.log"
    benchmark 2>&1 | tee "$LOG_ROOT/benchmark.log"
    score 2>&1 | tee "$LOG_ROOT/score.log"
    package 2>&1 | tee "$LOG_ROOT/package.log"
    echo "PASS: public contestant flow; submission=$WORK_ROOT/submission"
    ;;
  *) echo "unknown command: $COMMAND" >&2; exit 2 ;;
esac
