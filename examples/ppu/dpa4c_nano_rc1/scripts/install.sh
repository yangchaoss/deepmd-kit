#!/usr/bin/env bash
set -Eeuo pipefail
RC1_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
PYTHON="${DPA4C_CANDIDATE_PYTHON:?set DPA4C_CANDIDATE_PYTHON}"
BUILD_ROOT="${DPA4C_BUILD_ROOT:?set DPA4C_BUILD_ROOT outside the Git checkout}"
SITE="$($PYTHON -s -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
mkdir -p "$SITE/runner"
cp "$RC1_DIR/runner/__init__.py" "$RC1_DIR/runner/runtime_adapter.py" "$SITE/runner/"
env -u PYTHONPATH -u PYTHONHOME -u CONDA_PREFIX -u CONDA_DEFAULT_ENV \
  PYTHONNOUSERSITE=1 "$PYTHON" -s "$RC1_DIR/tools/install_runtime.py" \
  --build-json "$BUILD_ROOT/build.json" --output "$BUILD_ROOT/install.json"
