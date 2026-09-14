#!/usr/bin/env bash
set -Eeuo pipefail
RC1_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
PYTHON="${DPA4C_CANDIDATE_PYTHON:?set DPA4C_CANDIDATE_PYTHON}"
BUILD_ROOT="${DPA4C_BUILD_ROOT:?set DPA4C_BUILD_ROOT outside the Git checkout}"
mkdir -p "$BUILD_ROOT"
env -u PYTHONPATH -u PYTHONHOME -u CONDA_PREFIX -u CONDA_DEFAULT_ENV \
  PYTHONNOUSERSITE=1 PPU_SDK="${PPU_SDK:-/usr/local/PPU_SDK}" \
  CUDA_HOME="${CUDA_HOME:-/usr/local/PPU_SDK/CUDA_SDK}" \
  "$PYTHON" -s "$RC1_DIR/tools/preflight.py" \
  --model "${DPA4C_MODEL:?set DPA4C_MODEL}" \
  --structure "${DPA4C_STRUCTURE:?set DPA4C_STRUCTURE}" \
  --output "$BUILD_ROOT/preflight.json"
env -u PYTHONPATH -u PYTHONHOME -u CONDA_PREFIX -u CONDA_DEFAULT_ENV \
  PYTHONNOUSERSITE=1 PPU_SDK="${PPU_SDK:-/usr/local/PPU_SDK}" \
  CUDA_HOME="${CUDA_HOME:-/usr/local/PPU_SDK/CUDA_SDK}" \
  "$PYTHON" -s "$RC1_DIR/tools/build_runtime.py" \
  --source "$RC1_DIR/candidate/dpa4c_contest_ops.cu" \
  --build "$BUILD_ROOT/extension" --result "$BUILD_ROOT/build.json"
