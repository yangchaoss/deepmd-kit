#!/usr/bin/env bash
set -euo pipefail

RC1_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
exec "$RC1_DIR/scripts/contestant.sh" all \
  --model "${DPA4C_RC1_MODEL:?set DPA4C_RC1_MODEL}" \
  --structure "${DPA4C_RC1_STRUCTURE:?set DPA4C_RC1_STRUCTURE}" \
  --work-root "${DPA4C_RC1_WORK_ROOT:?set DPA4C_RC1_WORK_ROOT outside Git}"
