#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/../../../.." && pwd)"
echo "NOTICE: forwarding legacy entrypoint to contest/contest.sh" >&2
exec "$REPO_ROOT/contest/contest.sh" "$@"
