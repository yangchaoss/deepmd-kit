#!/usr/bin/env bash
set -Eeuo pipefail

# Resolve the entrypoint before deriving the repository root.  The runtime
# image exposes this file through /usr/local/bin/dpa4c-contestant-flow, so
# dirname "$0" is not sufficient when the entrypoint is a symlink.
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do
    SOURCE_DIR="$(CDPATH= cd -P -- "$(dirname -- "$SOURCE")" && pwd)"
    SOURCE="$(readlink -- "$SOURCE")"
    case "$SOURCE" in
        /*) ;;
        *) SOURCE="$SOURCE_DIR/$SOURCE" ;;
    esac
done
ROOT="$(CDPATH= cd -P -- "$(dirname -- "$SOURCE")/.." && pwd)"
exec python3 "$ROOT/contest/scripts/flow.py" "$@"
