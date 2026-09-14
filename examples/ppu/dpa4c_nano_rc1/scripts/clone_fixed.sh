#!/usr/bin/env bash
set -Eeuo pipefail

REPOSITORY=""
REF=""
CHECKOUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --repository) REPOSITORY="$2"; shift 2 ;;
    --ref) REF="$2"; shift 2 ;;
    --checkout) CHECKOUT="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ -z "$REPOSITORY" || ! "$REF" =~ ^[0-9a-f]{40}$ || -z "$CHECKOUT" ]]; then
  echo "usage: $0 --repository URL --ref 40_HEX_COMMIT --checkout NEW_DIR" >&2
  exit 2
fi
test ! -e "$CHECKOUT"
git clone --no-checkout "$REPOSITORY" "$CHECKOUT"
git -C "$CHECKOUT" checkout --detach "$REF"
test "$(git -C "$CHECKOUT" rev-parse HEAD)" = "$REF"
test -z "$(git -C "$CHECKOUT" status --porcelain=v1)"
printf 'CLONE_FIXED=PASS\nrepository=%s\ncommit=%s\ncheckout=%s\n' "$REPOSITORY" "$REF" "$CHECKOUT"
