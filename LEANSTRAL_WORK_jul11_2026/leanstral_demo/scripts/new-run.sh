#!/usr/bin/env bash
# Start a fresh working copy of a problem for a test or demo attempt.
# Usage: scripts/new-run.sh {B|A|Bfalse|Afalse}
# Prints the path of the new working file under runs/ (edit that, never the template).
set -euo pipefail
CDPATH= cd "$(dirname "$0")/.." >/dev/null

case "${1:-}" in
  B)      src=problems/B_RevComp.lean ;;
  A)      src=problems/A_Interval.lean ;;
  Bfalse) src=problems/B_RevComp_FALSE.lean ;;
  Afalse) src=problems/A_Interval_FALSE.lean ;;
  *) echo "usage: $0 {B|A|Bfalse|Afalse}" >&2; exit 1 ;;
esac

ts=$(date +%Y%m%d-%H%M%S)
dst="runs/${1}_${ts}.lean"
cp "$src" "$dst"
echo "$dst"
