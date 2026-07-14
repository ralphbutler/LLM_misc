#!/usr/bin/env bash
# Type-check a Lean file and print a clear verdict.
# Usage: scripts/check.sh runs/B_<timestamp>.lean
#   PASS       = fully verified, no sorry, no errors
#   INCOMPLETE = still has a `sorry`
#   FAIL       = compiler errors (e.g. a bogus proof was rejected)
set -uo pipefail
CDPATH= cd "$(dirname "$0")/.." >/dev/null

f="${1:-}"
[ -f "$f" ] || { echo "no such file: $f" >&2; exit 1; }

out=$(lake env lean "$f" 2>&1)
echo "$out"
echo "--------------------------------------------------"
if echo "$out" | grep -q "error:"; then
  echo "VERDICT: FAIL  (compiler rejected it)"; exit 2
elif echo "$out" | grep -qi "sorry"; then
  echo "VERDICT: INCOMPLETE  (sorry still present)"; exit 3
else
  echo "VERDICT: PASS  (fully verified — kernel-checked, no sorry)"; exit 0
fi
