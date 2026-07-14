#!/usr/bin/env bash
# Fair local Pass@N on a target: N INDEPENDENT fresh attempts at high temperature,
# a few feedback rounds each. SOLVED if ANY attempt compiles (stop at first success).
# This is the benchmark's own methodology (test-time scaling via sampling diversity),
# not a hint. Usage: scripts/best_of_n.sh {A|B|Afalse|Bfalse} [N] [rounds] [temp]
set -uo pipefail
CDPATH= cd "$(dirname "$0")/.." >/dev/null

target="${1:?usage: best_of_n.sh <target> [N] [rounds] [temp]}"
N="${2:-8}"; rounds="${3:-4}"; temp="${4:-0.8}"

echo "===== Pass@${N} on ${target}: temp=${temp}, <=${rounds} rounds/attempt, stop on first PASS ====="
for i in $(seq 1 "$N"); do
  f=$(scripts/new-run.sh "$target")
  echo ""; echo "########## ATTEMPT ${i}/${N}  ->  ${f} ##########"
  python3 scripts/prove.py "$f" "$rounds" "$temp" 2>&1 | tee "runs/${target}_bo${i}.log"
  if grep -q "SOLVED in" "runs/${target}_bo${i}.log"; then
    echo ""; echo "@@@@@ Pass@${N} SUCCESS on attempt ${i}: ${f} @@@@@"
    echo "$f" > "runs/${target}_bestofN_WINNER.txt"
    exit 0
  fi
done
echo ""; echo "@@@@@ Pass@${N}: NO attempt solved ${target} in ${N} tries @@@@@"
exit 1
