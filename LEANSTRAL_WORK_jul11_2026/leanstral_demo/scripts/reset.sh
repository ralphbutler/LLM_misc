#!/usr/bin/env bash
# Clear throwaway working copies. Templates in problems/ are never touched.
# Usage: scripts/reset.sh
set -euo pipefail
CDPATH= cd "$(dirname "$0")/.." >/dev/null
rm -f runs/*.lean
echo "cleared runs/ (templates in problems/ untouched)"
