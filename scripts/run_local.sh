#!/usr/bin/env bash
# Start the MEDSCRIPTAI-ONCO dashboard on this machine.
#
#   bash scripts/run_local.sh              # http://127.0.0.1:8000
#   bash scripts/run_local.sh 8080         # custom port
#
# Creates .venv on first run, installs the dependencies, then starts uvicorn.
set -euo pipefail

PORT="${1:-8000}"
cd "$(dirname "$0")/.."

# reuse an existing virtualenv in this folder rather than making a second one
VENV=""
for candidate in .venv venv *venv; do
  if [ -x "$candidate/bin/python" ]; then VENV="$candidate"; break; fi
done
if [ -z "$VENV" ]; then
  echo "==> creating .venv"
  python3 -m venv .venv
  VENV=".venv"
fi
echo "==> using $VENV"
# shellcheck disable=SC1090
source "$VENV/bin/activate"

echo "==> installing dependencies (first run takes a couple of minutes)"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

echo "==> starting the dashboard on http://127.0.0.1:${PORT}"
echo "    press Ctrl+C to stop"
exec python -m uvicorn api.app:app --reload --port "${PORT}"
