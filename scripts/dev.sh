#!/usr/bin/env bash
set -euo pipefail
# Give each background service its own process group for complete cleanup.
set -m
matchday_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$matchday_root"

if [ ! -x .venv/bin/python ]; then
  echo "Set up .venv using README.md first."
  exit 1
fi
if [ ! -d web/node_modules ]; then
  echo "Run npm --prefix web ci first."
  exit 1
fi
if [ ! -f models/current.json ]; then
  echo "No model found. Run .venv/bin/python -m premier_league_predictor train first."
  exit 1
fi

api_pid=""
ui_pid=""
cleanup() {
  trap - EXIT INT TERM
  if [ -n "$ui_pid" ]; then
    kill -TERM -- "-$ui_pid" 2>/dev/null || true
  fi
  if [ -n "$api_pid" ]; then
    kill -TERM -- "-$api_pid" 2>/dev/null || true
  fi
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

.venv/bin/python -m premier_league_predictor.api &
api_pid=$!
(cd web && exec npm run dev -- --port "${MATCHDAY_WEB_PORT:-3000}") &
ui_pid=$!
while kill -0 "$api_pid" 2>/dev/null && kill -0 "$ui_pid" 2>/dev/null; do
  sleep 1
done
# A service exited unexpectedly. The exit trap shuts down its sibling.
exit 1
