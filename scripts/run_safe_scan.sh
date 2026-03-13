#!/bin/zsh
set -euo pipefail

ROOT="/Volumes/new life /Websites/AI Hedge Fund"

cd "$ROOT"

export HOME="$ROOT"
export CREWAI_STORAGE_DIR="$ROOT/.crewai_storage"
export CREWAI_TRACING_ENABLED=false
export PYTHONPATH=src

exec "$ROOT/.venv311/bin/python" -m ai_hedge_fund.safe_scan \
  --auto-discover \
  --discovery-window-days 7 \
  --discovery-min-score 2.5 \
  --discovery-retry-attempts 2 \
  --execution-mode manual \
  --notify-on accepted
