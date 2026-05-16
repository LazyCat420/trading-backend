#!/bin/bash
# ============================================================
# Trading Cycle Backend — Entrypoint
# ============================================================
# Runs the autonomous trading cycle scheduler.
# Override behavior with environment variables:
#   CYCLE_ONCE=true     — run one cycle and exit
#   CYCLE_TICKERS=AAPL  — override ticker list
#   CYCLE_INTERVAL=30   — minutes between cycles
# ============================================================

set -e

echo "[cycle-backend] Python: $(/opt/venv/bin/python --version 2>&1 || echo 'NOT FOUND')"
echo "[cycle-backend] Starting trading cycle backend..."

ARGS=""

if [ "${CYCLE_ONCE}" = "true" ]; then
    ARGS="$ARGS --once"
fi

if [ -n "${CYCLE_TICKERS}" ]; then
    ARGS="$ARGS --tickers ${CYCLE_TICKERS}"
fi

if [ -n "${CYCLE_INTERVAL}" ]; then
    ARGS="$ARGS --interval ${CYCLE_INTERVAL}"
fi

exec /opt/venv/bin/python cycle_main.py $ARGS
