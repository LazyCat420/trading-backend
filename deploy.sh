#!/bin/bash
# ============================================================
# Trading Cycle Backend — Build & Deploy to Synology NAS
#
# Thin wrapper — all logic lives in ../deploy-kit/lib.sh
#
# Usage:
#   npm run deploy              # full deploy
#   npm run deploy -- --dry-run # validate without deploying
#   npm run deploy -- --skip-pull
#   npm run deploy -- --no-cache
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE_NAME="trading-cycle-backend"
DISPLAY_NAME="⚙️ Trading Cycle Backend"
SKIP_ENV_DEPLOY=true

PRE_BUILD() {
  local CENTRAL_ENV="${DEPLOY_KIT_DIR}/.env.deploy"
  if [ -f "$CENTRAL_ENV" ]; then
    set -a; source "$CENTRAL_ENV"; set +a
    info "Loaded deploy-kit/.env.deploy"
  fi
}

EXTRA_SSH_SYNC() {
  info "Syncing master .env from vault-service..."
  cat "${SCRIPT_DIR}/../vault-service/.env" | ssh "$DEPLOY_SSH_HOST" "cat > '${DEPLOY_COMPOSE_DIR}/.env'"
  ssh "$DEPLOY_SSH_HOST" "mkdir -p '${DEPLOY_COMPOSE_DIR}/logs' 2>/dev/null || sudo mkdir -p '${DEPLOY_COMPOSE_DIR}/logs'"
}

source "${SCRIPT_DIR}/../deploy-kit/lib.sh"
