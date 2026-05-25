#!/bin/bash
# deploy_changed.sh — Run multi-service deployment with permission bypasses for Rod's repos.
# Stored on LazyCat420's side to avoid editing shared/Rod deploy-kit files.

TRADING_SERVICE_DIR="$(cd "$(dirname "$0")" && pwd)"
DEPLOY_KIT_DIR="$(cd "${TRADING_SERVICE_DIR}/../deploy-kit" && pwd)"

# Define temporary modified script inside deploy-kit (cleaned up on exit)
TEMP_DEPLOY_SCRIPT="${DEPLOY_KIT_DIR}/deploy-all-custom.sh"

cleanup() {
  rm -f "$TEMP_DEPLOY_SCRIPT"
}
trap cleanup EXIT

# 1. Read the original deploy-all.sh
if [ ! -f "${DEPLOY_KIT_DIR}/deploy-all.sh" ]; then
  echo "Error: deploy-all.sh not found in deploy-kit" >&2
  exit 1
fi

# 2. Inject owner checks to skip building/syncing for rodrigo-barraza repos entirely
python3 -c '
import sys

with open(sys.argv[1], "r") as f:
    code = f.read()

# Target 1: Skip Rod libraries in the initial build/sync loop
target1 = """  for lib_id in "${LIBRARY_IDS[@]}"; do
    lib_dir="${ROOT_DIR}/${lib_id}"
    if [ ! -d "$lib_dir" ]; then"""

replacement1 = """  for lib_id in "${LIBRARY_IDS[@]}"; do
    lib_dir="${ROOT_DIR}/${lib_id}"
    _lib_remote=$(cd "$lib_dir" && git remote get-url origin 2>/dev/null || echo "")
    if [[ "$_lib_remote" == *"rodrigo-barraza"* ]]; then
      info "${lib_id}: Repository belongs to rodrigo-barraza. Skipping sync/build entirely."
      continue
    fi
    if [ ! -d "$lib_dir" ]; then"""

# Target 2: Skip Rod libraries in the inter-library dependency refresh loop
target2 = """  for lib_id in "${LIBRARY_IDS[@]}"; do
    lib_dir="${ROOT_DIR}/${lib_id}"
    [ -d "$lib_dir" ] || continue

    # Check if this library has any git-based deps on other libraries"""

replacement2 = """  for lib_id in "${LIBRARY_IDS[@]}"; do
    lib_dir="${ROOT_DIR}/${lib_id}"
    [ -d "$lib_dir" ] || continue
    _lib_remote=$(cd "$lib_dir" && git remote get-url origin 2>/dev/null || echo "")
    if [[ "$_lib_remote" == *"rodrigo-barraza"* ]]; then
      continue
    fi

    # Check if this library has any git-based deps on other libraries"""

if target1 not in code:
    print("Warning: target1 not found in deploy-all.sh")
if target2 not in code:
    print("Warning: target2 not found in deploy-all.sh")

code = code.replace(target1, replacement1)
code = code.replace(target2, replacement2)

with open(sys.argv[2], "w") as f:
    f.write(code)
' "${DEPLOY_KIT_DIR}/deploy-all.sh" "$TEMP_DEPLOY_SCRIPT"

chmod +x "$TEMP_DEPLOY_SCRIPT"

# 3. Execute the custom deploy script with all arguments forwarded
cd "$DEPLOY_KIT_DIR"
bash "./deploy-all-custom.sh" "$@"
