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

# 2. Inject owner checks to skip git operations for rodrigo-barraza repos
python3 -c '
import sys

with open(sys.argv[1], "r") as f:
    code = f.read()

# Replace the first instance (line 1056)
target1 = """    if $_lib_has_changes; then
      (cd "$lib_dir" && git add dist/ && git commit -m "build: rebuild dist/" --no-verify 2>&1) | sed '\''s/^/  /'\'' || true
      (cd "$lib_dir" && git push origin HEAD 2>&1) | sed '\''s/^/  /'\'' || true
      ok "${lib_id}: dist/ rebuilt and pushed"
    else"""

replacement1 = """    if $_lib_has_changes; then
      _lib_remote=$(cd "$lib_dir" && git remote get-url origin 2>/dev/null || echo "")
      if [[ "$_lib_remote" == *"rodrigo-barraza"* ]]; then
        info "${lib_id}: dist/ has changes but repository belongs to rodrigo-barraza. Skipping commit/push to prevent permission errors."
      else
        (cd "$lib_dir" && git add dist/ && git commit -m "build: rebuild dist/" --no-verify 2>&1) | sed '\''s/^/  /'\'' || true
        (cd "$lib_dir" && git push origin HEAD 2>&1) | sed '\''s/^/  /'\'' || true
        ok "${lib_id}: dist/ rebuilt and pushed"
      fi
    else"""

# Replace the second instance (line 1091)
target2 = """        if ! (cd "$lib_dir" && git diff --quiet dist/ 2>/dev/null) || \\
           [ -n "$(cd "$lib_dir" && git ls-files --others --exclude-standard dist/ 2>/dev/null)" ]; then
          (cd "$lib_dir" && git add dist/ && git commit -m "build: rebuild dist/ with updated deps" --no-verify 2>&1) | sed '\''s/^/  /'\'' || true
          (cd "$lib_dir" && git push origin HEAD 2>&1) | sed '\''s/^/  /'\'' || true
          ok "${lib_id}: rebuilt and pushed with fresh deps"
        else"""

replacement2 = """        if ! (cd "$lib_dir" && git diff --quiet dist/ 2>/dev/null) || \\
           [ -n "$(cd "$lib_dir" && git ls-files --others --exclude-standard dist/ 2>/dev/null)" ]; then
          _lib_remote=$(cd "$lib_dir" && git remote get-url origin 2>/dev/null || echo "")
          if [[ "$_lib_remote" == *"rodrigo-barraza"* ]]; then
            info "${lib_id}: dist/ has changes after dep refresh but repository belongs to rodrigo-barraza. Skipping commit/push."
          else
            (cd "$lib_dir" && git add dist/ && git commit -m "build: rebuild dist/ with updated deps" --no-verify 2>&1) | sed '\''s/^/  /'\'' || true
            (cd "$lib_dir" && git push origin HEAD 2>&1) | sed '\''s/^/  /'\'' || true
            ok "${lib_id}: rebuilt and pushed with fresh deps"
          fi
        else"""

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
