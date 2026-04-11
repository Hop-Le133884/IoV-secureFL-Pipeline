#!/usr/bin/env bash
# Removes generated/runtime artefacts, leaving source code and raw data intact.
# Use before a full re-run from: data_splits_gen.sh → jobs_gen.sh → start_fleet.sh
#
# Usage:
#   ./cleansing_job.sh             # dry-run (shows what would be deleted)
#   ./cleansing_job.sh --confirm   # actually delete

set -euo pipefail

DRY_RUN=true
if [[ "${1:-}" == "--confirm" ]]; then
    DRY_RUN=false
fi

# Targets
TARGETS=(
    # Python environment
    ".venv"
    "uv.lock"
    ".python-version"
    "__pycache__"

    # Generated data splits
    "data/IoV"

    # Generated job config
    "jobs/iov_double_rf_5_sites"

    # Entire NVFlare workspace (server + clients + admin provisioning)
    "workspace"

    # Old simulate-mode workspace
    "workspace_iov_double_rf"

    # Extracted model outputs
    "models"

    # Server nohup log
    "server.log"
)

# ── Run ───────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
if $DRY_RUN; then
    echo "=== DRY RUN (pass --confirm to actually delete) ==="
else
    echo "=== DELETING ==="
fi
echo ""

DELETED=0
SKIPPED=0

for target in "${TARGETS[@]}"; do
    full_path="$SCRIPT_DIR/$target"
    if compgen -G "$full_path" > /dev/null 2>&1; then
        if $DRY_RUN; then
            echo "  [would delete]  $target"
        else
            rm -rf $full_path
            echo "  [deleted]       $target"
        fi
        DELETED=$((DELETED + 1))
    else
        echo "  [not found]     $target"
        SKIPPED=$((SKIPPED + 1))
    fi
done

# NVFlare temp storage (/tmp/nvflare)
for tmp_dir in /tmp/nvflare/jobs-storage /tmp/nvflare/snapshot-storage; do
    if [[ -d "$tmp_dir" && -n "$(ls -A "$tmp_dir" 2>/dev/null)" ]]; then
        if $DRY_RUN; then
            echo "  [would delete]  $tmp_dir/*"
        else
            rm -rf "${tmp_dir:?}"/*
            echo "  [deleted]       $tmp_dir/*"
        fi
        DELETED=$((DELETED + 1))
    else
        echo "  [not found]     $tmp_dir/*"
        SKIPPED=$((SKIPPED + 1))
    fi
done

echo ""
if $DRY_RUN; then
    echo "Dry run complete. $DELETED item(s) would be removed ($SKIPPED not found)."
    echo "Run with --confirm to proceed: ./cleansing_job.sh --confirm"
else
    echo "Done. $DELETED item(s) removed ($SKIPPED not found)."
fi
echo ""
