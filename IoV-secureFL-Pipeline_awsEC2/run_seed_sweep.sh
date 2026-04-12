#!/usr/bin/env bash
# run_seed_sweep.sh — prepare one seed's data + job config for EC2 FL run.
#
# Run ONCE before first seed:
#   bash fleet_deployment.sh    ← installs venv + packages on all clients
#   bash network_provision.sh   ← generates certs, start server + admin
#
# Then for each seed (including the first):
#   SEED=42  bash run_seed_sweep.sh
#   SEED=123 bash run_seed_sweep.sh
#   ...
#
# After each run: submit job via admin console, wait, then run evaluate.sh

set -euo pipefail

REPO="$(realpath "$(dirname "$0")")"
cd "${REPO}"

SEED="${1:-${SEED:-}}"
if [[ -z "${SEED}" ]]; then
    echo "Error: SEED is required." >&2
    echo "Usage: SEED=42 bash run_seed_sweep.sh" >&2
    echo "   or: bash run_seed_sweep.sh 42" >&2
    exit 1
fi
DP_EPSILON="${DP_EPSILON:-80}"

echo "=== SEED=${SEED} | ε=${DP_EPSILON} ==="

echo "  [1/3] Generating data splits (new train/test signature split) ..."
SEED=${SEED} bash data_split_gen.sh ./data

echo "  [2/3] Deploying new training CSVs to fleet ..."
bash deploy_data.sh

echo "  [3/3] Generating job config ..."
DP_EPSILON=${DP_EPSILON} SEED=${SEED} bash jobs_gen.sh ./data

echo ""
echo "Done. Next steps:"
echo "  1. Submit the new job via the admin console"
echo "  2. Wait for completion"
echo "  3. Run: bash evaluate.sh <job_id>"

