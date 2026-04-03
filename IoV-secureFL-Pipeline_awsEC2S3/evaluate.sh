#!/usr/bin/env bash
# Extracts the global models from the latest completed NVFlare job and runs evaluation.
# Usage:
#   ./evaluate.sh                  → use latest job
#   ./evaluate.sh <job_id>         → use specific job ID

REPO="$(realpath "$(dirname "$0")")"
JOB_STORE="/tmp/nvflare/jobs-storage"
MODEL_DIR="${REPO}/models"

# Resolve job ID
if [[ -n "$1" ]]; then
    JOB_ID="$1"
else
    JOB_ID=$(ls -t "${JOB_STORE}" | head -1)
fi

if [[ -z "$JOB_ID" ]]; then
    echo "No completed jobs found in ${JOB_STORE}"
    exit 1
fi

WORKSPACE_ZIP="${JOB_STORE}/${JOB_ID}/workspace"
if [[ ! -f "$WORKSPACE_ZIP" ]]; then
    echo "Job workspace not found: ${WORKSPACE_ZIP}"
    exit 1
fi

echo "Job: ${JOB_ID}"
mkdir -p "${MODEL_DIR}"

# Extract both models from the job zip
python3 - <<EOF
import zipfile, sys
zip_path = "${WORKSPACE_ZIP}"
out_dir  = "${MODEL_DIR}"
models   = ["app_server/xgboost_model_inner.json", "app_server/xgboost_model_outer.json"]
with zipfile.ZipFile(zip_path) as z:
    found = z.namelist()
    for m in models:
        if m in found:
            data = z.read(m)
            dest = out_dir + "/" + m.split("/")[-1]
            open(dest, "wb").write(data)
            print(f"  Extracted: {dest}")
        else:
            print(f"  Missing:   {m}", file=sys.stderr)
            sys.exit(1)
EOF

if [[ $? -ne 0 ]]; then
    echo "Model extraction failed — re-run the FL job first."
    exit 1
fi

echo ""
source "${REPO}/.venv/bin/activate"
python3 "${REPO}/utils/model_validation.py" \
    --test_data  "${REPO}/data/processed/df_server_test.csv" \
    --workspace  "${MODEL_DIR}"
