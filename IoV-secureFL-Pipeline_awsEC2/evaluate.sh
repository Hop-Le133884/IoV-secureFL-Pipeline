#!/usr/bin/env bash
# Extracts the global models from the latest completed NVFlare job and runs evaluation.
# Usage:
#   ./evaluate.sh                  → use latest job
#   ./evaluate.sh <job_id>         → use specific job ID

REPO="$(realpath "$(dirname "$0")")"
JOB_STORE="/tmp/nvflare/jobs-storage"
MODEL_DIR="${REPO}/models"
REPORT_DIR="${REPO}/randomSEED_report"
REPORT_CSV="${REPORT_DIR}/SEEDs_report.csv"

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

# Run evaluation and tee output to a temp file for CSV parsing
EVAL_TMP=$(mktemp)
python3 "${REPO}/utils/model_validation.py" \
    --test_data  "${REPO}/data/processed/df_server_test.csv" \
    --workspace  "${MODEL_DIR}" \
    | tee "${EVAL_TMP}"

# Parse evaluation output and save metrics to SEEDs_report.csv
python3 - <<PYEOF
import re, csv, os, sys

eval_out = open("${EVAL_TMP}").read()
report_csv = "${REPORT_CSV}"
job_id = "${JOB_ID}"

CLASSES = ['BENIGN', 'DOS', 'GAS', 'RPM', 'SPEED', 'STEERING_WHEEL']
CSV_HEADER = (
    ['job_id', 'accuracy_pct', 'logloss', 'macro_f1_pct']
    + [f'prec_{c}'    for c in CLASSES]
    + [f'rec_{c}'     for c in CLASSES]
    + [f'f1_{c}'      for c in CLASSES]
    + [f'support_{c}' for c in CLASSES]
)

# ── Parse overall metrics ────────────────────────────────────────────
def extract(pattern, text, cast=float):
    m = re.search(pattern, text)
    return cast(m.group(1)) if m else None

accuracy  = extract(r'Overall Accuracy:\s+([\d.]+)%', eval_out)
logloss   = extract(r'Overall LogLoss:\s+([\d.]+)',   eval_out)
macro_f1  = extract(r'Macro F1-Score:\s+([\d.]+)%',  eval_out)

if None in (accuracy, logloss, macro_f1):
    print("  Warning: could not parse overall metrics from evaluation output.", file=sys.stderr)
    sys.exit(0)

# ── Parse per-class metrics from classification report ───────────────
#  Format:  CLASSNAME   prec   rec   f1   support
per_class = {}
for cls in CLASSES:
    pat = rf'\b{re.escape(cls)}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)'
    m = re.search(pat, eval_out)
    if m:
        per_class[cls] = {
            'prec':    round(float(m.group(1)) * 100, 4),
            'rec':     round(float(m.group(2)) * 100, 4),
            'f1':      round(float(m.group(3)) * 100, 4),
            'support': int(m.group(4)),
        }
    else:
        per_class[cls] = {'prec': '', 'rec': '', 'f1': '', 'support': ''}

# ── Build new metric dict ────────────────────────────────────────────
new_metrics = {
    'job_id':       job_id,
    'accuracy_pct': round(accuracy, 4),
    'logloss':      round(logloss, 6),
    'macro_f1_pct': round(macro_f1, 4),
}
for cls in CLASSES:
    new_metrics[f'prec_{cls}']    = per_class[cls]['prec']
    new_metrics[f'rec_{cls}']     = per_class[cls]['rec']
    new_metrics[f'f1_{cls}']      = per_class[cls]['f1']
    new_metrics[f'support_{cls}'] = per_class[cls]['support']

# ── Read existing CSV rows ───────────────────────────────────────────
rows = []
if os.path.exists(report_csv):
    with open(report_csv, newline='') as f:
        for row in csv.DictReader(f):
            rows.append({col: row.get(col, '') for col in CSV_HEADER})

# ── Append new row ───────────────────────────────────────────────────
rows.append(new_metrics)

os.makedirs(os.path.dirname(os.path.abspath(report_csv)), exist_ok=True)
with open(report_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=CSV_HEADER, extrasaction='ignore', restval='')
    writer.writeheader()
    writer.writerows(rows)

print(f"\n  -> Saved metrics for job={job_id}")
print(f"     {report_csv}")
PYEOF

rm -f "${EVAL_TMP}"
