#!/usr/bin/env bash
# Generate FL data splits from the 5x-capped federated dataset.
#
# Usage:  bash data_split_gen.sh ./data
#
# Inputs  (inside DATA_DIR):
#   processed/df_federated_5x.csv   — 5x-capped training data (built in notebook)
#
# Outputs (inside DATA_DIR):
#   processed/vehicle_site-N_train.csv  — one CSV per FL client site
#   processed/df_server_test.csv        — unique-signature server test set
#   IoV/data_splits/data_site-N.json    — JSON pointer files for the data loader

DATA_DIR=$(realpath "${1:-./data}")
FEDERATED_DATA="${DATA_DIR}/processed/df_federated_5x.csv"
OUTPUT_PATH="${DATA_DIR}/IoV/data_splits"
PROCESSED_DIR="${DATA_DIR}/processed"

if [ ! -f "${FEDERATED_DATA}" ]; then
    echo "Error: ${FEDERATED_DATA} not found."
    echo "Run the notebook 01_reproducing_exploration_baseline.ipynb first to generate it."
    exit 1
fi

echo "Generating FL data splits from ${FEDERATED_DATA} ..."

python3 utils/prepare_data_split.py \
    --federated_data_path "${FEDERATED_DATA}" \
    --site_num 5 \
    --out_path "${OUTPUT_PATH}" \
    --processed_dir "${PROCESSED_DIR}" \
    --alpha_benign 2.0 \
    --alpha_attack 0.5 \
    --seed 42

echo "Splits generated in ${OUTPUT_PATH}"
echo "Server test set: ${PROCESSED_DIR}/df_server_test.csv"
