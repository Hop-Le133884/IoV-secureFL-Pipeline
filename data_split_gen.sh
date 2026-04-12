#!/usr/bin/env bash
# Generate FL data splits from the 5x-capped federated dataset.
#
# Usage:  bash data_split_gen.sh ./data
#
# Inputs  (inside DATA_DIR):
#   processed/df_federated_100x.csv   — 100x-capped training data (built in notebook)
#
# Outputs (inside DATA_DIR):
#   processed/vehicle_site-N_train.csv  — one CSV per FL client site (train signatures only)
#   processed/df_server_test.csv        — held-out unique signatures (20% per class, never seen in training)
#   IoV/data_splits/data_site-N.json    — JSON pointer files for the data loader

DATA_DIR=$(realpath "${1:-./data}")
SEED="${SEED:-42}"
FEDERATED_DATA="${DATA_DIR}/processed/df_federated_100x.csv"
OUTPUT_PATH="${DATA_DIR}/IoV/data_splits"
PROCESSED_DIR="${DATA_DIR}/processed"

if [ ! -f "${FEDERATED_DATA}" ]; then
    echo "Error: ${FEDERATED_DATA} not found."
    echo "Run the notebook 01_reproducing_exploration_baseline.ipynb first to generate it."
    exit 1
fi

echo "Generating FL data splits from ${FEDERATED_DATA} (seed=${SEED}) ..."

python3 utils/prepare_data_split.py \
    --federated_data_path "${FEDERATED_DATA}" \
    --site_num 5 \
    --out_path "${OUTPUT_PATH}" \
    --processed_dir "${PROCESSED_DIR}" \
    --test_ratio 0.2 \
    --seed "${SEED}"

echo "Splits generated in ${OUTPUT_PATH}"
echo "Server test set: ${PROCESSED_DIR}/df_server_test.csv"
