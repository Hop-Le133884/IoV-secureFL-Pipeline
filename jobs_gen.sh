#!/usr/bin/env bash

# This dynamically converts "./data" into a full absolute path for this specific machine
DATA_DIR=$(realpath "${1:-./data}")

echo "Generating Double RF Job Configuration..."

python3 utils/prepare_job_config.py \
    --site_num 5 \
    --num_local_parallel_tree 20 \
    --max_depth 8 \
    --nthread 4 \
    --data_split_root "${DATA_DIR}/IoV/data_splits"

echo "IoV Double RF Job generated successfully."