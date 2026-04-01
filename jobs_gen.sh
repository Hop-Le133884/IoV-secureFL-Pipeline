#!/usr/bin/env bash

# This dynamically converts "./data" into a full absolute path for this specific machine
DATA_DIR=$(realpath "${1:-./data}")

# DP parameters (pass as env vars to override):
#   DP_EPSILON=1.0 bash jobs_gen.sh ./data   → enable DP with ε=1.0
#   DP_EPSILON=     bash jobs_gen.sh ./data   → disable DP (default)
#   SEED=123       bash jobs_gen.sh ./data   → set random seed (default: 42)
DP_EPSILON="${DP_EPSILON:-}"        # privacy budget ε — empty = no DP
DP_DELTA="${DP_DELTA:-1e-5}"        # failure probability δ
DP_CLIP_BOUND="${DP_CLIP_BOUND:-5.0}"  # leaf clipping bound C
SEED="${SEED:-42}"                  # random seed for XGBoost and DP noise

echo "Generating Double RF Job Configuration..."
if [ -n "${DP_EPSILON}" ]; then
    echo "  DP enabled: ε=${DP_EPSILON}, δ=${DP_DELTA}, clip_bound=${DP_CLIP_BOUND}"
else
    echo "  DP disabled (set DP_EPSILON to enable)"
fi

DP_ARGS=""
if [ -n "${DP_EPSILON}" ]; then
    DP_ARGS="--dp_epsilon ${DP_EPSILON} --dp_delta ${DP_DELTA} --dp_clip_bound ${DP_CLIP_BOUND}"
fi

python3 utils/prepare_job_config.py \
    --site_num 5 \
    --num_local_parallel_tree 20 \
    --max_depth 8 \
    --nthread 4 \
    --data_split_root "${DATA_DIR}/IoV/data_splits" \
    --seed "${SEED}" \
    ${DP_ARGS}

echo "IoV Double RF Job generated successfully."