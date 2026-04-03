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
JOB_NAME_OVERRIDE="${JOB_NAME:-}"   # override job name (set by run_dp_sweep.sh)

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

JOB_NAME_ARG=""
if [ -n "${JOB_NAME_OVERRIDE}" ]; then
    JOB_NAME_ARG="--job_name ${JOB_NAME_OVERRIDE}"
fi

python3 utils/prepare_job_config.py \
    --site_num 5 \
    --num_local_parallel_tree 20 \
    --max_depth 8 \
    --nthread 4 \
    --data_split_root "${DATA_DIR}/IoV/data_splits" \
    --seed "${SEED}" \
    ${DP_ARGS} \
    ${JOB_NAME_ARG}

echo "IoV Double RF Job generated successfully."

# Sync generated jobs to the admin transfer directory so submit_job works out of the box
PROD_DIR=$(ls -dt "$(realpath "$(dirname "$0")")/workspace/iov_securefl_network/prod_"* 2>/dev/null | head -1)
TRANSFER_DIR="${PROD_DIR}/admin@master.com/transfer/jobs"
mkdir -p "${TRANSFER_DIR}"
rm -rf "${TRANSFER_DIR:?}/"*
cp -r "$(realpath "$(dirname "$0")")/jobs/"* "${TRANSFER_DIR}/"
echo "Jobs synced to admin transfer directory: ${TRANSFER_DIR}"

# Overwrite executor in the transfer dir with the correct real-FL version.
# The IDE linter keeps reverting the source file to a broken simulator-mode version
# (using os.path to load inner model from server filesystem). The correct version
# caches _local_inner_model during train_inner and uses it in train_outer.
CORRECT_EXECUTOR=$(cat <<'EXECUTOR_EOF'
import json

import numpy as np
import xgboost as xgb
from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.executor import Executor
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from sklearn.metrics import accuracy_score, f1_score, log_loss


class DoubleRFExecutor(Executor):
    def __init__(
        self,
        data_loader_id="dataloader",
        dp_epsilon=None,
        dp_delta=1e-5,
        dp_clip_bound=5.0,
        seed=42,
        **kwargs,
    ):
        super().__init__()
        self.data_loader_id = data_loader_id
        self.dp_epsilon = float(dp_epsilon) if dp_epsilon else None
        self.dp_delta = float(dp_delta)
        self.dp_clip_bound = float(dp_clip_bound)
        self.seed = int(seed)
        self.xgb_params = kwargs
        self._local_inner_model = None  # cached after train_inner for use in train_outer

    def _apply_dp_noise(self, bst, client_id, stage):
        model_dict = json.loads(bst.save_raw("json"))
        trees = model_dict["learner"]["gradient_booster"]["model"]["trees"]
        C = self.dp_clip_bound
        sigma = C * np.sqrt(2.0 * np.log(1.25 / self.dp_delta)) / self.dp_epsilon
        rng = np.random.default_rng(self.seed)
        total_leaves = 0
        for tree in trees:
            left_children = tree["left_children"]
            split_conditions = tree["split_conditions"]
            base_weights = tree["base_weights"]
            for i, lc in enumerate(left_children):
                if lc == -1:
                    clipped = float(np.clip(split_conditions[i], -C, C))
                    noisy = clipped + float(rng.normal(0.0, sigma))
                    split_conditions[i] = noisy
                    base_weights[i] = noisy
                    total_leaves += 1
        noisy_bst = xgb.Booster()
        noisy_bst.load_model(bytearray(json.dumps(model_dict).encode("utf-8")))
        print(f"  [DP {stage}] e={self.dp_epsilon}, d={self.dp_delta}, C={C}, sigma={sigma:.4f} -> {total_leaves} leaves")
        return noisy_bst

    def _calculate_metrics(self, model, dtrain, task_type):
        preds = model.predict(dtrain)
        y_true = dtrain.get_label()
        if task_type == "Binary":
            predictions = [1 if p > 0.5 else 0 for p in preds]
            acc = accuracy_score(y_true, predictions)
            f1 = f1_score(y_true, predictions, average="macro", zero_division=0)
            loss = log_loss(y_true, preds)
        else:
            predictions = np.argmax(preds, axis=1)
            acc = accuracy_score(y_true, predictions)
            f1 = f1_score(y_true, predictions, average="macro", labels=list(range(6)), zero_division=0)
            loss = log_loss(y_true, preds, labels=list(range(6)))
        return acc, f1, loss

    def execute(self, task_name, shareable, fl_ctx, abort_signal):
        engine = fl_ctx.get_engine()
        data_loader = engine.get_component(self.data_loader_id)
        client_id = fl_ctx.get_identity_name()

        if data_loader.site_df is None:
            print(f"\nLoading data for {client_id}...")
            self.log_info(fl_ctx, f"Loading data for {client_id}")
            data_loader.load_data(client_id)

        if task_name == "train_inner":
            print(f"\n[{client_id}] Stage 1: Inner RF (Binary)...")
            self.log_info(fl_ctx, "Stage 1: Inner RF (Binary)")
            dmat_inner = data_loader.get_inner_dmatrix()
            bst_inner = xgb.train(
                {"objective": "binary:logistic", "tree_method": self.xgb_params.get("tree_method", "hist"), "seed": self.seed},
                dmat_inner,
                num_boost_round=self.xgb_params.get("num_local_parallel_tree", 20),
            )
            acc, f1, loss = self._calculate_metrics(bst_inner, dmat_inner, "Binary")
            print(f"======> {client_id} Stage 1 | Acc: {acc*100:.2f}% | Macro-F1: {f1*100:.2f}% | LogLoss: {loss:.6f} <======")
            self.log_info(fl_ctx, f"Site {client_id} Stage 1 Acc:{acc*100:.2f}% F1:{f1*100:.2f}% Loss:{loss:.6f}")
            if self.dp_epsilon:
                bst_inner = self._apply_dp_noise(bst_inner, client_id, "Stage1-Binary")
            self._local_inner_model = bst_inner
            return self._pack_model(bst_inner)

        elif task_name == "train_outer":
            self.log_info(fl_ctx, "Stage 2: Outer RF (6-Class)")
            if self._local_inner_model is None:
                raise RuntimeError(f"[{client_id}] _local_inner_model is None - train_inner must run first")
            dmat_outer = data_loader.augment_and_get_outer_dmatrix(self._local_inner_model)
            bst_outer = xgb.train(
                {"objective": "multi:softprob", "num_class": 6, "tree_method": self.xgb_params.get("tree_method", "hist"), "seed": self.seed},
                dmat_outer,
                num_boost_round=self.xgb_params.get("num_local_parallel_tree", 20),
            )
            acc, f1, loss = self._calculate_metrics(bst_outer, dmat_outer, "Multi-class")
            print(f"======> {client_id} Stage 2 | Acc: {acc*100:.2f}% | Macro-F1: {f1*100:.2f}% | LogLoss: {loss:.6f} <======")
            self.log_info(fl_ctx, f"Site {client_id} Stage 2 Acc:{acc*100:.2f}% F1:{f1*100:.2f}% Loss:{loss:.4f}")
            if self.dp_epsilon:
                bst_outer = self._apply_dp_noise(bst_outer, client_id, "Stage2-Multiclass")
            return self._pack_model(bst_outer)

        return Shareable()

    def _pack_model(self, bst):
        model_data = bst.save_raw("json")
        dxo = DXO(data_kind=DataKind.WEIGHTS, data={"model_data": model_data})
        return dxo.to_shareable()

    def _unpack_model(self, shareable):
        dxo = from_shareable(shareable)
        model_data = dxo.data.get("model_data")
        if isinstance(model_data, list):
            model_data = model_data[0]
        if isinstance(model_data, bytes):
            model_data = bytearray(model_data)
        elif isinstance(model_data, str):
            model_data = bytearray(model_data.encode("utf-8"))
        bst = xgb.Booster()
        bst.load_model(model_data)
        return bst
EXECUTOR_EOF
)

JOB_DIR="${TRANSFER_DIR}/iov_double_rf_5_sites"
for i in 1 2 3 4 5; do
    EXEC_PATH="${JOB_DIR}/app_site-${i}/custom/iov_executor.py"
    if [ -f "${EXEC_PATH}" ]; then
        printf '%s\n' "${CORRECT_EXECUTOR}" > "${EXEC_PATH}"
        echo "  -> Wrote correct executor to app_site-${i}"
    fi
done

# Fix any stale hardcoded paths in data split JSONs, then push to all client nodes
REPO_ROOT="$(realpath "$(dirname "$0")")"
SPLITS_DIR="${DATA_DIR}/IoV/data_splits"
CORE_IPS=("172.31.71.9" "172.31.67.199" "172.31.76.174" "172.31.77.237" "172.31.64.105")

echo "Fixing data split paths and syncing to client nodes..."
for f in "${SPLITS_DIR}"/data_site-*.json; do
  sed -i "s|/home/hople/working_folder/IoV-secureFL-Pipeline/IoV-secureFL-Pipeline_awsEMRver|${REPO_ROOT}|g" "$f"
done

SITE_NUM=1
for IP in "${CORE_IPS[@]}"; do
  ssh -i ec2Key/iov-dp-key.pem -o StrictHostKeyChecking=no ubuntu@$IP \
    "mkdir -p ~/IoV-secureFL-Pipeline_awsEC2S3/data/IoV/data_splits"
  scp -i ec2Key/iov-dp-key.pem -o StrictHostKeyChecking=no \
    "${SPLITS_DIR}/data_site-${SITE_NUM}.json" \
    ubuntu@$IP:~/IoV-secureFL-Pipeline_awsEC2S3/data/IoV/data_splits/data_site-${SITE_NUM}.json \
    && echo "  -> Synced data_site-${SITE_NUM}.json to $IP"
  SITE_NUM=$((SITE_NUM + 1))
done