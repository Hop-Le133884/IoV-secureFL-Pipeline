import json
import os

import numpy as np
import xgboost as xgb
from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.executor import Executor
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from sklearn.metrics import accuracy_score, log_loss


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
        """
        Args:
            data_loader_id: component ID of the IoVDataLoader.
            dp_epsilon:     DP privacy budget ε. None or 0 disables DP entirely.
            dp_delta:       DP failure probability δ (default 1e-5).
            dp_clip_bound:  Leaf value clipping bound C before noise injection.
                            Acts as the L∞ sensitivity of the output.
                            Recommended: 3.0 for Stage 1 (binary), 5.0 for Stage 2 (6-class).
            **kwargs:       XGBoost hyperparameters forwarded from the job config.
        """
        super().__init__()
        self.data_loader_id = data_loader_id
        self.dp_epsilon = float(dp_epsilon) if dp_epsilon else None
        self.dp_delta = float(dp_delta)
        self.dp_clip_bound = float(dp_clip_bound)
        self.seed = int(seed)
        self.xgb_params = kwargs

    # ------------------------------------------------------------------
    # Differential Privacy
    # ------------------------------------------------------------------

    def _apply_dp_noise(self, bst: xgb.Booster, client_id: str, stage: str) -> xgb.Booster:
        """Output perturbation: add calibrated Gaussian noise to XGBoost leaf values.

        Mechanism: Gaussian mechanism for (ε, δ)-DP.
            σ = C · √(2 ln(1.25 / δ)) / ε
        where C = dp_clip_bound is the L∞ sensitivity (each leaf is first clipped
        to [-C, C] to bound how much one training sample can shift a leaf value).

        Both `split_conditions` and `base_weights` are updated for leaf nodes
        since XGBoost stores the prediction score in both fields.
        """
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
                if lc == -1:  # leaf node
                    clipped = float(np.clip(split_conditions[i], -C, C))
                    noisy = clipped + float(rng.normal(0.0, sigma))
                    split_conditions[i] = noisy
                    base_weights[i] = noisy
                    total_leaves += 1

        noisy_bst = xgb.Booster()
        noisy_bst.load_model(bytearray(json.dumps(model_dict).encode("utf-8")))

        print(
            f"  [DP {stage}] ε={self.dp_epsilon}, δ={self.dp_delta}, "
            f"C={C}, σ={sigma:.4f} → noise added to {total_leaves} leaves"
        )
        return noisy_bst

    # ------------------------------------------------------------------
    # Metrics helpers
    # ------------------------------------------------------------------

    def _calculate_metrics(self, model, dtrain, task_type):
        preds = model.predict(dtrain)
        y_true = dtrain.get_label()
        if task_type == "Binary":
            predictions = [1 if p > 0.5 else 0 for p in preds]
            acc = accuracy_score(y_true, predictions)
            loss = log_loss(y_true, preds)
        else:
            predictions = np.argmax(preds, axis=1)
            acc = accuracy_score(y_true, predictions)
            loss = log_loss(y_true, preds)
        return acc, loss

    # ------------------------------------------------------------------
    # FL lifecycle
    # ------------------------------------------------------------------

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
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
                {"objective": "binary:logistic", "tree_method": self.xgb_params.get("tree_method", "hist"),
                 "seed": self.seed},
                dmat_inner,
                num_boost_round=self.xgb_params.get("num_local_parallel_tree", 20),
            )

            acc, loss = self._calculate_metrics(bst_inner, dmat_inner, "Binary")
            print(f"======> {client_id} Stage 1 | Acc: {acc*100:.2f}% | LogLoss: {loss:.6f} <======")
            self.log_info(fl_ctx, f"Site {client_id} Stage 1 (Binary) LogLoss: {loss:.6f}")

            if self.dp_epsilon:
                bst_inner = self._apply_dp_noise(bst_inner, client_id, "Stage1-Binary")

            return self._pack_model(bst_inner)

        elif task_name == "train_outer":
            self.log_info(fl_ctx, "Stage 2: Outer RF (6-Class)")

            # Derive workspace root from this file's location:
            # {workspace}/{site}/simulate_job/app_{site}/custom/iov_executor.py
            # 4 levels up from dirname(__file__) → workspace root
            workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
            inner_model_path = os.path.join(
                workspace_root, "server", "simulate_job", "app_server", "xgboost_model_inner.json"
            )
            global_inner_model = xgb.Booster()
            global_inner_model.load_model(inner_model_path)

            dmat_outer = data_loader.augment_and_get_outer_dmatrix(global_inner_model)
            bst_outer = xgb.train(
                {"objective": "multi:softprob", "num_class": 6, "tree_method": self.xgb_params.get("tree_method", "hist"),
                 "seed": self.seed},
                dmat_outer,
                num_boost_round=self.xgb_params.get("num_local_parallel_tree", 20),
            )

            acc, loss = self._calculate_metrics(bst_outer, dmat_outer, "Multi-class")
            print(f"======> {client_id} Stage 2 | Acc: {acc*100:.2f}% | LogLoss: {loss:.6f} <======")
            self.log_info(fl_ctx, f"Site {client_id} Stage 2 (Master) LogLoss: {loss:.4f}")

            if self.dp_epsilon:
                bst_outer = self._apply_dp_noise(bst_outer, client_id, "Stage2-Multiclass")

            return self._pack_model(bst_outer)

        return Shareable()

    # ------------------------------------------------------------------
    # Pack / unpack
    # ------------------------------------------------------------------

    def _pack_model(self, bst: xgb.Booster) -> Shareable:
        model_data = bst.save_raw("json")
        dxo = DXO(data_kind=DataKind.WEIGHTS, data={"model_data": model_data})
        return dxo.to_shareable()

    def _unpack_model(self, shareable: Shareable) -> xgb.Booster:
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
