"""
Custom XGBoost bagging aggregator for multi-class (multi:softprob) NVFlare FL.

NVFlare's built-in XGBBaggingAggregator has two bugs for multi:softprob models:

  Bug 1 — Stage contamination:
      aggregate() reads fl_ctx.GLOBAL_MODEL as its base, so Stage 1's 100-tree
      binary model leaks into Stage 2 as the starting point.

  Bug 2 — num_class blindness:
      It reads num_parallel_tree=20 from the client model JSON and copies only
      that many trees per client, ignoring the num_class multiplier.
      A multi:softprob client model with num_class=6, num_parallel_tree=20 has
      120 trees (6 × 20), but the built-in aggregator only copies 20 of them.

Result with built-in aggregator (5 clients, num_parallel_tree=20):
    Stage 2 expected:  5 × 120 = 600 trees
    Stage 2 actual:    100 (Stage 1 base) + 5 × 20 = 200 trees  ← corrupted

This aggregator fixes both issues:
  - Always starts fresh (never reads fl_ctx.GLOBAL_MODEL)
  - Copies ALL trees from every client contribution regardless of num_class
"""

import copy
import json

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.aggregator import Aggregator


class XGBMultiClassBaggingAggregator(Aggregator):
    def __init__(self):
        super().__init__()
        self._contributions: list[dict] = []

    def reset(self, fl_ctx: FLContext):
        self._contributions.clear()

    def accept(self, shareable: Shareable, fl_ctx: FLContext) -> bool:
        try:
            dxo = from_shareable(shareable)
            model_data = dxo.data.get("model_data")
            if model_data is None:
                self.log_error(fl_ctx, "No model_data in DXO")
                return False
            if isinstance(model_data, list):
                model_data = model_data[0]
            if isinstance(model_data, (bytes, bytearray)):
                model_json = json.loads(model_data)
            elif isinstance(model_data, str):
                model_json = json.loads(model_data)
            else:
                self.log_error(fl_ctx, f"Unexpected model_data type: {type(model_data)}")
                return False

            n_trees = len(model_json["learner"]["gradient_booster"]["model"]["trees"])
            client = fl_ctx.get_identity_name()
            self.log_info(fl_ctx, f"Accepted {n_trees} trees from {client}")
            self._contributions.append(model_json)
            return True
        except Exception as e:
            self.log_error(fl_ctx, f"accept() error: {e}")
            return False

    def aggregate(self, fl_ctx: FLContext) -> Shareable:
        if not self._contributions:
            raise RuntimeError("XGBMultiClassBaggingAggregator: no contributions received")

        # Use the first client's full model structure as the template:
        # preserves objective, num_class, num_parallel_tree, feature names, base_score, etc.
        agg_model = copy.deepcopy(self._contributions[0])
        model_body = agg_model["learner"]["gradient_booster"]["model"]

        all_trees: list = []
        all_tree_info: list = []

        for model_json in self._contributions:
            body = model_json["learner"]["gradient_booster"]["model"]
            all_trees.extend(body["trees"])
            n = len(body["trees"])
            all_tree_info.extend(body.get("tree_info", [0] * n))

        for i, tree in enumerate(all_trees):
            tree["id"] = i

        # Rebuild iteration_indptr: maps boosting round index → cumulative tree count.
        # Each client contributes one or more rounds; we append their round sizes in order.
        # Example: 5 clients × 1 round × 120 trees → [0, 120, 240, 360, 480, 600]
        new_indptr = [0]
        for model_json in self._contributions:
            body = model_json["learner"]["gradient_booster"]["model"]
            indptr = body.get("iteration_indptr", [0, len(body["trees"])])
            round_sizes = [indptr[i + 1] - indptr[i] for i in range(len(indptr) - 1)]
            for size in round_sizes:
                new_indptr.append(new_indptr[-1] + size)

        model_body["trees"] = all_trees
        model_body["tree_info"] = all_tree_info
        model_body["iteration_indptr"] = new_indptr
        model_body["gbtree_model_param"]["num_trees"] = str(len(all_trees))

        # Clear single-model prediction limits so all aggregated trees are used
        attrs = agg_model["learner"].get("attributes", {})
        attrs.pop("best_ntree_limit", None)
        attrs.pop("best_iteration", None)
        agg_model["learner"]["attributes"] = attrs

        self.log_info(
            fl_ctx,
            f"Aggregated {len(self._contributions)} clients → {len(all_trees)} trees total"
        )

        model_bytes = json.dumps(agg_model).encode("utf-8")
        dxo = DXO(data_kind=DataKind.WEIGHTS, data={"model_data": model_bytes})
        return dxo.to_shareable()
