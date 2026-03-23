import xgboost as xgb
from nvflare.apis.executor import Executor
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from sklearn.metrics import accuracy_score
import numpy as np
import os
from sklearn.metrics import accuracy_score, log_loss

class DoubleRFExecutor(Executor):
    def __init__(self, data_loader_id="dataloader", **kwargs):
        super().__init__()
        self.data_loader_id = data_loader_id
        self.xgb_params = kwargs

    def _calculate_metrics(self, model, dtrain, task_type):
        """Helper to log both accuracy and logloss."""
        preds = model.predict(dtrain)
        y_true = dtrain.get_label()
        
        if task_type == "Binary":
            # XGBoost binary outputs a 1D array of probabilities
            predictions = [1 if p > 0.5 else 0 for p in preds]
            acc = accuracy_score(y_true, predictions)
            loss = log_loss(y_true, preds)
        else:
            # XGBoost multi:softprob outputs a 2D array of probabilities
            predictions = np.argmax(preds, axis=1)
            acc = accuracy_score(y_true, predictions)
            loss = log_loss(y_true, preds)
            
        return acc, loss
    def _calculate_and_log_accuracy(self, model, dtrain, label_type):
        """Helper to log local training accuracy."""
        preds = model.predict(dtrain)
        # For XGBoost Random Forest, preds are often probabilities or class indices
        # We convert to crisp labels for accuracy calculation
        if len(preds.shape) > 1:  # Multi-class
            predictions = np.argmax(preds, axis=1)
        else:  # Binary or regression-style
            predictions = [1 if p > 0.5 else 0 for p in preds]
        
        y_true = dtrain.get_label()
        acc = accuracy_score(y_true, predictions)
        return acc

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        engine = fl_ctx.get_engine()
        data_loader = engine.get_component(self.data_loader_id)
        
        client_id = fl_ctx.get_identity_name()

        # LOADING
        if data_loader.site_df is None:
            print(f"\nLoading {client_id}/6 of 1.4M row of 6 stratified dataset into memory for {client_id}...")
            self.log_info(fl_ctx, f"{client_id}/6 of 1.4M row of 6 stratified into memory for {client_id}...")
            data_loader.load_data(client_id)

        if task_name == "train_inner":
            print(f"\n[{client_id}] Starting Stage 1: Inner RF (Binary)...")
            self.log_info(fl_ctx, "Stage 1: Inner RF (Binary)")
            dmat_inner = data_loader.get_inner_dmatrix()
            
            bst_inner = xgb.train(
                {"objective": "binary:logistic", "tree_method": self.xgb_params.get("tree_method", "hist")}, 
                dmat_inner, 
                num_boost_round=self.xgb_params.get("num_local_parallel_tree", 20)
            )

            # Calculate and Log Stage 1 Accuracy ---
            acc, loss = self._calculate_metrics(bst_inner, dmat_inner, "Binary")
            print(f"======> {client_id} Stage 1 Train | Acc: {acc * 100:.2f}% | LogLoss: {loss:.6f} <======")
            #train_acc = self._calculate_and_log_accuracy(bst_inner, dmat_inner, "Binary")
            #print(f"======> {client_id} Stage 1 Train Accuracy: {train_acc * 100:.2f}% <======")
            self.log_info(fl_ctx, f"====== Site {client_id} Stage 1 (Binary) Train Accuracy (LogLoss): {loss:.6f} ======")
            
            return self._pack_model(bst_inner)

        elif task_name == "train_outer":
            self.log_info(fl_ctx, "Stage 2: Outer RF (6-Class)")
            
            # Load the Stage 1 Expert directly from the server's clean persistence file
            global_inner_model = xgb.Booster()
            #inner_model_path = "/home/hople/working_folder/randomForest_nvflareXgboost_FL/workspace_iov_double_rf/server/simulate_job/app_server/xgboost_model_inner.json"
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
            inner_model_path = os.path.join(
                project_root,
                "workspace_iov_double_rf",
                "server",
                "simulate_job",
                "app_server",
                "xgboost_model_inner.json"
            )
            #How to Pivot my previous published paper using Random Forest on significantly deduplicated CICIoV2024?
            global_inner_model.load_model(inner_model_path)
            
            # Feature Engineering: Append Stage 1 probabilities
            dmat_outer = data_loader.augment_and_get_outer_dmatrix(global_inner_model)
            
            bst_outer = xgb.train(
                {"objective": "multi:softprob", "num_class": 6, "tree_method": self.xgb_params.get("tree_method", "hist")}, 
                dmat_outer, 
                num_boost_round=self.xgb_params.get("num_local_parallel_tree", 20)
            )
            
            # Calculate and Log Stage 1 Accuracy ---
            acc, loss = self._calculate_metrics(bst_outer, dmat_outer, "Multi-class")
            # Change the print text to "Stage 2" so your logs stay clean
            print(f"======> {client_id} Stage 2 Train | Acc: {acc * 100:.2f}% | LogLoss: {loss:.6f} <======")
            # Calculate and Log Stage 2 Accuracy ---
            #train_acc = self._calculate_and_log_accuracy(bst_outer, dmat_outer, "Multi-class")
            #print(f"======> {client_id} Stage 2 Train Accuracy: {train_acc * 100:.2f}% <======")
            self.log_info(fl_ctx, f"====== Site {client_id} Stage 2 (Master) Train Accuracy: {loss:.4f} ======")

            return self._pack_model(bst_outer)
            
        return Shareable()

    def _pack_model(self, bst: xgb.Booster) -> Shareable:
        model_data = bst.save_raw("json")
        dxo = DXO(data_kind=DataKind.WEIGHTS, data={"model_data": model_data})
        return dxo.to_shareable()

    def _unpack_model(self, shareable: Shareable) -> xgb.Booster:
        dxo = from_shareable(shareable)
        model_data = dxo.data.get("model_data")
        
        # 1. Unpack from list if NVFLARE wrapped it for network transport
        if isinstance(model_data, list):
            model_data = model_data[0]
            
        # 2. XGBoost strictly requires 'bytearray'
        if isinstance(model_data, bytes):
            model_data = bytearray(model_data)
        elif isinstance(model_data, str):
            model_data = bytearray(model_data.encode("utf-8"))
            
        bst = xgb.Booster()
        bst.load_model(model_data)
        return bst