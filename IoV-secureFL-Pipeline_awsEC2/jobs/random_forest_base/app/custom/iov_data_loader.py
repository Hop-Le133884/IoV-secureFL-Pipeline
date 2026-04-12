import json
import numpy as np
import pandas as pd
import xgboost as xgb
from nvflare.app_opt.xgboost.data_loader import XGBDataLoader

# All 9 CAN frame fields — ID is the primary DoS discriminator (CAN ID 291)
FEATURES = ['ID', 'DATA_0', 'DATA_1', 'DATA_2', 'DATA_3', 'DATA_4', 'DATA_5', 'DATA_6', 'DATA_7']

# Fixed global label map — must be consistent across all sites and the server.
# Non-IID splits mean some sites won't have all 6 classes locally, but the
# global model always outputs 6 probabilities, so labels must be fixed globally.
LABEL_MAP = {
    'BENIGN':         0,
    'DOS':            1,
    'GAS':            2,
    'RPM':            3,
    'SPEED':          4,
    'STEERING_WHEEL': 5,
}

class IoVDataLoader(XGBDataLoader):
    def __init__(self, data_split_filename):
        super().__init__()
        self.data_split_filename = data_split_filename
        self.site_df = None

    def load_data(self, client_id: str):
        with open(self.data_split_filename, 'r') as f:
            config = json.load(f)

        csv_path = config["csv_path"]
        self.site_df = pd.read_csv(csv_path)

        dmat = self.get_inner_dmatrix()
        return dmat, dmat

    @staticmethod
    def _balanced_weights(y):
        """Balanced sample weights: n_samples / (n_classes * count_per_class).
        Equivalent to sklearn class_weight='balanced'."""
        counts = y.value_counts()
        n_samples, n_classes = len(y), len(counts)
        class_w = {cls: n_samples / (n_classes * cnt) for cls, cnt in counts.items()}
        return y.map(class_w).values.astype(np.float32)

    def get_inner_dmatrix(self):
        X = self.site_df[FEATURES]
        y = self.site_df['is_attack']
        return xgb.DMatrix(X, label=y, weight=self._balanced_weights(y))

    def augment_and_get_outer_dmatrix(self, global_inner_model):
        X_inner = self.site_df[FEATURES]
        prob_attack = global_inner_model.predict(xgb.DMatrix(X_inner))
        self.site_df = self.site_df.copy()
        self.site_df['prob_ATTACK'] = (prob_attack > 0.5).astype(float)
        self.site_df['prob_BENIGN'] = (prob_attack <= 0.5).astype(float)

        augmented_features = FEATURES + ['prob_BENIGN', 'prob_ATTACK']
        X_outer = self.site_df[augmented_features]

        y_outer = self.site_df['specific_class'].astype(str).map(LABEL_MAP)
        print(f"\n--- Fixed Global Label Map: {LABEL_MAP} ---\n")
        print(f"    Local classes at this site: {sorted(self.site_df['specific_class'].unique().tolist())}")
        return xgb.DMatrix(X_outer, label=y_outer, weight=self._balanced_weights(y_outer))
