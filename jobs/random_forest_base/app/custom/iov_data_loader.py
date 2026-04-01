import json
import pandas as pd
import xgboost as xgb
from nvflare.app_opt.xgboost.data_loader import XGBDataLoader

# All 9 CAN frame fields — ID is the primary DoS discriminator (CAN ID 291)
FEATURES = ['ID', 'DATA_0', 'DATA_1', 'DATA_2', 'DATA_3', 'DATA_4', 'DATA_5', 'DATA_6', 'DATA_7']

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

        return self.get_inner_dmatrix(), self.get_inner_dmatrix()

    def get_inner_dmatrix(self):
        X = self.site_df[FEATURES]
        y = self.site_df['is_attack']
        return xgb.DMatrix(X, label=y)

    def augment_and_get_outer_dmatrix(self, global_inner_model):
        X_inner = self.site_df[FEATURES]
        prob_attack = global_inner_model.predict(xgb.DMatrix(X_inner))
        self.site_df = self.site_df.copy()
        self.site_df['prob_ATTACK'] = prob_attack

        augmented_features = FEATURES + ['prob_ATTACK']
        X_outer = self.site_df[augmented_features]

        unique_labels = sorted(self.site_df['specific_class'].astype(str).unique().tolist())
        dynamic_label_map = {label: idx for idx, label in enumerate(unique_labels)}
        print(f"\n--- Dynamic Label Map Applied: {dynamic_label_map} ---\n")

        y_outer = self.site_df['specific_class'].astype(str).map(dynamic_label_map)
        return xgb.DMatrix(X_outer, label=y_outer)
