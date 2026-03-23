import json
import pandas as pd
import xgboost as xgb
from nvflare.app_opt.xgboost.data_loader import XGBDataLoader
import os

class IoVDataLoader(XGBDataLoader):
    def __init__(self, data_split_filename, data_path=None):
        super().__init__()
        self.data_split_filename = data_split_filename

        if data_path is None:
            project_root = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
            self.data_path = os.path.join(project_root, "data", "CICIoV2024.csv")
        else: 
            self.data_path = data_path
        self.site_df = None

    def load_data(self, client_id: str):
        with open(self.data_split_filename, 'r') as f:
            config = json.load(f)
        
        indices = config["data_index"][client_id]["indices"]
        full_df = pd.read_csv(self.data_path)
        self.site_df = full_df.iloc[indices].copy()
        
        return self.get_inner_dmatrix(), self.get_inner_dmatrix()

    def get_inner_dmatrix(self):
        features = ['DATA_0', 'DATA_1', 'DATA_2', 'DATA_3', 'DATA_4', 'DATA_5', 'DATA_6', 'DATA_7']
        X = self.site_df[features]
        y = self.site_df['is_attack']
        return xgb.DMatrix(X, label=y)

    def augment_and_get_outer_dmatrix(self, global_inner_model):
        features = ['DATA_0', 'DATA_1', 'DATA_2', 'DATA_3', 'DATA_4', 'DATA_5', 'DATA_6', 'DATA_7']
        X_inner = self.site_df[features]
        
        # Generate probabilities from Stage 1 Global Expert
        prob_attack = global_inner_model.predict(xgb.DMatrix(X_inner))
        self.site_df['prob_ATTACK'] = prob_attack
        
        # Append new feature
        augmented_features = features + ['prob_ATTACK']
        X_outer = self.site_df[augmented_features]
        
        # Dynamically grabs exactly what is in the CSV and maps 0-5 alphabetically
        unique_labels = sorted(self.site_df['specific_class'].astype(str).unique().tolist())
        dynamic_label_map = {label: idx for idx, label in enumerate(unique_labels)}
        
        # Print the mapping to the console so we can see exactly how it matched
        print(f"\n--- Dynamic Label Map Applied: {dynamic_label_map} ---\n")
        
        y_outer = self.site_df['specific_class'].astype(str).map(dynamic_label_map)
        
        return xgb.DMatrix(X_outer, label=y_outer)