import pandas as pd
import xgboost as xgb
import numpy as np
import json
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, log_loss

def main():
    print("Loading Validation Dataset...")
    full_df = pd.read_csv("./data/CICIoV2024.csv")
    
    # Read the exact stratified validation indices we just created!
    with open("./data/IoV/data_splits/data_site-1.json", "r") as f:
        split_config = json.load(f)
    valid_indices = split_config["data_index"]["valid"]["indices"]
    
    df_valid = full_df.iloc[valid_indices].copy()

    features = ['DATA_0', 'DATA_1', 'DATA_2', 'DATA_3', 'DATA_4', 'DATA_5', 'DATA_6', 'DATA_7']
    X_inner = df_valid[features]
    
    label_map = {'BENIGN': 0, 'DOS': 1, 'GAS': 2, 'RPM': 3, 'SPEED': 4, 'STEERING_WHEEL': 5}
    y_true = df_valid['specific_class'].map(label_map)

    # Both models now live in the same clean directory!
    server_dir = "./workspace_iov_double_rf/server/simulate_job/app_server"

    print("Loading Stage 1 (Binary) Global Expert...")
    bst_inner = xgb.Booster()
    bst_inner.load_model(f"{server_dir}/xgboost_model_inner.json")

    print("Generating 'prob_ATTACK' feature...")
    dmat_inner = xgb.DMatrix(X_inner)
    prob_attack = bst_inner.predict(dmat_inner)
    
    df_valid['prob_ATTACK'] = prob_attack
    augmented_features = features + ['prob_ATTACK']
    X_outer = df_valid[augmented_features]

    print("Loading Stage 2 (6-Class) Global Master Model...")
    bst_outer = xgb.Booster()
    bst_outer.load_model(f"{server_dir}/xgboost_model_outer.json")

    print("Making Final Classifications...")
    dmat_outer = xgb.DMatrix(X_outer)
    outer_probs = bst_outer.predict(dmat_outer)
    y_pred = np.argmax(outer_probs, axis=1)
    
    logloss = log_loss(y_true, outer_probs)
    acc = accuracy_score(y_true, y_pred)
    # Added zero_division=0 to prevent warnings when a class is missing
    macro_f1 = f1_score(y_true, y_pred, average='macro', labels=[0,1,2,3,4,5])
    
    print("\n" + "="*55)
    print("      DOUBLE RANDOM FOREST EVALUATION METRICS")
    print("="*55)
    print(f"Overall Accuracy:  {acc * 100:.4f}%")
    print(f"Overall LogLoss:   {logloss:.6f}") # This is the "Confidence" metric
    print(f"Macro F1-Score:    {macro_f1 * 100:.4f}%\n")
    
    target_names = [k for k, v in sorted(label_map.items(), key=lambda item: item[1])]
    
    # FIX: Force Scikit-Learn to output the full 6-class matrix using the 'labels' argument
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names, labels=[0,1,2,3,4,5], zero_division=0))
    
    print("\nConfusion Matrix (Rows: True, Columns: Predicted):")
    print(confusion_matrix(y_true, y_pred, labels=[0,1,2,3,4,5]))
    print("="*55)

if __name__ == "__main__":
    main()