import argparse
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, log_loss

FEATURES = ['ID', 'DATA_0', 'DATA_1', 'DATA_2', 'DATA_3', 'DATA_4', 'DATA_5', 'DATA_6', 'DATA_7']

def main():
    parser = argparse.ArgumentParser(description="Evaluate global Double RF model on unique-signature test set")
    parser.add_argument("--test_data", type=str, default="./data/processed/df_server_test.csv",
                        help="Path to server test set (unique signatures)")
    parser.add_argument("--workspace", type=str,
                        default="./workspace_iov_double_rf/server/simulate_job/app_server",
                        help="Path to NVFlare server output directory with saved models")
    args = parser.parse_args()

    print(f"Loading test set: {args.test_data}")
    df_valid = pd.read_csv(args.test_data)
    print(f"  Test rows: {len(df_valid):,}  |  class distribution:")
    print(df_valid['specific_class'].value_counts().to_string(header=False))

    X_inner = df_valid[FEATURES]

    label_map = {'BENIGN': 0, 'DOS': 1, 'GAS': 2, 'RPM': 3, 'SPEED': 4, 'STEERING_WHEEL': 5}
    y_true = df_valid['specific_class'].map(label_map)

    print("\nLoading Stage 1 (Binary) Global Expert ...")
    bst_inner = xgb.Booster()
    bst_inner.load_model(f"{args.workspace}/xgboost_model_inner.json")

    print("Generating 'prob_BENIGN' and 'prob_ATTACK' features ...")
    prob_attack = bst_inner.predict(xgb.DMatrix(X_inner))
    df_valid = df_valid.copy()
    df_valid['prob_ATTACK'] = prob_attack
    df_valid['prob_BENIGN'] = 1.0 - prob_attack

    augmented_features = FEATURES + ['prob_BENIGN', 'prob_ATTACK']
    X_outer = df_valid[augmented_features]

    print("Loading Stage 2 (6-Class) Global Master Model ...")
    bst_outer = xgb.Booster()
    bst_outer.load_model(f"{args.workspace}/xgboost_model_outer.json")

    print("Making Final Classifications ...")
    outer_probs = bst_outer.predict(xgb.DMatrix(X_outer))
    y_pred = np.argmax(outer_probs, axis=1)

    logloss = log_loss(y_true, outer_probs)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', labels=[0, 1, 2, 3, 4, 5], zero_division=0)

    print("\n" + "=" * 60)
    print("      DOUBLE RANDOM FOREST — EVALUATION ON UNIQUE SIGNATURES")
    print("=" * 60)
    print(f"  Test set:          {args.test_data}")
    print(f"  Test rows:         {len(df_valid):,} (unique signatures only)")
    print(f"  Overall Accuracy:  {acc * 100:.4f}%")
    print(f"  Overall LogLoss:   {logloss:.6f}")
    print(f"  Macro F1-Score:    {macro_f1 * 100:.4f}%\n")

    target_names = [k for k, v in sorted(label_map.items(), key=lambda item: item[1])]
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names,
                                labels=[0, 1, 2, 3, 4, 5], zero_division=0))
    print("\nConfusion Matrix (Rows: True, Columns: Predicted):")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4, 5]))
    print("=" * 60)


if __name__ == "__main__":
    main()
