# src/federated/client_stage2.py
import argparse
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import warnings
from sklearn.metrics import classification_report

warnings.filterwarnings('ignore')

def load_stage2_data(node_id, num_nodes):
    print(f"Vehicle {node_id}: Loading private data for Stage 2...")

    # Load the federated dataset (same as Stage 1)
    df = pd.read_csv("./data/processed/df_federated_5x.csv")

    # Stratified split to give each vehicle a representative partition
    skf = StratifiedKFold(n_splits=num_nodes, shuffle=True, random_state=42)
    folds = list(skf.split(df, df['specific_class']))
    _, indices = folds[node_id]
    my_partition = df.iloc[indices].copy()

    # Raw CAN features
    feature_cols = ['DATA_0', 'DATA_1', 'DATA_2', 'DATA_3', 'DATA_4', 'DATA_5', 'DATA_6', 'DATA_7']
    X_raw = my_partition[feature_cols]

    # ────────────────────────────────────────────────
    # STAGE 1: Load the global binary RF model from Stage 1
    # This is the federated RF trained in client.py / server.py
    print("Loading Stage 1 Global Binary RF Model for feature engineering...")
    try:
        with open("./models/federated_global_rf.pkl", "rb") as f:
            stage1_model = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            "Stage 1 global model not found. Run Stage 1 first and ensure "
            "./models/federated_global_rf.pkl exists."
        )

    # Generate binary probabilities (prob_BENIGN, prob_ATTACK)
    base_probs = stage1_model.predict_proba(X_raw)
    my_partition['prob_BENIGN'] = base_probs[:, 0]   # probability of BENIGN
    my_partition['prob_ATTACK'] = base_probs[:, 1]   # probability of ATTACK

    # ────────────────────────────────────────────────
    # Augmented feature set = raw features + binary probabilities
    meta_feature_cols = feature_cols + ['prob_BENIGN', 'prob_ATTACK']
    X_augmented = my_partition[meta_feature_cols]

    # Encode multi-class target (specific_class → 0..5)
    le = LabelEncoder()
    y_encoded = le.fit_transform(my_partition['specific_class'])

    # Save encoder for later inverse transform / reporting
    with open(f"./models/label_encoder_node_{node_id}.pkl", "wb") as f:
        pickle.dump(le, f)

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_augmented, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    return X_train, X_test, y_train, y_test


def main():
    parser = argparse.ArgumentParser(description="Stage 2: XGBoost Federated Multi-Class Client")
    parser.add_argument("--node-id", type=int, required=True, help="Vehicle ID (0 to 4)")
    parser.add_argument("--num-nodes", type=int, default=5, help="Total vehicles")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_stage2_data(args.node_id, args.num_nodes)

    # ────────────────────────────────────────────────
    # XGBoost DMatrix (native format)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest  = xgb.DMatrix(X_test, label=y_test)

    # Federated XGBoost parameters
    params = {
        'objective': 'multi:softprob',
        'num_class': 6,                      # BENIGN + 5 attack types
        'eval_metric': 'mlogloss',
        'tree_method': 'hist',               # fast histogram method (CPU)
        # Federated settings
        'federated': 1,
        'federated_world_size': args.num_nodes,
        'federated_rank': args.node_id,
        'federated_server_address': '127.0.0.1:9091'  # Tracker / coordinator address
    }

    print(f"Vehicle {args.node_id}: Connecting to XGBoost Federated Tracker...")

    # Train federated model (blocks until all 5 clients connect)
    bst = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=30,                  # adjust as needed
        evals=[(dtest, 'eval')],
        early_stopping_rounds=10,
        verbose_eval=5
    )

    # ────────────────────────────────────────────────
    # Local evaluation
    pred_probs = bst.predict(dtest)                     # shape: [n_samples, 6]
    y_pred = np.argmax(pred_probs, axis=1)

    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    logloss = log_loss(y_test, pred_probs, labels=np.arange(6))

    print(f"\nVehicle {args.node_id} - Local Evaluation:")
    print(f"  Macro F1: {f1:.4f}")
    print(f"  Log Loss: {logloss:.4f}")

    # Optional: full report
    with open(f"./models/label_encoder_node_{args.node_id}.pkl", "rb") as f:
        le = pickle.load(f)

    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        zero_division=0,
        digits=4
    ))

    # Save local model for debugging
    bst.save_model(f'./models/local_xgb_node_{args.node_id}.json')

    # Only node 0 saves the final global model (all nodes have the same final model)
    if args.node_id == 0:
        print("Vehicle 0: Saving final federated XGBoost model...")
        bst.save_model('./models/federated_xgb_global.json')

    print(f"Vehicle {args.node_id} finished.")


if __name__ == "__main__":
    main()