#client_stage2.py
#import flwr as fl
import numpy as np
import pandas as pd
import pickle
import argparse
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')
    
def load_stage2_partition(node_id, num_nodes):
    print(f"Vehicle {node_id}: Loading private data...")
    df = pd.read_csv("./data/processed/df_federated_5x.csv")

    # split the data into chunks based on the number of vehicles
    skf = StratifiedKFold(n_splits=num_nodes, shuffle=True, random_state=42)
    folds = list(skf.split(df, df['specific_class']))
    _, vehcile_indices = folds[node_id]
    my_partition = df.iloc[vehcile_indices].copy()

    feature_cols = ['DATA_0', 'DATA_1', 'DATA_2', 'DATA_3', 'DATA_4', 'DATA_5', 'DATA_6', 'DATA_7']
    X_base = my_partition[feature_cols]


    # Loading the Stage 1 handoff to build 2 feature engineering cols
    print("Loading Stage 1 Global RF Model...")
    with open("./models/federated_global_rf.pkl", "rb") as f:
        stage1_model = pickle.load(f)


    base_probs = stage1_model.predict_proba(X_base)
    my_partition['prob_BENIGN'] = base_probs[:, 0]
    my_partition['prob_ATTACK'] = base_probs[:, 1]

    meta_feature_cols = feature_cols + ['prob_BENIGN', 'prob_ATTACK']
    X = my_partition[meta_feature_cols]

    # Encode the string target col
    le = LabelEncoder()
    y = le.fit_transform(my_partition['specific_class'])

    # Save the encoder for the final calssification report
    with open(f"./models/label_encoder_node_{node_id}.pkl", "wb") as f:
        pickle.dump(le, f)

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def main():
    # build parser arguments
    parser = argparse.ArgumentParser(description="Stage 2's Flower Client")
    parser.add_argument("--node-id", type=int, required=True, help="Partition ID 0 to 4 (car 0 to 4)")
    parser.add_argument("--num-nodes", type=int, default=5, help="Total number of nodes (cars)")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_stage2_partition(args.node_id, args.num_nodes)

    print(f"\nVehicle {args.node_id}: Data ready! Waking up C++ Networking Engine...")

    # CONNECTION: Now we securely lock into the gRPC Tracker
    xgb.collective.init(
        dmlc_communicator="federated",
        federated_server_address="127.0.0.1:9091",
        federated_world_size=args.num_nodes,
        federated_rank=args.node_id
    )

    print(f"Vehicle {args.node_id}: Connected to fleet! Syncing DMatrix dimensions...")

    # Initialize Native XGBoost in C++
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest  = xgb.DMatrix(X_test, label=y_test)

    # native histogram aggregation
    params = {
        'objective': 'multi:softprob',
        'num_class': 6,
        'eval_metric': 'mlogloss',
        'tree_method': 'hist', # Using histogram math for aggregate all cars nodes
    }

    print(f"Vehicle {args.node_id}: Connecting to Tracker and building trees natively...")

    # The code will pause and wait for all the cars to connect before it build a single gradient
    bst = xgb.train(params, dtrain, num_boost_round=20)

    # Evaluate the xgboost locally
    pred_probs = bst.predict(dtest)
    y_pred = np.argmax(pred_probs, axis=1)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    print(f"\nVEHICLE {args.node_id} Local Macro F1 Score: {f1:.4f}")

    # all nodes (cars) colaborate to build one tree, we only need to save model from one car
    if args.node_id == 0:
        print(f"\nVEHICLE 0: Saving the Global Federated XGBoost Meta Model to disk...")
        with open('./models/federated_META_xgb.pkl', 'wb') as f:
            pickle.dump(bst, f)

if __name__ == "__main__":
    main()

