# client.py
import flwr as fl
import numpy as np
import pandas as pd
import pickle
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,log_loss
from sklearn.model_selection import train_test_split, StratifiedKFold
import warnings

warnings.filterwarnings('ignore')

# Define the Flower Client
class IoVClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    # Freeze the model to send to Server
    def get_parameters(self, config):
        # Convert the entire Random Forest object into a byte string using pickle
        # Then wrap it in a numpy array so Flower can send it over the network
        return [np.array(pickle.dumps(self.model))]
    
    # Unfreeze the global model from Server send back to Client
    def set_parameters(self, parameters):
        # unpack the global model
        if parameters and len(parameters) > 0:
            self.model = pickle.loads(parameters[0].tobytes())

    # Train the model locally on the vehicle's private data
    def fit(self, parameters, config):
        print(f"Vehicle: Starting local training on {len(self.X_train)} real-world traffics...")
        self.model.fit(self.X_train, self.y_train)

        # Return the updated parameters, the number of data points, and any extra info
        return self.get_parameters(config), len(self.X_train), {}
    
    # Evaluate the global model on the vehicle's local test dataset
    def evaluate(self, parameters, config):
        print("Vehicle: Evaluating model...")
        # apply the global model from server sent to
        self.set_parameters(parameters)

        # Test it against local hidden data (the test dataset)
        y_pred = self.model.predict(self.X_test)
        f1 = f1_score(self.y_test, y_pred, average='macro', zero_division=0)

        # Log Loss
        #y_prob = self.model.predict_proba(self.X_test)[:, 1]
        #logLoss = log_loss(self.y_test, y_prob)
        # Return 3 arguments , loss, number of test points, and the F1 score
        return 0.0, len(self.X_test), {"macro_f1": f1}
    
def load_partition(node_id, num_nodes):
    print(f"Loading real IoV data for Vehicle Node {node_id}...")

    # Load dataset
    csv_path = "./data/processed/df_federated_5x.csv"
    df = pd.read_csv(csv_path)

    # checking 'label' binary
    if 'label' not in df.columns:
        df['label'] = (df['specific_class'] != 'BENIGN').astype(int)

    # shuffle the dataset using fix seed so all vehicles see the same initial shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # split the data into chunks based on the number of vehicles
    skf = StratifiedKFold(n_splits=num_nodes, shuffle=True, random_state=42)
    folds = list(skf.split(df, df['label']))
    _, vehicle_indices = folds[node_id]
    my_partition = df.iloc[vehicle_indices].copy()

    # extract features (DATA_0 to DATA_7) and the specific_class col
    feature_cols = ['DATA_0', 'DATA_1', 'DATA_2', 'DATA_3', 'DATA_4', 'DATA_5', 'DATA_6', 'DATA_7']
    X  = my_partition[feature_cols]
    y = my_partition['label']

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def main():
    # Setup argument parser so we can assign Vehicle IDs from the terminal
    parser = argparse.ArgumentParser(description="Flower Client")
    parser.add_argument("--node-id", type=int, required=True, help="Partition ID (0 to 4)")
    parser.add_argument("--num-nodes", type=int, default=5, help="Total number of vehicles")
    args = parser.parse_args()

    print(f"Initializing Vehicle Client {args.node_id}...")

    # Load this specific vehicle's slice of the data
    X_train, X_test, y_train, y_test = load_partition(args.node_id, args.num_nodes)
    model = RandomForestClassifier(
        n_estimators=20,
        max_depth=8,
        random_state=42
    )

    # Start the client and connect to the EMR server
    client = IoVClient(model, X_train, y_train, X_test, y_test)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)

if __name__ == "__main__":
    main()