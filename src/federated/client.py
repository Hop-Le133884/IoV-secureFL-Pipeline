# client.py
import flwr as fl
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
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

    # Get the current model weights (Not natively simple for Random Forets, but
    #required by Flower's architecture)
    def get_parameters(self, config):
        # For a standard neural network or logistic regression, we extract weights here,
        # *Note for Random Forest: Tree aggregation requires custom logic in FedAvg, but
        # this is the standard structural placeholder.
        return []
    
    # Train the model locally on the vehicle's private data
    def fit(self, parameters, config):
        print("Vehicle: Starting local training...")
        self.model.fit(self.X_train, self.y_train)

        # Return the updated parameters, the number of data points, and any extra info
        return self.get_parameters(config), len(self.X_train), {}
    
    # Evaluate the global model on the vehicle's local tet data
    def evaluate(self, parameters, config):
        print("Vehicle: Evaluating model...")
        y_pred = self.model.predict(self.X_test)
        f1 = f1_score(self.y_test, y_pred, average='macro', zero_division=0)

        # Return the loss (dummy value here), number of test points, and the F1 score
        return 0.0, len(self.X_test), {"macro_f1": f1}
    

def main():
    print("Initializing Vehicle Client...")


    # PLACEHOLDER: Load local vehicle data here
    # IN a real simulation, we would split your def_master in 
    # 01_reproducing_exploration_baseline jupyter notebooks into partitions
    X_train, X_test = np.random.rand(100, 9), np.random.rand(20, 9)
    y_train, y_test = np.random.randint(0, 2, 100), np.random.randint(0, 2, 20)

    # Initialize inner model as we did in 
    # 01_reproducing_exploration_baseline jupyter notebooks
    model = RandomForestClassifier(
        n_estimators=80,
        max_depth=7,
        random_state=42
    )

    # Start the client and connect to the EMR server
    client = IoVClient(model, X_train, y_train, X_test, y_test)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)

if __name__ == "__main__":
    main()