# client.py
import flwr as fl
import numpy as np
import pickle
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

    # Freeze the model to send to Server
    def get_parameters(self, config):
        # Convert the entire Random Forest object into a byte string using pickle
        # Then wrap it in a numpy array so Flower can send it over the network
        return [np.array(pickle.dumps(self.model))]
    
    # Unfreeze the model from Server send back to Client
    def set_parameters(self, parameters):
        # When the server sends the glued-together global model back, unpack it!
        if parameters and len(parameters) > 0:
            self.model = pickle.loads(parameters[0].tobytes())

    # Train the model locally on the vehicle's private data
    def fit(self, parameters, config):
        print("Vehicle: Starting local training...")
        self.model.fit(self.X_train, self.y_train)

        # Return the updated parameters, the number of data points, and any extra info
        return self.get_parameters(config), len(self.X_train), {}
    
    # Evaluate the global model on the vehicle's local test data
    def evaluate(self, parameters, config):
        print("Vehicle: Evaluating model...")
        # apply the global model from server sent to
        self.set_parameters(parameters)

        # Test it against local hidden data (the test set)
        y_pred = self.model.predict(self.X_test)
        f1 = f1_score(self.y_test, y_pred, average='macro', zero_division=0)

        # Return the loss, number of test points, and the F1 score
        return 0.0, len(self.X_test), {"macro_f1": f1}
    

def main():
    print("Initializing Vehicle Client...")

    # PLACEHOLDER: Load local vehicle data here
    # IN a real simulation, we would split your df_master in 
    # 01_reproducing_exploration_baseline jupyter notebooks into partitions
    X_train, X_test = np.random.rand(100, 9), np.random.rand(20, 9)
    y_train, y_test = np.random.randint(0, 2, 100), np.random.randint(0, 2, 20)

    model = RandomForestClassifier(
        n_estimators=10,
        max_depth=7,
        random_state=42
    )

    # Start the client and connect to the EMR server
    client = IoVClient(model, X_train, y_train, X_test, y_test)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)

if __name__ == "__main__":
    main()