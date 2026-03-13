#server.py
import flwr as fl
import pickle
import numpy as np
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters

# Define how the server should average the F1 scores from the vehicles
def weighted_average(metrics):
    # Multiply each F1 score by the number of test examples the vehicle had
    weighted_f1 = [num_examples * m["macro_f1"] for num_examples, m in metrics]
    # Sum up all the examples from all vehicles
    total_examples = sum(num_examples for num_examples, m in metrics)

    # Return the weighted average F1 score
    return {"macro_f1": sum(weighted_f1) / total_examples}

# Build the custom tree concatenation strategy
class RandomForestStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        # IF no clients succeeded, do nothing
        if not results:
            return None, {}
        
        print(f"\n Server: Concatenating trees together for Round {server_round} ...")

        global_model = None
        total_trees = []

        # Loop through the models sent by each vehicle
        for client_proxy, fit_res in results:
            # Unpack the raw network bytes sending by the client
            parameters = parameters_to_ndarrays(fit_res.parameters)

            # Unfreeze the specific vehicle's Random Forest using pickle
            client_model = pickle.loads(parameters[0].tobytes())

            if global_model is None:
                # Use the first client's model as our baseline structure
                global_model = client_model
            else:
                # for other clients, just extract their generated trees
                total_trees.extend(client_model.estimators_)

        # Add the collected trees to the global model
        if global_model is not None and len(total_trees) > 0:
            global_model.estimators_.extend(total_trees)
            # update the official tree count
            global_model.n_estimators = len(global_model.estimators_)
            print(f"Server: Global model successfully grew to {global_model.n_estimators} trees!")

        # Freeze the new Global model back into bytes
        global_model_bytes = pickle.dumps(global_model)

        # Pack it back into Flower's network format
        global_parameters = ndarrays_to_parameters([np.array(global_model_bytes)])

        # SAVING THE OPTIMAL MODEL
        print(f"SERVER: Saving Global Model for Round {server_round} to disk..")
        with open("../model/federated_global_rf.pkl", "wb") as f:
            pickle.dump(global_model, f)

        # Return the new global parameters to be sent back to the vehicles
        return global_parameters, {}

def main():
    print("Starting Flower Central Server (AWS EMR Node)...")

    # Using custom strategy above
    strategy = RandomForestStrategy(
        fraction_fit=1.0,           # Sample 100% of available clients for training
        fraction_evaluate=1.0,      # Sample 100% of available clients for evaluating
        min_fit_clients=2,          # need at least 2 vehicles connected to start training
        min_evaluate_clients=2,     # need at least 2 vehicles to evaluate
        min_available_clients=2,    # wait until 2 vehicles check in
        evaluate_metrics_aggregation_fn=weighted_average, # call weighted_average for macro f1 score average
    )

    # Start the server on port 8080
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3), # run 3 global rounds of training
        strategy=strategy,
    )

if __name__ == "__main__":
    main()