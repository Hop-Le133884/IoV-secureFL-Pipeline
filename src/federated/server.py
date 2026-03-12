#server.py
import flwr as fl

# Define how the server should average the F1 scores from the vehicles
def weighted_average(metrics):
    # Multiply each F1 score by the number of test examples the vehicle had
    weighted_f1 = [num_examples * m["macro_f1"] for num_examples, m in metrics]
    # Sum up all the examples from all vehicles
    total_examples = sum(num_examples for num_examples, m in metrics)

    # Return the weighted average F1 score
    return {"macro_f1": sum(weighted_f1) / total_examples}

def main():
    print("Starting Flower Central Server (AWS EMR Node)...")

    # Define the federated learning stategy
    # FedAvg (Federated Average) is the industry standard baseline
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,           # Sample 100% of available clients for training
        fraction_evaluate=1.0,      # Sample 100% of available clients for evaluating
        min_fit_clients=2,          # need at least 2 vehicles connected to start training
        min_evaluate_clients=2,     # need at least 2 vehciles to evaluate
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