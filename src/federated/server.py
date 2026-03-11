#server.py
import flwr as fl

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
    )

    # Start the server on port 8080
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3), # run 3 global rounds of training
        strategy=strategy,
    )

if __name__ == "__main__":
    main()