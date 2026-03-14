import flwr as fl
import pickle
# import numpy as np

def weighted_average(metrics):
    # Calculate avg f1
    macro_f1s = [num_examples * m['macro_f1'] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {'macro_f1': sum(macro_f1s / sum(examples))}


class XGBoostStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        # Let Flower's built-in FedAvg do the math averaging
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            # Unpack and save the winning Meta-Model
            model_bytes = fl.common.parameters_to_ndarrays(aggregated_parameters)[0].tobytes()
            global_xgb_model = pickle.loads(model_bytes)

            print(f"\nSERVER: Saving XGBoost Meta Model for Round {server_round}...")
            with open("./models/federated_META_xgb.pkl", "wb") as f:
                pickle.dump(global_xgb_model, f)


        return aggregated_parameters, aggregated_metrics
    


def main():
    print("Starting Central Server STAGE 2, XGBoost Meta Model...")

    strategy = XGBoostStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=5,
        min_evaluate_clients=5,
        min_available_clients=5,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=1),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()