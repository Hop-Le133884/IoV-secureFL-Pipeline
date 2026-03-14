# tracker on server cloud, stage 2
import xgboost as xgb
import xgboost.federated # FORCE python to load, otherwise will throw errors

def main():
    print("Starting Native XGBoost Federated Tracker on port 9091...")
    print("Waiting for exactly 5 vehicles to connect before starting the math...")

    # Open the secure router for 5 workers from 5 nodes (cars) on port 9091
    xgb.federated.run_federated_server(5, 9091)

if __name__ == "__main__":
    main()

