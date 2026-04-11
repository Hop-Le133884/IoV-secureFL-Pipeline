import argparse
import json
import xgboost as xgb


def diagnose(model_path: str, label: str):
    bst = xgb.Booster()
    bst.load_model(model_path)
    m = json.loads(bst.save_raw("json"))
    params = m["learner"]["gradient_booster"]["model"]["gbtree_model_param"]
    print(f"\n--- {label} ---")
    print(f"  num_parallel_tree in JSON : {params['num_parallel_tree']}")
    print(f"  num_trees in JSON         : {params['num_trees']}")
    print(f"  actual trees list length  : {len(m['learner']['gradient_booster']['model']['trees'])}")
    print(f"  num_boosted_rounds()      : {bst.num_boosted_rounds()}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose XGBoost global model tree counts")
    parser.add_argument(
        "--workspace", type=str,
        default="models",
        help="Path to directory containing xgboost_model_inner.json and xgboost_model_outer.json"
             " (Phase 2 default: ./workspace_iov_double_rf/server/simulate_job/app_server,"
             "  Phase 3 AWS:     ./models)"
    )
    args = parser.parse_args()

    diagnose(f"{args.workspace}/xgboost_model_inner.json", "Stage 1 — Binary (inner)")
    diagnose(f"{args.workspace}/xgboost_model_outer.json", "Stage 2 — 6-Class (outer)")
    print()


if __name__ == "__main__":
    main()
