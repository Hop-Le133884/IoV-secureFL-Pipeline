import json
import numpy as np

def check_leaf_values(model_path, label):
    with open(model_path, "r") as f:
        model_dict = json.load(f)

    trees = model_dict["learner"]["gradient_booster"]["model"]["trees"]
    leaves = []
    for tree in trees:
        left_children = tree["left_children"]
        split_conditions = tree["split_conditions"]
        for i, lc in enumerate(left_children):
            if lc == -1:  # leaf node
                leaves.append(float(split_conditions[i]))

    leaves = np.array(leaves)
    print(f"\n--- {label} ---")
    print(f"Total leaves : {len(leaves)}")
    print(f"Min          : {leaves.min():.4f}")
    print(f"Max          : {leaves.max():.4f}")
    print(f"95th pct     : {np.percentile(np.abs(leaves), 95):.4f}")
    print(f"99th pct     : {np.percentile(np.abs(leaves), 99):.4f}")
    print(f"100th pct    : {np.percentile(np.abs(leaves), 100):.4f}")
    print(f"% clipped by C=5 : {(np.abs(leaves) > 5).mean()*100:.2f}%")

check_leaf_values(
    "IoV-secureFL-Pipeline_awsEC2/models/xgboost_model_inner.json",
    "Stage 1 (Binary)"
)
check_leaf_values(
    "IoV-secureFL-Pipeline_awsEC2/models/xgboost_model_outer.json",
    "Stage 2 (6-class)"
)
