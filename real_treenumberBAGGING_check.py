import xgboost as xgb
import numpy as np
import json

X = np.random.randn(1000, 10)
y_binary = np.random.randint(0, 2, 1000)
y_multi  = np.random.randint(0, 6, 1000)

params_base = {"tree_method": "hist", "num_parallel_tree": 20, "learning_rate": 1.0, "seed": 42}

bst_binary = xgb.train({**params_base, "objective": "binary:logistic"},
                        xgb.DMatrix(X, label=y_binary), num_boost_round=1)

bst_multi  = xgb.train({**params_base, "objective": "multi:softprob", "num_class": 6},
                        xgb.DMatrix(X, label=y_multi),  num_boost_round=1)

for label, bst in [("binary:logistic", bst_binary), ("multi:softprob (6-class)", bst_multi)]:
    m = json.loads(bst.save_raw("json"))
    p = m["learner"]["gradient_booster"]["model"]["gbtree_model_param"]
    print(f"{label}: num_trees={p['num_trees']}, num_parallel_tree={p['num_parallel_tree']}, rounds={bst.num_boosted_rounds()}")
