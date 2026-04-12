import argparse
import json
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

SIGNATURE_COLS = ['ID', 'DATA_0', 'DATA_1', 'DATA_2', 'DATA_3', 'DATA_4', 'DATA_5', 'DATA_6', 'DATA_7']

def data_split_args_parser():
    parser = argparse.ArgumentParser(description="Generate FL data splits from df_federated_100x.csv")
    parser.add_argument("--federated_data_path", type=str, required=True,
                        help="Path to df_federated_100x.csv (100x-capped training data)")
    parser.add_argument("--site_num", type=int, default=5,
                        help="Number of FL client sites")
    parser.add_argument("--site_name_prefix", type=str, default="site-")
    parser.add_argument("--out_path", type=str, required=True,
                        help="Output directory for data_site-N.json split files")
    parser.add_argument("--processed_dir", type=str, default=None,
                        help="Directory for vehicle_site-N_train.csv and df_server_test.csv "
                             "(defaults to same directory as federated_data_path)")
    parser.add_argument("--test_ratio", type=float, default=0.2,
                        help="Fraction of unique signatures held out for the server test set "
                             "(default 0.2). Split is done BEFORE augmentation to prevent "
                             "data leakage — test signatures never appear in training data.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/test split and StratifiedKFold")
    return parser


def _train_test_split_unique(df: pd.DataFrame, test_ratio: float, seed: int):
    """
    Split unique signatures per class into train/test BEFORE augmentation.

    The full df_federated_100x.csv contains each unique signature repeated up to
    100x. Without this split, the test set would be drawn from the same signatures
    as the training data — the model would simply be evaluated on its own training
    patterns, inflating metrics.

    This function:
      1. Deduplicates df per class to find all unique signatures.
      2. Holds out `test_ratio` of unique signatures per class as the test set.
         Minimum 1 test sample and 1 train sample per class regardless of ratio.
      3. Filters the full (augmented) df to only keep rows whose signature is
         in the train split — these rows form the FL training pool.
      4. Returns df_train (augmented, train signatures only) and
         df_test (deduplicated, held-out signatures only).
    """
    rng = np.random.default_rng(seed)
    train_parts = []
    test_parts  = []

    print(f"Splitting unique signatures into train/test (test_ratio={test_ratio}, seed={seed}):")
    for cls in sorted(df['specific_class'].unique()):
        df_cls   = df[df['specific_class'] == cls]
        df_dedup = df_cls.drop_duplicates(subset=SIGNATURE_COLS).reset_index(drop=True)
        n_unique = len(df_dedup)

        n_test  = max(1, round(n_unique * test_ratio))
        n_test  = min(n_test, n_unique - 1)   # always keep ≥1 train signature
        n_train = n_unique - n_test

        perm       = rng.permutation(n_unique)
        test_sigs  = df_dedup.iloc[perm[:n_test]]
        train_sigs = df_dedup.iloc[perm[n_test:]]

        # Expand test set to ALL raw rows whose signature is in the held-out split.
        # Mirrors real-world deployment: the model sees repeated instances of the
        # same CAN frame pattern, not just one deduplicated row.
        # No leakage: test signatures are disjoint from train signatures.
        keep_test = test_sigs[SIGNATURE_COLS].copy()
        keep_test['_keep'] = True
        df_test_cls = (df_cls
                       .merge(keep_test, on=SIGNATURE_COLS, how='inner')
                       .drop(columns=['_keep']))
        test_parts.append(df_test_cls)
        test_rows = len(df_test_cls)

        # Filter augmented rows to train signatures only (merge on signature columns)
        keep = train_sigs[SIGNATURE_COLS].copy()
        keep['_keep'] = True
        df_train_cls = (df_cls
                        .merge(keep, on=SIGNATURE_COLS, how='inner')
                        .drop(columns=['_keep']))
        train_parts.append(df_train_cls)

        print(f"  {cls:16} | unique: {n_unique:>5}  →  train: {n_train:>4}  test: {n_test:>4}"
              f"  (train rows: {len(df_train_cls):>6}, test rows: {test_rows:>6})")

    df_train = pd.concat(train_parts, ignore_index=True)
    df_test  = pd.concat(test_parts,  ignore_index=True)
    return df_train, df_test


def main():
    parser = data_split_args_parser()
    args = parser.parse_args()

    processed_dir = args.processed_dir or os.path.dirname(os.path.abspath(args.federated_data_path))

    print(f"Loading {args.federated_data_path} ...")
    df = pd.read_csv(args.federated_data_path)
    print(f"  Loaded {len(df):,} rows, columns: {df.columns.tolist()}")

    # ── 1. Held-out test set (unique signatures never seen during training) ──
    df_train, df_test = _train_test_split_unique(df, args.test_ratio, args.seed)
    test_path = os.path.join(processed_dir, "df_server_test.csv")
    df_test.to_csv(test_path, index=False)
    print(f"\nServer test set  ({len(df_test):,} held-out unique signatures) → {test_path}")
    print(f"FL training pool ({len(df_train):,} augmented rows from train signatures)\n")

    # ── 2. Client training shards (StratifiedKFold — IID) ───────────────────
    skf = StratifiedKFold(n_splits=args.site_num, shuffle=True, random_state=args.seed)
    os.makedirs(args.out_path, exist_ok=True)

    print(f"Splitting {len(df_train):,} train rows into {args.site_num} stratified (IID) client shards ...")
    for shard_idx, (_, fold_idx) in enumerate(skf.split(df_train, df_train['specific_class']), 1):
        site_name = f"{args.site_name_prefix}{shard_idx}"
        shard = df_train.iloc[fold_idx].copy()

        csv_name = f"vehicle_{site_name}_train.csv"
        csv_path = os.path.join(processed_dir, csv_name)
        shard.to_csv(csv_path, index=False)

        json_data = {
            "csv_path":      os.path.abspath(csv_path),
            "test_csv_path": os.path.abspath(test_path),
            "site":          site_name,
            "n_rows":        len(shard),
            "test_ratio":    args.test_ratio,
            "class_counts":  shard['specific_class'].value_counts().to_dict()
        }
        json_path = os.path.join(args.out_path, f"data_{site_name}.json")
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=4)

        print(f"  {site_name}: {len(shard):>5} rows → {csv_path}")
        print(f"           classes: { {k: v for k, v in sorted(json_data['class_counts'].items())} }")

    print(f"\nSplit JSON files → {args.out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
