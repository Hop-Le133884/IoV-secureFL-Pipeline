import argparse
import json
import os
import pandas as pd
import numpy as np

SIGNATURE_COLS = ['ID', 'DATA_0', 'DATA_1', 'DATA_2', 'DATA_3', 'DATA_4', 'DATA_5', 'DATA_6', 'DATA_7']
BENIGN_CLASS   = 'BENIGN'

# Non-IID Dirichlet concentration parameters:
#   BENIGN (majority, 96% of data): higher alpha → more balanced across sites
#   Attack classes:                 lower alpha  → heterogeneous (sites miss whole classes)
ALPHA_BENIGN = 15.0
ALPHA_ATTACK = 0.2


def data_split_args_parser():
    parser = argparse.ArgumentParser(description="Generate non-IID FL data splits from df_federated_100x.csv")
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
    parser.add_argument("--alpha_benign", type=float, default=ALPHA_BENIGN,
                        help=f"Dirichlet alpha for BENIGN class (default {ALPHA_BENIGN})")
    parser.add_argument("--alpha_attack", type=float, default=ALPHA_ATTACK,
                        help=f"Dirichlet alpha for attack classes (default {ALPHA_ATTACK})")
    parser.add_argument("--test_ratio", type=float, default=0.2,
                        help="Fraction of unique signatures held out for the server test set "
                             "(default 0.2). Split is done BEFORE augmentation to prevent "
                             "data leakage — test signatures never appear in training data.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
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
         df_test (all raw rows from held-out signatures — mirrors real-world
         deployment where the same CAN frame pattern appears multiple times).
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

        # Filter augmented rows to train signatures only (merge on signature columns)
        keep = train_sigs[SIGNATURE_COLS].copy()
        keep['_keep'] = True
        df_train_cls = (df_cls
                        .merge(keep, on=SIGNATURE_COLS, how='inner')
                        .drop(columns=['_keep']))
        train_parts.append(df_train_cls)

        print(f"  {cls:16} | unique: {n_unique:>5}  →  train: {n_train:>4}  test: {n_test:>4}"
              f"  (train rows: {len(df_train_cls):>6}, test rows: {len(df_test_cls):>6})")

    df_train = pd.concat(train_parts, ignore_index=True)
    df_test  = pd.concat(test_parts,  ignore_index=True)
    return df_train, df_test


def dirichlet_noniid_split(df, n_sites, alpha_benign, alpha_attack, seed):
    """
    Non-IID split using class-specific Dirichlet concentration.

    BENIGN uses alpha_benign (15.0) — high alpha keeps BENIGN roughly balanced
    across sites, avoiding pathological splits (e.g. one site getting 16 rows).

    Attack classes use alpha_attack (0.5) — low alpha makes each site specialise
    in a subset of attack types, mimicking real vehicles on different routes that
    encounter different attack surfaces. Many sites will have zero samples of
    certain minority attack classes, which is intentional and realistic.
    """
    rng     = np.random.default_rng(seed)
    classes = sorted(df['specific_class'].unique())
    site_indices = [[] for _ in range(n_sites)]

    for cls in classes:
        alpha   = alpha_benign if cls == BENIGN_CLASS else alpha_attack
        cls_idx = df[df['specific_class'] == cls].index.tolist()
        rng.shuffle(cls_idx)

        proportions = rng.dirichlet(alpha * np.ones(n_sites))
        splits      = (np.cumsum(proportions) * len(cls_idx)).astype(int)
        splits      = np.minimum(splits, len(cls_idx))

        prev = 0
        for i, end in enumerate(splits):
            site_indices[i].extend(cls_idx[prev:end])
            prev = end

    return site_indices


def print_split_stats(site_dfs, classes, alpha_benign, alpha_attack):
    attack_classes = [c for c in classes if c != BENIGN_CLASS]
    print(f"\nNon-IID split  (BENIGN α={alpha_benign}, attack α={alpha_attack})")
    print(f"{'Site':<10}", end="")
    for cls in classes:
        print(f"{cls:>16}", end="")
    print(f"{'Total':>8}   {'Benign%':>8}   Missing attack classes")
    print("-" * (10 + 16 * len(classes) + 50))
    for i, sdf in enumerate(site_dfs):
        benign_n   = len(sdf[sdf['specific_class'] == BENIGN_CLASS])
        benign_pct = benign_n / len(sdf) * 100 if len(sdf) > 0 else 0
        missing    = [c for c in attack_classes if len(sdf[sdf['specific_class'] == c]) == 0]
        print(f"site-{i+1:<5}", end="")
        for cls in classes:
            print(f"{len(sdf[sdf['specific_class']==cls]):>16}", end="")
        print(f"{len(sdf):>8}   {benign_pct:>7.1f}%   {missing if missing else '-'}")
    print()


def main():
    parser = data_split_args_parser()
    args   = parser.parse_args()

    processed_dir = args.processed_dir or os.path.dirname(os.path.abspath(args.federated_data_path))

    print(f"Loading {args.federated_data_path} ...")
    df = pd.read_csv(args.federated_data_path)
    print(f"  Loaded {len(df):,} rows, columns: {df.columns.tolist()}")

    # ── 1. Held-out test set (unique signatures never seen during training) ──
    df_train, df_test = _train_test_split_unique(df, args.test_ratio, args.seed)
    test_path = os.path.join(processed_dir, "df_server_test.csv")
    df_test.to_csv(test_path, index=False)
    print(f"\nServer test set  ({len(df_test):,} rows from held-out signatures) → {test_path}")
    print(f"FL training pool ({len(df_train):,} augmented rows from train signatures)\n")

    # ── 2. Non-IID Dirichlet client shards (from train pool only) ───────────
    classes      = sorted(df_train['specific_class'].unique())
    site_indices = dirichlet_noniid_split(
        df_train, args.site_num, args.alpha_benign, args.alpha_attack, args.seed
    )
    site_dfs = [df_train.loc[idx].reset_index(drop=True) for idx in site_indices]

    print_split_stats(site_dfs, classes, args.alpha_benign, args.alpha_attack)

    os.makedirs(args.out_path, exist_ok=True)
    for i, sdf in enumerate(site_dfs):
        site_name = f"{args.site_name_prefix}{i + 1}"

        csv_name = f"vehicle_{site_name}_train.csv"
        csv_path = os.path.join(processed_dir, csv_name)
        sdf.to_csv(csv_path, index=False)

        json_data = {
            "csv_path":      os.path.abspath(csv_path),
            "test_csv_path": os.path.abspath(test_path),
            "site":          site_name,
            "n_rows":        len(sdf),
            "alpha_benign":  args.alpha_benign,
            "alpha_attack":  args.alpha_attack,
            "test_ratio":    args.test_ratio,
            "class_counts":  sdf['specific_class'].value_counts().to_dict()
        }
        json_path = os.path.join(args.out_path, f"data_{site_name}.json")
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=4)

        print(f"  {site_name}: {len(sdf):>5} rows → {csv_path}")
        print(f"           classes: { {k: v for k, v in sorted(json_data['class_counts'].items())} }")

    print(f"\nSplit JSON files → {args.out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
