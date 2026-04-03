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
ALPHA_BENIGN = 2.0
ALPHA_ATTACK = 0.1


def data_split_args_parser():
    parser = argparse.ArgumentParser(description="Generate non-IID FL data splits from df_federated_5x.csv")
    parser.add_argument("--federated_data_path", type=str, required=True,
                        help="Path to df_federated_5x.csv (5x-capped training data)")
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
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    return parser


def _make_server_test_set(df: pd.DataFrame) -> pd.DataFrame:
    """Strict per-class deduplication — mirrors the paper's unique-signature benchmark."""
    classes = df['specific_class'].unique()
    parts = []
    print("Building server test set (strict dedup per class):")
    for cls in sorted(classes):
        df_cls   = df[df['specific_class'] == cls]
        df_dedup = df_cls.drop_duplicates(subset=SIGNATURE_COLS)
        print(f"  {cls:16} | 5x-capped: {len(df_cls):>5} → unique: {len(df_dedup):>5}")
        parts.append(df_dedup)
    return pd.concat(parts, ignore_index=True)


def dirichlet_noniid_split(df, n_sites, alpha_benign, alpha_attack, seed):
    """
    Non-IID split using class-specific Dirichlet concentration.

    BENIGN uses alpha_benign (2.0) — high alpha keeps BENIGN roughly balanced
    across sites, avoiding pathological splits (e.g. one site getting 16 rows).

    Attack classes use alpha_attack (0.1) — low alpha makes each site specialise
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

    # ── 1. Server test set (unique signatures) ──────────────────────────────
    df_test   = _make_server_test_set(df)
    test_path = os.path.join(processed_dir, "df_server_test.csv")
    df_test.to_csv(test_path, index=False)
    print(f"\nServer test set ({len(df_test):,} unique signatures) → {test_path}")

    # ── 2. Non-IID Dirichlet client shards ──────────────────────────────────
    classes      = sorted(df['specific_class'].unique())
    site_indices = dirichlet_noniid_split(
        df, args.site_num, args.alpha_benign, args.alpha_attack, args.seed
    )
    site_dfs = [df.loc[idx].reset_index(drop=True) for idx in site_indices]

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
