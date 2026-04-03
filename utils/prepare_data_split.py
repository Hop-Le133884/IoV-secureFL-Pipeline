import argparse
import json
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

SIGNATURE_COLS = ['ID', 'DATA_0', 'DATA_1', 'DATA_2', 'DATA_3', 'DATA_4', 'DATA_5', 'DATA_6', 'DATA_7']

def data_split_args_parser():
    parser = argparse.ArgumentParser(description="Generate FL data splits from df_federated_5x.csv")
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
    return parser


def _make_server_test_set(df: pd.DataFrame) -> pd.DataFrame:
    """Strict per-class deduplication — mirrors the paper's unique-signature benchmark."""
    classes = df['specific_class'].unique()
    parts = []
    print("Building server test set (strict dedup per class):")
    for cls in sorted(classes):
        df_cls = df[df['specific_class'] == cls]
        df_dedup = df_cls.drop_duplicates(subset=SIGNATURE_COLS)
        print(f"  {cls:16} | 5x-capped: {len(df_cls):>5} → unique: {len(df_dedup):>5}")
        parts.append(df_dedup)
    return pd.concat(parts, ignore_index=True)


def main():
    parser = data_split_args_parser()
    args = parser.parse_args()

    processed_dir = args.processed_dir or os.path.dirname(os.path.abspath(args.federated_data_path))

    print(f"Loading {args.federated_data_path} ...")
    df = pd.read_csv(args.federated_data_path)
    print(f"  Loaded {len(df):,} rows, columns: {df.columns.tolist()}")

    # ── 1. Server test set (unique signatures) ──────────────────────────────
    df_test = _make_server_test_set(df)
    test_path = os.path.join(processed_dir, "df_server_test.csv")
    df_test.to_csv(test_path, index=False)
    print(f"\nServer test set ({len(df_test):,} unique signatures) → {test_path}")

    # ── 2. Client training shards (StratifiedKFold — IID) ───────────────────
    skf = StratifiedKFold(n_splits=args.site_num, shuffle=True, random_state=42)
    os.makedirs(args.out_path, exist_ok=True)

    print(f"\nSplitting {len(df):,} rows into {args.site_num} stratified (IID) client shards ...")
    for shard_idx, (_, fold_idx) in enumerate(skf.split(df, df['specific_class']), 1):
        site_name = f"{args.site_name_prefix}{shard_idx}"
        shard = df.iloc[fold_idx].copy()

        csv_name = f"vehicle_{site_name}_train.csv"
        csv_path = os.path.join(processed_dir, csv_name)
        shard.to_csv(csv_path, index=False)

        json_data = {
            "csv_path": os.path.abspath(csv_path),
            "test_csv_path": os.path.abspath(test_path),
            "site": site_name,
            "n_rows": len(shard),
            "class_counts": shard['specific_class'].value_counts().to_dict()
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
