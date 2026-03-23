import argparse
import json
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

def data_split_args_parser():
    parser = argparse.ArgumentParser(description="Generate Stratified Data Split for IoV Dataset")
    parser.add_argument("--data_path", type=str, required=True, help="Path to CICIoV2024.csv")
    parser.add_argument("--site_num", type=int, required=True, help="Total number of sites (vehicles)")
    parser.add_argument("--site_name_prefix", type=str, default="site-", help="Site name prefix")
    parser.add_argument("--size_valid", type=int, default=140822, help="Validation size")
    parser.add_argument("--target_col", type=str, default="specific_class", help="Column to stratify on")
    parser.add_argument("--out_path", type=str, required=True, help="Output path for the data split json files")
    parser.add_argument("--data_split_root", type=str, default="./data/IoV/data_splits", help="Path to data splits")
    return parser

def main():
    parser = data_split_args_parser()
    args = parser.parse_args()

    print(f"Loading dataset labels from {args.data_path}...")
    df_labels = pd.read_csv(args.data_path, usecols=[args.target_col])
    
    print("Performing Stratified Validation Split...")
    # 1. Stratify 10% into validation, 90% into training
    indices = np.arange(len(df_labels))
    train_idx, valid_idx = train_test_split(
        indices, 
        test_size=args.size_valid, 
        stratify=df_labels[args.target_col], 
        random_state=42
    )
    
    valid_indices = valid_idx.tolist()
    train_df_labels = df_labels.iloc[train_idx]
    
    # 2. Stratify the remaining training data across the 5 sites
    skf = StratifiedKFold(n_splits=args.site_num, shuffle=True, random_state=42)
    print(f"Performing stratified split for {args.site_num} sites...")
    
    base_json_data = {
        "data_path": args.data_path,
        "data_index": {
            "valid": {"indices": valid_indices}
        }
    }

    for site, (_, fold_idx) in enumerate(skf.split(train_df_labels, train_df_labels[args.target_col])):
        site_id = f"{args.site_name_prefix}{site + 1}"
        
        # Map the fold indices back to the original absolute CSV row numbers
        actual_indices = train_idx[fold_idx].tolist()
        
        site_json = base_json_data.copy()
        site_json["data_index"][site_id] = {"indices": actual_indices}
        
        os.makedirs(args.out_path, exist_ok=True)
        output_file = os.path.join(args.out_path, f"data_{site_id}.json")
        with open(output_file, "w") as f:
            json.dump(site_json, f, indent=4)
        
        print(f"  > Generated {output_file} with {len(actual_indices)} stratified samples.")

if __name__ == "__main__":
    main()