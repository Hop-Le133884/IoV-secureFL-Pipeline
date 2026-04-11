import argparse
import json
import os
import pathlib
import shutil
from nvflare.apis.fl_constant import JobConstants

JOB_CONFIGS_ROOT = "jobs"
BASE_FOLDER = "random_forest_base"

def job_config_args_parser():
    parser = argparse.ArgumentParser(description="Generate Double RF configs for IoV dataset")
    parser.add_argument("--data_split_root", type=str, required=True, help="Path to stratified data splits")
    parser.add_argument("--site_num", type=int, default=5, help="Total number of vehicles")
    parser.add_argument("--site_name_prefix", type=str, default="site-", help="Site name prefix")
    parser.add_argument("--num_local_parallel_tree", type=int, default=20, help="Trees per stage per site")
    parser.add_argument("--max_depth", type=int, default=8, help="Maximum depth of a tree")
    parser.add_argument("--local_subsample", type=float, default=0.8, help="Row subsample rate per tree (RF bagging)")
    parser.add_argument("--colsample_bynode", type=float, default=0.8, help="Feature subsample rate per split (RF feature randomness)")
    parser.add_argument("--nthread", type=int, default=4, help="nthread for xgboost")
    parser.add_argument("--dp_epsilon", type=float, default=None, help="DP privacy budget ε (None = no DP)")
    parser.add_argument("--dp_delta", type=float, default=1e-5, help="DP failure probability δ")
    parser.add_argument("--dp_clip_bound", type=float, default=5.0, help="Leaf value clipping bound C (L∞ sensitivity)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for XGBoost training and DP noise")
    parser.add_argument("--job_name", type=str, default=None, help="Override auto-generated job name")
    return parser

def _read_json(filename):
    with open(filename, "r") as f:
        return json.load(f)

def _write_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

def _get_job_name(args) -> str:
    return f"iov_double_rf_{args.site_num}_sites"

def _gen_deploy_map(num_sites: int, site_name_prefix: str) -> dict:
    deploy_map = {"app_server": ["server"]}
    for i in range(1, num_sites + 1):
        deploy_map[f"app_{site_name_prefix}{i}"] = [f"{site_name_prefix}{i}"]
    return deploy_map

def _update_server_config(config: dict, args):
    config["min_clients"] = args.site_num
    for wf in config["workflows"]:
        wf["args"]["min_clients"] = args.site_num

def _update_client_config(config: dict, args, site_name: str):
    exec_args = config["executors"][0]["executor"]["args"]
    exec_args["num_local_parallel_tree"] = args.num_local_parallel_tree
    exec_args["max_depth"] = args.max_depth
    exec_args["local_subsample"] = args.local_subsample
    exec_args["colsample_bynode"] = args.colsample_bynode
    exec_args["nthread"] = args.nthread
    exec_args["dp_epsilon"] = args.dp_epsilon
    exec_args["dp_delta"] = args.dp_delta
    exec_args["dp_clip_bound"] = args.dp_clip_bound
    exec_args["seed"] = args.seed

    data_split_path = os.path.join(args.data_split_root, f"data_{site_name}.json")
    config["components"][0]["args"] = {"data_split_filename": data_split_path}

def main():
    parser = job_config_args_parser()
    args = parser.parse_args()
    job_name = args.job_name if args.job_name else _get_job_name(args)
    
    src_job_path = pathlib.Path(JOB_CONFIGS_ROOT) / BASE_FOLDER
    src_custom_path = src_job_path / "app" / "custom"
    
    dst_job_path = pathlib.Path(JOB_CONFIGS_ROOT) / job_name
    os.makedirs(dst_job_path, exist_ok=True)

    meta = _read_json(src_job_path / JobConstants.META_FILE)
    meta["name"] = job_name
    meta["deploy_map"] = _gen_deploy_map(args.site_num, args.site_name_prefix)
    _write_json(meta, dst_job_path / JobConstants.META_FILE)

    os.makedirs(dst_job_path / "app_server" / "config", exist_ok=True)
    os.makedirs(dst_job_path / "app_server" / "custom", exist_ok=True)
    server_cfg = _read_json(src_job_path / "app" / "config" / JobConstants.SERVER_JOB_CONFIG)
    _update_server_config(server_cfg, args)
    _write_json(server_cfg, dst_job_path / "app_server" / "config" / JobConstants.SERVER_JOB_CONFIG)
    # Copy server-side controller script
    shutil.copy(src_custom_path / "broadcast_inner_model.py", dst_job_path / "app_server" / "custom" / "broadcast_inner_model.py")

    for i in range(1, args.site_num + 1):
        site_name = f"{args.site_name_prefix}{i}"
        app_dir = dst_job_path / f"app_{site_name}"
        os.makedirs(app_dir / "config", exist_ok=True)
        os.makedirs(app_dir / "custom", exist_ok=True)
        
        client_cfg = _read_json(src_job_path / "app" / "config" / JobConstants.CLIENT_JOB_CONFIG)
        _update_client_config(client_cfg, args, site_name)
        _write_json(client_cfg, app_dir / "config" / JobConstants.CLIENT_JOB_CONFIG)
        
        for script in ["iov_data_loader.py", "iov_executor.py"]:
            src_file = src_custom_path / script
            if src_file.exists():
                shutil.copy(src_file, app_dir / "custom" / script)

if __name__ == "__main__":
    main()