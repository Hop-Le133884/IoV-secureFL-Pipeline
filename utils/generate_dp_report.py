"""
Phase 2 DP Privacy-Utility Tradeoff — Sequential Sweep Runner.

Runs the full FL pipeline for each epsilon × seed combination sequentially,
prints all output live, then saves averaged results to CSV.

Usage:
    python utils/generate_dp_report.py --epsilon "inf; 100; 80; 70; 50; 20; 1"
    python utils/generate_dp_report.py --epsilon "inf; 100" --seeds "42,456,1000"
    python utils/generate_dp_report.py --epsilon "inf; 100" --csv_out reports/dp_tradeoff.csv
"""

import argparse
import csv
import math
import os
import re
import shutil
import subprocess

JOB_NAME  = "iov_double_rf_5_sites"
WORKSPACE = "workspace_iov_double_rf"
SEEDS_DEFAULT = [42, 456, 1000]


def sigma_from_epsilon(epsilon, delta=1e-5, clip=5.0):
    if epsilon == float("inf"):
        return 0.0
    return clip * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon


def parse_epsilons(raw):
    results = []
    for part in raw.strip().rstrip(";").split(";"):
        part = part.strip()
        if not part:
            continue
        if part.lower() in ("inf", "∞"):
            results.append(float("inf"))
        else:
            val = float(part)
            if val <= 0:
                print(f"Warning: ε={val} invalid — skipping.")
                continue
            results.append(val)
    return results


def cleanup():
    for path in [WORKSPACE, f"jobs/{JOB_NAME}"]:
        if os.path.exists(path):
            shutil.rmtree(path)


def run_one(epsilon, seed):
    """Run full pipeline for one (epsilon, seed). Returns (f1_pct, acc_pct)."""
    env = os.environ.copy()
    env["SEED"] = str(seed)
    if epsilon == float("inf"):
        env.pop("DP_EPSILON", None)
    else:
        env["DP_EPSILON"] = str(epsilon)

    # Step 1: jobs_gen — output goes directly to terminal
    print("\n--- Step 1/3: jobs_gen.sh ---")
    r = subprocess.run(["bash", "jobs_gen.sh", "./data"], env=env)
    if r.returncode != 0:
        raise RuntimeError("jobs_gen.sh failed")

    # Step 2: NVFlare simulation — output goes directly to terminal
    print("\n--- Step 2/3: run_experiment_simulator.sh ---")
    r = subprocess.run(["bash", "run_experiment_simulator.sh"], env=env)
    if r.returncode != 0:
        cleanup()
        raise RuntimeError("run_experiment_simulator.sh failed")

    # Step 3: Validation — capture to parse F1/Acc, then print
    print("\n--- Step 3/3: model_validation.py ---")
    r = subprocess.run(
        ["python", "utils/model_validation.py"], env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    print(r.stdout)

    cleanup()

    if r.returncode != 0:
        raise RuntimeError("model_validation.py failed")

    f1_match  = re.search(r"Macro F1-Score:\s+([\d.]+)%", r.stdout)
    acc_match = re.search(r"Overall Accuracy:\s+([\d.]+)%", r.stdout)
    if not f1_match or not acc_match:
        raise RuntimeError("Could not parse F1/Accuracy from validation output")

    return float(f1_match.group(1)), float(acc_match.group(1))


def compute_stats(values):
    n    = len(values)
    mean = sum(values) / n
    std  = (sum((v - mean) ** 2 for v in values) / n) ** 0.5
    return mean, std


def save_csv(all_results, seeds, csv_path):
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epsilon", "sigma"]
            + [f"f1_seed_{s}"  for s in seeds] + ["f1_mean",  "f1_std"]
            + [f"acc_seed_{s}" for s in seeds] + ["acc_mean", "acc_std"]
        )
        for eps, runs in all_results:
            f1_vals  = [r[1] for r in runs]
            acc_vals = [r[2] for r in runs]
            valid_f1  = [v for v in f1_vals  if not math.isnan(v)]
            valid_acc = [v for v in acc_vals if not math.isnan(v)]
            mf1,  sf1  = compute_stats(valid_f1)  if valid_f1  else (float("nan"), float("nan"))
            macc, sacc = compute_stats(valid_acc) if valid_acc else (float("nan"), float("nan"))
            eps_str = "inf" if eps == float("inf") else str(eps)
            writer.writerow(
                [eps_str, f"{sigma_from_epsilon(eps):.4f}"]
                + [f"{v:.4f}" if not math.isnan(v) else "FAIL" for v in f1_vals]
                + [f"{mf1:.4f}", f"{sf1:.4f}"]
                + [f"{v:.4f}" if not math.isnan(v) else "FAIL" for v in acc_vals]
                + [f"{macc:.4f}", f"{sacc:.4f}"]
            )
    print(f"\nCSV saved → {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Run DP sweep and save CSV report")
    parser.add_argument("--epsilon", type=str, required=True,
                        help="Semicolon-separated ε values. E.g. \"inf; 100; 50; 1\"")
    parser.add_argument("--seeds", type=str,
                        default=",".join(str(s) for s in SEEDS_DEFAULT),
                        help="Comma-separated seeds (default: 42,456,1000)")
    parser.add_argument("--csv_out", type=str, default="reports/dp_tradeoff.csv",
                        help="Output CSV path (default: reports/dp_tradeoff.csv)")
    args = parser.parse_args()

    epsilons = parse_epsilons(args.epsilon)
    seeds    = [int(s.strip()) for s in args.seeds.split(",")]
    total    = len(epsilons) * len(seeds)

    if not epsilons:
        parser.error("No valid epsilon values provided.")

    print(f"DP sweep: {len(epsilons)} epsilon(s) × {len(seeds)} seed(s) = {total} sequential runs")
    print(f"Seeds: {seeds}")

    all_results = []
    run = 0
    for eps in epsilons:
        runs = []
        for seed in seeds:
            run += 1
            label = "∞" if eps == float("inf") else str(eps)
            print(f"\n{'='*62}")
            print(f"  [{run}/{total}]  ε={label}  seed={seed}  σ={sigma_from_epsilon(eps):.4f}")
            print(f"{'='*62}")
            try:
                f1, acc = run_one(eps, seed)
                runs.append((seed, f1, acc))
                print(f"\n  → F1={f1:.2f}%  Acc={acc:.2f}%")
            except RuntimeError as e:
                print(f"  [FAIL] {e}")
                runs.append((seed, float("nan"), float("nan")))
        all_results.append((eps, runs))

    save_csv(all_results, seeds, args.csv_out)


if __name__ == "__main__":
    main()
