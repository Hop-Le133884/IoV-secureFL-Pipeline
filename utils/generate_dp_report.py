"""
Phase 2 DP Privacy-Utility Tradeoff — Multi-Seed Sweep Runner & Report Generator.

For each epsilon value, runs the full FL pipeline 5 times with different seeds,
averages F1/Accuracy across runs, then saves a CSV report.

Usage:
    python utils/generate_dp_report.py --epsilon "inf; 100; 60; 50; 10; 5; 1"
    python utils/generate_dp_report.py --epsilon "inf; 100; 50" --runs 3
    python utils/generate_dp_report.py --epsilon "inf; 100; 50" --csv_out reports/dp_results.csv
"""

import argparse
import csv
import math
import os
import re
import subprocess

SEEDS = [42, 123, 456, 789, 1000]


def sigma_from_epsilon(epsilon, delta=1e-5, clip=5.0):
    """Gaussian mechanism σ = C · √(2 ln(1.25/δ)) / ε"""
    if epsilon == float("inf"):
        return 0.0
    return clip * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon


def parse_epsilons(raw: str):
    """Parse semicolon-separated epsilon values. Ignores empty tokens and trailing semicolons."""
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
                print(f"  Warning: ε={val} is invalid for the Gaussian mechanism — skipping.")
                continue
            results.append(val)
    return results


def run_single(epsilon, seed):
    """Run the full pipeline for one (epsilon, seed) pair. Returns (macro_f1_pct, accuracy_pct)."""
    env = os.environ.copy()
    env["SEED"] = str(seed)

    if epsilon == float("inf"):
        env.pop("DP_EPSILON", None)
    else:
        env["DP_EPSILON"] = str(epsilon)

    # Step 1: Generate job configs
    r = subprocess.run(["bash", "jobs_gen.sh", "./data"], env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"jobs_gen.sh failed (ε={epsilon}, seed={seed}):\n{r.stderr}")

    # Step 2: NVFlare simulation
    r = subprocess.run(["bash", "run_experiment_simulator.sh"], env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"run_experiment_simulator.sh failed (ε={epsilon}, seed={seed}):\n{r.stderr}")

    # Step 3: Model validation — capture output for parsing
    r = subprocess.run(["python", "utils/model_validation.py"], env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"model_validation.py failed (ε={epsilon}, seed={seed}):\n{r.stderr}")

    f1_match  = re.search(r"Macro F1-Score:\s+([\d.]+)%", r.stdout)
    acc_match = re.search(r"Overall Accuracy:\s+([\d.]+)%", r.stdout)
    if not f1_match or not acc_match:
        raise RuntimeError(f"Could not parse F1/Accuracy (ε={epsilon}, seed={seed})")

    return float(f1_match.group(1)), float(acc_match.group(1))


def run_epsilon(epsilon, seeds):
    """Run all seeds for one epsilon. Returns list of (f1, acc) per seed."""
    sigma = sigma_from_epsilon(epsilon)
    label = "∞ (no DP)" if epsilon == float("inf") else str(epsilon)

    print(f"\n{'='*62}")
    print(f"  ε = {label}  |  σ = {sigma:.4f}  |  {len(seeds)} runs")
    print(f"{'='*62}")

    per_seed = []
    for i, seed in enumerate(seeds, 1):
        print(f"  Run {i}/{len(seeds)}  seed={seed} ...", flush=True)
        f1, acc = run_single(epsilon, seed)
        per_seed.append((seed, f1, acc))
        print(f"    F1={f1:.2f}%  Acc={acc:.2f}%")

    return per_seed


def compute_stats(values):
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    std = variance ** 0.5
    return mean, std


def print_table(rows):
    """rows: list of (epsilon, sigma, mean_f1, std_f1, mean_acc, std_acc)"""
    header = (f"{'ε':>12}  {'σ':>6}  "
              f"{'F1 mean':>9}  {'F1 std':>7}  "
              f"{'Acc mean':>9}  {'Acc std':>7}")
    sep = "-" * len(header)
    print(sep)
    print("  Phase 2 — DP Privacy-Utility Tradeoff  (averaged over 5 seeds)")
    print(sep)
    print(header)
    print(sep)
    for eps, sigma, mf1, sf1, macc, sacc in rows:
        eps_str = "∞ (no DP)" if eps == float("inf") else str(eps)
        print(f"{eps_str:>12}  {sigma:>6.2f}  "
              f"{mf1:>8.2f}%  {sf1:>6.2f}%  "
              f"{macc:>8.2f}%  {sacc:>6.2f}%")
    print(sep)


def save_csv(rows, all_runs, csv_path):
    """Save detailed results (per-seed + averaged) to CSV."""
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    n_seeds = len(SEEDS)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        seed_f1_cols  = [f"f1_seed_{s}"  for s in SEEDS[:n_seeds]]
        seed_acc_cols = [f"acc_seed_{s}" for s in SEEDS[:n_seeds]]
        writer.writerow(
            ["epsilon", "sigma"]
            + seed_f1_cols + ["f1_mean", "f1_std"]
            + seed_acc_cols + ["acc_mean", "acc_std"]
        )

        for (eps, sigma, mf1, sf1, macc, sacc), per_seed in zip(rows, all_runs):
            eps_str = "inf" if eps == float("inf") else str(eps)
            f1_vals  = [r[1] for r in per_seed]
            acc_vals = [r[2] for r in per_seed]
            writer.writerow(
                [eps_str, f"{sigma:.4f}"]
                + [f"{v:.4f}" for v in f1_vals]  + [f"{mf1:.4f}", f"{sf1:.4f}"]
                + [f"{v:.4f}" for v in acc_vals] + [f"{macc:.4f}", f"{sacc:.4f}"]
            )

    print(f"CSV report saved → {csv_path}")


def print_summary(rows):
    baseline = next(((eps, sigma, mf1, sf1, macc, sacc)
                     for eps, sigma, mf1, sf1, macc, sacc in rows
                     if eps == float("inf")), None)
    dp = [(eps, sigma, mf1, sf1, macc, sacc)
          for eps, sigma, mf1, sf1, macc, sacc in rows
          if eps != float("inf")]

    print("\n── Phase 2 Summary ──────────────────────────────────────")
    if baseline:
        baseline_f1 = baseline[2]
        print(f"  No-DP baseline:    Macro F1 = {baseline_f1:.2f}% ± {baseline[3]:.2f}%")
        if dp:
            best = max(dp, key=lambda r: r[2])
            usable = [r for r in dp if r[2] >= 0.95 * baseline_f1]
            strongest = max(usable, key=lambda r: 1 / r[0]) if usable else None
            print(f"  Best DP result:    ε = {best[0]}  "
                  f"(σ={best[1]:.2f}, F1={best[2]:.2f}% ± {best[3]:.2f}%,  "
                  f"Δ={best[2]-baseline_f1:+.2f}%)")
            if strongest:
                print(f"  Strongest ε ≥95%:  ε = {strongest[0]}  "
                      f"(F1={strongest[2]:.2f}% ± {strongest[3]:.2f}%)")
    print("─────────────────────────────────────────────────────────\n")


def main():
    parser = argparse.ArgumentParser(description="Run multi-seed DP sweep and generate Phase 2 report")
    parser.add_argument(
        "--epsilon", type=str, required=True,
        help="Semicolon-separated ε values. Use 'inf' for no-DP baseline. "
             "Example: \"inf; 100; 50; 10; 5; 1\""
    )
    parser.add_argument(
        "--runs", type=int, default=5,
        help="Number of seeds per epsilon (default: 5, max: %(default)s seeds available)"
    )
    parser.add_argument(
        "--csv_out", type=str, default="reports/dp_tradeoff.csv",
        help="Output path for the CSV report (default: reports/dp_tradeoff.csv)"
    )
    args = parser.parse_args()

    epsilons = parse_epsilons(args.epsilon)
    if not epsilons:
        parser.error("No valid epsilon values provided.")

    seeds = SEEDS[:args.runs]
    total_runs = len(epsilons) * len(seeds)
    print(f"DP sweep: {len(epsilons)} epsilon(s) × {len(seeds)} seeds = {total_runs} total runs")
    print(f"Seeds: {seeds}")
    print("Pipeline per run: jobs_gen.sh → run_experiment_simulator.sh → model_validation.py\n")

    summary_rows = []
    all_runs = []

    for eps in epsilons:
        sigma = sigma_from_epsilon(eps)
        per_seed = run_epsilon(eps, seeds)
        f1_vals  = [r[1] for r in per_seed]
        acc_vals = [r[2] for r in per_seed]
        mf1,  sf1  = compute_stats(f1_vals)
        macc, sacc = compute_stats(acc_vals)
        summary_rows.append((eps, sigma, mf1, sf1, macc, sacc))
        all_runs.append(per_seed)

    print("\n\n")
    print_table(summary_rows)
    print_summary(summary_rows)
    save_csv(summary_rows, all_runs, args.csv_out)


if __name__ == "__main__":
    main()
