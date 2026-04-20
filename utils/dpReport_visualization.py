"""
DP Tradeoff Visualization — plots mean F1 and Accuracy per epsilon from dp_tradeoff.csv.

Usage:
    python utils/dpReport_visualization.py
    python utils/dpReport_visualization.py --csv reports/dp_tradeoff.csv --out reports/dp_tradeoff.png
"""

import argparse
import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def load_csv(csv_path):
    """Returns list of (epsilon_float, sigma, mean_f1, std_f1, mean_acc, std_acc)."""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items()}
            eps = float("inf") if row["epsilon"].lower() == "inf" else float(row["epsilon"])
            rows.append((
                eps,
                float(row["sigma"]),
                float(row["f1_mean"]),
                float(row["f1_std"]),
                float(row["acc_mean"]),
                float(row["acc_std"]),
            ))
    # Sort by epsilon descending so x-axis goes high→low (weak→strong privacy)
    rows.sort(key=lambda r: r[0] if r[0] != float("inf") else 1e18, reverse=True)
    return rows


def make_plot(rows, out_path):
    baseline = next((r for r in rows if r[0] == float("inf")), None)
    dp_rows  = [r for r in rows if r[0] != float("inf")]

    epsilons  = [r[0] for r in dp_rows]
    sigmas    = [r[1] for r in dp_rows]
    mean_f1s  = [r[2] for r in dp_rows]
    std_f1s   = [r[3] for r in dp_rows]
    mean_accs = [r[4] for r in dp_rows]
    std_accs  = [r[5] for r in dp_rows]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Phase 2 — Federated Double RF with Differential Privacy\n"
        "Privacy-Utility Tradeoff (Gaussian Mechanism on XGBoost Leaf Values)",
        fontsize=12, y=1.01,
    )

    # ── Left: Mean F1 & Accuracy vs ε (log scale, inverted) ─────────────────
    ax = axes[0]
    if baseline:
        ax.axhline(baseline[2], color="grey", linestyle="--", linewidth=1,
                   label=f"No-DP baseline  F1={baseline[2]:.2f}% ± {baseline[3]:.2f}%")

    ax.errorbar(epsilons, mean_f1s, yerr=std_f1s,
                fmt="o-", color="steelblue", linewidth=2, markersize=7,
                capsize=4, label="Macro F1 (mean ± std)")
    ax.errorbar(epsilons, mean_accs, yerr=std_accs,
                fmt="s--", color="tomato", linewidth=1.5, markersize=5,
                capsize=4, label="Accuracy (mean ± std)")

    for eps, f1 in zip(epsilons, mean_f1s):
        ax.annotate(f"ε={eps}", (eps, f1),
                    textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8, color="steelblue")

    ax.set_xscale("log")
    ax.set_xlabel("Privacy Budget ε  (← stronger privacy | weaker privacy →)", fontsize=10)
    ax.set_ylabel("Score (%)", fontsize=10)
    ax.set_title("Utility vs Privacy Budget (ε)", fontsize=11)
    ax.set_ylim(-5, 110)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    # ── Right: Mean F1 vs σ ──────────────────────────────────────────────────
    ax2 = axes[1]
    if baseline:
        ax2.axhline(baseline[2], color="grey", linestyle="--", linewidth=1,
                    label=f"No-DP baseline ({baseline[2]:.2f}%)")

    ax2.errorbar(sigmas, mean_f1s, yerr=std_f1s,
                 fmt="o-", color="steelblue", linewidth=2, markersize=7,
                 capsize=4, label="Macro F1 (mean ± std)")

    for sigma, eps, f1 in zip(sigmas, epsilons, mean_f1s):
        ax2.annotate(f"ε={eps}\nσ={sigma:.2f}", (sigma, f1),
                     textcoords="offset points", xytext=(6, 0),
                     fontsize=7.5, color="steelblue", va="center")

    ax2.set_xlabel("Noise σ (Gaussian std applied to leaf values)", fontsize=10)
    ax2.set_ylabel("Macro F1 (%)", fontsize=10)
    ax2.set_title("Utility vs Noise Level (σ)", fontsize=11)
    ax2.set_ylim(-5, 110)
    ax2.legend(fontsize=8)
    ax2.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot DP tradeoff from CSV report")
    parser.add_argument("--csv", type=str, default="DP_SEED_report/dp_tradeoff.csv",
                        help="Path to dp_tradeoff.csv (default: DP_SEED_report/dp_tradeoff.csv)")
    parser.add_argument("--out", type=str, default="DP_SEED_report/dp_tradeoff.png",
                        help="Output image path (default: DP_SEED_report/dp_tradeoff.png)")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: {args.csv} not found.")
        raise SystemExit(1)

    rows = load_csv(args.csv)
    make_plot(rows, args.out)


if __name__ == "__main__":
    main()
