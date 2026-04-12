"""
Bar chart: compare F1 and Accuracy across 10 seeds in dp_tradeoff.csv.
Also appends an 'average' column (mean of f1_mean and acc_mean) to the CSV.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

HERE   = Path(__file__).parent
CSV    = HERE / "../DP_SEED_report/seeds_comparing.csv"
OUTPNG = HERE / "../DP_SEED_report/seeds_comparing.png"

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV)
row = df.iloc[0]

SEEDS = [42, 123, 456, 789, 1234, 2025, 99, 7, 314, 1000]
f1s   = [row[f"f1_seed_{s}"]  for s in SEEDS]
accs  = [row[f"acc_seed_{s}"] for s in SEEDS]

f1_avg  = row["f1_mean"]
acc_avg = row["acc_mean"]

# ── Add 'average' column to CSV if not already present ────────────────────────
if "average" not in df.columns:
    df["average"] = (df["f1_mean"] + df["acc_mean"]) / 2
    df.to_csv(CSV, index=False)
    print(f"Added 'average' column to {CSV.name}")

# ── Plot ──────────────────────────────────────────────────────────────────────
labels = [f"seed\n{s}" for s in SEEDS] + ["Average"]
f1_vals  = f1s  + [f1_avg]
acc_vals = accs + [acc_avg]

x     = np.arange(len(labels))
width = 0.38

fig, ax = plt.subplots(figsize=(14, 6))

bars_f1  = ax.bar(x - width/2, f1_vals,  width, label="Macro F1 (%)",  color="steelblue",  edgecolor="white")
bars_acc = ax.bar(x + width/2, acc_vals, width, label="Accuracy (%)",  color="darkorange", edgecolor="white")

# Colour the Average group differently
for bar in (bars_f1[-1], bars_acc[-1]):
    bar.set_color("tomato")
    bar.set_alpha(0.85)

# Value labels on bars
for bar, val in zip(list(bars_f1) + list(bars_acc), f1_vals + acc_vals):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.4,
        f"{val:.1f}",
        ha="center", va="bottom", fontsize=7.5, rotation=0
    )

# Reference lines for averages
ax.axhline(f1_avg,  color="steelblue",  linestyle="--", linewidth=1.1, alpha=0.6,
           label=f"F1 mean = {f1_avg:.2f}%")
ax.axhline(acc_avg, color="darkorange", linestyle="--", linewidth=1.1, alpha=0.6,
           label=f"Acc mean = {acc_avg:.2f}%")

# Separator line before Average group
ax.axvline(len(SEEDS) - 0.5, color="grey", linestyle=":", linewidth=1)

epsilon = row["epsilon"]
sigma   = row["sigma"]
ax.set_title(
    f"FL + DP_80 — Per-Seed F1 & Accuracy  (ε={epsilon}, σ={sigma})\n"
    f"F1: mean={f1_avg:.2f}%  std={row['f1_std']:.2f}%  |  "
    f"Acc: mean={acc_avg:.2f}%  std={row['acc_std']:.2f}%",
    fontsize=12
)
ax.set_ylabel("Score (%)", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylim(max(0, min(f1_vals + acc_vals) - 8), min(107, max(f1_vals + acc_vals) + 6))
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPNG, dpi=150)
plt.show()
print(f"Saved → {OUTPNG}")
