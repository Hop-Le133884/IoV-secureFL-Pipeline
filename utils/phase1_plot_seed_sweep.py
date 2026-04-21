"""
Phase 1 — Centralized Double RF, 10-Seed Bar Chart.
Reads reports/phase1_seed_sweep.csv and saves reports/phase1_seed_sweep.png.

Usage:
    python utils/phase1_plot_seed_sweep.py
"""

import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = "reports/phase1_seed_sweep.csv"
OUT_PATH = "reports/phase1_seed_sweep.png"

seeds, means = [], []
with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if not row["seed"].strip().lstrip("-").isdigit():
            continue
        seeds.append(int(row["seed"]))
        means.append(float(row["seed_mean_f1"]))

avg = np.mean(means)
std = np.std(means)

labels = [f"seed={s}" for s in seeds] + ["Average"]
values = means + [avg]
colors = ["steelblue"] * len(means) + ["tomato"]

fig, ax = plt.subplots(figsize=(13, 5))
bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8)

for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
            f"{val:.2f}%", ha="center", va="bottom", fontsize=9)

ax.axhline(avg, color="tomato", linestyle="--", linewidth=1.2, alpha=0.7)
ax.set_ylabel("Macro F1 Score (%)")
ax.set_title(
    "Phase 1 — Centralized Double RF (sklearn), Macro F1 Across 10 Seeds\n"
    f"Mean = {avg:.2f}%  |  Std = {std:.2f}%"
)
ax.set_ylim(max(0, min(values) - 8), min(105, max(values) + 6))
ax.tick_params(axis="x", rotation=20)
plt.tight_layout()
os.makedirs(os.path.dirname(os.path.abspath(OUT_PATH)), exist_ok=True)
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT_PATH}")
print(f"Mean: {avg:.2f}%  |  Std: {std:.2f}%  |  Min: {min(means):.2f}%  |  Max: {max(means):.2f}%")
