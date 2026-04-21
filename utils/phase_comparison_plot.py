"""
Three-Phase F1 Comparison Chart.
Plots Phase 1 (centralized), Phase 2 (federated, no-DP), Phase 2 (ε=80),
and Phase 3 (real EC2, non-IID, ε=80) side-by-side with error bars.

Usage:
    python utils/phase_comparison_plot.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_PATH = "reports/phase_comparison.png"

phases = [
    "Phase 1\nCentralized\nDouble RF",
    "Phase 2\nFederated\nNo-DP",
    "Phase 2\nFederated\nε=100",
    "Phase 2\nFederated\nε=80",
    "Phase 2\nFederated\nε=50",
    "Phase 2\nFederated\nε=20",
    "Phase 2\nFederated\nε=1",
    "Phase 3\nReal EC2\nNon-IID ε=80",
]
means = [79.97, 64.97, 60.35, 54.43, 46.52, 32.63, 12.31, 48.19]
stds  = [ 4.57, 10.11, 10.26, 16.33, 17.00, 20.63, 14.74, 18.56]

colors = [
    "#2196F3",   # Phase 1 — blue
    "#4CAF50",   # Phase 2 no-DP — green
    "#8BC34A",   # Phase 2 ε=100
    "#FFC107",   # Phase 2 ε=80
    "#FF9800",   # Phase 2 ε=50
    "#FF5722",   # Phase 2 ε=20
    "#F44336",   # Phase 2 ε=1
    "#9C27B0",   # Phase 3 — purple
]

x = np.arange(len(phases))
fig, ax = plt.subplots(figsize=(15, 6))
bars = ax.bar(x, means, yerr=stds, color=colors, edgecolor="white",
              linewidth=0.8, capsize=5, error_kw={"elinewidth": 1.5, "ecolor": "black"})

for bar, m, s in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.8,
            f"{m:.1f}%\n±{s:.1f}%", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(phases, fontsize=9)
ax.set_ylabel("Macro F1 Score (%)", fontsize=11)
ax.set_ylim(0, 115)
ax.set_title(
    "Cross-Phase Macro F1 Comparison — Centralized → Federated → DP → Real EC2\n"
    "(Mean ± Std across 10 seeds each)",
    fontsize=12,
)
ax.axhline(79.97, color="#2196F3", linestyle=":", linewidth=1.0, alpha=0.6,
           label="Phase 1 centralized baseline (79.97%)")
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle=":", alpha=0.4)

plt.tight_layout()
os.makedirs(os.path.dirname(os.path.abspath(OUT_PATH)), exist_ok=True)
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT_PATH}")
