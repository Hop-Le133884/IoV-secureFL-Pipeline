import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

HERE   = Path(__file__).parent
SEEDS  = [42, 123, 456, 1000, 314, 7, 99, 789, 1234, 2000]
CSV    = HERE / "../randomSEED_report/SEEDs_report.csv"
OUTPNG = HERE / "../randomSEED_report/seed_sweep_f1.png"

df     = pd.read_csv(CSV)
f1s    = df["macro_f1_pct"].tolist()
avg    = np.mean(f1s)

labels = [f"seed={s}" for s in SEEDS[:len(f1s)]] + ["Average"]
values = f1s + [avg]
colors = ["steelblue"] * len(f1s) + ["tomato"]

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8)

for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{val:.2f}%", ha="center", va="bottom", fontsize=9)

ax.axhline(avg, color="tomato", linestyle="--", linewidth=1.2, alpha=0.7)
ax.set_ylabel("Macro F1 Score (%)")
ax.set_title("FL + DP (ε=80) — Macro F1 Across 10 Seeds\n"
             f"Mean = {avg:.2f}%  |  Std = {np.std(f1s):.2f}%")
ax.set_ylim(max(0, min(values) - 5), min(100, max(values) + 5))
ax.tick_params(axis="x", rotation=20)
plt.tight_layout()
plt.savefig(OUTPNG, dpi=150)
plt.show()
print(f"Mean: {avg:.2f}%  |  Std: {np.std(f1s):.2f}%  |  Min: {min(f1s):.2f}%  |  Max: {max(f1s):.2f}%")
