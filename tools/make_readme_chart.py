"""Draw docs/img/walk_forward.png from the walk-forward table in docs/engineering-log.md.

The numbers are copied from that table (Premier League, refit per season, booster + logistic
blend): they are results on real data, so this script needs none.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
BASE_RATE = [0.1688, 0.1749, 0.2047, 0.1954, 0.1809]
PR_AUC = [0.2178, 0.2367, 0.2699, 0.2738, 0.2439]
LIFT = [1.290, 1.354, 1.318, 1.401, 1.348]

fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=160)
x = range(len(SEASONS))
w = 0.38
ax.bar([i - w / 2 for i in x], BASE_RATE, w, label="base rate (predict the average for everyone)",
       color="#b8c0cc")
ax.bar([i + w / 2 for i in x], PR_AUC, w, label="model (PR-AUC)", color="#c9a227")
for i, (b, m, lift) in enumerate(zip(BASE_RATE, PR_AUC, LIFT)):
    ax.text(i + w / 2, m + 0.004, f"{lift:.2f}x", ha="center", va="bottom", fontsize=8)
ax.set_xticks(list(x))
ax.set_xticklabels(SEASONS, fontsize=8)
ax.set_ylim(0, 0.32)
ax.set_ylabel("precision-recall AUC", fontsize=8)
ax.tick_params(axis="y", labelsize=8)
ax.set_title("Walk-forward, one season held out at a time (Premier League)", fontsize=10, loc="left")
ax.legend(fontsize=7.5, frameon=False, loc="upper left")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
out = Path(__file__).resolve().parents[1] / "docs" / "img" / "walk_forward.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out)
print("wrote", out)
