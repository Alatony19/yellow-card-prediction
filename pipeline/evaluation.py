"""Metrics and plots for the card model.

Two families of metric, because they answer different questions:

*probability quality* - log loss, Brier, expected calibration error and the
  reliability curve.  These are what matter if a predicted 0.22 is ever going to
  be compared with a market price.

*ranking quality* - PR-AUC plus precision/recall at K **within a match**.  A
  match has 20 outfield starters and typically 3 of them get carded, so
  "precision@3 per match" is the honest read on a top-3 shortlist.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             precision_recall_curve, roc_auc_score)


# --------------------------------------------------------------------------- #
# probability quality
# --------------------------------------------------------------------------- #
def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """Mean |observed - predicted| across equal-count probability bins."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.digitize(y_prob, edges[1:-1], right=True)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        ece += m.mean() * abs(y_true[m].mean() - y_prob[m].mean())
    return float(ece)


def calibration_table(y_true, y_prob, n_bins: int = 10) -> pd.DataFrame:
    """Reliability table: predicted vs observed rate per probability decile."""
    df = pd.DataFrame({"y": np.asarray(y_true, dtype=float),
                       "p": np.asarray(y_prob, dtype=float)})
    df["bin"] = pd.qcut(df["p"], n_bins, labels=False, duplicates="drop")
    out = df.groupby("bin").agg(
        n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean"),
        p_low=("p", "min"), p_high=("p", "max"),
    ).reset_index(drop=True)
    return out


# --------------------------------------------------------------------------- #
# ranking quality, evaluated per match
# --------------------------------------------------------------------------- #
def topk_per_match(df: pd.DataFrame, prob_col: str, target_col: str,
                   k: int, match_col: str = "match_id") -> dict:
    """Precision/recall at K computed inside each match, then averaged.

    Ties are broken deterministically by probability then player, so the numbers
    are reproducible.
    """
    d = df[[match_col, prob_col, target_col]].copy()
    d["_rank"] = d.groupby(match_col)[prob_col].rank(method="first", ascending=False)
    picked = d[d["_rank"] <= k]
    per_match = picked.groupby(match_col)[target_col].agg(["sum", "size"])
    totals = d.groupby(match_col)[target_col].sum()
    precision = (per_match["sum"] / per_match["size"]).mean()
    recall = (per_match["sum"] / totals.replace(0, np.nan)).mean()
    hit_rate = (per_match["sum"] > 0).mean()
    return {
        f"precision@{k}": float(precision),
        f"recall@{k}": float(recall),
        f"hit_rate@{k}": float(hit_rate),
    }


def evaluate(df: pd.DataFrame, prob_col: str, target_col: str,
             name: str = "model", k_values=(1, 3, 5)) -> dict:
    """All metrics for one set of predictions."""
    y = df[target_col].to_numpy(dtype=float)
    p = np.clip(df[prob_col].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    base = y.mean()
    out = {
        "model": name,
        "n": int(len(y)),
        "base_rate": float(base),
        "pr_auc": float(average_precision_score(y, p)),
        "pr_auc_lift": float(average_precision_score(y, p) / base),
        "roc_auc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
        "mean_pred": float(p.mean()),
    }
    for k in k_values:
        out.update(topk_per_match(df, prob_col, target_col, k))
    return out


def comparison_table(results: list[dict]) -> pd.DataFrame:
    cols = ["model", "n", "pr_auc", "pr_auc_lift", "roc_auc", "log_loss", "brier",
            "ece", "precision@1", "precision@3", "precision@5", "recall@5",
            "hit_rate@3", "base_rate", "mean_pred"]
    df = pd.DataFrame(results)
    return df[[c for c in cols if c in df.columns]]


# --------------------------------------------------------------------------- #
# plots (matplotlib only, no seaborn dependency)
# --------------------------------------------------------------------------- #
def plot_pr_curves(frames: dict, target_col: str, ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4.5))
    base = None
    for label, (df, prob_col) in frames.items():
        y = df[target_col].to_numpy(dtype=float)
        p = df[prob_col].to_numpy(dtype=float)
        precision, recall, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        ax.plot(recall, precision, lw=1.8, label=f"{label} (AP={ap:.3f})")
        base = y.mean()
    if base is not None:
        ax.axhline(base, ls="--", c="grey", lw=1, label=f"base rate ({base:.3f})")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Precision-recall")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    return ax


def plot_calibration(y_true, y_prob, n_bins: int = 10, ax=None, label: str = "model"):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    tbl = calibration_table(y_true, y_prob, n_bins)
    ax.plot([0, tbl["observed"].max() * 1.1], [0, tbl["observed"].max() * 1.1],
            ls="--", c="grey", lw=1, label="perfect")
    ax.plot(tbl["predicted"], tbl["observed"], "o-", lw=1.8, label=label)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_title(f"Reliability (ECE = {expected_calibration_error(y_true, y_prob, n_bins):.4f})")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    return ax


def plot_feature_importance(model, feature_names, top_n: int = 25, ax=None):
    import matplotlib.pyplot as plt
    booster = model.get_booster()
    score = booster.get_score(importance_type="gain")
    imp = pd.Series({f: score.get(f, 0.0) for f in feature_names})
    imp = imp.sort_values(ascending=True).tail(top_n)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 0.32 * len(imp) + 1))
    ax.barh(imp.index, imp.to_numpy(), color="#3b6ea5")
    ax.set_xlabel("gain")
    ax.set_title("Feature importance (gain)")
    ax.grid(alpha=.3, axis="x")
    return ax
