"""The joint distribution of (home goals, away goals), and every market read off it.

Two independent Poisson variables get the shape of a football score roughly
right but systematically misprice the four lowest scorelines: real matches
produce more 0-0 and 1-1, and fewer 1-0 and 0-1, than independence implies.
Dixon & Coles (1997) correct exactly those four cells with a single extra
parameter ``rho``, which is what this module implements.

Everything downstream - 1X2, over/under any line, multigol ranges, both teams
to score, correct score - is a projection of the same matrix, so two quoted
probabilities can never contradict one another.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import poisson

__all__ = [
    "score_matrix", "market_probability", "all_markets", "fit_rho",
    "top_scores", "matrix_summary", "rho_bounds",
]


def rho_bounds(lam_h: float, lam_a: float) -> tuple[float, float]:
    """Range of ``rho`` that keeps all four corrected cells non-negative."""
    lo = max(-1.0 / max(lam_h, 1e-9), -1.0 / max(lam_a, 1e-9))
    hi = min(1.0 / max(lam_h * lam_a, 1e-9), 1.0)
    return lo, hi


def _tau(max_goals: int, lam_h: float, lam_a: float, rho: float) -> np.ndarray:
    """Dixon-Coles low-score correction, as a multiplicative mask."""
    t = np.ones((max_goals + 1, max_goals + 1), dtype=float)
    t[0, 0] = 1.0 - lam_h * lam_a * rho
    t[0, 1] = 1.0 + lam_h * rho
    t[1, 0] = 1.0 + lam_a * rho
    t[1, 1] = 1.0 - rho
    return t


def score_matrix(lam_h: float, lam_a: float, rho: float = 0.0,
                 max_goals: int = 10) -> np.ndarray:
    """P(home = h, away = a) for h, a in 0..max_goals; rows are home goals.

    The matrix is renormalised after truncation and after the correction, so it
    sums to exactly 1 and every market read off it is a genuine probability.
    """
    lam_h = float(max(lam_h, 1e-6))
    lam_a = float(max(lam_a, 1e-6))
    k = np.arange(max_goals + 1)
    m = np.outer(poisson.pmf(k, lam_h), poisson.pmf(k, lam_a))
    if rho:
        lo, hi = rho_bounds(lam_h, lam_a)
        m = m * _tau(max_goals, lam_h, lam_a, float(np.clip(rho, lo, hi)))
    m = np.clip(m, 0.0, None)
    total = m.sum()
    return m / total if total > 0 else m


# --------------------------------------------------------------------------- #
# projections
# --------------------------------------------------------------------------- #
def _grids(n: int):
    idx = np.arange(n)
    return idx[:, None] + np.zeros((1, n), dtype=int), np.zeros((n, 1), dtype=int) + idx[None, :]


def market_probability(m: np.ndarray, kind: str, arg=None) -> float:
    """Project the score matrix onto one market.

    kind            arg                   meaning
    --------------  --------------------  -----------------------------------
    result          "H" / "D" / "A"       match result
    total_over      2.5                   total goals above the line
    total_under     2.5                   total goals below the line
    total_range     (1, 3)                total goals in [lo, hi] inclusive
    team_over       ("home", 1.5)         that side's goals above the line
    team_range      ("home", 1, 3)        that side's goals in [lo, hi]
    btts            None                  both teams score
    correct_score   (2, 1)                an exact scoreline
    """
    n = m.shape[0]
    h, a = _grids(n)
    if kind == "result":
        if arg == "H":
            return float(m[h > a].sum())
        if arg == "D":
            return float(np.trace(m))
        if arg == "A":
            return float(m[h < a].sum())
        raise ValueError(f"result arg must be H/D/A, got {arg!r}")
    if kind == "total_over":
        return float(m[(h + a) > float(arg)].sum())
    if kind == "total_under":
        return float(m[(h + a) < float(arg)].sum())
    if kind == "total_range":
        lo, hi = arg
        return float(m[((h + a) >= lo) & ((h + a) <= hi)].sum())
    if kind == "btts":
        return float(m[1:, 1:].sum())
    if kind == "team_over":
        side, line = arg
        g = h if side == "home" else a
        return float(m[g > float(line)].sum())
    if kind == "team_range":
        side, lo, hi = arg
        g = h if side == "home" else a
        return float(m[(g >= lo) & (g <= hi)].sum())
    if kind == "correct_score":
        hh, aa = arg
        return float(m[hh, aa]) if hh < n and aa < n else 0.0
    raise ValueError(f"unknown market kind {kind!r}")


def all_markets(m: np.ndarray, markets) -> dict[str, float]:
    return {name: market_probability(m, kind, arg) for name, kind, arg in markets}


def top_scores(m: np.ndarray, n: int = 6) -> list[tuple[int, int, float]]:
    """The n most likely exact scorelines, most likely first."""
    flat = np.argsort(m, axis=None)[::-1][:n]
    return [(int(i // m.shape[1]), int(i % m.shape[1]), float(m.flat[i])) for i in flat]


def matrix_summary(m: np.ndarray) -> dict[str, float]:
    """Expected goals implied by the matrix itself, after the rho correction."""
    n = m.shape[0]
    h, a = _grids(n)
    return {
        "exp_home_goals": float((m * h).sum()),
        "exp_away_goals": float((m * a).sum()),
        "exp_total_goals": float((m * (h + a)).sum()),
        "exp_supremacy": float((m * (h - a)).sum()),
    }


# --------------------------------------------------------------------------- #
# fitting rho
# --------------------------------------------------------------------------- #
def fit_rho(home_goals, away_goals, lam_h, lam_a) -> float:
    """MLE for the low-score correction, given per-match expected goals.

    Only the four lowest cells contribute, so this is a clean one-dimensional
    problem.  Fitted on the training seasons and then held fixed: it describes a
    property of football, not of any one season.
    """
    hg = np.asarray(home_goals, dtype=int)
    ag = np.asarray(away_goals, dtype=int)
    lh = np.asarray(lam_h, dtype=float)
    la = np.asarray(lam_a, dtype=float)

    def neg_ll(rho: float) -> float:
        t = np.ones(len(hg))
        m00 = (hg == 0) & (ag == 0)
        m01 = (hg == 0) & (ag == 1)
        m10 = (hg == 1) & (ag == 0)
        m11 = (hg == 1) & (ag == 1)
        t[m00] = 1.0 - lh[m00] * la[m00] * rho
        t[m01] = 1.0 + lh[m01] * rho
        t[m10] = 1.0 + la[m10] * rho
        t[m11] = 1.0 - rho
        if np.any(t <= 0):
            return 1e9
        return -float(np.sum(np.log(t)))

    res = minimize_scalar(neg_ll, bounds=(-0.35, 0.35), method="bounded")
    return float(res.x)
