"""Dixon-Coles team ratings: the classical benchmark for football scorelines.

Each club gets an attack and a defence rating; the league gets one home
advantage and one low-score correction.  For a match between i at home and j
away,

    lambda_home = exp(attack_i - defence_j + home)
    lambda_away = exp(attack_j - defence_i)

fitted by maximum likelihood with an exponential time decay, so a result from
last month counts for more than one from two seasons ago.

This matters as more than a baseline.  A gradient-boosted model reads form off
rolling averages, which blur *who* the form was against; Dixon-Coles solves for
strength and schedule jointly, so a side that has just played the top four is
not mistaken for a side in decline.  The two disagree in useful ways, which is
why modelling.py blends them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class DixonColes:
    teams: list[str]
    attack: np.ndarray
    defence: np.ndarray
    home_adv: float
    rho: float
    n_matches: int = 0
    converged: bool = True
    _index: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._index = {t: i for i, t in enumerate(self.teams)}

    def rate(self, home_team: str, away_team: str) -> tuple[float, float]:
        """Expected goals for one fixture.  Unknown clubs fall back to average."""
        i = self._index.get(home_team)
        j = self._index.get(away_team)
        atk_i = self.attack[i] if i is not None else 0.0
        def_i = self.defence[i] if i is not None else float(self.defence.mean())
        atk_j = self.attack[j] if j is not None else 0.0
        def_j = self.defence[j] if j is not None else float(self.defence.mean())
        return (float(np.exp(atk_i - def_j + self.home_adv)),
                float(np.exp(atk_j - def_i)))

    def rate_many(self, home_teams, away_teams) -> tuple[np.ndarray, np.ndarray]:
        out = [self.rate(h, a) for h, a in zip(home_teams, away_teams)]
        arr = np.asarray(out, dtype=float)
        return arr[:, 0], arr[:, 1]

    def table(self) -> pd.DataFrame:
        return pd.DataFrame({
            "team": self.teams, "attack": self.attack, "defence": self.defence,
        }).sort_values("attack", ascending=False).reset_index(drop=True)

    @property
    def params(self) -> np.ndarray:
        return np.concatenate([self.attack, self.defence,
                               [self.home_adv, self.rho]])


def _neg_log_likelihood(params, n, hi, ai, hg, ag, w):
    attack = params[:n]
    defence = params[n:2 * n]
    home_adv = params[2 * n]
    rho = params[2 * n + 1]
    # attack is only identified up to a constant; centring it inside the
    # objective removes the flat direction without a hard constraint
    attack = attack - attack.mean()

    lam_h = np.exp(np.clip(attack[hi] - defence[ai] + home_adv, -6, 4))
    lam_a = np.exp(np.clip(attack[ai] - defence[hi], -6, 4))

    ll = hg * np.log(lam_h) - lam_h + ag * np.log(lam_a) - lam_a

    tau = np.ones_like(ll)
    m00 = (hg == 0) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m10 = (hg == 1) & (ag == 0)
    m11 = (hg == 1) & (ag == 1)
    tau[m00] = 1.0 - lam_h[m00] * lam_a[m00] * rho
    tau[m01] = 1.0 + lam_h[m01] * rho
    tau[m10] = 1.0 + lam_a[m10] * rho
    tau[m11] = 1.0 - rho
    if np.any(tau <= 1e-10):
        return 1e10
    ll = ll + np.log(tau)
    return -float(np.sum(w * ll))


def fit(matches: pd.DataFrame, xi: float = 0.0018,
        ref_time=None, warm_start: np.ndarray | None = None,
        teams: list[str] | None = None) -> DixonColes:
    """Fit on one row per match: home_team, away_team, home_goals, away_goals, kickoff.

    ``xi`` is the decay per day; ``ref_time`` is the moment the ratings are for
    (defaults to the last kickoff in the data).  Pass ``warm_start`` with the
    previous fit's parameters to make a sequence of refits cheap - successive
    matchweeks move the solution very little.
    """
    m = matches.dropna(subset=["home_goals", "away_goals"])
    if teams is None:
        teams = sorted(set(m["home_team"]) | set(m["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    keep = m["home_team"].isin(idx) & m["away_team"].isin(idx)
    m = m[keep]
    hi = m["home_team"].map(idx).to_numpy()
    ai = m["away_team"].map(idx).to_numpy()
    hg = m["home_goals"].to_numpy(dtype=float)
    ag = m["away_goals"].to_numpy(dtype=float)

    ref = pd.Timestamp(ref_time) if ref_time is not None else m["kickoff"].max()
    days = (ref - m["kickoff"]).dt.total_seconds().to_numpy() / 86400.0
    w = np.exp(-xi * np.clip(days, 0, None))

    if warm_start is not None and len(warm_start) == 2 * n + 2:
        x0 = np.asarray(warm_start, dtype=float).copy()
    else:
        x0 = np.concatenate([np.zeros(n), np.zeros(n), [0.25], [-0.05]])

    bounds = [(-3.0, 3.0)] * n + [(-3.0, 3.0)] * n + [(-1.0, 1.0), (-0.25, 0.25)]
    res = minimize(_neg_log_likelihood, x0,
                   args=(n, hi, ai, hg, ag, w),
                   method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 400, "ftol": 1e-9})

    attack = res.x[:n] - res.x[:n].mean()
    return DixonColes(teams=list(teams), attack=attack, defence=res.x[n:2 * n],
                      home_adv=float(res.x[2 * n]), rho=float(res.x[2 * n + 1]),
                      n_matches=len(m), converged=bool(res.success))


def rolling_ratings(matches: pd.DataFrame, xi: float = 0.0018,
                    min_matches: int = 200, verbose: bool = False) -> pd.DataFrame:
    """Refit once per matchweek, each time on matches that kicked off before it.

    This is the honest way to score history: the rating used for a fixture never
    saw that fixture, or any other played at the same time.  Warm starting from
    the previous matchweek keeps 300-odd refits to a couple of minutes.
    """
    m = matches.sort_values("kickoff", kind="mergesort").reset_index(drop=True)
    all_teams = sorted(set(m["home_team"]) | set(m["away_team"]))
    groups = list(m.groupby(["season", "matchday"], sort=True))
    groups.sort(key=lambda kv: kv[1]["kickoff"].min())

    out, warm = [], None
    for (season, matchday), grp in groups:
        cutoff = grp["kickoff"].min()
        past = m[m["kickoff"] < cutoff]
        if len(past) < min_matches:
            continue
        model = fit(past, xi=xi, ref_time=cutoff, warm_start=warm, teams=all_teams)
        warm = model.params
        lam_h, lam_a = model.rate_many(grp["home_team"], grp["away_team"])
        out.append(pd.DataFrame({
            "match_id": grp["match_id"].to_numpy(),
            "dc_lambda_home": lam_h,
            "dc_lambda_away": lam_a,
            "dc_home_adv": model.home_adv,
            "dc_rho": model.rho,
            "dc_train_matches": model.n_matches,
        }))
        if verbose:
            print(f"  {season} mw{matchday:02d}: {model.n_matches:4d} past matches, "
                  f"home_adv {model.home_adv:.3f}, rho {model.rho:.3f}", flush=True)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
