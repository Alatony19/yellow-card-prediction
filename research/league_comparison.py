"""One global card model, or five local ones?

The question cannot wait for the scrape
---------------------------------------
Scraping FBref for Serie A, the Bundesliga, La Liga and Ligue 1 is tens of
hours of browser time.  Before spending it, it is worth knowing what the answer
is likely to be - and most of the evidence is already on disk, because
football-data.co.uk publishes, for every match in all five leagues since 2009,
the referee, the fouls, the yellow and red cards for each side, and the closing
1X2 line.  That is not player-level, so it cannot train the card model, but it
is exactly the right data to answer the *architecture* question:

    Do the five leagues differ in the LEVEL of cards, in the STRUCTURE that
    generates them, or in both?

Because those two answers imply opposite designs:

* **Level only.**  Pool everything into one model and give each league its own
  intercept.  Serie A's extra bookings are then a constant the model learns
  once, and every league's data trains the same shape - five times the rows for
  the part that is hard to estimate.  This is the architecture ``team_model.py``
  already implements for a different purpose: a league-level rate multiplied by
  a within-match share.
* **Structure too.**  Fit separately, because the coefficient a pooled model
  settles on would be wrong everywhere.

What is measured here
---------------------
1. **Levels.**  Cards per match, per team, home vs away, fouls, cards per foul.
2. **Structure.**  A Poisson regression of a team's yellow cards on the market's
   read of the fixture - how close it is, whether this side is favoured - fitted
   once per league.  If the slopes agree while the intercepts differ, the
   structure is shared.
3. **Transfer.**  The decisive test: fit on four leagues, predict the fifth.
   Once with the foreign intercept, once re-anchored to the held-out league's
   own level, and finally against a pooled model that sees all five leagues and
   gives each its own intercept - the architecture actually being proposed.

   The anchor is taken from the **most recent training season only**, not from
   the whole training history, and that detail turned out to decide the
   experiment.  League card levels drift hard: Serie A fell from 2.55 yellows
   per team in 2019-20 to 1.85 in 2025-26 while the Premier League rose from
   1.52 to 1.87.  A career anchor therefore aims at a level the league has left
   behind, and re-anchoring on it makes the prediction worse rather than
   better.  ``team_model.py`` already anchors on the validation season for the
   same reason; this follows it.

Usage
-----
    python research/league_comparison.py
    python research/league_comparison.py --seasons 2017-2018 ... 2025-2026
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

V3_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3_DIR))
import paths  # noqa: E402,F401 - puts pipeline/, serve/, research/ on sys.path

import config
import download_odds

DATA_DIR = config.ODDS_ROOT

LEAGUES = {
    "Premier League": "premier_league",
    "Serie A": "serie_a",
    "La Liga": "la_liga",
    "Bundesliga": "bundesliga",
    "Ligue 1": "ligue_1",
}

# Price columns, best first.  Closing ("C") where available, otherwise the
# pre-match average; seasons before 2019-20 only carry the Betbrain names.
CLOSE_1X2 = [("AvgCH", "AvgCD", "AvgCA"), ("AvgH", "AvgD", "AvgA"),
             ("BbAvH", "BbAvD", "BbAvA"), ("B365H", "B365D", "B365A")]


def _first_available(df, groups):
    for cols in groups:
        if all(c in df.columns for c in cols) and df[list(cols)].notna().any().all():
            return list(cols)
    return None


def load_league(league_dir: str, seasons) -> pd.DataFrame:
    """Two rows per match - one per side - with cards, fouls and the price."""
    frames = []
    for season in seasons:
        path = DATA_DIR / league_dir / download_odds.season_to_file(season)
        if not path.exists():
            continue
        raw = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        raw = raw[raw["HomeTeam"].notna() & raw["AwayTeam"].notna()].copy()
        if "HY" not in raw.columns:
            continue

        cols = _first_available(raw, CLOSE_1X2)
        if cols is None:
            continue
        h, d, a = (pd.to_numeric(raw[c], errors="coerce") for c in cols)
        inv = np.vstack([1 / h, 1 / d, 1 / a])
        p = inv / inv.sum(axis=0)

        base = pd.DataFrame({
            "season": season,
            "match_key": np.arange(len(raw)),
            "referee": raw.get("Referee", pd.Series(index=raw.index, dtype=object)),
            "p_home": p[0], "p_draw": p[1], "p_away": p[2],
        })
        for side, is_home in (("H", 1), ("A", 0)):
            other = "A" if side == "H" else "H"
            t = base.copy()
            t["is_home"] = is_home
            t["team"] = raw[f"{'Home' if is_home else 'Away'}Team"].values
            t["yellows"] = pd.to_numeric(raw[f"{side}Y"], errors="coerce").values
            t["reds"] = pd.to_numeric(raw.get(f"{side}R"), errors="coerce").values
            t["fouls"] = pd.to_numeric(raw.get(f"{side}F"), errors="coerce").values
            t["opp_fouls"] = pd.to_numeric(raw.get(f"{other}F"), errors="coerce").values
            frames.append(t)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["p_for"] = np.where(out["is_home"] == 1, out["p_home"], out["p_away"])
    out["p_against"] = np.where(out["is_home"] == 1, out["p_away"], out["p_home"])
    # The same two constructions the card model's odds features use, so the
    # conclusion here transfers directly to that feature set.
    out["closeness"] = 1.0 - (out["p_home"] - out["p_away"]).abs()
    out["expectancy"] = out["p_for"] + 0.5 * out["p_draw"]
    out["supremacy"] = out["p_for"] - out["p_against"]
    return out.dropna(subset=["yellows", "closeness"])


def levels_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in data.items():
        home = df[df["is_home"] == 1]
        away = df[df["is_home"] == 0]
        rows.append({
            "league": name,
            "team_matches": len(df),
            "yel_per_team": df["yellows"].mean(),
            "cards_per_match": df["yellows"].mean() * 2 + df["reds"].mean() * 2,
            "yel_home": home["yellows"].mean(),
            "yel_away": away["yellows"].mean(),
            "away_minus_home": away["yellows"].mean() - home["yellows"].mean(),
            "reds_per_team": df["reds"].mean(),
            "fouls_per_team": df["fouls"].mean(),
            "cards_per_foul": df["yellows"].sum() / max(df["fouls"].sum(), 1),
            "referees": df["referee"].nunique(),
        })
    return pd.DataFrame(rows)


def fit_poisson(df: pd.DataFrame, cols: list[str]):
    """Poisson GLM of yellows on ``cols``; returns (intercept, coefs)."""
    from sklearn.linear_model import PoissonRegressor
    X = df[cols].to_numpy(dtype=float)
    y = df["yellows"].to_numpy(dtype=float)
    m = PoissonRegressor(alpha=1e-6, max_iter=1000)
    m.fit(X, y)
    return m


def poisson_deviance(y, mu) -> float:
    y = np.asarray(y, dtype=float)
    mu = np.clip(np.asarray(mu, dtype=float), 1e-9, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / mu), 0.0)
    return float(2.0 * np.mean(term - (y - mu)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", nargs="+", default=config.ALL_SEASONS)
    args = ap.parse_args()

    data = {name: load_league(d, args.seasons) for name, d in LEAGUES.items()}
    data = {k: v for k, v in data.items() if len(v)}
    pd.set_option("display.width", 220)

    # ---- 1. levels --------------------------------------------------------- #
    print("=== 1. LEVELS: how much do the five leagues differ in raw rate? ===")
    lev = levels_table(data)
    print(lev.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    spread = lev["yel_per_team"].max() / lev["yel_per_team"].min()
    print(f"\n  highest league books {spread:.2f}x as many yellows per team as the lowest")

    # ---- 2. structure ------------------------------------------------------ #
    cols = ["closeness", "supremacy", "is_home"]
    print("\n=== 2. STRUCTURE: Poisson slopes on the market's read, per league ===")
    print("    (intercept = level; slopes = shape.  Shared shape -> pool.)")
    rows = []
    for name, df in data.items():
        m = fit_poisson(df, cols)
        rows.append({"league": name, "intercept": m.intercept_,
                     **{c: v for c, v in zip(cols, m.coef_)}})
    struct = pd.DataFrame(rows)
    print(struct.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print("\n  coefficient spread across leagues (max - min):")
    for c in ["intercept"] + cols:
        v = struct[c]
        print(f"    {c:<12} {v.max() - v.min():+.4f}   "
              f"(relative to |mean| {abs(v.mean()):.4f}: "
              f"{abs(v.max() - v.min()) / max(abs(v.mean()), 1e-9):.1%})")

    # ---- 3. transfer ------------------------------------------------------- #
    print("\n=== 3. TRANSFER: fit on four leagues, predict the fifth ===")
    print("    lower deviance is better; 'local' is the ceiling a private model reaches")
    test_seasons = set(config.VALID_SEASONS) | set(config.TEST_SEASONS)
    out = []
    for name, df in data.items():
        te = df[df["season"].isin(test_seasons)]
        tr_local = df[~df["season"].isin(test_seasons)]
        others = pd.concat([d[~d["season"].isin(test_seasons)]
                            for k, d in data.items() if k != name],
                           ignore_index=True)
        if not (len(te) and len(tr_local) and len(others)):
            continue

        local = fit_poisson(tr_local, cols)
        foreign = fit_poisson(others, cols)

        mu_local = local.predict(te[cols].to_numpy(dtype=float))
        mu_foreign = foreign.predict(te[cols].to_numpy(dtype=float))

        # Re-anchoring: keep the foreign SHAPE, replace its LEVEL with the one
        # the held-out league showed most recently.  Anchoring on the whole
        # training history instead aims at a level these leagues have drifted
        # away from, and measurably hurts - see the module docstring.
        anchor_season = sorted(tr_local["season"].unique())[-1]
        recent = tr_local[tr_local["season"] == anchor_season]
        anchor = (recent["yellows"].mean()
                  / foreign.predict(recent[cols].to_numpy(dtype=float)).mean())
        mu_anchored = mu_foreign * anchor

        # The architecture actually proposed: one model over all five leagues,
        # each with its own intercept.  Implemented as league dummies so the
        # shape is estimated on five leagues' rows at once.
        pooled_tr = pd.concat(
            [d[~d["season"].isin(test_seasons)].assign(league=k)
             for k, d in data.items()], ignore_index=True)
        dummies = sorted(data)
        for lg in dummies[1:]:
            pooled_tr[f"lg_{lg}"] = (pooled_tr["league"] == lg).astype(float)
        pooled_cols = cols + [f"lg_{lg}" for lg in dummies[1:]]
        pooled = fit_poisson(pooled_tr, pooled_cols)
        te_p = te.copy()
        for lg in dummies[1:]:
            te_p[f"lg_{lg}"] = float(name == lg)
        mu_pooled = pooled.predict(te_p[pooled_cols].to_numpy(dtype=float))
        # ... and the same pooled model re-levelled to the held-out league's
        # most recent season, which is what a shipped version would do.
        rec_p = recent.copy()
        for lg in dummies[1:]:
            rec_p[f"lg_{lg}"] = float(name == lg)
        pooled_anchor = (recent["yellows"].mean()
                         / pooled.predict(rec_p[pooled_cols].to_numpy(dtype=float)).mean())

        out.append({
            "held-out league": name, "test_rows": len(te),
            "dev_local": poisson_deviance(te["yellows"], mu_local),
            "dev_foreign": poisson_deviance(te["yellows"], mu_foreign),
            "dev_anchored": poisson_deviance(te["yellows"], mu_anchored),
            "dev_pooled": poisson_deviance(te["yellows"], mu_pooled),
            "dev_pooled_anch": poisson_deviance(te["yellows"],
                                                mu_pooled * pooled_anchor),
            "anchor": anchor,
        })
    tr = pd.DataFrame(out)
    print(tr.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n  mean deviance vs the locally fitted model "
          "(negative = beats a league's own model):")
    for c in ["dev_foreign", "dev_anchored", "dev_pooled", "dev_pooled_anch"]:
        d = tr[c] - tr["dev_local"]
        print(f"    {c:<16} {d.mean():+.5f}   "
              f"{(d < 0).sum()}/{len(tr)} leagues better than local")

    outdir = config.RESEARCH_DIR
    lev.to_csv(outdir / "league_levels.csv", index=False)
    struct.to_csv(outdir / "league_structure.csv", index=False)
    tr.to_csv(outdir / "league_transfer.csv", index=False)
    print(f"\nwrote league_levels.csv, league_structure.csv, league_transfer.csv "
          f"to {outdir}")


if __name__ == "__main__":
    main()
