"""Verification gates for the served model (the five-league ensemble).

Run with:  python tests/test_serving.py          (from the v3 directory)

``test_parity`` is the important one: a played 2026-27 match is re-scored
through ``predict.predict_fixture`` as if it were upcoming - team sheets, the
archive's own price, no result - and every player's served probability must
equal the one the same model gives that player's row in the training dataset.
Any train/serve skew (the "UPCOMING" season bug of 2026-09-28 was one) shows up
here as a non-zero difference.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

V3_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V3_DIR))
import paths  # noqa: E402,F401 - puts pipeline/, serve/, research/ on sys.path

import card_market  # noqa: E402
import ensemble  # noqa: E402
import pooled  # noqa: E402
import predict  # noqa: E402
from match_model import odds as odds_mod  # noqa: E402

FAILURES: list[str] = []
SEASON = "2026-2027"


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f"  -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def replay(ds: pd.DataFrame, league: str) -> tuple[dict, pd.DataFrame]:
    """Kwargs for predict_fixture reproducing the league's last played match."""
    players, _teams, _cfg, lodds = pooled.league_history(league)
    sub = ds[(ds["league"].astype(str) == league) & (ds["season"] == SEASON)]
    mid = sub["match_id"].iloc[-1]
    m = players[players["match_id"] == mid]
    home = m.loc[m.side == "home", "team"].iloc[0]
    away = m.loc[m.side == "away", "team"].iloc[0]

    def xi(side):
        s = m[(m.side == side) & m.is_starter]
        gk = s[s.pos_group_actual == "GK"]["player"].tolist()
        return gk[:1] + s[s.pos_group_actual != "GK"]["player"].tolist()

    if lodds is None:
        lodds = odds_mod.load_odds(seasons=[SEASON], verbose=False)
    o = lodds[(lodds.home_team == home) & (lodds.away_team == away)
              & (lodds.season == SEASON)].iloc[0]
    kwargs = dict(
        home_team=home, away_team=away, kickoff=m["kickoff"].iloc[0],
        referee=m["referee"].iloc[0], home_xi=xi("home"), away_xi=xi("away"),
        home_formation=m.loc[m.side == "home", "formation"].iloc[0],
        away_formation=m.loc[m.side == "away", "formation"].iloc[0],
        matchweek=int(sub.loc[sub.match_id == mid, "matchweek"].iloc[0]),
        ages=dict(zip(m.player, m.age_years)),
        odds=(1 / o.p_home, 1 / o.p_draw, 1 / o.p_away),
        odds_ou=(1 / o.p_over25, 1 / (1 - o.p_over25)), league=league)
    return kwargs, sub[sub.match_id == mid].copy()


def test_parity(model, ds: pd.DataFrame) -> None:
    for league in pooled.ALL_LEAGUES:
        kwargs, rows = replay(ds, league)
        served = predict.predict_fixture(model=model, **kwargs)
        rows["p_ds"] = model.predict_proba(rows)
        j = served.merge(rows[["player", "p_ds"]], on="player")
        diff = float((j["card_probability"] - j["p_ds"]).abs().max())
        check(f"parity {league}", len(served) == 20 and len(j) == 20 and diff < 1e-6,
              f"{kwargs['home_team']} v {kwargs['away_team']}: {len(j)}/20 rows, "
              f"max |diff| {diff:.2e}")


def test_output_contract(model, ds: pd.DataFrame) -> None:
    kwargs, _ = replay(ds, pooled.PL)
    out = predict.predict_fixture(model=model, **kwargs)
    p = out["card_probability"]
    check("probabilities in (0, 1)", bool(((p > 0) & (p < 1)).all()))
    check("sorted by card_probability", bool(p.is_monotonic_decreasing))
    check("bag range brackets its members",
          bool((out["bag_lo"] <= out["bag_hi"]).all() and (out["bag_sd"] >= 0).all()))
    check("expected bookings plausible", 1.5 < p.sum() < 7.0, f"sum {p.sum():.2f}")


def test_blend_is_the_prediction(model, ds: pd.DataFrame) -> None:
    rows = ds[ds["season"] == SEASON].head(400)
    comp = model.components(rows)
    manual = sum(model.weights[c] * comp[f"p_{c}"] for c in ensemble.COMPONENTS)
    check("weights sum to 1", abs(sum(model.weights.values()) - 1) < 1e-9,
          str(model.weights))
    check("predict_proba == weighted components",
          bool(np.allclose(model.predict_proba(rows), np.clip(manual, 1e-6, 1 - 1e-6))))


def test_card_market(model, ds: pd.DataFrame) -> None:
    kwargs, _ = replay(ds, "serie_a")
    pred = predict.predict_fixture(model=model, **kwargs)
    odds = {r.player: round(1 / max(r.card_probability, 0.05), 2)
            for r in pred.itertuples()}
    cmp, meta = card_market.compare(pred, odds)
    check("card_market prices a non-PL fixture",
          len(cmp) == 20 and not meta["unmatched_price_names"]
          and abs(meta["model_expected_cards"] - pred["card_probability"].sum()) < 1e-9)


def main() -> int:
    try:
        model = ensemble.load()
    except FileNotFoundError as exc:
        print(f"SKIPPED: this gate scores the trained five-league ensemble, which is not built here ({exc}).\n"
              "It needs the real data for all five leagues - see README, 'Getting the data'.")
        return 0
    ds = pd.read_parquet(pooled.dataset_path(pooled.FINAL_FIRST_SEASON))
    print(f"ensemble: {len(model.members)} members, weights {model.weights}, "
          f"trained on {model.trained_on[0]}..{model.trained_on[-1]}\n")
    test_blend_is_the_prediction(model, ds)
    test_output_contract(model, ds)
    test_card_market(model, ds)
    test_parity(model, ds)
    print("\n" + "=" * 62)
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED: " + ", ".join(FAILURES))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
