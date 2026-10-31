"""Score one fixture against the market's own player-card prices.

``predict_fixture`` answers "who is most likely to be booked?".  That is not
the question a price comparison asks.  That asks "where does the model
disagree with the market?" - and the model's favourite is very often the
market's favourite too, at a price that already contains the whole signal.
Ranking and value are different orderings and this module computes the second
one.

Two prices, two different jobs
------------------------------
The **1X2 price** is a model *input*.  ``config.ODDS_PROFILE`` puts the
de-vigged closing line into the feature vector, so the fixture must be priced
before it can be scored at all; ``predict_fixture(odds=...)`` handles that.

The **player card price** is a *comparison*, never an input.  It is what the
model is being measured against, so feeding it in would be circular.

De-vigging without the other side of the book
---------------------------------------------
A player card market is two-way - "carded" against "not carded" - and the
honest de-vig divides each implied probability by the pair's sum.  When both
sides are supplied (``no`` alongside ``yes``) that is what happens, per market,
and it is the accurate route.

Usually only the "yes" price is quoted, and then each market's margin is
unknown.  The fallback anchors the *field*: the implied probabilities of all
twenty starters are scaled by a single constant so that they sum to an
independent estimate of the match's expected bookings.  Three sources for that
anchor, in order of preference:

  ``--total-cards N``   the match cards line, if you have one.  Best: it is the
                        market's own opinion, from a market with a much thinner
                        margin than the player props.
  ``--margin M``        a flat overround, e.g. 0.12, if you know the book's.
  (default)             the model's own expected card count.

The default is deliberately circular about *level* and deliberately honest
about *shape*.  Scaling to the model's own sum means model and market agree by
construction on how many cards the match produces, so every remaining
difference is about which players collect them.  That is the only comparison
the within-match ranking actually uses, and it is the one the model is good at - the
match-level total is what the odds features already import from the market.
The output flags which anchor was used; do not read the ``edge`` column as a
real edge over the market when the anchor was ``model``, only as a disagreement about shape.

Input format
------------
A JSON file (or ``--price`` pairs on the command line):

    {
      "home": "Fulham", "away": "Manchester Utd",
      "kickoff": "2026-09-20 16:30", "matchweek": 5,
      "league": "premier_league",             # optional; any of the big five
      "referee": "Michael Oliver",
      "odds": {"home": 3.10, "draw": 3.45, "away": 2.30},
      "odds_ou": {"over": 1.95, "under": 1.90},
      "home_formation": "4-2-3-1", "away_formation": "4-3-3",
      "home_xi": ["Bernd Leno", "..."],
      "away_xi": ["Altay Bayindir", "..."],
      "card_odds": {"Calvin Bassey": 3.75, "Casemiro": 2.50, "...": 0}
    }

The goalkeeper is dropped by the model, so give eleven names per side and
expect twenty rows back.  Names must match the dataset's spelling; anything
unmatched is reported rather than silently ignored, because a quietly dropped
name is a quietly wrong shortlist.

Usage
-----
    python serve/card_market.py --match match.json
    python serve/card_market.py --match match.json --total-cards 4.5
    python serve/card_market.py --match match.json --margin 0.12
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import config, predict
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import paths  # noqa: E402,F401 - puts pipeline/, serve/, research/ on sys.path
    import config
    import predict


def implied_from_prices(card_odds: dict, no_odds: dict | None = None
                        ) -> tuple[pd.Series, str]:
    """Decimal prices -> implied probabilities, de-vigged where possible.

    Returns the series and the name of the de-vig route actually taken, so the
    caller can label the output honestly.
    """
    yes = pd.Series({k: 1.0 / float(v) for k, v in card_odds.items()},
                    dtype=float)
    if no_odds:
        no = pd.Series({k: 1.0 / float(v) for k, v in no_odds.items()},
                       dtype=float)
        both = yes.index.intersection(no.index)
        if len(both):
            devigged = yes.copy()
            devigged.loc[both] = yes[both] / (yes[both] + no[both])
            return devigged, "two-sided"
    return yes, "raw"


def anchor_field(implied: pd.Series, route: str, model_expected: float,
                 total_cards: float | None, margin: float | None
                 ) -> tuple[pd.Series, str]:
    """Scale raw implied probabilities onto a defensible level.

    No-op when the prices were already de-vigged two-sidedly - that estimate is
    per-market and correct, and rescaling it would throw the accuracy away.
    """
    if route == "two-sided":
        return implied, "two-sided (no rescale)"
    s = float(implied.sum())
    if s <= 0:
        return implied, "none"
    if total_cards is not None:
        return implied * (float(total_cards) / s), f"market total {total_cards}"
    if margin is not None:
        return implied / (1.0 + float(margin)), f"flat margin {margin}"
    return implied * (model_expected / s), "model expectation"


def compare(pred: pd.DataFrame, card_odds: dict, no_odds: dict | None = None,
            total_cards: float | None = None, margin: float | None = None,
            prob_col: str = "card_probability") -> tuple[pd.DataFrame, dict]:
    """Join model probabilities to market probabilities and rank by edge."""
    implied, route = implied_from_prices(card_odds, no_odds)
    model_expected = float(pred[prob_col].sum())
    market, anchor = anchor_field(implied, route, model_expected,
                                  total_cards, margin)

    out = pred.copy()
    out["market_prob"] = out["player"].map(market)
    out["fair_odds"] = 1.0 / out[prob_col]
    out["offered_odds"] = out["player"].map(
        {k: float(v) for k, v in card_odds.items()})

    # Edge as the model states it: the expected return per unit at the offered
    # price.  >0 means the model thinks the price is too long.
    out["edge"] = out[prob_col] * out["offered_odds"] - 1.0
    out["d_vs_market"] = out[prob_col] - out["market_prob"]

    unmatched = sorted(set(card_odds) - set(pred["player"]))
    unpriced = sorted(set(pred["player"]) - set(card_odds))
    meta = {
        "devig_route": route,
        "anchor": anchor,
        "model_expected_cards": model_expected,
        "market_expected_cards": float(market.sum()),
        "raw_implied_sum": float(implied.sum()),
        "unmatched_price_names": unmatched,
        "unpriced_players": unpriced,
    }
    return out.sort_values("edge", ascending=False), meta


# Players with five or fewer prior appearances are the one slice where the
# model is measurably weaker: segments.py puts their ROC-AUC at 0.536 against
# 0.600 overall, and their base rate at 21.0% against a mean prediction of
# 17.5%.  Both halves matter and they point opposite ways.  The bias says the
# model under-calls newcomers, so a thin-history name near the top is probably
# not *too* high; the AUC says the ordering inside that slice is close to
# random, so the model is not the reason to prefer one newcomer over another.
# An apparent edge resting on such a row is resting on the model's weakest segment.
THIN_HISTORY_APPS = 5


def flag_thin_history(rows: pd.DataFrame) -> pd.Series:
    """True where the row sits in the slice segments.py measured as weak."""
    apps = rows.get("plr_apps_prior")
    thin = pd.Series(False, index=rows.index)
    if apps is not None:
        thin |= apps.fillna(0) <= THIN_HISTORY_APPS
    if "pos_group" in rows:
        thin |= rows["pos_group"].astype(str).eq("UNK")
    return thin


def run(spec: dict, total_cards: float | None, margin: float | None,
        top: int, model=None) -> pd.DataFrame:
    odds = spec.get("odds")
    if not odds:
        raise SystemExit(
            "this fixture needs a 1X2 price: the shipped model carries odds "
            f"features (profile {config.ODDS_PROFILE!r}) and without a price "
            "they arrive NaN, a state the booster never saw in training.")
    ou = spec.get("odds_ou")

    # One model for ranking and for pricing: the five-league ensemble, whose
    # logistic weight was chosen walk-forward alongside the rest (ensemble.py).
    # The old split - a blend for prices, the plain booster for shortlists - is gone.
    pred = predict.predict_fixture(
        model=model, league=spec.get("league", "premier_league"),
        home_team=spec["home"], away_team=spec["away"],
        kickoff=spec["kickoff"], referee=spec.get("referee"),
        home_xi=spec["home_xi"], away_xi=spec["away_xi"],
        home_formation=spec.get("home_formation"),
        away_formation=spec.get("away_formation"),
        matchweek=spec.get("matchweek"),
        odds=(odds["home"], odds["draw"], odds["away"]),
        odds_ou=((ou["over"], ou["under"]) if ou else None),
    )

    card_odds = spec.get("card_odds")
    if not card_odds:
        print("\nno card_odds supplied - model ranking only\n")
        print(pred.head(top).to_string(float_format=lambda v: f"{v:.4f}"))
        return pred

    cmp, meta = compare(pred, card_odds, spec.get("card_odds_no"),
                        total_cards, margin)

    print(f"\n{spec['home']} v {spec['away']}  -  {spec['kickoff']}")
    print(f"referee: {spec.get('referee') or 'unknown (league prior)'}")
    print(f"de-vig: {meta['devig_route']}   anchor: {meta['anchor']}")
    print(f"expected bookings - model {meta['model_expected_cards']:.2f}, "
          f"market {meta['market_expected_cards']:.2f} "
          f"(raw implied sum {meta['raw_implied_sum']:.2f})")
    if meta["unmatched_price_names"]:
        print(f"WARNING price names not in the XI: {meta['unmatched_price_names']}")
    if meta["unpriced_players"]:
        print(f"note: no price for {len(meta['unpriced_players'])} starters: "
              f"{meta['unpriced_players']}")

    cmp["thin"] = flag_thin_history(cmp)
    cols = ["player", "team", "pos_group", "card_probability", "market_prob",
            "offered_odds", "fair_odds", "edge", "thin"]
    cols = [c for c in cols if c in cmp.columns]
    print("\n--- ranked by edge (model prob x offered price - 1) ---")
    print(cmp[cols].head(top).to_string(index=False,
                                        float_format=lambda v: f"{v:.4f}"))
    print("\n--- ranked by raw model probability ---")
    print(cmp[cols].sort_values("card_probability", ascending=False).head(top)
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if meta["anchor"] == "model expectation":
        print("\nanchor was the model's own total, so 'edge' reflects "
              "disagreement about WHICH player, not about how many cards.")

    flagged = cmp[cmp["thin"]].nlargest(3, "edge")
    if len(flagged):
        print(f"\nthin history (<={THIN_HISTORY_APPS} prior apps or unknown "
              f"position): {', '.join(flagged['player'])}")
        print("  segments.py measures this slice at ROC-AUC 0.536 vs 0.600 "
              "overall - the model under-calls newcomers (21.0% booked vs "
              "17.5% predicted), so the level is if anything low, but the "
              "ordering within the slice is near random.  Treat a large edge "
              "here as thin evidence, not a strong signal.")
    return cmp


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match", required=True, help="JSON file for one fixture")
    ap.add_argument("--total-cards", type=float, default=None,
                    help="match cards line, used to anchor the field")
    ap.add_argument("--margin", type=float, default=None,
                    help="flat book overround, e.g. 0.12")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--out", default=None, help="write the full table to CSV")
    args = ap.parse_args()

    spec = json.loads(Path(args.match).read_text(encoding="utf-8"))
    cmp = run(spec, args.total_cards, args.margin, args.top)
    if args.out:
        cmp.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
