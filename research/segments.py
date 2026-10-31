"""Where the card model is right and where it is wrong.

The question "when is the model accurate?" has two distinct answers and they
have different exploits, so they are measured separately throughout:

*discrimination* - can it tell the carded player from the rest, inside this
  slice?  Measured with ROC-AUC, which does not move when the base rate moves,
  plus PR-AUC lift for the top-K shortlist reading.  A slice where discrimination
  collapses is a slice to stop selecting in.

*calibration* - is the level right?  Measured as mean(p) - mean(y), in
  percentage points.  A slice with good discrimination but a biased level is
  worth *more*, not less: the ranking is fine and the bias is a correction.

Segment discovery runs on the earlier walk-forward seasons and every candidate
is re-measured on the later ones before anything is built on it.  A segment
that only exists in the discovery window is a multiple-comparisons artefact,
and with ~17 dimensions x ~4 bins there are enough comparisons for several.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import config, oos as oos_mod
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import paths  # noqa: E402,F401 - puts pipeline/, serve/, research/ on sys.path
    import config
    import oos as oos_mod

from sklearn.metrics import average_precision_score, roc_auc_score

DISCOVERY_SEASONS = ["2021-2022", "2022-2023", "2023-2024"]
CONFIRM_SEASONS = ["2024-2025", "2025-2026"]

# A slice smaller than this is not evidence of anything.
MIN_SEGMENT_ROWS = 1500


# --------------------------------------------------------------------------- #
# the slicing dimensions
# --------------------------------------------------------------------------- #
def _qbin(s: pd.Series, q: int, labels) -> pd.Series:
    """Quantile bins that tolerate ties and missing values."""
    out = pd.qcut(s.rank(method="first"), q, labels=labels)
    return out.astype(object).where(s.notna(), "missing")


def add_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Every slice is computable strictly before kick-off - no exceptions.

    ``p``-derived slices (the model's own confidence, the predicted match total)
    are included because they are also known pre-match, and they turn out to be
    among the more informative dimensions.
    """
    d = df.copy()

    # --- who the player is ------------------------------------------------- #
    d["seg_pos"] = d["pos_group"].astype(str)
    d["seg_player_rate"] = _qbin(d["plr_cards_p90_eb"], 5,
                                 ["rate_1_low", "rate_2", "rate_3", "rate_4",
                                  "rate_5_high"])
    d["seg_history"] = pd.cut(d["plr_apps_prior"], [-1, 5, 20, 60, 150, 1e9],
                              labels=["apps_0-5", "apps_6-20", "apps_21-60",
                                      "apps_61-150", "apps_150+"]).astype(str)
    d["seg_age"] = pd.cut(d["plr_age_years"], [0, 23, 27, 31, 99],
                          labels=["age_u23", "age_23-27", "age_27-31", "age_31+"]
                          ).astype(object).where(d["plr_age_years"].notna(), "missing")
    d["seg_minutes"] = pd.cut(d["plr_mins_roll"], [-1, 60, 75, 85, 91],
                              labels=["mins_lt60", "mins_60-75", "mins_75-85",
                                      "mins_85plus"]).astype(str)

    # --- what kind of match it is ------------------------------------------ #
    d["seg_referee"] = _qbin(d["ref_cards_pm_eb"], 4,
                             ["ref_1_lenient", "ref_2", "ref_3", "ref_4_strict"])
    d["seg_difficulty"] = _qbin(d["ctx_match_difficulty"], 4,
                                ["diff_1_easy", "diff_2", "diff_3", "diff_4_hard"])
    d["seg_mismatch"] = _qbin(d["ctx_elo_diff"].abs(), 4,
                              ["gap_1_even", "gap_2", "gap_3", "gap_4_lopsided"])
    d["seg_fixture_temper"] = _qbin(d["ctx_fixture_cards_pm_eb"], 4,
                                    ["fix_1_calm", "fix_2", "fix_3", "fix_4_fiery"])
    d["seg_venue"] = np.where(d["is_home"] == 1, "home", "away")
    d["seg_stage"] = pd.cut(d["matchweek"], [0, 10, 20, 29, 38],
                            labels=["mw_1-10", "mw_11-20", "mw_21-29",
                                    "mw_30-38"]).astype(str)
    d["seg_stakes"] = _qbin(d["ctx_stakes"], 3,
                            ["stakes_low", "stakes_mid", "stakes_high"])
    d["seg_congestion"] = pd.cut(d["ctx_matches_last_14d"], [-1, 2, 3, 99],
                                 labels=["cong_le2", "cong_3", "cong_4plus"]).astype(str)
    d["seg_table"] = pd.cut(d["ctx_position"], [0, 6, 13, 21],
                            labels=["pos_top6", "pos_mid", "pos_bottom"]
                            ).astype(object).where(d["ctx_position"].notna(), "missing")

    # --- what the model itself says ---------------------------------------- #
    d["seg_confidence"] = _qbin(d["p"], 5, ["p_1_low", "p_2", "p_3", "p_4", "p_5_high"])
    d["seg_match_total"] = _qbin(d["p_sum_in_match"], 4,
                                 ["tot_1_quiet", "tot_2", "tot_3", "tot_4_busy"])
    d["seg_season"] = d["season"]
    return d


SEGMENT_DIMS = [
    "seg_pos", "seg_player_rate", "seg_history", "seg_age", "seg_minutes",
    "seg_referee", "seg_difficulty", "seg_mismatch", "seg_fixture_temper",
    "seg_venue", "seg_stage", "seg_stakes", "seg_congestion", "seg_table",
    "seg_confidence", "seg_match_total", "seg_season",
]


# --------------------------------------------------------------------------- #
# metrics for one slice
# --------------------------------------------------------------------------- #
def slice_metrics(d: pd.DataFrame, prob_col: str = "p",
                  target_col: str | None = None) -> dict:
    target_col = target_col or config.TARGET_COL
    y = d[target_col].to_numpy(dtype=float)
    p = np.clip(d[prob_col].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    base = y.mean()
    out = {"n": int(len(y)), "n_matches": int(d["match_id"].nunique()),
           "base_rate": float(base), "mean_p": float(p.mean()),
           "bias_pp": float((p.mean() - base) * 100)}
    if 0 < base < 1:
        ap = average_precision_score(y, p)
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(ap)
        out["lift"] = float(ap / base)
    else:
        out["roc_auc"] = out["pr_auc"] = out["lift"] = np.nan
    return out


def auc_ci(d: pd.DataFrame, prob_col: str = "p", target_col: str | None = None,
           n_boot: int = 400, seed: int = 0) -> tuple:
    """Bootstrap CI for ROC-AUC, resampling *matches* rather than rows.

    Rows inside a match are not independent - one bad-tempered game books
    several players at once - so a row-level bootstrap would report an interval
    far too narrow.
    """
    target_col = target_col or config.TARGET_COL
    rng = np.random.default_rng(seed)
    matches = d["match_id"].to_numpy()
    uniq = np.unique(matches)
    idx_by_match = {m: np.flatnonzero(matches == m) for m in uniq}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_match[m] for m in pick])
        s = d.iloc[idx]
        y = s[target_col].to_numpy(dtype=float)
        if y.min() == y.max():
            continue
        vals.append(roc_auc_score(y, s[prob_col].to_numpy(dtype=float)))
    if not vals:
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def topk_in_slice(d: pd.DataFrame, k: int = 3, target_col: str | None = None) -> dict:
    """Top-K precision for the rows of this slice.

    The ranking used is the one from the *full* match (``rank_in_match``), so a
    player-level slice never silently re-ranks a truncated field.
    """
    target_col = target_col or config.TARGET_COL
    if "rank_in_match" not in d:
        return {}
    sel = d[d["rank_in_match"] <= k]
    per = sel.groupby("match_id")[target_col].agg(["sum", "size"])
    if not len(per):
        return {f"precision@{k}": np.nan, f"hit@{k}": np.nan}
    return {f"precision@{k}": float((per["sum"] / per["size"]).mean()),
            f"hit@{k}": float((per["sum"] > 0).mean())}


# --------------------------------------------------------------------------- #
# the report
# --------------------------------------------------------------------------- #
def segment_report(d: pd.DataFrame, dims=None, prob_col: str = "p",
                   min_rows: int = MIN_SEGMENT_ROWS, ci: bool = False) -> pd.DataFrame:
    """One row per (dimension, level) with discrimination and calibration."""
    dims = dims or SEGMENT_DIMS
    overall = slice_metrics(d, prob_col)
    rows = []
    for dim in dims:
        for level, sub in d.groupby(dim, sort=False):
            if len(sub) < min_rows:
                continue
            m = slice_metrics(sub, prob_col)
            m["dim"], m["level"] = dim, str(level)
            m["auc_vs_overall"] = m["roc_auc"] - overall["roc_auc"]
            m["lift_vs_overall"] = m["lift"] - overall["lift"]
            m.update(topk_in_slice(sub))
            if ci:
                lo, hi = auc_ci(sub, prob_col)
                m["auc_lo"], m["auc_hi"] = lo, hi
                m["auc_sig"] = bool(lo > 0.5) if np.isfinite(lo) else False
            rows.append(m)
    cols = ["dim", "level", "n", "n_matches", "base_rate", "mean_p", "bias_pp",
            "roc_auc", "auc_vs_overall", "pr_auc", "lift", "lift_vs_overall",
            "precision@3", "hit@3"]
    if ci:
        cols += ["auc_lo", "auc_hi", "auc_sig"]
    return pd.DataFrame(rows)[cols].sort_values(["dim", "level"]).reset_index(drop=True)


def discovery_vs_confirmation(d: pd.DataFrame, dims=None, prob_col: str = "p",
                              min_rows: int = 800) -> pd.DataFrame:
    """The only table that matters: does a segment effect replicate?

    ``auc_vs_overall`` is measured on the discovery seasons and again on the
    confirmation seasons.  A segment is interesting only if the two agree in
    sign and in rough size.
    """
    disc = d[d["season"].isin(DISCOVERY_SEASONS)]
    conf = d[d["season"].isin(CONFIRM_SEASONS)]
    a = segment_report(disc, dims, prob_col, min_rows).set_index(["dim", "level"])
    b = segment_report(conf, dims, prob_col, min_rows).set_index(["dim", "level"])
    j = a.join(b, lsuffix="_disc", rsuffix="_conf", how="inner").reset_index()
    j["replicates"] = (np.sign(j["auc_vs_overall_disc"])
                       == np.sign(j["auc_vs_overall_conf"]))
    j["bias_replicates"] = (np.sign(j["bias_pp_disc"])
                            == np.sign(j["bias_pp_conf"]))
    keep = ["dim", "level", "n_disc", "n_conf",
            "roc_auc_disc", "roc_auc_conf",
            "auc_vs_overall_disc", "auc_vs_overall_conf", "replicates",
            "bias_pp_disc", "bias_pp_conf", "bias_replicates",
            "lift_disc", "lift_conf"]
    return j[keep].sort_values("auc_vs_overall_disc").reset_index(drop=True)


def replicating_effects(d: pd.DataFrame, dims=None, prob_col: str = "p",
                        min_rows: int = 800) -> pd.DataFrame:
    """Segments whose discrimination effect held sign across the two windows."""
    rc = discovery_vs_confirmation(d, dims, prob_col, min_rows)
    r = rc[rc["replicates"]].copy()
    r["avg_effect"] = (r["auc_vs_overall_disc"] + r["auc_vs_overall_conf"]) / 2
    return r.reindex(r["avg_effect"].abs().sort_values(ascending=False).index)


# --------------------------------------------------------------------------- #
# what a user actually does with this
# --------------------------------------------------------------------------- #
def betting_policy(d: pd.DataFrame, dims=None, k: int = 1,
                   target_col: str | None = None,
                   min_picks: int = 150) -> pd.DataFrame:
    """Hit rate of the model's top-K picks, restricted to each segment.

    This is the only framing that maps onto the actual decision.  Ranking metrics
    average over the whole field; a user only ever picks the top of it, so the
    question is "when I take the model's number-one name, how often does that
    slot get carded, and does the answer depend on the conditions?".

    ``strike`` is the raw hit rate of the picks.  ``edge_vs_field`` compares it
    with the base rate *of the same slice*, which is the honest control: a
    segment can look strong simply by being a card-heavy slice, and that is
    worth nothing - the market prices that too.
    """
    target_col = target_col or config.TARGET_COL
    dims = dims or SEGMENT_DIMS
    picks_all = d[d["rank_in_match"] <= k]
    overall = float(picks_all[target_col].mean())
    rows = []
    for dim in dims:
        for level, sub in d.groupby(dim, sort=False):
            picks = sub[sub["rank_in_match"] <= k]
            if len(picks) < min_picks:
                continue
            strike = float(picks[target_col].mean())
            field = float(sub[target_col].mean())
            rows.append({
                "dim": dim, "level": str(level), "n_picks": int(len(picks)),
                "strike": strike, "field_base": field,
                "edge_vs_field": strike / field if field > 0 else np.nan,
                "strike_vs_overall": strike - overall,
                # break-even decimal odds for this strike rate
                "fair_odds": 1.0 / strike if strike > 0 else np.nan,
            })
    out = pd.DataFrame(rows).sort_values("strike", ascending=False)
    return out.reset_index(drop=True)


def policy_replication(d: pd.DataFrame, dims=None, k: int = 1,
                       min_picks: int = 60) -> pd.DataFrame:
    """The same policy table, discovery window against confirmation window.

    A segment worth selecting has to beat the field in *both*.  Almost none do.
    """
    a = betting_policy(d[d["season"].isin(DISCOVERY_SEASONS)], dims, k,
                       min_picks=min_picks).set_index(["dim", "level"])
    b = betting_policy(d[d["season"].isin(CONFIRM_SEASONS)], dims, k,
                       min_picks=min_picks).set_index(["dim", "level"])
    j = a.join(b, lsuffix="_disc", rsuffix="_conf", how="inner").reset_index()
    j["both_beat_field"] = ((j["edge_vs_field_disc"] > 1.0)
                            & (j["edge_vs_field_conf"] > 1.0))
    j["avg_strike"] = (j["strike_disc"] + j["strike_conf"]) / 2
    keep = ["dim", "level", "n_picks_disc", "n_picks_conf", "strike_disc",
            "strike_conf", "avg_strike", "edge_vs_field_disc",
            "edge_vs_field_conf", "both_beat_field"]
    return j[keep].sort_values("avg_strike", ascending=False).reset_index(drop=True)




# --------------------------------------------------------------------------- #
# a composite "act / do not act" rule
# --------------------------------------------------------------------------- #
# Chosen on the discovery seasons only, from the levels whose top-1 strike beat
# the overall top-1 strike in BOTH windows.  Each condition is a plain pre-match
# fact, and the rule is deliberately a conjunction of three rather than a fitted
# model - with 1,900 matches, anything cleverer is fitting noise.
FAVOURABLE = {
    "seg_minutes": ["mins_60-75", "mins_75-85"],   # rotation-proof but not 90-minute
    "seg_history": ["apps_21-60", "apps_61-150", "apps_150+"],
    "seg_referee": ["ref_2", "ref_3", "ref_4_strict"],
}
UNFAVOURABLE = {
    "seg_pos": ["CB"],              # top-1 CB picks strike 0.177
    "seg_referee": ["ref_1_lenient"],
    "seg_minutes": ["mins_85plus"],
}


def apply_rule(d: pd.DataFrame, rule: dict, how: str = "all") -> pd.Series:
    """Boolean mask for rows satisfying ``rule`` (all conditions, or any)."""
    masks = [d[dim].astype(str).isin(levels) for dim, levels in rule.items()]
    if not masks:
        return pd.Series(True, index=d.index)
    m = masks[0]
    for x in masks[1:]:
        m = (m & x) if how == "all" else (m | x)
    return m


def composite_policy(d: pd.DataFrame, k: int = 1,
                     target_col: str | None = None) -> pd.DataFrame:
    """Strike rate of the top-K pick under the composite rule, by window.

    Reported per window so the discovery/confirmation gap is visible rather than
    averaged away.  ``coverage`` is the share of matches that still produce a
    pick - a rule that lifts the strike rate by filtering out four matches in
    five has not necessarily made anyone money.
    """
    target_col = target_col or config.TARGET_COL
    picks = d[d["rank_in_match"] <= k].copy()
    picks["_fav"] = apply_rule(picks, FAVOURABLE, "all")
    picks["_unfav"] = apply_rule(picks, UNFAVOURABLE, "any")

    rows = []
    for win, seasons in [("discovery", DISCOVERY_SEASONS),
                         ("confirmation", CONFIRM_SEASONS),
                         ("pooled", DISCOVERY_SEASONS + CONFIRM_SEASONS)]:
        w = picks[picks["season"].isin(seasons)]
        n_matches = w["match_id"].nunique()
        for name, sel in [("all picks", w),
                          ("favourable (3 conditions)", w[w["_fav"]]),
                          ("unfavourable (any)", w[w["_unfav"]]),
                          ("neither", w[~w["_fav"] & ~w["_unfav"]])]:
            if not len(sel):
                continue
            strike = float(sel[target_col].mean())
            rows.append({"window": win, "rule": name, "n_picks": len(sel),
                         "coverage": len(sel) / max(n_matches, 1),
                         "strike": strike,
                         "fair_odds": 1.0 / strike if strike else np.nan})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Segment analysis of the card model.")
    ap.add_argument("--rebuild-oos", action="store_true")
    ap.add_argument("--ci", action="store_true", help="bootstrap CIs (slow)")
    args = ap.parse_args()

    d = add_segments(oos_mod.load_oos(rebuild=args.rebuild_oos))
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 300)

    o = slice_metrics(d)
    print(f"pooled out-of-sample: {o['n']:,} rows / {o['n_matches']:,} matches")
    print(f"  base {o['base_rate']:.4f} | mean p {o['mean_p']:.4f} | "
          f"ROC-AUC {o['roc_auc']:.4f} | lift {o['lift']:.4f}\n")

    rep = segment_report(d, ci=args.ci)
    print("=== all segments (pooled 5 seasons) ===")
    print(rep.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n=== replicating discrimination effects, largest first ===")
    r = replicating_effects(d)
    print(r.head(25).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("")
    print("=== selection policy: strike rate of the top-1 pick, by segment ===")
    pol = betting_policy(d, k=1)
    print(pol.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("")
    print("=== does a policy segment replicate? ===")
    polr = policy_replication(d, k=1)
    print(polr.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    pol.to_csv(config.RESEARCH_DIR / "betting_policy.csv", index=False)

    rep.to_csv(config.RESEARCH_DIR / "segment_report.csv", index=False)
    discovery_vs_confirmation(d).to_csv(
        config.RESEARCH_DIR / "segment_replication.csv", index=False)
    print(f"\nwrote segment_report.csv and segment_replication.csv "
          f"to {config.RESEARCH_DIR}")


if __name__ == "__main__":
    main()
