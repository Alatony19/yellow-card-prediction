"""Correctness gates for the match-result pipeline.

Three kinds of check, in the order they matter:

* **Coherence** - the score matrix is a probability distribution and the markets
  read off it cannot contradict each other.  If this breaks, every quoted number
  is wrong at once.
* **Recovery** - the fitting code, run on data simulated from known parameters,
  gets those parameters back.  This is what separates "the optimiser converged"
  from "the optimiser converged on the right answer".
* **Leakage** - a feature attached to match M is unchanged when every match
  after M is deleted from the archive.  Same gate as the card pipeline.

Run:  python tests/test_match_model.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import build_match_table as bmt   # noqa: E402
import config                      # noqa: E402
import modelling                   # noqa: E402
import odds as odds_mod            # noqa: E402
import ratings                     # noqa: E402
import score_matrix as sm          # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


# --------------------------------------------------------------------------- #
# 1. coherence of the score matrix
# --------------------------------------------------------------------------- #
def test_score_matrix() -> None:
    print("\n--- score matrix coherence ---")
    cases = [(1.6, 1.1, -0.08), (0.7, 2.4, 0.05), (3.1, 0.4, 0.0), (1.0, 1.0, -0.2)]
    sums, hda, ou, rng, btts, mean_ok = [], [], [], [], [], []
    for lh, la, rho in cases:
        m = sm.score_matrix(lh, la, rho=rho, max_goals=config.MAX_GOALS)
        sums.append(abs(m.sum() - 1.0))
        hda.append(abs(sum(sm.market_probability(m, "result", r)
                           for r in "HDA") - 1.0))
        ou.append(abs(sm.market_probability(m, "total_over", 2.5)
                      + sm.market_probability(m, "total_under", 2.5) - 1.0))
        # home 1-3 plus home 0 plus home 4+ must be the whole distribution
        p13 = sm.market_probability(m, "team_range", ("home", 1, 3))
        p0 = 1.0 - sm.market_probability(m, "team_over", ("home", 0.5))
        p4 = sm.market_probability(m, "team_over", ("home", 3.5))
        rng.append(abs(p13 + p0 + p4 - 1.0))
        # both teams score, computed a second way
        pa0 = 1.0 - sm.market_probability(m, "team_over", ("away", 0.5))
        btts.append(abs(sm.market_probability(m, "btts")
                        - (1.0 - p0 - pa0 + m[0, 0])))
        if rho == 0.0:
            s = sm.matrix_summary(m)
            mean_ok.append(abs(s["exp_home_goals"] - lh) < 5e-3
                           and abs(s["exp_away_goals"] - la) < 5e-3)

    check("score matrix sums to 1", max(sums) < 1e-12, f"max error {max(sums):.2e}")
    check("home + draw + away = 1", max(hda) < 1e-12, f"max error {max(hda):.2e}")
    check("over 2.5 + under 2.5 = 1", max(ou) < 1e-12, f"max error {max(ou):.2e}")
    check("multigol 1-3 partitions with 0 and 4+", max(rng) < 1e-12,
          f"max error {max(rng):.2e}")
    check("both-teams-score agrees with its complement form", max(btts) < 1e-12,
          f"max error {max(btts):.2e}")
    check("with rho = 0 the matrix mean equals lambda", all(mean_ok))

    # the Dixon-Coles correction must move the four cells in the known direction
    m0 = sm.score_matrix(1.5, 1.2, rho=0.0)
    mn = sm.score_matrix(1.5, 1.2, rho=-0.10)
    check("negative rho lifts 0-0 and 1-1, lowers 1-0 and 0-1",
          mn[0, 0] > m0[0, 0] and mn[1, 1] > m0[1, 1]
          and mn[1, 0] < m0[1, 0] and mn[0, 1] < m0[0, 1])

    # Truncation must be harmless at the most lopsided rates the data produces,
    # not merely at an average one - the matrix is renormalised after cutting
    # the tail, so a fat tail contaminates every cell.
    worst = 0.0
    for lh, la in [(2.0, 1.8), (4.2, 0.4), (3.5, 3.2), (4.5, 2.0)]:
        cut = sm.score_matrix(lh, la, rho=0.0, max_goals=config.MAX_GOALS)
        ref = sm.score_matrix(lh, la, rho=0.0, max_goals=22)
        for kind, arg in [("total_over", 2.5), ("team_range", ("home", 1, 3)),
                          ("result", "H"), ("btts", None)]:
            worst = max(worst, abs(sm.market_probability(cut, kind, arg)
                                   - sm.market_probability(ref, kind, arg)))
    check(f"truncating at {config.MAX_GOALS} goals does not move the markets",
          worst < 1e-4, f"largest move {worst:.2e} across four extreme fixtures")


# --------------------------------------------------------------------------- #
# 2. the fitters recover known parameters
# --------------------------------------------------------------------------- #
def test_rho_recovery() -> None:
    print("\n--- parameter recovery ---")
    rng = np.random.default_rng(0)
    n = 60_000
    lam_h = rng.uniform(0.8, 2.2, n)
    lam_a = rng.uniform(0.6, 1.9, n)
    true_rho = -0.09
    hg = np.empty(n, dtype=int)
    ag = np.empty(n, dtype=int)
    for i in range(n):
        m = sm.score_matrix(lam_h[i], lam_a[i], rho=true_rho, max_goals=8)
        k = rng.choice(m.size, p=m.ravel())
        hg[i], ag[i] = divmod(k, m.shape[1])
    got = sm.fit_rho(hg, ag, lam_h, lam_a)
    check("fit_rho recovers a planted rho", abs(got - true_rho) < 0.02,
          f"planted {true_rho:+.3f}, recovered {got:+.3f}")


def test_dixon_coles_recovery() -> None:
    rng = np.random.default_rng(1)
    teams = [f"T{i:02d}" for i in range(20)]
    attack = rng.normal(0, 0.35, 20)
    attack -= attack.mean()
    defence = rng.normal(0, 0.30, 20)
    home_adv = 0.22

    rows = []
    base = pd.Timestamp("2020-08-01")
    k = 0
    for _rep in range(8):                      # 8 double round robins
        for i in range(20):
            for j in range(20):
                if i == j:
                    continue
                lh = np.exp(attack[i] - defence[j] + home_adv)
                la = np.exp(attack[j] - defence[i])
                rows.append({
                    "match_id": f"m{k}",
                    "home_team": teams[i], "away_team": teams[j],
                    "home_goals": rng.poisson(lh), "away_goals": rng.poisson(la),
                    "kickoff": base + pd.Timedelta(days=k // 10),
                })
                k += 1
    sim = pd.DataFrame(rows)

    # xi = 0 so every simulated match counts equally, as they were generated
    model = ratings.fit(sim, xi=0.0)
    atk_err = float(np.abs(model.attack - attack).max())
    def_err = float(np.abs(model.defence - defence).max())
    check("Dixon-Coles recovers planted attack ratings", atk_err < 0.12,
          f"max abs error {atk_err:.3f}")
    check("Dixon-Coles recovers planted defence ratings", def_err < 0.12,
          f"max abs error {def_err:.3f}")
    check("Dixon-Coles recovers the home advantage",
          abs(model.home_adv - home_adv) < 0.05,
          f"planted {home_adv:.3f}, recovered {model.home_adv:.3f}")
    check("Dixon-Coles attack ratings are centred",
          abs(float(model.attack.mean())) < 1e-8)


def test_implied_lambdas_roundtrip() -> None:
    truth = [(1.9, 1.0), (1.2, 1.4), (2.6, 0.7), (0.9, 2.1)]
    ph, pdr, po = [], [], []
    for lh, la in truth:
        m = sm.score_matrix(lh, la, rho=0.0)
        ph.append(sm.market_probability(m, "result", "H"))
        pdr.append(sm.market_probability(m, "result", "D"))
        po.append(sm.market_probability(m, "total_over", 2.5))
    got_h, got_a = odds_mod.implied_lambdas(ph, pdr, po)
    err = max(max(abs(got_h[i] - truth[i][0]), abs(got_a[i] - truth[i][1]))
              for i in range(len(truth)))
    check("market probabilities invert back to the lambdas that made them",
          err < 1e-3, f"max abs error {err:.2e}")


def test_devig() -> None:
    p = odds_mod.devig_proportional(1 / np.array([2.0]), 1 / np.array([3.5]),
                                    1 / np.array([4.0]))
    total = float(sum(x[0] for x in p))
    check("de-vigged probabilities sum to 1", abs(total - 1.0) < 1e-12,
          f"sum {total:.12f}")


# --------------------------------------------------------------------------- #
# 3. the built tables
# --------------------------------------------------------------------------- #
def test_tables(tm: pd.DataFrame, match: pd.DataFrame) -> None:
    print("\n--- built tables ---")
    check("two team rows per match",
          bool((tm.groupby("match_id").size() == 2).all()),
          f"{tm['match_id'].nunique():,} matches")
    check("exactly one home and one away row per match",
          bool((tm.groupby("match_id")["is_home"].sum() == 1).all()))
    check("match table has one row per match",
          len(match) == match["match_id"].nunique() == tm["match_id"].nunique(),
          f"{len(match):,} rows")
    check("goals are non-negative integers",
          bool((match["home_goals"] >= 0).all() and (match["away_goals"] >= 0).all()
               and (match["home_goals"] % 1 == 0).all()))
    check("result column agrees with the scoreline",
          bool((np.select([match.home_goals > match.away_goals,
                           match.home_goals == match.away_goals],
                          ["H", "D"], default="A") == match["result"]).all()))

    # the independent odds feed is a second opinion on every scoreline
    both = match[match["fd_home_goals"].notna()]
    agree = float(((both["home_goals"] == both["fd_home_goals"])
                   & (both["away_goals"] == both["fd_away_goals"])).mean())
    check("scorelines agree with football-data.co.uk", agree == 1.0,
          f"{agree:.4%} of {len(both):,} matches")
    # Every match must join the closing line.  This is the only external check
    # the project has on club identity, and it is what caught "Newcastle" v
    # "Newcastle Utd" (74 fixtures) and "Coventry"/"Hull" v "Coventry City"/
    # "Hull City" (7 fixtures).  Do not weaken it.
    #
    # The one legitimate absence is the tail of the live season: football-data
    # publishes weekly, so the newest round has been played on FBref before it
    # has a price.  That shows as a *contiguous trailing block* by kickoff.
    # Name drift shows as gaps scattered through the history instead, so the
    # gate below still fails on drift while tolerating a stale feed.
    priced = match["mkt_lambda_home"].notna()
    n_missing = int((~priced).sum())
    last_priced = match.loc[priced, "kickoff"].max()
    trailing_only = bool((match.loc[~priced, "kickoff"] > last_priced).all())
    check("every match carries a closing line, bar an unpublished tail",
          n_missing == 0 or trailing_only,
          f"{n_missing} missing"
          + (f", all after the last priced match ({last_priced:%Y-%m-%d})"
             if n_missing else ""))
    check("completed seasons are fully priced",
          bool(match.loc[~match["season"].isin(config.TEST_SEASONS),
                         "mkt_lambda_home"].notna().all()),
          f"{int(match.loc[~match['season'].isin(config.TEST_SEASONS), 'mkt_lambda_home'].isna().sum())} missing")

    feats = modelling.feature_names(config)
    post_match = {"goals_for", "goals_against", "shot_xg", "opp_shot_xg", "points",
                  "team_yellow", "shots", "shots_on_target", "poss_share",
                  "home_goals", "away_goals", "total_goals", "result"}
    leaked = sorted(post_match & set(feats))
    check("no post-match quantity is used as a feature", not leaked, f"{leaked}")
    check("feature count is small", len(feats) <= 30, f"{len(feats)} features")

    tr = set(config.TRAIN_SEASONS)
    check("train, validation and test seasons are disjoint",
          not (tr & set(config.VALID_SEASONS)) and not (tr & set(config.TEST_SEASONS))
          and not (set(config.VALID_SEASONS) & set(config.TEST_SEASONS)))
    check("validation precedes test", max(config.VALID_SEASONS) < min(config.TEST_SEASONS))


# --------------------------------------------------------------------------- #
# 4. leakage
# --------------------------------------------------------------------------- #
def test_no_leakage(tm: pd.DataFrame) -> None:
    print("\n--- leakage gate (rebuild rolling form from truncated history) ---")
    cutoff = tm["kickoff"].quantile(0.6)
    past = tm[tm["kickoff"] < cutoff].copy()

    def rebuild(src: pd.DataFrame) -> pd.DataFrame:
        form = bmt.build_team_form(src)
        out = bmt._asof(src, form, by="team")
        cols = [c for c in out.columns if c.startswith("f_")]
        return out[["match_id", "team"] + cols].sort_values(
            ["match_id", "team"]).reset_index(drop=True)

    full = rebuild(tm)
    trunc = rebuild(past)
    keys = set(zip(past["match_id"], past["team"]))
    full_sub = full[[k in keys for k in zip(full["match_id"], full["team"])]]
    full_sub = full_sub.reset_index(drop=True)

    check("truncated rebuild has the same rows",
          len(full_sub) == len(trunc), f"{len(full_sub):,} vs {len(trunc):,}")
    check("truncated rebuild has the same row order",
          bool((full_sub["match_id"].to_numpy() == trunc["match_id"].to_numpy()).all()
               and (full_sub["team"].to_numpy() == trunc["team"].to_numpy()).all()))
    num = [c for c in trunc.columns if c.startswith("f_")]
    diffs = {c: float(np.nanmax(np.abs(full_sub[c].to_numpy() - trunc[c].to_numpy())))
             for c in num}
    worst = max(diffs.values()) if diffs else 0.0
    check("NO ROLLING FEATURE USES FUTURE INFORMATION", worst < 1e-12,
          f"largest change after deleting later matches: {worst:.2e} "
          f"over {len(num)} features")

    # and the Dixon-Coles rolling ratings must never see their own matchweek
    dc_path = config.ARTIFACT_DIR / "dc_ratings.parquet"
    if dc_path.exists():
        dc = pd.read_parquet(dc_path)
        check("rolling ratings cover most matches but never the first ones",
              len(dc) < len(tm["match_id"].unique()) and len(dc) > 2000,
              f"{len(dc):,} of {tm['match_id'].nunique():,} matches rated "
              f"(the earliest are used only to warm the ratings up)")


def main() -> None:
    tm = pd.read_parquet(config.TEAM_MATCH_PATH)
    match = pd.read_parquet(config.MATCH_DATASET_PATH)
    print(f"team-match rows: {len(tm):,}   matches: {len(match):,}")

    test_score_matrix()
    test_rho_recovery()
    test_dixon_coles_recovery()
    test_implied_lambdas_roundtrip()
    test_devig()
    test_tables(tm, match)
    test_no_leakage(tm)

    print("\n" + "=" * 62)
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED: " + ", ".join(FAILURES))
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
