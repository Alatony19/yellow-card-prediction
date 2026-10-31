"""Verification gates for pipeline/history_features.py.

Run with:  python tests/test_history_features.py

* leakage    - rebuild the features from a history truncated at a cut-off date;
               every row before the cut-off must be identical.
* geometry   - the slot read from (formation, table order) must agree with the
               positions FBref prints: left-backs left, right-backs right,
               centre-backs at the back, forwards at the front.
* identity   - one birth date per in-league player key (bar namesakes), and
               linking across leagues actually links.
* brute force - a player's cross-league minutes recomputed by hand.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

V3_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V3_DIR))
import paths  # noqa: E402,F401

import history_features as hf  # noqa: E402
import pooled  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f"  -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def candidate_rows(tables: dict, a: pd.DataFrame, seasons: list[str]) -> pd.DataFrame:
    rows = []
    for lg, t in tables.items():
        s = t[t["is_starter"] & (t["pos_group_actual"] != "GK") & t["season"].isin(seasons)]
        rows.append(s[["match_id", "team", "opponent", "player", "player_key",
                       "kickoff", "Age"]].assign(league=lg))
    return pd.concat(rows, ignore_index=True)


def test_leakage(tables: dict, a: pd.DataFrame) -> None:
    seasons = ["2024-2025", "2025-2026"]
    cand = candidate_rows(tables, a, seasons)
    cut = pd.Timestamp("2025-11-01")
    pri = hf.default_priors(a)           # priors are fixed scalars, as in features.py
    full = hf.attach(cand, a, pri)
    early = cand[cand["kickoff"] < cut]
    trunc = hf.attach(early, a[a["kickoff"] < cut], pri)
    f, t = full.loc[early.index, hf.ALL_FEATURES], trunc[hf.ALL_FEATURES]
    same = np.isclose(f.to_numpy(float), t.to_numpy(float), equal_nan=True, atol=1e-12)
    bad = [c for c, ok in zip(hf.ALL_FEATURES, same.all(axis=0)) if not ok]
    check("no future information (truncated history)", not bad,
          f"{len(early):,} rows before {cut.date()}" + (f"; moved: {bad}" if bad else ""))


def test_geometry(tables: dict) -> None:
    p = tables[pooled.PL]
    slots = hf.slot_coordinates(p)
    j = p.join(slots, how="inner")
    first = j["pos_raw"].fillna("").str.split(",").str[0]
    m = j.groupby(first)[["x", "depth"]].mean()
    check("slot known for nearly every outfield start",
          slots["x"].notna().mean() > 0.98, f"{slots['x'].notna().mean():.3%}")
    check("left-backs on the left, right-backs on the right",
          m.loc["LB", "x"] < 0.25 and m.loc["RB", "x"] > 0.75,
          f"LB {m.loc['LB', 'x']:.2f}, RB {m.loc['RB', 'x']:.2f}")
    check("centre-backs at the back, forwards at the front",
          m.loc["CB", "depth"] < 0.05 and m.loc["FW", "depth"] > 0.8,
          f"CB depth {m.loc['CB', 'depth']:.2f}, FW {m.loc['FW', 'depth']:.2f}")


def test_identity(a: pd.DataFrame) -> None:
    spread = a.dropna(subset=["dob"]).groupby(["league", "player_key"])["dob"].nunique()
    check("one birth date per player key (namesakes aside)", (spread > 1).mean() < 0.01,
          f"{(spread > 1).mean():.3%} of keys carry two people")
    multi = a.groupby("ident")["league"].nunique()
    check("players linked across leagues", (multi > 1).sum() > 1000,
          f"{(multi > 1).sum():,} identities in 2+ leagues")


def test_bruteforce(tables: dict, a: pd.DataFrame) -> None:
    s = a[a["ident"].notna()].sort_values(["ident", "kickoff"], kind="mergesort")
    prev = s.groupby("ident")["league"].shift()
    n_before = s.groupby("ident").cumcount()
    # a start that is the player's first match after a league switch, with at
    # least a season's worth of appearances behind him
    switch = s[(s["league"] != prev) & prev.notna() & s["is_starter"] & (n_before >= 30)]
    r = switch[switch["season"] >= "2022-2023"].iloc[len(switch) // 3]
    h = s[s["ident"] == r["ident"]]
    cand = pd.DataFrame([{"league": r["league"], "match_id": r["match_id"], "team": r["team"],
                          "opponent": "", "player": r["player"], "player_key": r["player_key"],
                          "kickoff": r["kickoff"], "Age": None}])
    got = float(hf.attach(cand, a)["plr_xl_m90"].iloc[0])
    before = h[h["kickoff"] < r["kickoff"]]
    want = float(before.loc[before["league"] != r["league"], "minutes"].sum() / 90.0)
    check("brute-force cross-league minutes", abs(got - want) < 1e-9,
          f"{r['player']}: {got:.2f} vs {want:.2f} other-league 90s")


def main() -> int:
    try:
        hf.load_history()
    except FileNotFoundError as exc:
        print(f"SKIPPED: this gate needs the five-league base tables ({exc.filename or exc}).\n"
              "It needs the real data for all five leagues - see README, 'Getting the data'.")
        return 0
    tables = hf.base_tables()
    a = hf.load_history()
    test_geometry(tables)
    test_identity(a)
    test_bruteforce(tables, a)
    test_leakage(tables, a)
    print("\n" + "=" * 62)
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED: " + ", ".join(FAILURES))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
