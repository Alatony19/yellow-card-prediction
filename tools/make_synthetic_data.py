"""Generate a synthetic Premier-League-shaped archive, so the pipeline and its tests run
on a fresh clone without any scraped data.

Everything here is invented: the clubs, the players, the referees and the scores. The
files have exactly the layout the real FBref scrape and the football-data.co.uk download
produce, so the *real* parser (`pipeline/build_base_table.py`), feature build
(`pipeline/features.py`) and leakage gates (`tests/test_pipeline.py`) run unchanged on it.

    python tools/make_synthetic_data.py                  # writes data_synthetic/
    python tools/demo.py                                 # generate + build + test, in one go

The synthetic world has just enough structure to be learnable - players differ in how
often they are booked, defenders and holding midfielders are booked more, underdogs and
tight matches are dirtier, referees differ a little - but none of it is real football, so
no number produced from it means anything about real matches.

Layout written (relative to ``--out``):

    fbref/premier_league/<season>/match_day_N/<Home>_<Away>/{main,lineups,officials,
                                                              home_player_stats,
                                                              away_player_stats}.csv
    odds/premier_league/<YYYY>-<YY>.csv
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

V3_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3_DIR))
import paths  # noqa: E402,F401 - puts pipeline/, serve/, research/ on sys.path

import config  # noqa: E402

CLUBS = [
    "Ashford", "Brackley", "Carnwell", "Dunmere", "Eastbury", "Fenwick", "Garrowby",
    "Halworth", "Ivybridge", "Jarrow Vale", "Kingsmead", "Lowmoor", "Marlbury", "Northvale",
    "Oakhaven", "Pennick", "Queensgate", "Redholm", "Stonebridge", "Thornley",
    "Upminster Park", "Valeford", "Westcombe", "Yarrowmoor", "Zennor", "Aldermoor",
]
FIRST = ["Aiden", "Basil", "Cormac", "Dario", "Elio", "Fabrice", "Gunnar", "Hector", "Ivo",
         "Jonas", "Kasper", "Luca", "Matteo", "Nils", "Otto", "Pavel", "Quentin", "Rafael",
         "Stefan", "Tomas", "Ulrich", "Viktor", "Willem", "Xavier", "Yannick", "Zoran",
         "Adrian", "Bruno", "Cedric", "Damian", "Emil", "Felix", "Gregor", "Hugo", "Isak",
         "Julian", "Karim", "Leandro", "Marek", "Nikola", "Oscar", "Pedro", "Rhys", "Silas",
         "Tobias", "Uriel", "Vasco", "Wesley", "Yusuf", "Zeno"]
LAST = ["Abrams", "Bellweather", "Castellan", "Dorsey", "Eklund", "Farrow", "Galvez",
        "Hartigan", "Imrie", "Jankovic", "Kessler", "Lindqvist", "Moreau", "Nakamura",
        "Ortega", "Petrov", "Quill", "Rasmussen", "Saldana", "Thorne", "Ulloa", "Varga",
        "Whitlock", "Xander", "Yilmaz", "Zielinski", "Ashdown", "Brandt", "Calloway",
        "Delacroix", "Esposito", "Fairweather", "Grimaldi", "Holloway", "Ivanov", "Jurgens",
        "Kowalczyk", "Lambert", "Marchetti", "Novak", "Okonkwo", "Pelletier", "Rinaldi",
        "Sorensen", "Tamburri", "Underhill", "Vasquez", "Wexler", "Yamada", "Zimmer",
        "Aldridge", "Bannister", "Cordova", "Dunleavy", "Eastwood", "Fontaine", "Gallagher",
        "Hendry", "Iglesias", "Jessop", "Kavanagh", "Lockhart", "Mahoney", "Nyberg"]
REFEREES = [f"{f} {l}" for f, l in zip(
    ["Arden", "Bryce", "Corwin", "Dale", "Edric", "Floyd", "Garth", "Heath", "Ingram", "Jory",
     "Kemp", "Lyle", "Merrick", "Nolan", "Orson", "Perry", "Quincy", "Royce", "Sloane",
     "Tavis", "Vance", "Walden", "Yorick", "Zane", "Alder"],
    ["Pryce", "Hollis", "Draycott", "Marlowe", "Ashby", "Kincaid", "Voss", "Brennan", "Calder",
     "Fenner", "Gresham", "Haldane", "Jessup", "Lorimer", "Maddox", "Norwood", "Overton",
     "Pembroke", "Radley", "Sutcliffe", "Tennant", "Upton", "Wetherby", "Yardley", "Zeller"])]

FORMATIONS = {
    "4-3-3": ["GK", "CB", "CB", "LB", "RB", "DM", "CM", "CM", "W", "W", "FW"],
    "4-2-3-1": ["GK", "CB", "CB", "LB", "RB", "DM", "DM", "AM", "W", "W", "FW"],
    "4-4-2": ["GK", "CB", "CB", "LB", "RB", "CM", "CM", "W", "W", "FW", "FW"],
    "3-5-2": ["GK", "CB", "CB", "CB", "DM", "CM", "CM", "LB", "RB", "FW", "FW"],
}
SQUAD_SHAPE = {"GK": 3, "CB": 4, "LB": 2, "RB": 2, "DM": 3, "CM": 3, "AM": 2, "W": 4, "FW": 3}
# yellow cards per 90 by position, before the player's own propensity
CARD_BASE = {"GK": 0.03, "CB": 0.17, "LB": 0.14, "RB": 0.14, "DM": 0.20, "CM": 0.13,
             "AM": 0.09, "W": 0.085, "FW": 0.10}
FOUL_BASE = {"GK": 0.1, "CB": 0.95, "LB": 0.8, "RB": 0.8, "DM": 1.3, "CM": 1.0,
             "AM": 0.8, "W": 0.7, "FW": 1.0}
GOAL_W = {"GK": 0.0, "CB": 0.5, "LB": 0.6, "RB": 0.6, "DM": 0.5, "CM": 1.5, "AM": 3.0,
          "W": 3.0, "FW": 5.0}
KICKOFFS = ["12:30", "15:00", "15:00", "15:00", "15:00", "17:30", "16:30", "20:00"]
FIRST_SATURDAY = {  # the Saturday each season opens on (any consistent choice would do)
    2017: dt.date(2017, 8, 12), 2018: dt.date(2018, 8, 11), 2019: dt.date(2019, 8, 10),
    2020: dt.date(2020, 9, 12), 2021: dt.date(2021, 8, 14), 2022: dt.date(2022, 8, 6),
    2023: dt.date(2023, 8, 12), 2024: dt.date(2024, 8, 17), 2025: dt.date(2025, 8, 16),
    2026: dt.date(2026, 8, 15),
}
PARTIAL_SEASON = {"2026-2027": 6}   # matchweeks played so far in the live season


def flat(n: str) -> str:
    return "".join(ch for ch in n.lower() if ch.isalnum())


class World:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.shot_rng = np.random.default_rng(seed + 1)   # separate stream: shots never perturb the rest
        names = [flat(c) for c in CLUBS]
        assert not any(a != b and a in b for a in names for b in names), "club names nest"
        self.club_att = {c: self.rng.normal(0, 0.22) for c in CLUBS}
        self.club_def = {c: self.rng.normal(0, 0.22) for c in CLUBS}
        self.club_strict = {c: self.rng.normal(0, 0.12) for c in CLUBS}
        self.ref_strict = {r: self.rng.normal(0, 0.16) for r in REFEREES}
        self.used_names: set[str] = set()
        self.next_shirt = {}
        self.squads: dict[str, list[dict]] = {}
        self.table: list[str] = list(CLUBS[:20])
        for c in CLUBS:
            self.squads[c] = self._new_squad()

    # ------------------------------------------------------------ players
    def _new_player(self, pos: str) -> dict:
        while True:
            name = f"{self.rng.choice(FIRST)} {self.rng.choice(LAST)}"
            if name not in self.used_names:
                self.used_names.add(name)
                break
        return {
            "name": name, "pos": pos,
            "quality": float(self.rng.normal(0, 1)),
            "prop": float(math.exp(self.rng.normal(-0.125, 0.5))),   # card propensity, mean ~1
            "born": 1988 + int(self.rng.integers(0, 15)),
        }

    def _new_squad(self) -> list[dict]:
        return [self._new_player(p) for p, n in SQUAD_SHAPE.items() for _ in range(n)]

    def new_season(self, first: bool) -> None:
        if first:
            return
        strength = {c: self.club_att[c] - self.club_def[c] + self.rng.normal(0, 0.15)
                    for c in self.table}
        ranked = sorted(self.table, key=strength.get)
        relegated, stay = ranked[:3], ranked[3:]
        pool = [c for c in CLUBS if c not in self.table]
        promoted = list(self.rng.choice(pool, 3, replace=False))
        for c in promoted:      # promoted sides start weaker
            self.club_att[c] -= 0.1
            self.club_def[c] += 0.1
        self.table = stay + promoted
        for c in CLUBS:         # a little drift, and squad turnover
            self.club_att[c] += self.rng.normal(0, 0.06)
            self.club_def[c] += self.rng.normal(0, 0.06)
            squad = self.squads[c]
            for i in self.rng.choice(len(squad), 4, replace=False):
                squad[i] = self._new_player(squad[i]["pos"])

    # ------------------------------------------------------------ fixtures
    def fixtures(self, clubs: list[str]) -> list[list[tuple[str, str]]]:
        """Circle-method round robin: 38 rounds, every pair home and away once."""
        order = list(self.rng.permutation(clubs))
        n = len(order)
        rounds = []
        for r in range(n - 1):
            pairs = []
            for i in range(n // 2):
                a, b = order[i], order[n - 1 - i]
                pairs.append((a, b) if (r + i) % 2 == 0 else (b, a))
            rounds.append(pairs)
            order = [order[0]] + [order[-1]] + order[1:-1]
        second = [[(b, a) for a, b in rnd] for rnd in rounds]
        allr = rounds + second
        idx = self.rng.permutation(len(allr))
        return [allr[i] for i in idx]

    def lambdas(self, home: str, away: str) -> tuple[float, float]:
        lh = math.exp(0.22 + 0.25 + self.club_att[home] - self.club_def[away])
        la = math.exp(0.22 + self.club_att[away] - self.club_def[home])
        return lh, la

    # ------------------------------------------------------------ one side
    def pick_xi(self, club: str, formation: str):
        squad = self.squads[club]
        used: set[int] = set()
        xi = []
        for slot in FORMATIONS[formation]:
            cands = [i for i, p in enumerate(squad)
                     if p["pos"] == slot and i not in used and self.rng.random() < 0.93]
            if not cands:
                cands = [i for i, p in enumerate(squad) if i not in used
                         and (p["pos"] != "GK") == (slot != "GK")]
            w = np.array([math.exp(0.7 * squad[i]["quality"]) for i in cands])
            i = int(self.rng.choice(cands, p=w / w.sum()))
            used.add(i)
            xi.append((i, slot))
        rest = [i for i in range(len(squad)) if i not in used]
        gk = [i for i in rest if squad[i]["pos"] == "GK"]
        field = [i for i in rest if squad[i]["pos"] != "GK"]
        bench = gk[:1] + list(self.rng.choice(field, 6, replace=False))
        return xi, [int(b) for b in bench]


def season_files(world: World, season: str, out: Path, max_rounds: int | None) -> list[dict]:
    start_year = int(season[:4])
    rounds = world.fixtures(world.table)
    n_rounds = PARTIAL_SEASON.get(season, len(rounds))
    if max_rounds:
        n_rounds = min(n_rounds, max_rounds)
    first_sat = FIRST_SATURDAY[start_year]
    base = out / "fbref" / "premier_league" / season
    odds_rows = []
    sub_cap = 3 if start_year < 2022 else 5          # the five-substitution rule
    for mw in range(1, n_rounds + 1):
        date0 = first_sat + dt.timedelta(days=7 * (mw - 1))
        for home, away in rounds[mw - 1]:
            day = date0 + dt.timedelta(days=int(world.rng.choice([0, 0, 0, 1, 1, 2])))
            ko = str(world.rng.choice(KICKOFFS))
            ref = str(world.rng.choice(REFEREES))
            lh, la = world.lambdas(home, away)
            goals = {home: int(world.rng.poisson(lh)), away: int(world.rng.poisson(la))}
            tight = 1.0 - abs(lh - la) / (lh + la)
            sides = {}
            for club, opp, is_home, lam, lam_opp in ((home, away, True, lh, la), (away, home, False, la, lh)):
                fmt = str(world.rng.choice(list(FORMATIONS)))
                xi, bench = world.pick_xi(club, fmt)
                sides[club] = (fmt, xi, bench, is_home, lam, lam_opp)
            stats = {}
            for club in (home, away):
                fmt, xi, bench, is_home, lam, lam_opp = sides[club]
                stats[club] = side_rows(world, club, xi, bench, goals[club], lam < lam_opp, tight, ref, sub_cap)
            folder = base / f"match_day_{mw}" / f"{home}_{away}"
            write_match(world, folder, mw, day, ko, ref, home, away, sides, stats)
            if start_year <= 2024:      # FBref stopped publishing shot-level xG after 2024-25
                write_shots(world, folder, home, away, goals, lh, la, stats)
            odds_rows.append(odds_row(world, season, day, ko, home, away, goals, lh, la, ref))
    return odds_rows


def side_rows(world, club, xi, bench, team_goals, underdog, tight, ref, sub_cap):
    rng, squad = world.rng, world.squads[club]
    ctx = math.exp(0.12 * underdog + 0.10 * (tight - 0.8) + world.club_strict[club]
                   + world.ref_strict[ref])
    n_subs = int(rng.choice(range(0, sub_cap + 1), p=_sub_probs(sub_cap)))
    non_gk = [k for k, (_, slot) in enumerate(xi) if slot != "GK"]
    replaced = sorted(rng.choice(non_gk, min(n_subs, len(non_gk)), replace=False)) if n_subs else []
    bench_out = [b for b in bench if squad[b]["pos"] != "GK"]
    rng.shuffle(bench_out)
    plays: list[list[tuple[int, int, int]]] = []   # per slot: [(squad_idx, minute_on, minute_off)]
    for k, (i, slot) in enumerate(xi):
        slot_plays = [[i, 0, 90]]
        if k in replaced and bench_out:
            m = int(rng.integers(46, 89))
            j = bench_out.pop()
            slot_plays[0][2] = m
            slot_plays.append([j, m, 90])
            if rng.random() < 0.04 and bench_out:       # a substitute substituted
                m2 = int(rng.integers(m + 3, 90)) if m + 3 < 90 else 90
                if m2 < 90:
                    k2 = bench_out.pop()
                    slot_plays[1][2] = m2
                    slot_plays.append([k2, m2, 90])
        plays.append(slot_plays)
    rows = []
    for slot_plays in plays:
        for seat, (i, on, off) in enumerate(slot_plays):
            p = squad[i]
            pos = p["pos"] if seat == 0 else p["pos"]
            minutes = off - on
            red = int(rng.random() < 0.0035 * ctx and pos != "GK")
            if red and seat == 0 and len(slot_plays) == 1:
                minutes = int(rng.integers(20, 89))
            lam_y = CARD_BASE[pos] * 1.18 * p["prop"] * ctx * minutes / 90.0
            yellow = int(rng.random() < 1.0 - math.exp(-lam_y))
            fouls = int(rng.poisson(FOUL_BASE[pos] * (p["prop"] ** 0.6) * ctx * minutes / 90.0))
            rows.append({"idx": i, "name": p["name"], "pos": pos, "born": p["born"],
                         "minutes": minutes, "yellow": yellow, "red": red, "fouls": fouls,
                         "fouled": int(rng.poisson(0.9 * minutes / 90.0)),
                         "crosses": int(rng.poisson((1.6 if pos in ("W", "LB", "RB") else 0.4) * minutes / 90)),
                         "tklw": int(rng.poisson((1.1 if pos in ("DM", "CM", "CB") else 0.6) * minutes / 90)),
                         "ints": int(rng.poisson(0.9 * minutes / 90)),
                         "off": int(rng.poisson((0.5 if pos in ("FW", "W") else 0.04) * minutes / 90)),
                         "shots": int(rng.poisson((1.4 if pos in ("FW", "W", "AM") else 0.3) * minutes / 90)),
                         "seat": seat, "is_start": seat == 0, "goals": 0})
    w = np.array([GOAL_W[r["pos"]] * r["minutes"] for r in rows], dtype=float)
    if team_goals and w.sum() > 0:
        for g in rng.choice(len(rows), team_goals, p=w / w.sum()):
            rows[int(g)]["goals"] += 1
    return rows, xi, bench, plays


def _sub_probs(cap):
    base = np.array([0.04, 0.05, 0.12, 0.30, 0.25, 0.24][: cap + 1], dtype=float)
    return base / base.sum()


def write_match(world, folder, mw, day, ko, ref, home, away, sides, stats):
    folder.mkdir(parents=True, exist_ok=True)
    rng = world.rng
    pd.DataFrame([{
        "date_text": day.strftime("%A %B %d, %Y"), "venue_date": day.isoformat(), "venue_time": ko,
        "venue_epoch": 0, "local_time": ko, "competition": "Premier League", "matchweek": mw,
        "team_a": home, "team_b": away, "attendance": int(rng.integers(18000, 60000)),
        "venue": f"{home} Park", "referee": ref, "ar1": "A. Assistant", "ar2": "B. Assistant",
        "fourth_official": "C. Fourth", "var": "D. Video",
    }]).to_csv(folder / "main.csv", index=False)
    pd.DataFrame({"role": ["referee", "ar1", "ar2", "fourth_official", "var"],
                  "official": [ref, "A. Assistant", "B. Assistant", "C. Fourth", "D. Video"]}
                 ).to_csv(folder / "officials.csv", index=False)

    # lineups.csv: two 3-column blocks, XI then a Bench separator then the bench
    blocks = {}
    for club, side in ((home, "home"), (away, "away")):
        fmt, xi, bench, *_ = sides[club]
        rows_, _xi, _bench, plays = stats[club]
        squad = world.squads[club]
        shirt = {}
        for n, (i, _slot) in enumerate(xi):
            shirt[i] = n + 1
        for n, b in enumerate(bench):
            shirt[b] = 12 + n
        lines = [(shirt[i], squad[i]["name"], side) for i, _ in xi]
        lines.append(("Bench", "Bench", side))
        lines += [(shirt[b], squad[b]["name"], side) for b in bench]
        blocks[side] = (fmt, lines)
    n = max(len(blocks["home"][1]), len(blocks["away"][1]))
    pad = lambda ls: ls + [("", "", "")] * (n - len(ls))  # noqa: E731
    h, a = pad(blocks["home"][1]), pad(blocks["away"][1])
    hh, ah = f"{home} ({blocks['home'][0]})", f"{away} ({blocks['away'][0]})"
    with open(folder / "lineups.csv", "w", encoding="utf-8", newline="") as fh:
        fh.write(f'{hh},{hh}.1,Side,{ah},{ah}.1,Side\n')
        for x, y in zip(h, a):
            fh.write(",".join(_cell(v) for v in (*x, *y)) + "\n")

    cols = ["Player", "#", "Nation", "Pos", "Age", "Min", "Performance_Gls", "Performance_Ast",
            "Performance_PK", "Performance_PKatt", "Performance_Sh", "Performance_SoT",
            "Performance_CrdY", "Performance_CrdR", "Performance_Fls", "Performance_Fld",
            "Performance_Off", "Performance_Crs", "Performance_TklW", "Performance_Int",
            "Performance_OG", "Performance_PKwon", "Performance_PKcon"]
    for club, fname in ((home, "home_player_stats.csv"), (away, "away_player_stats.csv")):
        rows_, xi, bench, plays = stats[club]
        shirt = {i: n + 1 for n, (i, _) in enumerate(xi)}
        shirt.update({b: 12 + n for n, b in enumerate(bench)})
        # the sheet is printed in slot order - starter, then whoever replaced him -
        # which is how substitution chains are recovered downstream
        recs = [[r["name"], float(shirt[r["idx"]]), "xx XXX",
                 (("LW" if r["idx"] % 2 else "RW") if r["pos"] == "W" else r["pos"]),
                 f"{2026 - r['born']}-{int(rng.integers(0, 365)):03d}", r["minutes"], r["goals"], 0, 0, 0,
                 r["shots"], min(r["shots"], int(rng.integers(0, 2))), r["yellow"], r["red"],
                 r["fouls"], r["fouled"], r["off"], r["crosses"], r["tklw"], r["ints"], 0, "", ""]
                for r in rows_]
        df = pd.DataFrame(recs, columns=cols)
        totals = {c: "" for c in cols}
        totals["Player"] = f"{len(df)} Players"
        for c in ("Min", "Performance_Gls", "Performance_CrdY", "Performance_CrdR", "Performance_Fls"):
            totals[c] = int(pd.to_numeric(df[c]).sum())
        pd.concat([df, pd.DataFrame([totals])]).to_csv(folder / fname, index=False)


def write_shots(world, folder, home, away, goals, lh, la, stats):
    """conbined_shots_stats.csv (sic): one row per shot, with xG, post-shot xG and outcome."""
    rng = world.shot_rng
    rows = []
    for club, lam in ((home, lh), (away, la)):
        n = max(goals[club], int(rng.poisson(5.5 + 3.5 * lam)))
        if n == 0:
            continue
        raw = rng.exponential(1.0, n)
        xg = np.clip(raw / raw.sum() * lam * math.exp(rng.normal(0, 0.1)), 0.01, 0.95)
        outcome = ["Goal"] * goals[club] + list(rng.choice(
            ["Saved", "Blocked", "Off Target", "Wayward"], n - goals[club], p=[0.3, 0.3, 0.25, 0.15]))
        rng.shuffle(outcome)
        starters = [r["name"] for r in stats[club][0] if r["is_start"] and r["pos"] != "GK"]
        for k in range(n):
            rows.append({"Minute": int(rng.integers(1, 91)), "Player": str(rng.choice(starters)),
                         "Squad": club, "xG": round(float(xg[k]), 2),
                         "PSxG": round(float(xg[k]) * 1.1, 2) if outcome[k] in ("Goal", "Saved") else 0.0,
                         "Outcome": outcome[k]})
    pd.DataFrame(rows).sort_values("Minute").to_csv(folder / "conbined_shots_stats.csv", index=False)


def _cell(v) -> str:
    s = str(v)
    return f'"{s}"' if "," in s else s


def poisson_pmf(lam, k):
    return math.exp(-lam) * lam ** k / math.factorial(k)


def odds_row(world, season, day, ko, home, away, goals, lh, la, ref):
    rng = world.rng
    lh2, la2 = lh * math.exp(rng.normal(0, 0.04)), la * math.exp(rng.normal(0, 0.04))   # the market is close, not exact
    ph = pd_ = pa = po = 0.0
    for i in range(11):
        for j in range(11):
            p = poisson_pmf(lh2, i) * poisson_pmf(la2, j)
            if i > j:
                ph += p
            elif i == j:
                pd_ += p
            else:
                pa += p
            if i + j > 2:
                po += p
    tot = ph + pd_ + pa
    ph, pd_, pa = ph / tot, pd_ / tot, pa / tot
    margin = 1.055
    price = lambda p: round(1.0 / (p * margin), 2)  # noqa: E731
    ftr = "H" if goals[home] > goals[away] else ("D" if goals[home] == goals[away] else "A")
    return {"Div": "E0", "Date": day.strftime("%d/%m/%Y"), "Time": ko, "HomeTeam": home, "AwayTeam": away,
            "FTHG": goals[home], "FTAG": goals[away], "FTR": ftr, "Referee": ref,
            "AvgH": price(ph), "AvgD": price(pd_), "AvgA": price(pa),
            "Avg>2.5": price(po / tot), "Avg<2.5": price(1 - po / tot)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Write a synthetic archive in the real layout.")
    ap.add_argument("--out", default=str(V3_DIR / "data_synthetic"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-rounds", type=int, default=None,
                    help="cap matchweeks per season, for a quick run (the tests need the full 38)")
    args = ap.parse_args()
    out = Path(args.out)
    world = World(args.seed)
    for k, season in enumerate(config.ALL_SEASONS):
        world.new_season(first=(k == 0))
        rows = season_files(world, season, out, args.max_rounds)
        odds_dir = out / "odds" / "premier_league"
        odds_dir.mkdir(parents=True, exist_ok=True)
        start, end = season.split("-")
        pd.DataFrame(rows).to_csv(odds_dir / f"{start}-{end[2:]}.csv", index=False)
        print(f"{season}: {len(rows)} matches")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
