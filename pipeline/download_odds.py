"""Fetch football-data.co.uk closing-price archives for the big five leagues.

The Premier League archive was already in hand (``data/premier_league/``); this
script generalises how it got there so the other four leagues arrive in exactly
the same shape, and so a new season is one command rather than a manual save.

Why this source
---------------
football-data.co.uk is the only free archive that publishes, for every match
since the 1990s, both the **closing** 1X2 line and the closing over/under 2.5
line from a market average.  Closing prices matter more than the identity of
the book: the closing line is the best-informed price of the day, and
``match_model/odds.py`` inverts the pair of them into the scoring rates the
market implies.  It also ships the referee name and the match's own card counts
(HY/AY/HR/AR), which is a free cross-check on the FBref scrape.

What it does NOT ship is a player-level card market.  Nobody publishes historic
"player to be booked" prices for free; they exist only behind paid feeds
(Betfair historical data, OddsPortal scrapes).  So the odds here inform the
model, they do not give it a benchmark to be scored against at player level.

Layout written
--------------
    <repo>/data/<league_dir>/<YYYY-YY>.csv

matching the existing ``data/premier_league/2024-25.csv`` exactly, so
``odds.load_odds(odds_dir=...)`` reads any league without a code change.

Usage
-----
    python pipeline/download_odds.py                     # all five leagues, all seasons
    python pipeline/download_odds.py --leagues serie_a bundesliga
    python pipeline/download_odds.py --seasons 2024-2025 2025-2026
    python pipeline/download_odds.py --force             # re-fetch even if present
"""
from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

V3_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = V3_DIR / "data" / "odds"

BASE_URL = "https://www.football-data.co.uk/mmz4281/{code}/{div}.csv"

# league key -> (football-data division code, directory name, display name)
LEAGUES = {
    "premier_league": ("E0", "premier_league", "Premier League"),
    "serie_a": ("I1", "serie_a", "Serie A"),
    "la_liga": ("SP1", "la_liga", "La Liga"),
    "bundesliga": ("D1", "bundesliga", "Bundesliga"),
    "ligue_1": ("F1", "ligue_1", "Ligue 1"),
}

# The card pipeline's ALL_SEASONS plus the earlier years the odds archive has,
# so a future burn-in extension does not need a second download.  The range runs
# one season past the last completed one: a season that has not started yet
# answers 200 with a header-only file, which download_league detects and skips
# rather than writing, so an over-long range costs nothing.
SEASONS = [f"{y}-{y + 1}" for y in range(2009, 2028)]

# A browser user-agent: the plain urllib default is answered with 403.
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def season_to_code(season: str) -> str:
    """"2024-2025" -> "2425", the football-data URL fragment."""
    start, end = season.split("-")
    return f"{start[2:]}{end[2:]}"


def season_to_file(season: str) -> str:
    """"2024-2025" -> "2024-25.csv", matching the existing archive naming."""
    start, end = season.split("-")
    return f"{start}-{end[2:]}.csv"


def fetch(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def download_league(league: str, seasons: list[str], force: bool,
                    delay: float) -> dict:
    div, dirname, display = LEAGUES[league]
    out_dir = DATA_DIR / dirname
    out_dir.mkdir(parents=True, exist_ok=True)

    written, skipped, missing = 0, 0, []
    for season in seasons:
        path = out_dir / season_to_file(season)
        if path.exists() and not force:
            skipped += 1
            continue
        url = BASE_URL.format(code=season_to_code(season), div=div)
        try:
            blob = fetch(url)
        except urllib.error.HTTPError as exc:
            missing.append(f"{season} (HTTP {exc.code})")
            continue
        except Exception as exc:                      # noqa: BLE001
            missing.append(f"{season} ({type(exc).__name__})")
            continue

        # A season that has not started yet answers 200 with a header-only file;
        # writing it would poison the archive with an empty season.
        try:
            probe = pd.read_csv(pd.io.common.BytesIO(blob), encoding="utf-8-sig")
        except Exception as exc:                      # noqa: BLE001
            missing.append(f"{season} (unparseable: {type(exc).__name__})")
            continue
        if "HomeTeam" not in probe.columns or probe["HomeTeam"].notna().sum() == 0:
            missing.append(f"{season} (no matches yet)")
            continue

        path.write_bytes(blob)
        written += 1
        time.sleep(delay)

    return {"league": display, "dir": dirname, "written": written,
            "skipped": skipped, "missing": missing}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--leagues", nargs="+", default=list(LEAGUES),
                    choices=list(LEAGUES))
    ap.add_argument("--seasons", nargs="+", default=SEASONS)
    ap.add_argument("--force", action="store_true",
                    help="re-download seasons already on disk")
    ap.add_argument("--delay", type=float, default=0.5)
    args = ap.parse_args()

    print(f"football-data.co.uk -> {DATA_DIR}")
    print(f"{len(args.leagues)} leagues x {len(args.seasons)} seasons\n")

    reports = []
    for league in args.leagues:
        rep = download_league(league, args.seasons, args.force, args.delay)
        reports.append(rep)
        line = (f"{rep['league']:<16} {rep['written']:>3} written  "
                f"{rep['skipped']:>3} already there")
        if rep["missing"]:
            line += f"  {len(rep['missing'])} unavailable"
        print(line)
        for m in rep["missing"]:
            print(f"                   - {m}")

    total = sum(r["written"] for r in reports)
    print(f"\n{total} season files written.")


if __name__ == "__main__":
    main()
