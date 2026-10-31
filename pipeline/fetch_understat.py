"""Pull Understat team-match statistics for the big five leagues.

Understat publishes, for every league match since 2014-15, each side's xG,
non-penalty xG, PPDA (opponent passes allowed per defensive action - lower
means harder pressing) and deep completions.  None of it is on FBref's reduced
2025-26 report, and pressing is a direct read on how often a side engages the
ball carrier, which is where fouls - and cards - come from.

Run it from its own virtual environment (``pip install soccerdata``), because
``soccerdata`` pulls in package versions that clash with the selenium that
``scrape_fbref.py`` drives:

    python pipeline/fetch_understat.py
    python pipeline/fetch_understat.py --seasons 2627

Writes ``artifacts/understat/team_match.csv`` (CSV, not parquet: that
environment has no pyarrow).  ``understat_features.py`` joins it onto the card pipeline.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import soccerdata as sd

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "understat" / "team_match.csv"

LEAGUES = {
    "ENG-Premier League": "premier_league",
    "ITA-Serie A": "serie_a",
    "ESP-La Liga": "la_liga",
    "GER-Bundesliga": "bundesliga",
    "FRA-Ligue 1": "ligue_1",
}
SEASONS = [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(2017, 2027)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seasons", nargs="+", default=SEASONS)
    args = ap.parse_args()

    frames = []
    for sd_league, key in LEAGUES.items():
        for season in args.seasons:
            # The current season must be re-read, not served from cache.
            live = season == SEASONS[-1]
            u = sd.Understat(leagues=sd_league, seasons=season, no_cache=live)
            tm = u.read_team_match_stats().reset_index()
            tm["league_key"] = key
            tm["season_label"] = f"20{season[:2]}-20{season[2:]}"
            frames.append(tm)
            print(f"{key:<15} {season}: {len(tm)} matches", flush=True)

    out = pd.concat(frames, ignore_index=True)
    if OUT.exists() and args.seasons != SEASONS:
        old = pd.read_csv(OUT)
        old = old[~old["season_label"].isin(out["season_label"].unique())]
        out = pd.concat([old, out], ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(out):,} matches")


if __name__ == "__main__":
    main()
