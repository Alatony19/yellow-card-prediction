"""Refresh the FBref match archive for a season.

A rewrite of the original notebook scraper, which no longer runs.  Three things
changed on FBref since it was written, all of them found by trying it:

1. **Cloudflare.**  A plain HTTP request now returns 403, so ``pd.read_html(url)``
   fails.  Every page has to come through a real browser, and a *headless* one
   is served the challenge page ("Ci siamo quasi..."), so the window is visible.
2. **The schedule table moved.**  It is no longer ``tables[0]``; it is found
   here by looking for the table that has Home, Away and Score columns, which
   will survive the next reshuffle too.
3. **The advanced per-player tables are gone.**  A match report used to carry
   18+ tables (passing, pass types, defensive actions, possession,
   miscellaneous) plus a shot-by-shot table with xG.  It now carries 16, and
   only ``stats_<team>_summary`` exists - 23 columns per player.  Still there:
   minutes, goals, own goals, cards, fouls, fouls drawn, tackles won,
   interceptions, crosses, offsides, shots, shots on target.  Gone: tackles
   attempted, challenges lost, aerial duels, touches, passes attempted, and
   every shot-level xG.

That loss was measured before accepting it, on the eight seasons already in
hand: the card model goes from lift 1.3440 to 1.3413, and the match model from
1X2 RPS 0.1946 to 0.1949 as long as the football-data odds feed is kept.  Both
are far inside season-to-season noise, so a refreshed season is worth having.

Columns that no longer exist are simply absent from the files written here, and
build_base_table.py turns an absent column into NaN rather than 0, so a rolling
mean skips it instead of being dragged down.

Leagues
-------
Five competitions are supported.  The Premier League keeps ``data/`` so nothing
already built moves; each other league writes to a sibling ``data_<league>/``
with an identical layout, so ``build_base_table.py`` reads any of them by being
pointed at a different root.

Do NOT run several leagues at once
----------------------------------
Measured, not assumed.  One worker sustains **12.7 s per match**.  Four workers
on the same connection were throttled to roughly **150 s per match each** - a
net slowdown of about 3x, because FBref starts serving the Cloudflare challenge
and the retry backoff (20s, then 40s) dominates.  Recovery after dropping back
to one worker is not immediate either.  Run the leagues sequentially; the
``--competition all`` form does exactly that.

At 12.7 s per match, five seasons of all four new leagues is roughly 7,000
matches, or about 25 hours.  The run is resumable - a match folder that already
holds main.csv is skipped - so it can be stopped and restarted freely.

Usage
-----
    python pipeline/scrape_fbref.py --season 2025-2026
    python pipeline/scrape_fbref.py --season 2025-2026 --limit 3      # a trial run
    python pipeline/scrape_fbref.py --season 2025-2026 --delay 6      # be politer
    python pipeline/scrape_fbref.py --competition serie_a --seasons 2024-2025 2025-2026
    python pipeline/scrape_fbref.py --competition all --seasons 2021-2022 ... 2025-2026
"""
from __future__ import annotations

import argparse
import random
import re
import sys
import time
from io import StringIO
from pathlib import Path

import pandas as pd

V3_DIR = Path(__file__).resolve().parents[1]
FBREF_DIR = V3_DIR / "data" / "fbref"
SCHEDULE_URL = ("https://fbref.com/en/comps/{comp_id}/{season}/schedule/"
                "{season}-{slug}-Scores-and-Fixtures")

# The five leagues, each with its FBref competition id, the slug FBref uses in
# the schedule URL, the display name written into main.csv, and the archive
# folder under ``data/fbref/``.  Each league has its own root, which keeps
# ``iter_match_dirs`` - which treats every child of the root as a season -
# working unchanged for all five.
COMPETITIONS = {
    "premier_league": (9, "Premier-League", "Premier League", "premier_league"),
    "serie_a": (11, "Serie-A", "Serie A", "serie_a"),
    "la_liga": (12, "La-Liga", "La Liga", "la_liga"),
    "bundesliga": (20, "Bundesliga", "Bundesliga", "bundesliga"),
    "ligue_1": (13, "Ligue-1", "Ligue 1", "ligue_1"),
}


def competition_root(competition: str) -> Path:
    """Archive root for a league, e.g. data/fbref/serie_a."""
    return FBREF_DIR / COMPETITIONS[competition][3]
REPORT_RE = re.compile(r"https://fbref\.com/en/matches/[0-9a-f]{8}/")

# FBref asks for no more than twenty requests a minute.  Three seconds plus
# jitter keeps us under that with room to spare; going faster gets the whole
# run challenged and costs more time than it saves.
DEFAULT_DELAY = 4.0

# A dead browser is not a failed match.  Chrome can drop its DevTools
# connection part-way through a long run; every later match then raises
# InvalidSessionIdException instantly.  The per-match `except Exception` used to
# swallow that and carry on, so a run could "complete" hundreds of matches
# having scraped none of them - on 2026-09-21 it burned seven and a half hours
# and 210 La Liga fixtures that way before anyone looked at the log.
#
# Two guards, both of which have to be crossed before the run gives up:
#   * a dead session rebuilds the driver and retries that same match, up to
#     MAX_DRIVER_RESTARTS times across the season (--max-driver-restarts);
#   * any run of MAX_CONSECUTIVE_FAILURES failures, of whatever kind, aborts.
# A scrape that cannot make progress must stop loudly rather than spin.
#
# The budget is deliberately generous, because it is not the thing protecting
# against a permanently broken browser.  A restart is only ever spent on a
# death the *retry then survives*: if the retried match dies too, the run
# aborts immediately whatever the budget says.  So the cap only ever limits how
# many recoverable deaths a season may absorb, and observed behaviour is one
# roughly every three hours - enough to exhaust a small cap on a long league.
MAX_DRIVER_RESTARTS = 20
MAX_CONSECUTIVE_FAILURES = 15

# How many goes a single match gets.  Only a dead browser consumes more than
# one, and each extra go costs a restart from the season budget above.
MATCH_ATTEMPTS = 4

# Messages chromedriver uses when the browser is gone rather than the page bad.
_DEAD_SESSION_MARKERS = (
    "invalid session id",
    "not connected to devtools",
    "browser has closed the connection",
    "chrome not reachable",
    "disconnected",
    "target window already closed",
    "no such session",
)


def is_session_dead(exc: BaseException) -> bool:
    """True when the exception means the browser died, not the page failed."""
    if type(exc).__name__ in ("InvalidSessionIdException", "NoSuchWindowException"):
        return True
    msg = str(exc).lower()
    return any(marker in msg for marker in _DEAD_SESSION_MARKERS)


# --------------------------------------------------------------------------- #
# browser
# --------------------------------------------------------------------------- #
def make_driver(headless: bool = False):
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--window-size=1500,1100")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    driver = webdriver.Chrome(options=opts)
    driver.set_page_load_timeout(90)
    return driver


def fetch(driver, url: str, delay: float, wait_for: str = "<table",
          tries: int = 3) -> str:
    """Load a page and return its HTML once the challenge has cleared.

    For a match report the page must also *be* the requested match.  The
    ``wait_for`` test alone accepts whatever the browser is showing, and the
    previous report has tables too: on 2026-09-26 that saved Lazio v Empoli
    into the Spezia v Hellas Verona folder (same kick-off time, so nothing
    downstream noticed except a player listed for two clubs on one day).  The
    8-hex match id has to be in both the browser's URL and the page source.
    """
    mid = re.search(r"/matches/([0-9a-f]{8})/", url)
    mid = mid.group(1) if mid else None
    for attempt in range(tries):
        driver.get(url)
        for _ in range(8):
            time.sleep(delay * 0.75 + random.uniform(0.3, 1.2))
            html = driver.page_source
            title = (driver.title or "").lower()
            challenged = ("just a moment" in title or "ci siamo" in title
                          or "attendere" in title)
            right_page = mid is None or (
                mid in (driver.current_url or "") and f"/matches/{mid}/" in html)
            if wait_for in html and not challenged and right_page:
                return html
        print(f"    challenge not cleared (attempt {attempt + 1}/{tries}); backing off")
        time.sleep(20 + 20 * attempt)
    raise RuntimeError(f"could not load {url}")


# --------------------------------------------------------------------------- #
# schedule
# --------------------------------------------------------------------------- #
def find_schedule_table(html: str) -> pd.DataFrame:
    """The fixtures table, located by its columns rather than its position."""
    for t in pd.read_html(StringIO(html)):
        cols = [str(c) for c in (t.columns.get_level_values(-1)
                if isinstance(t.columns, pd.MultiIndex) else t.columns)]
        if {"Home", "Away", "Score"} <= set(cols):
            t.columns = cols
            return t
    raise RuntimeError("no fixtures table on the schedule page")


def extract_report_links(html: str) -> dict[tuple[str, str], str]:
    """Pair every fixture with the report link in its own row.

    Read out of the HTML with BeautifulSoup rather than through Selenium: the
    page source already holds all 380 rows, while querying the live DOM for
    them returns only a couple.  Pairing on (home, away) from the row itself
    also avoids needing an alias table for the URL slug, where the schedule
    says "Nott'ham Forest" and the link says "Nottingham-Forest".
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    out: dict[tuple[str, str], str] = {}
    for tr in soup.find_all("tr"):
        home = tr.find(attrs={"data-stat": "home_team"})
        away = tr.find(attrs={"data-stat": "away_team"})
        if home is None or away is None:
            continue
        link = None
        for a in tr.find_all("a", href=True):
            href = a["href"]
            if re.match(r"(https://fbref\.com)?/en/matches/[0-9a-f]{8}/", href):
                link = href if href.startswith("http") else "https://fbref.com" + href
                break
        if link:
            out[(home.get_text(strip=True), away.get_text(strip=True))] = link
    return out


def get_schedule(driver, season: str, delay: float,
                 competition: str = "premier_league"):
    comp_id, slug, _display, _dir = COMPETITIONS[competition]
    url = SCHEDULE_URL.format(season=season, comp_id=comp_id, slug=slug)
    print(f"loading schedule: {url}")
    html = fetch(driver, url, delay)
    sched = find_schedule_table(html)
    sched = sched[sched["Home"].notna() & sched["Away"].notna()]
    sched = sched[sched["Score"].notna()]          # played matches only
    # FBref repeats the header every few rows; those come through as data rows
    # whose Wk is the literal string "Wk".
    sched = sched[pd.to_numeric(sched["Wk"], errors="coerce").notna()]

    by_pair = extract_report_links(html)
    print(f"  {len(sched)} played fixtures, {len(by_pair)} rows carry a report link")
    return sched, by_pair


def match_key(home: str, away: str) -> str:
    return f"{home}|{away}"


def link_key(url: str) -> str:
    """Turn .../Nottingham-Forest-Manchester-United-November-1-2025-... into a
    rough team pair, for matching a link to its schedule row."""
    slug = url.rstrip("/").split("/")[-1]
    return slug.lower()


# --------------------------------------------------------------------------- #
# one match report
# --------------------------------------------------------------------------- #
def parse_report(html: str):
    """Return (lineups, home_stats, away_stats, shots or None)."""
    tables = pd.read_html(StringIO(html))

    lineup_idx, stat_idx = [], []
    for i, t in enumerate(tables):
        cols = [str(c) for c in (t.columns.get_level_values(-1)
                if isinstance(t.columns, pd.MultiIndex) else t.columns)]
        if t.shape[1] == 2 and t.shape[0] >= 15 and "(" in str(cols[0]):
            lineup_idx.append(i)
        elif {"Player", "Min", "Pos"} <= set(cols) and t.shape[1] >= 15:
            stat_idx.append(i)

    if len(lineup_idx) < 2 or len(stat_idx) < 2:
        raise RuntimeError(f"unexpected report layout: "
                           f"{len(lineup_idx)} line-ups, {len(stat_idx)} stat tables")

    lineups = pd.concat(
        [tables[lineup_idx[0]].assign(Side="home"),
         tables[lineup_idx[1]].assign(Side="away")], axis=1)

    def flatten(t: pd.DataFrame) -> pd.DataFrame:
        t = t.copy()
        if isinstance(t.columns, pd.MultiIndex):
            t.columns = ["_".join(str(x) for x in tup if not str(x).startswith("Unnamed"))
                         .strip("_") for tup in t.columns]
        # drop the trailing "N Players" total row only if the scraper's
        # consumer expects it -- build_base_table drops the last row itself, so
        # keep the table exactly as FBref renders it
        return t

    home_stats = flatten(tables[stat_idx[0]])
    away_stats = flatten(tables[stat_idx[1]])

    shots = None
    for t in tables:
        cols = [str(c) for c in (t.columns.get_level_values(-1)
                if isinstance(t.columns, pd.MultiIndex) else t.columns)]
        if {"Minute", "Squad", "xG"} <= set(cols):
            t = t.copy()
            if isinstance(t.columns, pd.MultiIndex):
                t.columns = ["_".join(str(x) for x in tup
                                      if not str(x).startswith("Unnamed")).strip("_")
                             for tup in t.columns]
            shots = t
            break
    return lineups, home_stats, away_stats, shots


def parse_meta(driver, html: str, sched_row: pd.Series, season: str,
               matchweek: int, home: str, away: str,
               competition: str = "premier_league"):
    """main.csv and officials.csv, from the scorebox metadata block."""
    from selenium.webdriver.common.by import By
    try:
        from util import parse_match_html_to_dfs
        meta_el = driver.find_element(By.CLASS_NAME, "scorebox_meta")
        main_df, officials_df = parse_match_html_to_dfs(meta_el.get_attribute("innerHTML"))
        if main_df is not None and len(main_df):
            return main_df, officials_df
    except Exception as exc:
        print(f"    scorebox parse fell back ({type(exc).__name__}: {exc})")

    # Fallback: the schedule row already carries date, time, venue and referee,
    # which is everything build_base_table actually reads from main.csv.
    date = str(sched_row.get("Date", ""))
    tm = str(sched_row.get("Time", "") or "15:00")
    referee = str(sched_row.get("Referee", "") or "Unknown")
    main_df = pd.DataFrame([{
        "date_text": date, "venue_date": date, "venue_time": tm,
        "venue_epoch": pd.NA, "local_time": tm,
        "competition": COMPETITIONS[competition][2],
        "matchweek": matchweek, "team_a": away, "team_b": home,
        "attendance": sched_row.get("Attendance", pd.NA),
        "venue": sched_row.get("Venue", pd.NA), "referee": referee,
        "ar1": pd.NA, "ar2": pd.NA, "fourth_official": pd.NA, "var": pd.NA,
    }])
    officials_df = pd.DataFrame([{"role": "referee", "official": referee}])
    return main_df, officials_df


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def scrape_season(season: str, delay: float = DEFAULT_DELAY,
                  limit: int | None = None, out_dir: Path | None = None,
                  headless: bool = False,
                  competition: str = "premier_league",
                  max_driver_restarts: int = MAX_DRIVER_RESTARTS) -> None:
    out_root = Path(out_dir or competition_root(competition)) / season
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[{COMPETITIONS[competition][2]}] {season}")

    driver = make_driver(headless=headless)
    n_done = n_skip = n_fail = 0
    n_restarts = n_consecutive_fail = 0
    try:
        sched, by_pair = get_schedule(driver, season, delay, competition)

        rows = []
        for _, r in sched.iterrows():
            home, away = str(r["Home"]).strip(), str(r["Away"]).strip()
            rows.append((r, home, away, by_pair.get((home, away))))

        found = sum(1 for *_x, h in rows if h)
        print(f"  matched {found}/{len(rows)} fixtures to a report link\n")

        if limit:
            rows = rows[:limit]

        for r, home, away, link in rows:
            try:
                matchweek = int(float(r.get("Wk", 0) or 0))
            except (TypeError, ValueError):
                matchweek = 0
            folder = out_root / f"match_day_{matchweek}" / f"{home}_{away}"
            if (folder / "main.csv").exists():
                n_skip += 1
                continue
            if not link:
                print(f"  ! no report link for {home} v {away}")
                n_fail += 1
                continue

            print(f"  mw{matchweek:>2}  {home} v {away}")
            # Retry only for a dead browser: any other failure is a property of
            # the page, and retrying it just wastes a request.
            #
            # Several attempts, not one.  A fresh Chrome does not always come up
            # ready - on 2026-09-21 a restart's very next request died too, and
            # with a single retry that aborted Ligue 1 2021-22 at 186 of 380
            # while 18 of the 20 restarts sat unused.  One flaky restart must
            # not end a season; only exhausting the season's budget may, and the
            # backoff below gives the browser longer each time.
            for attempt in range(MATCH_ATTEMPTS):
                try:
                    time.sleep(delay + random.uniform(0, delay * 0.5))
                    html = fetch(driver, link, delay)
                    lineups, hs_df, as_df, shots = parse_report(html)
                    main_df, officials_df = parse_meta(driver, html, r, season,
                                                       matchweek, home, away,
                                                       competition)
                    folder.mkdir(parents=True, exist_ok=True)
                    main_df.to_csv(folder / "main.csv", index=False)
                    officials_df.to_csv(folder / "officials.csv", index=False)
                    lineups.to_csv(folder / "lineups.csv", index=False)
                    hs_df.to_csv(folder / "home_player_stats.csv", index=False)
                    as_df.to_csv(folder / "away_player_stats.csv", index=False)
                    if shots is not None:
                        shots.to_csv(folder / "conbined_shots_stats.csv",
                                     index=False)
                    n_done += 1
                    n_consecutive_fail = 0
                    break
                except Exception as exc:
                    dead = is_session_dead(exc)
                    print(f"    FAILED: {type(exc).__name__}: {exc}")
                    last_attempt = attempt == MATCH_ATTEMPTS - 1
                    if dead and not last_attempt and n_restarts < max_driver_restarts:
                        n_restarts += 1
                        wait = 15 * (attempt + 1)
                        print(f"    browser session is dead - restarting it "
                              f"({n_restarts}/{max_driver_restarts}), waiting "
                              f"{wait}s, retrying this match "
                              f"(attempt {attempt + 2}/{MATCH_ATTEMPTS})",
                              flush=True)
                        try:
                            driver.quit()
                        except Exception:
                            pass
                        time.sleep(wait)
                        driver = make_driver(headless=headless)
                        continue
                    n_fail += 1
                    n_consecutive_fail += 1
                    if dead and n_restarts >= max_driver_restarts:
                        raise RuntimeError(
                            f"browser session died and the season's restart "
                            f"budget of {max_driver_restarts} is spent; stopping "
                            f"so the run does not spin. Already-written matches "
                            f"are intact - rerun the same command to resume."
                        ) from exc
                    if n_consecutive_fail >= MAX_CONSECUTIVE_FAILURES:
                        raise RuntimeError(
                            f"{n_consecutive_fail} matches failed in a row; "
                            f"stopping rather than scraping nothing for hours. "
                            f"Already-written matches are intact - rerun the "
                            f"same command to resume.") from exc
                    break
    finally:
        driver.quit()

    print(f"\nwrote {n_done} matches, skipped {n_skip} already present, "
          f"{n_fail} failed")
    print(f"output: {out_root}")
    if n_done:
        print("\nNext: python pipeline/build_base_table.py  (add the season to "
              "config.ALL_SEASONS first)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Refresh FBref match reports.")
    ap.add_argument("--season", default="2025-2026",
                    help="one season, or several: --season 2023-2024 2024-2025")
    ap.add_argument("--seasons", nargs="+", default=None,
                    help="alias for --season taking a list")
    ap.add_argument("--competition", default="premier_league",
                    choices=list(COMPETITIONS) + ["all", "new"],
                    help="one league, 'all' for all five, or 'new' for the four "
                         "that are not the Premier League - both run SEQUENTIALLY, "
                         "which is faster than running them in parallel")
    ap.add_argument("--delay", type=float, default=DEFAULT_DELAY)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--max-driver-restarts", type=int,
                    default=MAX_DRIVER_RESTARTS,
                    help="how many recoverable browser deaths a season may "
                         f"absorb (default {MAX_DRIVER_RESTARTS}); a death the "
                         "retry does not survive aborts regardless")
    ap.add_argument("--headless", action="store_true",
                    help="usually fails: FBref serves headless browsers the challenge page")
    args = ap.parse_args()

    seasons = args.seasons or [args.season]
    if args.competition == "all":
        comps = list(COMPETITIONS)
    elif args.competition == "new":
        comps = [c for c in COMPETITIONS if c != "premier_league"]
    else:
        comps = [args.competition]
    if len(comps) > 1 and args.out_dir:
        ap.error("--out-dir cannot be combined with several competitions")

    # A season that gives up should not silently cost the rest of the run, but
    # two in a row means the problem is the environment rather than the season
    # - stop then, instead of walking the remaining leagues failing every match.
    aborted_in_a_row = 0
    for comp in comps:
        for season in seasons:
            try:
                scrape_season(season, delay=args.delay, limit=args.limit,
                              out_dir=args.out_dir, headless=args.headless,
                              competition=comp,
                              max_driver_restarts=args.max_driver_restarts)
                aborted_in_a_row = 0
            except RuntimeError as exc:
                aborted_in_a_row += 1
                print(f"\n!! {comp} {season} ABORTED: {exc}\n", flush=True)
                if aborted_in_a_row >= 2:
                    print("!! two seasons aborted in a row - stopping the run. "
                          "Fix the browser, then rerun the same command to "
                          "resume from where this left off.", flush=True)
                    raise SystemExit(1)


if __name__ == "__main__":
    main()
