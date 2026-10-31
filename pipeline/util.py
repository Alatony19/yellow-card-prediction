import re, html
from bs4 import BeautifulSoup
import pandas as pd
from typing import Sequence, Tuple


def flatten_columns(
    df: pd.DataFrame,
    sep: str = "_",
    use_level_names_for_blank: bool = True,
    lower: bool = False,
    strip_whitespace: bool = True,
    remove_unnamed: bool = True,
    empty_fallback: str = "col",
    dedupe: bool = True,
) -> pd.DataFrame:
    """
    Return a copy of `df` with flattened, single-level column names.

    - sep: string used to join levels (e.g., "_", ".", "__").
    - use_level_names_for_blank: if a level value is blank/None, use that level's name instead.
    - lower: lowercase final column names.
    - strip_whitespace: collapse internal whitespace and strip ends.
    - remove_unnamed: drop values like 'Unnamed: x_level_y'.
    - empty_fallback: name to use if a column results in an empty label.
    - dedupe: if True, add numeric suffixes (_2, _3, …) to resolve duplicates.
    """
    out = df.copy()
    cols = out.columns

    def _clean_piece(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return ""
        s = str(x)
        if remove_unnamed and s.startswith("Unnamed"):
            return ""
        return s

    if isinstance(cols, pd.MultiIndex):
        level_names = cols.names or [None] * cols.nlevels
        new_cols = []
        for tup in cols:
            parts = []
            for i, val in enumerate(tup):
                piece = _clean_piece(val)
                if piece == "" and use_level_names_for_blank and level_names[i]:
                    piece = str(level_names[i])
                if piece != "":
                    parts.append(piece)
            label = sep.join(parts)
            if strip_whitespace:
                label = re.sub(r"\s+", " ", label).strip()
            if lower:
                label = label.lower()
            if label == "":
                label = empty_fallback
            new_cols.append(label)
    else:
        new_cols = []
        for c in cols:
            s = _clean_piece(c)
            if strip_whitespace:
                s = re.sub(r"\s+", " ", s).strip()
            if lower:
                s = s.lower()
            if s == "":
                s = empty_fallback
            new_cols.append(s)

    # De-duplicate while preserving order
    if dedupe:
        seen = {}
        deduped = []
        for c in new_cols:
            if c not in seen:
                seen[c] = 1
                deduped.append(c)
            else:
                seen[c] += 1
                deduped.append(f"{c}{sep}{seen[c]}")
        new_cols = deduped

    out.columns = pd.Index(new_cols)
    return out





def build_player_stats(
    tables,
    home_range: Tuple[int, int] = (3, 8),
    away_range: Tuple[int, int] = (10, 15),
    keys: Sequence[str] = ('Player', '#', 'Nation', 'Pos', 'Age', 'Min'),
    how: str = 'inner',
    suffix_template: str = "__t{idx}",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge tables[home_range[0]..home_range[1]] into home_player_stats and
    tables[away_range[0]..away_range[1]] into away_player_stats (inclusive, 0-based).

    - `keys` are matched case/whitespace-insensitively and MultiIndex-safe.
    - `how`: join policy between each step ('inner' keeps only rows common to all).
    - Overlapping non-key columns are suffixed using `suffix_template` with table index.
    Returns: (home_player_stats, away_player_stats)
    """

    def _canon(s: object) -> str:
        s = "" if s is None else str(s)
        s = (s.replace("\u00A0", " ").replace("\u2007", " ").replace("\u202F", " "))
        return re.sub(r"\s+", " ", s).strip().lower()

    def _flatten_if_multiindex(df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.columns, pd.MultiIndex):
            new_cols = []
            for tup in df.columns:
                parts = []
                for x in tup:
                    if x is None:
                        continue
                    sx = str(x)
                    if sx.startswith("Unnamed"):
                        continue
                    sx = re.sub(r"\s+", " ", sx).strip()
                    if sx:
                        parts.append(sx)
                new_cols.append("_".join(parts) if parts else "col")
            df = df.copy()
            df.columns = pd.Index(new_cols)
        return df

    def _resolve_keys(df: pd.DataFrame, desired_keys: Sequence[str]):
        df = _flatten_if_multiindex(df)
        cmap = {_canon(c): c for c in df.columns}
        resolved = {}
        missing = []
        for k in desired_keys:
            ck = _canon(k)
            if ck in cmap:
                resolved[k] = cmap[ck]
            else:
                missing.append(k)
        return df, resolved, missing

    def _merge_range(rng: Tuple[int, int]) -> pd.DataFrame:
        start, end = rng
        if start > end:
            raise ValueError(f"Invalid range {rng}: start > end")

        picked = []
        for idx in range(start, end + 1):
            df, resolved, missing = _resolve_keys(tables[idx], keys)
            if missing:
                have = sorted({_canon(c) for c in df.columns})
                need = sorted({_canon(k) for k in keys})
                raise ValueError(
                    f"Table at index {idx} missing join columns after normalization: {missing}\n"
                    f"Needed canon: {need}\nHave canon sample: {have[:25]}..."
                )
            picked.append((idx, df, resolved))

        base_idx, base, res0 = picked[0]
        base = base.rename(columns={v: k for k, v in res0.items()})
        base = base.drop_duplicates(subset=list(keys)).set_index(list(keys))

        for idx, df, resolved in picked[1:]:
            right = df.rename(columns={v: k for k, v in resolved.items()})
            right = right.drop_duplicates(subset=list(keys)).set_index(list(keys))

            overlap = base.columns.intersection(right.columns)
            if len(overlap) > 0:
                right = right.rename(columns={c: f"{c}{suffix_template.format(idx=idx)}" for c in overlap})

            base = base.join(right, how=how)

        return base.reset_index()

    home_player_stats = _merge_range(home_range)
    away_player_stats = _merge_range(away_range)
    return home_player_stats, away_player_stats





def parse_match_html_to_dfs(table: str):
    """
    Parse the provided FBref-style match info HTML block into:
      1) A single-row pandas DataFrame with core fields
      2) A (role, official) DataFrame for the officials (or None if not found)
    """
    # Decode HTML entities (e.g., &nbsp;)
    s = html.unescape(table)
    soup = BeautifulSoup(s, "html.parser")

    row = {}

    # --- Date & times ---
    date_link = soup.select_one("div > strong > a")
    row["date_text"] = date_link.get_text(strip=True) if date_link else None

    vt = soup.select_one("span.venuetime")
    row["venue_date"]  = vt.get("data-venue-date") if vt else None
    row["venue_time"]  = vt.get("data-venue-time") if vt else None
    row["venue_epoch"] = int(vt.get("data-venue-epoch")) if (vt and vt.get("data-venue-epoch")) else None

    local = soup.select_one("span.localtime")
    if local:
        m = re.search(r"(\d{1,2}:\d{2})", local.get_text(" ", strip=True))
        row["local_time"] = m.group(1) if m else local.get_text(" ", strip=True)
    else:
        row["local_time"] = None

    # --- Competition & matchweek ---
    comp_a = soup.find("a", href=re.compile(r"^/en/comps/"))
    row["competition"] = comp_a.get_text(strip=True) if comp_a else None

    # Try to grab "(Matchweek N)" that follows the competition link
    matchweek = None
    if comp_a:
        # Prefer text immediately after the <a>
        tail = ""
        sib = comp_a.next_sibling
        while sib and isinstance(sib, str) and not tail.strip():
            tail += str(sib)
            sib = sib.next_sibling
        if not tail:
            # Fallback: search in parent text
            tail = comp_a.parent.get_text(" ", strip=True)
        m = re.search(r"Matchweek\s+(\d+)", tail, re.I)
        matchweek = int(m.group(1)) if m else None
    row["matchweek"] = matchweek

    # --- Teams (from Head-to-Head link text) ---
    h2h_a = soup.find("a", href=re.compile(r"/stathead/matchup/teams/"))
    row["team_a"] = row["team_b"] = None
    if h2h_a:
        h2h_text = h2h_a.get_text(" ", strip=True)
        h2h_text = re.sub(r"\s*Historical Head-to-Head\s*", "", h2h_text, flags=re.I)
        parts = re.split(r"\s+vs\.?\s+", h2h_text)
        if len(parts) == 2:
            row["team_a"], row["team_b"] = parts[0].strip(), parts[1].strip()
        else:
            row["teams_text"] = h2h_text  # fallback if format differs

    # --- Attendance ---
    att_label = soup.find("small", string=re.compile(r"Attendance", re.I))
    row["attendance"] = None
    if att_label:
        att_val = att_label.find_next("small")
        # Ensure we didn't just re-grab the label itself
        if att_val and att_val is not att_label:
            digits = re.sub(r"[^\d]", "", att_val.get_text())
            row["attendance"] = int(digits) if digits else None

    # --- Venue ---
    venue_label = soup.find("small", string=re.compile(r"Venue", re.I))
    row["venue"] = None
    if venue_label:
        venue_val = venue_label.find_next("small")
        if venue_val and venue_val is not venue_label:
            row["venue"] = venue_val.get_text(" ", strip=True)

    # --- Officials ---
    officials_label = soup.find("small", string=re.compile(r"Officials", re.I))
    officials_map = {}
    if officials_label:
        officials_small = officials_label.find_next("small")
        tokens = []
        if officials_small:
            spans = officials_small.find_all("span")
            if spans:
                tokens = [sp.get_text(" ", strip=True) for sp in spans]
            else:
                # Fallback: split by middle dot or commas
                tokens = [t.strip() for t in re.split(r"[·|,]", officials_small.get_text(" ", strip=True)) if t.strip()]

        role_map = {
            "referee": "referee",
            "ar1": "ar1",
            "ar2": "ar2",
            "4th": "fourth_official",
            "var": "var",
            "avar": "avar"
        }
        for tok in tokens:
            m = re.match(r"(.+?)\s*\(([^)]+)\)", tok)
            if m:
                name = m.group(1).strip()
                role = m.group(2).strip().lower()
                col = role_map.get(role, role)
                officials_map[col] = name

    # Merge officials into the main row
    row.update(officials_map)

    # Build DataFrames
    main_df = pd.DataFrame([row])
    officials_df = (
        pd.DataFrame([{"role": k, "official": v} for k, v in officials_map.items()])
        if officials_map else None
    )
    return main_df, officials_df


