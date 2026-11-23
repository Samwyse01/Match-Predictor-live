"""Plug extra results/fixtures (e.g. an export from Futbol24) into the pipeline.

This module does NOT scrape futbol24.com. It reads CSV files you provide, so you stay in control of
how the data was obtained (their terms, an export, a licensed feed). Because Futbol24 exports can
differ, column names are configurable and team names can be reconciled with an alias file.

Alias file (CSV, two columns):   alias,canonical
    Man Utd,Man United
    Nott'm Forest,Nott'm Forest
Column map file (JSON): keys are our names, values are the column names in your file:
    {"HomeTeam": "Home", "AwayTeam": "Away", "FTHG": "HomeGoals", "FTAG": "AwayGoals"}
"""
import json
import pandas as pd

DEFAULT_MAP = {c: c for c in
               ["Date", "League", "HomeTeam", "AwayTeam", "FTHG", "FTAG",
                "HC", "AC", "HS", "AS", "HST", "AST", "OddsH", "OddsD", "OddsA"]}
STD_COLS = ["Date", "League", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
            "OddsH", "OddsD", "OddsA", "HS", "AS", "HST", "AST", "HC", "AC"]


def _parse_dates(s: pd.Series) -> pd.Series:
    d = pd.to_datetime(s, format="ISO8601", errors="coerce")
    rest = d.isna()
    if rest.any():
        d[rest] = pd.to_datetime(s[rest], dayfirst=True, errors="coerce")
    return d


def _load(path: str, col_map: dict | None, aliases: dict) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="utf-8-sig")
    m = {**DEFAULT_MAP, **(col_map or {})}
    df = raw.rename(columns={v: k for k, v in m.items() if v in raw.columns})
    missing = [c for c in ("Date", "HomeTeam", "AwayTeam") if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}. Fix them with --col-map (see src/futbol24.py).")
    df["Date"] = _parse_dates(df["Date"])
    df = df.dropna(subset=["Date"]).copy()
    for col in ("HomeTeam", "AwayTeam"):
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].map(lambda t: aliases.get(t.lower(), t))
    if "League" not in df:
        df["League"] = "EXT"
    return df


def _standardise(df: pd.DataFrame, results: bool) -> pd.DataFrame:
    for c in STD_COLS:
        if c not in df.columns:
            df[c] = float("nan")
    if results:
        df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
        df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
        df = df.dropna(subset=["FTHG", "FTAG"]).copy()
        df["FTR"] = ["H" if h > a else "D" if h == a else "A" for h, a in zip(df.FTHG, df.FTAG)]
    return df[STD_COLS if results else ["Date", "League", "HomeTeam", "AwayTeam", "OddsH", "OddsD", "OddsA"]]


def _read_aliases(path: str | None) -> dict:
    if not path:
        return {}
    a = pd.read_csv(path)
    return {str(k).strip().lower(): str(v).strip() for k, v in zip(a["alias"], a["canonical"])}


def augment(hist, fixtures, results_csv=None, fixtures_csv=None, aliases_csv=None, col_map_json=None):
    """Merge extra results/fixtures into the main data. The main source wins on duplicate matches."""
    aliases = _read_aliases(aliases_csv)
    col_map = json.load(open(col_map_json)) if col_map_json else None
    known = set(hist.HomeTeam) | set(hist.AwayTeam)

    if results_csv:
        extra = _standardise(_load(results_csv, col_map, aliases), results=True)
        unknown = (set(extra.HomeTeam) | set(extra.AwayTeam)) - known
        if unknown:
            print(f"[futbol24] {len(unknown)} team names not found in main data; add them to the alias file "
                  f"if they are the same clubs: {sorted(unknown)[:15]}")
        n0 = len(hist)
        hist = (pd.concat([hist, extra], ignore_index=True)
                .drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"], keep="first")
                .sort_values("Date", kind="stable").reset_index(drop=True))
        print(f"[futbol24] results: +{len(hist) - n0} new matches")

    if fixtures_csv:
        extra = _standardise(_load(fixtures_csv, col_map, aliases), results=False)
        n0 = len(fixtures)
        fixtures = (pd.concat([fixtures, extra], ignore_index=True)
                    .drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"], keep="first")
                    .reset_index(drop=True))
        print(f"[futbol24] fixtures: +{len(fixtures) - n0} new fixtures")
    return hist, fixtures
