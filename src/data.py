"""Download and clean football-data.co.uk CSVs (results + bookmaker odds)."""
from pathlib import Path
import io
import pandas as pd
import requests

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
BASE = "https://www.football-data.co.uk"
ODDS_SETS = [("B365H", "B365D", "B365A"), ("AvgH", "AvgD", "AvgA"), ("PSH", "PSD", "PSA")]


def _get(url: str, cache: Path | None = None, refresh: bool = False) -> pd.DataFrame:
    if cache and cache.exists() and not refresh:
        return pd.read_csv(cache, encoding="latin-1")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(r.content)
    return pd.read_csv(io.BytesIO(r.content), encoding="latin-1")


def add_odds(df: pd.DataFrame) -> pd.DataFrame:
    """Pick the first odds set that exists; expose as OddsH/OddsD/OddsA."""
    df = df.copy()
    for h, d, a in ODDS_SETS:
        if {h, d, a} <= set(df.columns):
            df["OddsH"], df["OddsD"], df["OddsA"] = df[h], df[d], df[a]
            return df
    df["OddsH"] = df["OddsD"] = df["OddsA"] = float("nan")
    return df


def clean(df: pd.DataFrame, league: str) -> pd.DataFrame:
    df = df.dropna(subset=["HomeTeam", "AwayTeam"]).copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"])
    df["League"] = league
    df = add_odds(df)
    keep = ["Date", "League", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "OddsH", "OddsD", "OddsA",
            "HS", "AS", "HST", "AST", "HC", "AC"]  # shots, shots on target, corners
    return df[[c for c in keep if c in df.columns]]


def load_history(leagues: list[str], seasons: list[str], refresh: bool = False) -> pd.DataFrame:
    frames = []
    for lg in leagues:
        for s in seasons:
            try:
                raw = _get(f"{BASE}/mmz4281/{s}/{lg}.csv", RAW / f"{lg}_{s}.csv", refresh)
            except Exception as e:  # season/league may not exist
                print(f"skip {lg} {s}: {e}")
                continue
            frames.append(clean(raw, lg))
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["FTR", "FTHG", "FTAG"])
    return df.sort_values("Date").reset_index(drop=True)


def load_fixtures(leagues: list[str]) -> pd.DataFrame:
    raw = _get(f"{BASE}/fixtures.csv")
    raw = raw[raw["Div"].isin(leagues)].rename(columns={"Div": "League"})
    raw = raw.dropna(subset=["HomeTeam", "AwayTeam"]).copy()
    raw["Date"] = pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce")
    raw = add_odds(raw)
    keep = ["Date", "League", "HomeTeam", "AwayTeam", "OddsH", "OddsD", "OddsA"]
    return raw[keep].dropna(subset=["Date"]).reset_index(drop=True)
