"""Permanent log of predictions, with the real results attached once the matches are played."""
from pathlib import Path
import pandas as pd

LOG = Path(__file__).resolve().parent.parent / "docs" / "prediction_log.csv"
LOG_COLS = ["date", "league", "home", "away", "p_home", "p_draw", "p_away", "pick",
            "odds_home", "odds_draw", "odds_away", "edge_on", "best_edge", "value_flag", "predicted_at"]
RESULT_COLS = ["home_goals", "away_goals", "result"]


def update_log(preds: pd.DataFrame, hist: pd.DataFrame, path=LOG, today=None) -> pd.DataFrame:
    """Add new predictions (the FIRST prediction for a match is kept), then attach results from `hist`."""
    path, today = Path(path), pd.Timestamp(today or pd.Timestamp.today().date())
    if preds.empty:
        new = pd.DataFrame(columns=LOG_COLS)
    else:
        new = preds.copy()
        new["date"] = new["date"].astype(str)
        new["predicted_at"] = str(today.date())
        for c in LOG_COLS:
            if c not in new:
                new[c] = pd.NA
        new = new[LOG_COLS]
    if path.exists() and path.stat().st_size > 0:
        old = pd.read_csv(path, dtype={"date": str})
        log = pd.concat([old[LOG_COLS], new], ignore_index=True)
    else:
        log = new
    log = log.drop_duplicates(["date", "home", "away"], keep="first")

    res = hist[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]].copy()
    res["date"] = res["Date"].dt.strftime("%Y-%m-%d")
    res = res.rename(columns={"HomeTeam": "home", "AwayTeam": "away", "FTHG": "home_goals",
                              "FTAG": "away_goals", "FTR": "result"})
    res = res[["date", "home", "away"] + RESULT_COLS].drop_duplicates(["date", "home", "away"])
    log = log.merge(res, on=["date", "home", "away"], how="left")

    stale = log["result"].isna() & (pd.to_datetime(log["date"]) < today - pd.Timedelta(days=14))
    log = log[~stale].sort_values("date").reset_index(drop=True)  # postponed/never-played rows drop out
    for c in ("home_goals", "away_goals"):
        log[c] = log[c].astype("Int64")
    path.parent.mkdir(exist_ok=True)
    log.to_csv(path, index=False)
    return log
