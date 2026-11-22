"""Fit on all history, predict upcoming fixtures (result + optional corners/shots totals)."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from . import data, model as M
from .config import EDGE_THRESHOLD, LABELS, LEAGUES, LINES, MIN_GAMES, STAT_COLS, season_codes
from .features import BASE_FEATURES, FEATURES, build_dataset, totals_features

OUT = Path(__file__).resolve().parent.parent / "docs"


def predict_fixtures(hist, fixtures, kind="logreg", edge=EDGE_THRESHOLD, use_h2h=True) -> pd.DataFrame:
    ds, fb = build_dataset(hist)
    ds = ds[(ds.n_home >= MIN_GAMES) & (ds.n_away >= MIN_GAMES)]
    mdl = M.fit(ds, kind, FEATURES if use_h2h else BASE_FEATURES)

    rows = []
    for f in fixtures.itertuples():
        x = fb.features(f.HomeTeam, f.AwayTeam, f.Date)
        p = M.predict_proba(mdl, pd.DataFrame([x]))[0]
        rec = {"date": f.Date.date(), "league": f.League, "home": f.HomeTeam, "away": f.AwayTeam,
               "p_home": p[0], "p_draw": p[1], "p_away": p[2],
               "pick": LABELS[int(p.argmax())],
               "h2h_meetings_weight": x["h2h_w"],
               "low_confidence": min(x["n_home"], x["n_away"]) < MIN_GAMES}
        odds = np.array([f.OddsH, f.OddsD, f.OddsA], dtype=float)
        if not np.isnan(odds).any():
            book = M.implied_probs(pd.DataFrame([odds]))[0]
            gap = p - book
            k = int(gap.argmax())
            rec.update({"odds_home": odds[0], "odds_draw": odds[1], "odds_away": odds[2],
                        "best_edge": gap[k], "edge_on": LABELS[k],
                        "value_flag": bool(gap[k] > edge and not rec["low_confidence"])})
        rows.append(rec)
    return pd.DataFrame(rows)


def predict_totals(hist, fixtures, use_h2h=True, lines=LINES) -> pd.DataFrame:
    """Expected total corners / shots / shots on target plus P(over) and fair odds for each line.

    There is no free source of bookmaker odds for these markets, so compare the fair odds
    below with what your bookmaker offers by hand: a bet only has value if its odds are
    HIGHER than the fair odds.
    """
    ds, fb = build_dataset(hist)
    ds = ds[(ds.n_home >= MIN_GAMES) & (ds.n_away >= MIN_GAMES)]
    models = {}
    for stat in STAT_COLS:
        col = f"y_{stat}"
        if col in ds and ds[col].notna().sum() > 200:
            models[stat] = M.TotalsModel(stat, totals_features(stat, use_h2h)).fit(ds)

    rows = []
    for f in fixtures.itertuples():
        x = fb.features(f.HomeTeam, f.AwayTeam, f.Date)
        X = pd.DataFrame([x])
        low = min(x["n_home"], x["n_away"]) < MIN_GAMES
        for stat, tm in models.items():
            mu = float(tm.mean(X)[0])
            for line in lines[stat]:
                po = float(tm.prob_over(mu, line))
                rows.append({"date": f.Date.date(), "league": f.League, "home": f.HomeTeam, "away": f.AwayTeam,
                             "stat": stat, "expected_total": mu, "line": line,
                             "p_over": po, "p_under": 1 - po,
                             "fair_over_odds": 1 / po if po > 0 else np.inf,
                             "fair_under_odds": 1 / (1 - po) if po < 1 else np.inf,
                             "low_confidence": low})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--leagues", nargs="+", default=list(LEAGUES))
    ap.add_argument("--seasons", type=int, default=6)
    ap.add_argument("--model", default="logreg", choices=["logreg", "gbm"])
    ap.add_argument("--no-h2h", action="store_true", help="turn head-to-head features off")
    ap.add_argument("--totals", action="store_true", help="also predict total corners and shots")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--extra-results", help="CSV of extra past results (e.g. a Futbol24 export)")
    ap.add_argument("--extra-fixtures", help="CSV of extra upcoming fixtures")
    ap.add_argument("--aliases", help="CSV (alias,canonical) to match team names across sources")
    ap.add_argument("--col-map", help="JSON mapping our column names to the ones in your CSV files")
    a = ap.parse_args()
    if a.demo:
        from .synthetic import generate
        hist, fixtures = generate()
    else:
        hist = data.load_history(a.leagues, season_codes(a.seasons))
        fixtures = data.load_fixtures(a.leagues)
    if a.extra_results or a.extra_fixtures:
        from .futbol24 import augment
        hist, fixtures = augment(hist, fixtures, a.extra_results, a.extra_fixtures, a.aliases, a.col_map)
    OUT.mkdir(exist_ok=True)
    out = predict_fixtures(hist, fixtures, a.model, use_h2h=not a.no_h2h)
    out.to_csv(OUT / "predictions.csv", index=False)
    print(out.round(3).to_string(index=False))
    if a.totals:
        tot = predict_totals(hist, fixtures, use_h2h=not a.no_h2h)
        tot.to_csv(OUT / "totals_predictions.csv", index=False)
        print("\n", tot[tot.line.isin([9.5, 24.5])].round(3).to_string(index=False))
