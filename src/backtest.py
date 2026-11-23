"""Chronological backtests: train on the past, test on the future."""
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from . import data, model as M
from .config import EDGE_THRESHOLD, LINES, MIN_GAMES, STAT_COLS, season_codes
from .features import BASE_FEATURES, FEATURES, build_dataset, totals_features


def run(df: pd.DataFrame, test_frac: float = 0.25, kind: str = "logreg",
        edge: float = EDGE_THRESHOLD, use_h2h: bool = True) -> dict:
    d, _ = build_dataset(df)
    d = d[(d.n_home >= MIN_GAMES) & (d.n_away >= MIN_GAMES)].sort_values("Date")
    cut = int(len(d) * (1 - test_frac))
    train, test = d.iloc[:cut], d.iloc[cut:]
    mdl = M.fit(train, kind, FEATURES if use_h2h else BASE_FEATURES)
    p = M.predict_proba(mdl, test)
    y = test["FTR"].map(M.Y_MAP).to_numpy()

    res = {"train_matches": len(train), "test_matches": len(test),
           "model_logloss": log_loss(y, p, labels=[0, 1, 2])}
    base = np.tile(train["FTR"].map(M.Y_MAP).value_counts(normalize=True).sort_index().to_numpy(), (len(test), 1))
    res["baseline_logloss"] = log_loss(y, base, labels=[0, 1, 2])

    has_odds = test[["OddsH", "OddsD", "OddsA"]].notna().all(axis=1).to_numpy()
    if has_odds.any():
        odds = test.loc[has_odds, ["OddsH", "OddsD", "OddsA"]]
        book = M.implied_probs(odds)
        yb, pm = y[has_odds], p[has_odds]
        res["bookmaker_logloss"] = log_loss(yb, book, labels=[0, 1, 2])
        res["model_logloss_on_odds_subset"] = log_loss(yb, pm, labels=[0, 1, 2])
        # Flat 1-unit bets wherever the model sees an edge over the no-vig bookmaker probability
        gap = pm - book
        pick = gap.argmax(axis=1)
        bet = gap.max(axis=1) > edge
        o = odds.to_numpy()[np.arange(len(odds)), pick]
        profit = np.where(pick == yb, o - 1, -1.0)[bet]
        res["value_bets"] = int(bet.sum())
        res["value_bet_roi"] = float(profit.mean()) if bet.any() else float("nan")
    return res


def run_totals(df: pd.DataFrame, stat: str, test_frac: float = 0.25, use_h2h: bool = True) -> dict:
    """Backtest the expected-total model for corners / shots / shots on target."""
    target = f"y_{stat}"
    d, _ = build_dataset(df)
    if target not in d:
        return {}
    d = d[(d.n_home >= MIN_GAMES) & (d.n_away >= MIN_GAMES)].dropna(subset=[target]).sort_values("Date")
    cut = int(len(d) * (1 - test_frac))
    train, test = d.iloc[:cut], d.iloc[cut:]
    tm = M.TotalsModel(stat, totals_features(stat, use_h2h)).fit(train)
    mu, y = tm.mean(test), test[target].to_numpy()

    team_avg = ((test[f"home_{stat}_for"] + test[f"away_{stat}_against"]
                 + test[f"away_{stat}_for"] + test[f"home_{stat}_against"]) / 2).to_numpy()
    res = {"test_matches": len(test),
           "mae_model": float(np.abs(y - mu).mean()),
           "mae_league_mean": float(np.abs(y - train[target].mean()).mean()),
           "mae_team_averages": float(np.abs(y - team_avg).mean()),
           "dispersion_alpha": tm.alpha}
    med = train[target].median()
    line = min(LINES[stat], key=lambda L: abs(L - med))
    p_over = np.clip(tm.prob_over(mu, line), 1e-6, 1 - 1e-6)
    over = (y > line).astype(int)
    rate = np.clip(float((train[target] > line).mean()), 1e-6, 1 - 1e-6)
    res["line"] = line
    res["over_logloss_model"] = log_loss(over, p_over, labels=[0, 1])
    res["over_logloss_baseline"] = log_loss(over, np.full(len(over), rate), labels=[0, 1])
    return res


def _print(res: dict, prefix: str = ""):
    for k, v in res.items():
        print(f"{prefix}{k:32s} {v:.4f}" if isinstance(v, float) else f"{prefix}{k:32s} {v}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--leagues", nargs="+", default=["E0"])
    ap.add_argument("--seasons", type=int, default=6)
    ap.add_argument("--model", default="logreg", choices=["logreg", "gbm"])
    ap.add_argument("--totals", action="store_true", help="also backtest corners/shots totals")
    ap.add_argument("--demo", action="store_true", help="use synthetic data (offline)")
    ap.add_argument("--extra-results", help="CSV of extra past results (e.g. a Futbol24 export)")
    ap.add_argument("--aliases", help="CSV (alias,canonical) to match team names across sources")
    ap.add_argument("--col-map", help="JSON mapping our column names to the ones in your CSV file")
    a = ap.parse_args()
    if a.demo:
        from .synthetic import generate
        hist, _ = generate()
    else:
        hist = data.load_history(a.leagues, season_codes(a.seasons))
    if a.extra_results:
        from .futbol24 import augment
        hist, _ = augment(hist, pd.DataFrame(columns=["Date", "League", "HomeTeam", "AwayTeam", "OddsH", "OddsD", "OddsA"]),
                          a.extra_results, None, a.aliases, a.col_map)

    print("== Match result (1X2), with head-to-head ==")
    _print(run(hist, kind=a.model, use_h2h=True))
    print("\n== Same model WITHOUT head-to-head (ablation) ==")
    r = run(hist, kind=a.model, use_h2h=False)
    print(f"{'model_logloss':32s} {r['model_logloss']:.4f}")
    print("\nLower log loss is better. If model_logloss is not below bookmaker_logloss, there is no edge.")
    print("If H2H does not lower log loss, it adds nothing beyond Elo and form.")

    if a.totals:
        for stat in STAT_COLS:
            print(f"\n== Total {stat} ==")
            res = run_totals(hist, stat, use_h2h=True)
            if not res:
                print("no data for this stat")
                continue
            _print(res)
            print(f"{'mae_model_without_h2h':32s} {run_totals(hist, stat, use_h2h=False)['mae_model']:.4f}")
