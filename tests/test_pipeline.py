import datetime as dt
import numpy as np
import pandas as pd
from src.backtest import run, run_totals
from src.features import FeatureBuilder
from src.predict import predict_fixtures, predict_totals
from src.synthetic import generate

D0 = pd.Timestamp("2024-01-01")


def test_no_leakage_first_match_is_neutral():
    hist, _ = generate(seasons=1)
    feats = FeatureBuilder().transform(hist)
    assert feats.iloc[0]["elo_diff"] == 60  # only home advantage, nothing learned yet
    assert feats.iloc[0]["h2h_w"] == 0      # no previous meetings


def test_probabilities_sum_to_one():
    hist, fx = generate(seasons=3)
    out = predict_fixtures(hist, fx)
    assert np.allclose(out[["p_home", "p_draw", "p_away"]].sum(axis=1), 1)


def test_backtest_runs_and_beats_baseline():
    hist, _ = generate(seasons=4)
    r = run(hist)
    assert r["model_logloss"] < r["baseline_logloss"]
    assert np.isfinite(run(hist, use_h2h=False)["model_logloss"])


def test_h2h_direction_and_decay():
    fb = FeatureBuilder()
    assert fb.features("A", "B", D0)["h2h_w"] == 0
    fb.update("A", "B", 3, 0, D0, {"HC": 7, "AC": 3})
    soon = fb.features("A", "B", D0 + pd.Timedelta(days=30))
    flipped = fb.features("B", "A", D0 + pd.Timedelta(days=30))   # same pair, roles reversed
    later = fb.features("A", "B", D0 + pd.Timedelta(days=3650))
    assert soon["h2h_pts"] > 0 > flipped["h2h_pts"]
    assert abs(later["h2h_pts"]) < abs(soon["h2h_pts"])           # old meetings fade
    assert 10.0 < soon["h2h_corners_total"] < 10.5                 # 10 corners seen, shrunk toward the 10.5 prior


def test_totals_predictions_are_valid():
    hist, fx = generate(seasons=4)
    tot = predict_totals(hist, fx)
    assert set(tot.stat) == {"corners", "shots", "shots_on_target"}
    assert tot.p_over.between(0, 1).all() and (tot.expected_total > 0).all()
    one = tot[(tot.stat == "corners") & (tot.home == fx.HomeTeam[0]) & (tot.away == fx.AwayTeam[0])]
    assert one.sort_values("line").p_over.is_monotonic_decreasing  # higher line -> lower P(over)


def test_totals_backtest_runs():
    hist, _ = generate(seasons=4)
    for stat in ("corners", "shots", "shots_on_target"):
        r = run_totals(hist, stat)
        assert np.isfinite(r["mae_model"]) and np.isfinite(r["over_logloss_model"])


def test_external_csv_merge_with_aliases_and_column_map(tmp_path=None):
    import json, tempfile, os
    from src.futbol24 import augment
    hist, fx = generate(seasons=2)
    d = tempfile.mkdtemp()
    last = hist.iloc[-1]
    # one duplicate of an existing match (different team spelling + column names) and two new ones
    rows = [
        {"Day": last.Date.strftime("%Y-%m-%d"), "Home": "t" + last.HomeTeam[1:].upper(), "Away": last.AwayTeam,
         "HG": 9, "AG": 9},
        {"Day": "2099-01-01", "Home": "Team00", "Away": "Team01", "HG": 2, "AG": 1},
        {"Day": "2099-01-08", "Home": "Team01", "Away": "Team00", "HG": 0, "AG": 0},
    ]
    pd.DataFrame(rows).to_csv(os.path.join(d, "res.csv"), index=False)
    pd.DataFrame({"Day": ["2099-02-01"], "Home": ["Team02"], "Away": ["Team03"]}).to_csv(os.path.join(d, "fx.csv"), index=False)
    pd.DataFrame({"alias": ["t" + last.HomeTeam[1:].upper()], "canonical": [last.HomeTeam]}).to_csv(os.path.join(d, "al.csv"), index=False)
    json.dump({"Date": "Day", "HomeTeam": "Home", "AwayTeam": "Away", "FTHG": "HG", "FTAG": "AG"}, open(os.path.join(d, "m.json"), "w"))
    h2, f2 = augment(hist, fx, os.path.join(d, "res.csv"), os.path.join(d, "fx.csv"),
                     os.path.join(d, "al.csv"), os.path.join(d, "m.json"))
    assert len(h2) == len(hist) + 2          # duplicate dropped (main source wins), 2 new kept
    assert len(f2) == len(fx) + 1
    assert h2.loc[(h2.Date == last.Date) & (h2.HomeTeam == last.HomeTeam) & (h2.AwayTeam == last.AwayTeam), "FTHG"].iloc[0] == last.FTHG
    assert h2.FTR.isin(["H", "D", "A"]).all()
    assert np.isfinite(run(h2)["model_logloss"])  # merged data flows through the pipeline


def test_prediction_log_keeps_first_and_attaches_results():
    import os, tempfile
    from src.track import update_log
    hist, _ = generate(seasons=3)
    train, hold = hist.iloc[:-20], hist.tail(20)
    fx = hold[["Date", "League", "HomeTeam", "AwayTeam", "OddsH", "OddsD", "OddsA"]].reset_index(drop=True)
    preds = predict_fixtures(train, fx)
    path = os.path.join(tempfile.mkdtemp(), "log.csv")
    day = hold.Date.min()
    log1 = update_log(preds, train, path, today=day)
    assert len(log1) > 0 and log1.result.isna().all()            # nothing played yet
    again = preds.copy(); again["p_home"] = 0.99
    log2 = update_log(again, hist, path, today=day + pd.Timedelta(days=1))
    assert len(log2) == len(log1) and (log2.p_home != 0.99).all()  # first prediction is kept
    assert log2.result.notna().sum() >= 15                       # real results attached
