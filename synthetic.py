"""Offline synthetic league so the pipeline can be tested without internet.

A small head-to-head effect is baked into corners so the H2H code path has something to find.
Nothing learned from this data says anything about real football.
"""
import numpy as np
import pandas as pd
from math import exp, factorial


def _outcome_probs(lh, la, n=8):
    pm = lambda lam, k: exp(-lam) * lam**k / factorial(k)
    h = d = a = 0.0
    for i in range(n):
        for j in range(n):
            p = pm(lh, i) * pm(la, j)
            if i > j:
                h += p
            elif i == j:
                d += p
            else:
                a += p
    s = h + d + a
    return h / s, d / s, a / s


def generate(seasons=6, teams=20, margin=0.05, seed=0, n_fixtures=10):
    rng = np.random.default_rng(seed)
    names = [f"Team{i:02d}" for i in range(teams)]
    strength = rng.normal(0, 0.3, teams)
    m = rng.normal(0, 0.10, (teams, teams))
    pair_fx = (m + m.T) / 2  # symmetric: how corner-heavy a pairing is
    rows, day = [], pd.Timestamp("2020-08-15")
    for _ in range(seasons):
        strength += rng.normal(0, 0.05, teams)  # strengths drift
        pairs = [(i, j) for i in range(teams) for j in range(teams) if i != j]
        rng.shuffle(pairs)
        for n, (i, j) in enumerate(pairs):
            d = strength[i] - strength[j]
            lh, la = exp(0.25 + d) * 1.3, exp(-d) * 1.1
            hg, ag = rng.poisson(lh), rng.poisson(la)
            ph, pd_, pa = _outcome_probs(lh, la)
            noisy = np.array([ph, pd_, pa]) * rng.normal(1, 0.03, 3)
            odds = 1 / (noisy / noisy.sum() * (1 + margin))
            rows.append({"Date": day + pd.Timedelta(days=n // 10), "League": "SYN",
                         "HomeTeam": names[i], "AwayTeam": names[j], "FTHG": hg, "FTAG": ag,
                         "FTR": "H" if hg > ag else "D" if hg == ag else "A",
                         "OddsH": odds[0], "OddsD": odds[1], "OddsA": odds[2],
                         "HC": rng.poisson(5.5 * exp(0.3 * d + pair_fx[i, j])),
                         "AC": rng.poisson(4.5 * exp(-0.3 * d + pair_fx[i, j])),
                         "HS": rng.poisson(13.5 * exp(0.4 * d)), "AS": rng.poisson(10.5 * exp(-0.4 * d)),
                         "HST": rng.poisson(4.6 * exp(0.4 * d)), "AST": rng.poisson(3.4 * exp(-0.4 * d))})
        day += pd.Timedelta(days=380 + 40)
    df = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
    fx = df.tail(n_fixtures)[["Date", "League", "HomeTeam", "AwayTeam", "OddsH", "OddsD", "OddsA"]].reset_index(drop=True)
    return df.iloc[:-n_fixtures].reset_index(drop=True), fx
