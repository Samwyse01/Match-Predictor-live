"""Elo, form, head-to-head and corner/shot features. Every row uses ONLY pre-kickoff information."""
from collections import defaultdict, deque
from math import log

import pandas as pd

from .config import (DEFAULT_TOTAL, H2H_HALF_LIFE_DAYS, H2H_MAX_MEETINGS, H2H_PRIOR,
                     STAT_COLS, STAT_WINDOW)

BASE_FEATURES = ["elo_diff", "form_pts_diff", "home_gf", "home_ga", "away_gf", "away_ga", "rest_diff"]
H2H_FEATURES = ["h2h_pts", "h2h_gd", "h2h_w"]
FEATURES = BASE_FEATURES + H2H_FEATURES
DEFAULT_AVG = (1.4, 1.4, 1.4)  # points, goals for, goals against for unknown teams


def totals_features(stat: str, use_h2h: bool = True) -> list[str]:
    cols = [f"home_{stat}_for", f"home_{stat}_against", f"away_{stat}_for", f"away_{stat}_against", "abs_elo_diff"]
    if use_h2h:
        cols.append(f"h2h_{stat}_total")
    return cols


def _sign(x):
    return (x > 0) - (x < 0)


class FeatureBuilder:
    def __init__(self, k: float = 20, home_adv: float = 60, window: int = 5, init: float = 1500):
        self.k, self.home_adv, self.window, self.init = k, home_adv, window, init
        self.elo = defaultdict(lambda: init)
        self.recent = defaultdict(lambda: deque(maxlen=window))  # (pts, gf, ga)
        self.stat_recent = {s: defaultdict(lambda: deque(maxlen=STAT_WINDOW)) for s in STAT_COLS}
        self.meetings = defaultdict(lambda: deque(maxlen=H2H_MAX_MEETINGS))
        self.last_date = {}
        self.games = defaultdict(int)

    # ---- team form -------------------------------------------------------
    def _avg(self, team):
        r = self.recent[team]
        if not r:
            return DEFAULT_AVG
        n = len(r)
        return tuple(sum(x[i] for x in r) / n for i in range(3))

    def _stat_avg(self, stat, team):
        r = self.stat_recent[stat][team]
        if not r:
            half = DEFAULT_TOTAL[stat] / 2
            return half, half
        n = len(r)
        return sum(x[0] for x in r) / n, sum(x[1] for x in r) / n

    def _rest(self, team, date):
        last = self.last_date.get(team)
        return 7 if last is None else min((date - last).days, 14)

    # ---- head-to-head over time -----------------------------------------
    def _h2h(self, home, away, date) -> dict:
        """Time-decayed head-to-head from the point of view of the *home* team in this fixture."""
        sw = s_pts = s_gd = 0.0
        tot_w = {s: 0.0 for s in STAT_COLS}
        tot_sum = {s: 0.0 for s in STAT_COLS}
        for m_date, m_home, _m_away, hg, ag, totals in self.meetings[tuple(sorted((home, away)))]:
            w = 0.5 ** (max((date - m_date).days, 0) / H2H_HALF_LIFE_DAYS)
            gd = (hg - ag) if m_home == home else (ag - hg)
            sw += w
            s_pts += w * 3 * _sign(gd)
            s_gd += w * gd
            for stat, tot in totals.items():
                tot_w[stat] += w
                tot_sum[stat] += w * tot
        out = {"h2h_pts": s_pts / (sw + H2H_PRIOR), "h2h_gd": s_gd / (sw + H2H_PRIOR), "h2h_w": sw}
        for stat in STAT_COLS:
            out[f"h2h_{stat}_total"] = (tot_sum[stat] + H2H_PRIOR * DEFAULT_TOTAL[stat]) / (tot_w[stat] + H2H_PRIOR)
        return out

    # ---- public API --------------------------------------------------------
    def features(self, home, away, date) -> dict:
        hp, hgf, hga = self._avg(home)
        ap, agf, aga = self._avg(away)
        f = {
            "elo_diff": self.elo[home] + self.home_adv - self.elo[away],
            "abs_elo_diff": abs(self.elo[home] - self.elo[away]),
            "form_pts_diff": hp - ap,
            "home_gf": hgf, "home_ga": hga, "away_gf": agf, "away_ga": aga,
            "rest_diff": self._rest(home, date) - self._rest(away, date),
            "n_home": self.games[home], "n_away": self.games[away],
        }
        f.update(self._h2h(home, away, date))
        for stat in STAT_COLS:
            f[f"home_{stat}_for"], f[f"home_{stat}_against"] = self._stat_avg(stat, home)
            f[f"away_{stat}_for"], f[f"away_{stat}_against"] = self._stat_avg(stat, away)
        return f

    @staticmethod
    def _parse_extra(extra) -> dict:
        """{stat: (home_value, away_value)} for the stats that are present and numeric."""
        out = {}
        for stat, (hc, ac) in STAT_COLS.items():
            h, a = (extra or {}).get(hc), (extra or {}).get(ac)
            if h is None or a is None or pd.isna(h) or pd.isna(a):
                continue
            out[stat] = (float(h), float(a))
        return out

    def update(self, home, away, hg, ag, date, extra=None):
        diff = self.elo[home] + self.home_adv - self.elo[away]
        exp_home = 1 / (1 + 10 ** (-diff / 400))
        score = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        mult = log(abs(hg - ag) + 1) + 1
        delta = self.k * mult * (score - exp_home)
        self.elo[home] += delta
        self.elo[away] -= delta
        hpts = 3 if hg > ag else 1 if hg == ag else 0
        apts = 3 if ag > hg else 1 if hg == ag else 0
        self.recent[home].append((hpts, hg, ag))
        self.recent[away].append((apts, ag, hg))

        stats = self._parse_extra(extra)
        for stat, (h, a) in stats.items():
            self.stat_recent[stat][home].append((h, a))
            self.stat_recent[stat][away].append((a, h))
        self.meetings[tuple(sorted((home, away)))].append(
            (date, home, away, hg, ag, {s: h + a for s, (h, a) in stats.items()}))

        self.last_date[home] = self.last_date[away] = date
        self.games[home] += 1
        self.games[away] += 1

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Walk matches in date order; features first, then update state (no leakage)."""
        stat_cols = [c for pair in STAT_COLS.values() for c in pair]
        out = []
        for row in df.sort_values("Date", kind="stable").itertuples():
            f = self.features(row.HomeTeam, row.AwayTeam, row.Date)
            f["_idx"] = row.Index
            out.append(f)
            extra = {c: getattr(row, c, None) for c in stat_cols}
            self.update(row.HomeTeam, row.AwayTeam, row.FTHG, row.FTAG, row.Date, extra)
        return pd.DataFrame(out).set_index("_idx").sort_index()


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add y_corners / y_shots / y_shots_on_target (home + away totals) where the data has them."""
    df = df.copy()
    for stat, (h, a) in STAT_COLS.items():
        if h in df.columns and a in df.columns:
            df[f"y_{stat}"] = df[h] + df[a]
    return df


def build_dataset(df: pd.DataFrame):
    """Return (history with pre-match features + targets, the FeatureBuilder holding end-of-history state)."""
    fb = FeatureBuilder()
    feats = fb.transform(df)
    return add_targets(df.join(feats)), fb
