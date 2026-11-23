import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import LABELS
from .features import FEATURES

Y_MAP = {l: i for i, l in enumerate(LABELS)}


# ---------------------------------------------------------------- 1X2 result model
def make_model(kind: str = "logreg"):
    if kind == "gbm":
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=0)
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))


def fit(df: pd.DataFrame, kind: str = "logreg", features: list[str] = FEATURES):
    model = make_model(kind)
    model.fit(df[features], df["FTR"].map(Y_MAP))
    return model


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X[list(model.feature_names_in_)])  # columns ordered H, D, A


def implied_probs(odds: pd.DataFrame) -> np.ndarray:
    """Convert decimal odds (H, D, A) to probabilities with the bookmaker margin removed."""
    raw = 1.0 / odds.to_numpy(dtype=float)
    return raw / raw.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------- totals (corners / shots)
class TotalsModel:
    """Poisson regression for the expected total, negative binomial for over/under probabilities."""

    def __init__(self, stat: str, features: list[str]):
        self.stat, self.features = stat, features
        self.target = f"y_{stat}"
        self.pipe = make_pipeline(StandardScaler(), PoissonRegressor(alpha=1e-3, max_iter=500))
        self.alpha = 0.0  # over-dispersion: var = mu + alpha * mu^2

    def fit(self, df: pd.DataFrame):
        d = df.dropna(subset=[self.target] + self.features)
        self.pipe.fit(d[self.features], d[self.target])
        mu, y = self.pipe.predict(d[self.features]), d[self.target].to_numpy()
        self.alpha = max(0.0, float(np.mean((y - mu) ** 2 - mu) / np.mean(mu**2)))
        return self

    def mean(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipe.predict(X[self.features])

    def prob_over(self, mu, line: float):
        """P(total > line) for a half-line such as 9.5, i.e. total >= 10."""
        k = int(np.floor(line))
        mu = np.asarray(mu, dtype=float)
        if self.alpha < 1e-6:
            return 1 - poisson.cdf(k, mu)
        n = 1 / self.alpha
        return 1 - nbinom.cdf(k, n, n / (n + mu))
