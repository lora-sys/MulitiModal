"""Small tabular baselines; every preprocessing fit stays inside its train fold."""

import numpy as np

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class MaskImputer(SimpleImputer):
    """Always emit one missingness bit per input, including unseen missingness."""

    def __init__(self):
        super().__init__(strategy="median", keep_empty_features=True)

    def transform(self, X):
        missing = np.isnan(np.asarray(X, dtype=float)).astype(float)
        values = super().transform(X)
        return np.concatenate([values, missing], axis=1)


def make_model(spec: dict, seed: int) -> Pipeline:
    params = spec["params"]
    if spec["model"] == "naive":
        estimator = DummyRegressor(strategy="mean")
    elif spec["model"] == "elastic_net":
        estimator = MultiOutputRegressor(ElasticNet(max_iter=20000, random_state=seed, **params))
    elif spec["model"] == "random_forest":
        estimator = RandomForestRegressor(random_state=seed, n_jobs=1, **params)
    else:
        from xgboost import XGBRegressor
        estimator = MultiOutputRegressor(XGBRegressor(random_state=seed, n_jobs=1, objective="reg:squarederror", **params))
    return Pipeline([("missing", MaskImputer()),
                     ("scale", StandardScaler()), ("model", estimator)])
