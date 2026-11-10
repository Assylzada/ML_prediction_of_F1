"""
Общие вещи для pipeline.py и advanced.py (вынесены, чтобы не было циклических импортов):
препроцессор, константы и список "контекстных" колонок.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42

# Колонки, которые НЕ являются признаками обычных моделей (их отбрасывает ColumnTransformer),
# но нужны моделям, которые работают на уровне гонки (ranking, multilevel logit).
CONTEXT_COLUMNS = ["raceId", "driverRef", "constructorRef", "position"]

# Последовательности для рекуррентных сетей: последние SEQ_LEN гонок пилота (только прошлое).
SEQ_LEN = 5
SEQ_FEATURES = ["positionOrder", "grid_clean", "dnf", "points", "podium"]
SEQ_COLUMNS = [f"seq{t}_{f}" for t in range(1, SEQ_LEN + 1) for f in SEQ_FEATURES]  # t=1 - самая свежая гонка

class Winsorizer(BaseEstimator, TransformerMixin):
    """Обрезка выбросов по квантилям (считаются только на train)."""

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lower_ = np.nanquantile(X, self.lower, axis=0)
        self.upper_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def create_preprocessor(continuous, binary, categorical):
    continuous_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("winsor", Winsorizer(0.01, 0.99)),
        ("scaler", StandardScaler()),
    ])
    binary_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("cont", continuous_pipe, continuous),
        ("bin", binary_pipe, binary),
        ("cat", categorical_pipe, categorical),
    ])