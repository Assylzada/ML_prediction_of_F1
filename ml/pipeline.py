import os
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# работает и как `from ml.pipeline import run_ml`, и как `python ml/pipeline.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml import advanced as adv                                         # noqa: E402
from ml.common import (                                                # noqa: E402
    CONTEXT_COLUMNS, RANDOM_STATE, SEQ_COLUMNS, SEQ_FEATURES, SEQ_LEN, create_preprocessor,
)

# Необязательные библиотеки: если установлены, модели добавляются в сравнение сами
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

warnings.filterwarnings("ignore")

MIN_YEAR = 2003          # современная эра: стабильные регламенты, есть квалификации
TEST_RACE_FRACTION = 0.20  # последние 20% гонок по времени - тест
PRIOR_PODIUM_RATE = 0.15   # априорная доля подиумов (3 из ~20) для сглаживания


# ============================================================
# LOAD DATA
# ============================================================

def load_data(data_path):
    # В датасете Ergast пропуски записаны как \N
    def read(name):
        return pd.read_csv(os.path.join(data_path, name), na_values=["\\N"])

    return (
        read("results.csv"),
        read("races.csv"),
        read("qualifying.csv"),
        read("drivers.csv"),
        read("constructors.csv"),
    )


# ============================================================
# MERGE DATASETS
# ============================================================

def merge_data(results, races, qualifying, drivers, constructors):
    df = results[
        ["raceId", "driverId", "constructorId", "grid",
         "position", "positionOrder", "points"]
    ].copy()

    df = df.merge(
        races[["raceId", "year", "round", "circuitId", "name", "date"]]
        .rename(columns={"name": "race_name"}),
        on="raceId", how="left",
    )

    quali = (
        qualifying[["raceId", "driverId", "position"]]
        .rename(columns={"position": "qualifying_position"})
        .drop_duplicates(["raceId", "driverId"])
    )
    df = df.merge(quali, on=["raceId", "driverId"], how="left")

    df = df.merge(
        drivers[["driverId", "driverRef", "dob"]],
        on="driverId", how="left",
    )
    df = df.merge(
        constructors[["constructorId", "constructorRef"]],
        on="constructorId", how="left",
    )
    return df


# ============================================================
# TASK 3 + TASK 4: DATA PREPARATION + FEATURE ENGINEERING
# ============================================================

def _roll_prev(series, window):
    """Среднее за последние `window` значений, НЕ включая текущее (shift(1))."""
    return series.shift(1).rolling(window, min_periods=1).mean()


def add_history_features(df):
    """Признаки формы/силы. Только прошлое -> без утечки."""
    df = df.copy()

    for col in ["year", "round", "circuitId", "grid", "position",
                "positionOrder", "points", "qualifying_position"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
    df["points"] = df["points"].fillna(0)

    # Цель: подиум (позиция 1-3). NaN (сход/дисквалификация) -> 0
    df["podium"] = (df["position"] <= 3).astype(int)
    df["dnf"] = df["position"].isna().astype(int)

    # Хронологический порядок гонок
    race_order = (
        df[["raceId", "date", "round"]].drop_duplicates("raceId")
        .sort_values(["date", "round", "raceId"])
        .reset_index(drop=True)
    )
    race_order["race_order"] = np.arange(len(race_order))
    df = df.merge(race_order[["raceId", "race_order"]], on="raceId", how="left")

    df = df.drop_duplicates(["raceId", "driverId"])
    df = df.sort_values(["race_order", "driverId"]).reset_index(drop=True)

    # ---------------- Пилот ----------------
    g = df.groupby("driverId")
    df["drv_prev_races"] = g.cumcount()
    prior_podiums = g["podium"].cumsum() - df["podium"]
    k = 5  # сила сглаживания
    df["drv_podium_rate"] = (
        (prior_podiums + PRIOR_PODIUM_RATE * k) / (df["drv_prev_races"] + k)
    )
    df["drv_form_finish5"] = g["positionOrder"].transform(lambda s: _roll_prev(s, 5))
    df["drv_podium_rate10"] = g["podium"].transform(lambda s: _roll_prev(s, 10))
    df["drv_dnf_rate10"] = g["dnf"].transform(lambda s: _roll_prev(s, 10))

    # Очки в чемпионате ДО этой гонки (в долях от лидера - не зависит от системы очков)
    df["drv_season_pts"] = (
        df.groupby(["driverId", "year"])["points"].cumsum() - df["points"]
    )
    leader = df.groupby("raceId")["drv_season_pts"].transform("max")
    df["drv_pts_share"] = (df["drv_season_pts"] / leader.replace(0, np.nan)).fillna(0)
    df["drv_champ_rank_pct"] = df.groupby("raceId")["drv_season_pts"].rank(
        pct=True, ascending=False
    )
    df["drv_form_rank_pct"] = df.groupby("raceId")["drv_form_finish5"].rank(
        pct=True, ascending=True
    )

    # История пилота на этой трассе
    gc = df.groupby(["driverId", "circuitId"])
    df["drv_circuit_starts"] = gc.cumcount()
    df["drv_circuit_avg_finish"] = gc["positionOrder"].transform(
        lambda s: s.shift(1).expanding().mean()
    )

    # ---------------- Команда ----------------
    cr = (
        df.groupby(["constructorId", "raceId"], as_index=False)
        .agg(
            race_order=("race_order", "first"),
            year=("year", "first"),
            c_finish=("positionOrder", "mean"),
            c_podium=("podium", "mean"),
            c_points=("points", "sum"),
        )
        .sort_values(["race_order", "constructorId"])
        .reset_index(drop=True)
    )
    cg = cr.groupby("constructorId")
    cr["con_form_finish5"] = cg["c_finish"].transform(lambda s: _roll_prev(s, 5))
    cr["con_podium_rate10"] = cg["c_podium"].transform(lambda s: _roll_prev(s, 10))
    cr["con_season_pts"] = (
        cr.groupby(["constructorId", "year"])["c_points"].cumsum() - cr["c_points"]
    )
    c_leader = cr.groupby("raceId")["con_season_pts"].transform("max")
    cr["con_pts_share"] = (cr["con_season_pts"] / c_leader.replace(0, np.nan)).fillna(0)
    cr["con_season_rank_pct"] = cr.groupby("raceId")["con_season_pts"].rank(
        pct=True, ascending=False
    )
    cr["con_form_rank_pct"] = cr.groupby("raceId")["con_form_finish5"].rank(
        pct=True, ascending=True
    )
    df = df.merge(
        cr[["constructorId", "raceId", "con_form_finish5", "con_podium_rate10",
            "con_pts_share", "con_season_rank_pct", "con_form_rank_pct"]],
        on=["constructorId", "raceId"], how="left",
    )

    # ---------------- Старт ----------------
    df["pit_lane_start"] = (df["grid"] == 0).astype(int)
    df["grid_nan"] = df["grid"].where(df["grid"] > 0)
    race_max = df.groupby("raceId")["grid_nan"].transform("max")
    df["grid_clean"] = df["grid_nan"].fillna(race_max + 1)
    df["grid_clean"] = df["grid_clean"].fillna(df["grid_clean"].median())
    field_size = df.groupby("raceId")["driverId"].transform("count")
    df["grid_pct"] = df["grid_clean"] / field_size
    df["front_row"] = (df["grid_clean"] <= 2).astype(int)
    df["top6_start"] = (df["grid_clean"] <= 6).astype(int)
    df["top10_start"] = (df["grid_clean"] <= 10).astype(int)
    df["grid_minus_team_avg"] = df["grid_clean"] - df.groupby(
        ["raceId", "constructorId"]
    )["grid_clean"].transform("mean")

    df.loc[df["qualifying_position"] <= 0, "qualifying_position"] = np.nan
    df["quali_missing"] = df["qualifying_position"].isna().astype(int)
    df["grid_vs_quali"] = df["qualifying_position"] - df["grid_clean"]

    df["driver_age"] = (df["date"] - df["dob"]).dt.days / 365.25

    # Категориальный признак (кодируется OneHot)
    df["grid_zone"] = np.select(
        [
            df["pit_lane_start"] == 1,
            df["grid_clean"] == 1,
            df["grid_clean"] == 2,
            df["grid_clean"] <= 6,
            df["grid_clean"] <= 10,
            df["grid_clean"] <= 15,
        ],
        ["pit_lane", "pole", "front_row", "top6", "top10", "midfield"],
        default="back",
    )

    # Последовательности для LSTM/GRU/TCN: последние SEQ_LEN гонок пилота (shift -> только прошлое)
    gd = df.groupby("driverId")
    seq = {f"seq{t}_{f}": gd[f].shift(t)
           for t in range(1, SEQ_LEN + 1) for f in SEQ_FEATURES}
    df = pd.concat([df, pd.DataFrame(seq, index=df.index)], axis=1)
    return df


CONTINUOUS_FEATURES = [
    "grid_clean", "grid_pct", "qualifying_position", "grid_vs_quali",
    "grid_minus_team_avg",
    "driver_age", "drv_prev_races", "drv_podium_rate", "drv_form_finish5",
    "drv_podium_rate10", "drv_dnf_rate10", "drv_pts_share", "drv_champ_rank_pct",
    "drv_form_rank_pct", "drv_circuit_starts", "drv_circuit_avg_finish",
    "con_form_finish5", "con_podium_rate10", "con_pts_share",
    "con_season_rank_pct", "con_form_rank_pct",
]
BINARY_FEATURES = [
    "pit_lane_start", "front_row", "top6_start", "top10_start", "quali_missing",
]
CATEGORICAL_FEATURES = ["grid_zone"]

META_COLUMNS = [
    "raceId", "race_order", "year", "round", "race_name",
    "driverRef", "constructorRef", "position", "grid",
]


def prepare_dataset(df, min_year=MIN_YEAR):
    df = add_history_features(df)
    df = df[df["year"] >= min_year].reset_index(drop=True)

    features = CONTINUOUS_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
    dataset = df[features + META_COLUMNS + SEQ_COLUMNS + ["podium"]].copy()
    return dataset, features


def time_split(dataset, test_fraction=TEST_RACE_FRACTION):
    """Train = более ранние гонки, Test = самые свежие гонки."""
    orders = np.sort(dataset["race_order"].unique())
    cutoff = orders[int(len(orders) * (1 - test_fraction))]
    train_mask = dataset["race_order"] < cutoff
    return dataset[train_mask].copy(), dataset[~train_mask].copy()


PROTECTED_FEATURES = {"grid_clean"}  # ключевые признаки, которые нельзя удалять


def select_features(X_train, y_train, continuous, corr_threshold=0.95):
    """
    Task 4: отбор признаков по корреляции.
    Из пары сильно коррелирующих признаков (|r| > 0.95) оставляем тот,
    у которого выше |корреляция| с целевой переменной.
    """
    target_corr = X_train[continuous].apply(
        lambda c: c.corr(y_train)
    ).abs().fillna(0)
    corr_matrix = X_train[continuous].corr().abs()

    dropped = set()
    for i, a in enumerate(continuous):
        for b in continuous[i + 1:]:
            if a in dropped or b in dropped:
                continue
            if corr_matrix.loc[a, b] > corr_threshold:
                if a in PROTECTED_FEATURES:
                    dropped.add(b)
                elif b in PROTECTED_FEATURES:
                    dropped.add(a)
                else:
                    dropped.add(a if target_corr[a] < target_corr[b] else b)

    kept = [c for c in continuous if c not in dropped]
    return kept, sorted(dropped), target_corr.sort_values(ascending=False)


# ============================================================
# PREPROCESSOR (missing values, outliers, scaling, encoding)
# ============================================================

# ============================================================
# MODELS + GRIDS
# ============================================================

MODEL_FAMILY = {
    "Logistic Regression": "Linear",
    "KNN": "Instance-based",
    "SVM (RBF)": "Kernel",
    "Neural Network (MLP)": "Neural network",
    "Random Forest": "Bagging ensemble",
    "Extra Trees": "Bagging ensemble",
    "AdaBoost": "Boosting ensemble",
    "Gradient Boosting": "Boosting ensemble",
    "XGBoost": "Boosting ensemble",
    "LightGBM": "Boosting ensemble",
    "Stacking Ensemble": "Stacking ensemble",
    "PCA + Logistic Regression": "Dimensionality reduction",
    "EDNN-style (2 networks)": "Neural network",
    "Random Forest (balanced)": "Bagging ensemble",
    "LightGBM (balanced)": "Boosting ensemble",
    "Pairwise Ranker": "Learning to rank",
    "Multilevel Rank-Logit": "Bayesian multilevel",
    **{label: "Sequence model" for label in
       ("LSTM (last 5 races)", "GRU (last 5 races)", "Bi-LSTM (last 5 races)",
        "TCN (last 5 races)", "CNN-BiLSTM (last 5 races)")},
}

# модели, которым нужен контекст целой гонки (raceId/пилот/команда) или последовательность
# прошлых гонок: для одиночного what-if предсказания они не подходят
CONTEXT_MODELS = {"Pairwise Ranker", "Multilevel Rank-Logit",
                  "LSTM (last 5 races)", "GRU (last 5 races)", "Bi-LSTM (last 5 races)",
                  "TCN (last 5 races)", "CNN-BiLSTM (last 5 races)"}
# медленные модели: режим quick их пропускает
SLOW_MODELS = {"SVM (RBF)", "Stacking Ensemble", "EDNN-style (2 networks)", "Extra Trees",
               "LSTM (last 5 races)", "GRU (last 5 races)", "Bi-LSTM (last 5 races)",
               "TCN (last 5 races)", "CNN-BiLSTM (last 5 races)"}
SEQ_LABELS = {"lstm": "LSTM (last 5 races)", "gru": "GRU (last 5 races)",
              "bilstm": "Bi-LSTM (last 5 races)", "tcn": "TCN (last 5 races)",
              "cnn_bilstm": "CNN-BiLSTM (last 5 races)"}


def family_of(name):
    base = re.sub(r" Tuned.*$", "", name)
    return MODEL_FAMILY.get(base, "Other")


def build_models(continuous, binary, categorical):
    """
    Сравниваются модели из 6 семейств. Итоговый выбор делается по кросс-валидации
    (PR-AUC), а не "на глаз". XGBoost / LightGBM добавляются, только если установлены.
    """
    def pipe(model):
        # у каждой модели СВОЙ препроцессор
        return Pipeline([
            ("preprocessor", create_preprocessor(continuous, binary, categorical)),
            ("model", model),
        ])

    hgb_kwargs = dict(
        learning_rate=0.03, max_iter=150, max_depth=3, min_samples_leaf=50,
        l2_regularization=1.0, early_stopping=False, random_state=RANDOM_STATE)

    models = {
        "Logistic Regression": pipe(
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        "KNN": pipe(KNeighborsClassifier(n_neighbors=25, weights="distance")),
        "SVM (RBF)": pipe(
            SVC(C=1.0, probability=True, random_state=RANDOM_STATE)),
        "Neural Network (MLP)": pipe(
            MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-2, max_iter=500,
                          early_stopping=True, random_state=RANDOM_STATE)),
        "Random Forest": pipe(
            RandomForestClassifier(
                n_estimators=300, min_samples_leaf=5, max_features="sqrt",
                random_state=RANDOM_STATE, n_jobs=1)),
        "Extra Trees": pipe(
            ExtraTreesClassifier(
                n_estimators=400, min_samples_leaf=5, max_features="sqrt",
                random_state=RANDOM_STATE, n_jobs=1)),
        "AdaBoost": pipe(
            AdaBoostClassifier(n_estimators=200, learning_rate=0.1,
                               random_state=RANDOM_STATE)),
        "Gradient Boosting": pipe(HistGradientBoostingClassifier(**hgb_kwargs)),
    }

    if XGBClassifier is not None:
        models["XGBoost"] = pipe(XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=1))
    if LGBMClassifier is not None:
        models["LightGBM"] = pipe(LGBMClassifier(
            n_estimators=200, learning_rate=0.03, num_leaves=8, min_child_samples=40,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=RANDOM_STATE, n_jobs=1, verbose=-1))

    # Стекинг: мета-модель (LR) учится на предсказаниях RF, Gradient Boosting и LR
    models["Stacking Ensemble"] = pipe(StackingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(
                n_estimators=200, min_samples_leaf=5, random_state=RANDOM_STATE)),
            ("gb", HistGradientBoostingClassifier(**hgb_kwargs)),
            ("lr", LogisticRegression(max_iter=2000)),
        ],
        final_estimator=LogisticRegression(max_iter=2000),
        cv=3,
    ))

    # ---- методы из литобзора (ml/advanced.py) ----
    models["PCA + Logistic Regression"] = adv.build_pca_logreg(continuous, binary, categorical)
    models["EDNN-style (2 networks)"] = adv.build_ednn(continuous, binary, categorical)
    models["Random Forest (balanced)"] = pipe(RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5, max_features="sqrt",
        class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=1))
    if LGBMClassifier is not None:
        models["LightGBM (balanced)"] = pipe(LGBMClassifier(
            n_estimators=200, learning_rate=0.03, num_leaves=8, min_child_samples=40,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1, verbose=-1))

    cols = dict(continuous=continuous, binary=binary, categorical=categorical)
    # эти модели сами делают предобработку внутри и берут контекст гонки из X
    models["Pairwise Ranker"] = Pipeline([("model", adv.PairwiseRankClassifier(**cols))])
    models["Multilevel Rank-Logit"] = Pipeline([("model", adv.MultilevelRankLogit(**cols))])
    if adv.TORCH_OK:
        for arch, label in SEQ_LABELS.items():
            models[label] = Pipeline([("model", adv.TorchSeqClassifier(arch=arch, **cols))])
    return models


def available_models():
    """Метаданные моделей для интерфейса (какие есть, какие медленные, чего не хватает)."""
    probe = build_models(["grid_clean"], [], [])
    meta = [{"name": n, "family": MODEL_FAMILY.get(n, "Other"), "slow": n in SLOW_MODELS,
             "available": True, "tunable": n in PARAM_GRIDS} for n in probe]
    missing = []
    if XGBClassifier is None:
        missing.append({"name": "XGBoost", "hint": "pip install xgboost"})
    if LGBMClassifier is None:
        missing += [{"name": "LightGBM", "hint": "pip install lightgbm"},
                    {"name": "LightGBM (balanced)", "hint": "pip install lightgbm"}]
    if not adv.TORCH_OK:
        missing += [{"name": n, "hint": "pip install torch"} for n in SEQ_LABELS.values()]
    for m in missing:
        meta.append({"name": m["name"], "family": MODEL_FAMILY.get(m["name"], "Other"),
                     "slow": m["name"] in SLOW_MODELS, "available": False,
                     "hint": m["hint"], "tunable": m["name"] in PARAM_GRIDS})
    return meta


PARAM_GRIDS = {
    "Logistic Regression": {
        "model__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10],
        "model__class_weight": [None, "balanced"],
    },
    "KNN": {
        "model__n_neighbors": [15, 25, 40, 60],
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2],
    },
    "Neural Network (MLP)": {
        "model__hidden_layer_sizes": [(16,), (32, 16)],
        "model__alpha": [1e-3, 1e-2, 1e-1],
    },
    "Random Forest": {
        "model__n_estimators": [300, 500],
        "model__max_depth": [6, 10, None],
        "model__min_samples_leaf": [3, 10, 25],
        "model__max_features": ["sqrt", 0.5],
    },
    "Extra Trees": {
        "model__n_estimators": [300, 500],
        "model__max_depth": [6, 10, None],
        "model__min_samples_leaf": [3, 10, 25],
    },
    "AdaBoost": {
        "model__n_estimators": [100, 200, 400],
        "model__learning_rate": [0.05, 0.1, 0.5],
    },
    "Gradient Boosting": {
        "model__learning_rate": [0.02, 0.03, 0.05],
        "model__max_depth": [2, 3, 4],
        "model__min_samples_leaf": [30, 50, 80],
        "model__max_iter": [100, 150, 250],
        "model__l2_regularization": [0.0, 1.0, 5.0],
    },
    "XGBoost": {
        "model__max_depth": [2, 3, 4],
        "model__learning_rate": [0.03, 0.06],
        "model__n_estimators": [150, 300],
    },
    "LightGBM": {
        "model__num_leaves": [4, 8, 16],
        "model__learning_rate": [0.03, 0.06],
        "model__n_estimators": [150, 300],
    },
    "PCA + Logistic Regression": {
        "pca__n_components": [0.80, 0.90, 0.95],
        "model__C": [0.03, 0.3, 3],
    },
    "Random Forest (balanced)": {
        "model__max_depth": [6, 10, None],
        "model__min_samples_leaf": [3, 10, 25],
    },
    "Pairwise Ranker": {"model__C": [0.03, 0.1, 0.3, 1, 3]},
    "Multilevel Rank-Logit": {
        "model__sigma_driver": [0.3, 1.0],
        "model__sigma_team": [0.3, 1.0],
        "model__l2": [0.3, 3.0],
    },
    # SVM, Stacking, EDNN и сети на PyTorch не настраиваются: SVM слишком медленный, Stacking - уже ансамбль
}


# ============================================================
# METRICS
# ============================================================

def best_threshold(y_true, proba):
    """Порог, максимизирующий F1 (подбирается ТОЛЬКО на train OOF)."""
    thresholds = np.linspace(0.05, 0.90, 86)
    scores = [f1_score(y_true, proba >= t, zero_division=0) for t in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def compute_metrics(y_true, proba, threshold, name, frame=None):
    pred = (proba >= threshold).astype(int)
    hit = None
    if frame is not None:
        hit = top3_hit_rate(frame.assign(_proba=proba), "_proba")
    return {
        "Model": name,
        "Accuracy": round(accuracy_score(y_true, pred), 4),
        "Precision": round(precision_score(y_true, pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, pred, zero_division=0), 4),
        "F1": round(f1_score(y_true, pred, zero_division=0), 4),
        "ROC_AUC": round(roc_auc_score(y_true, proba), 4),
        "PR_AUC": round(average_precision_score(y_true, proba), 4),
        "Top3_Hit": hit,
        "Brier": round(float(brier_score_loss(y_true, np.clip(proba, 0, 1))), 4),
        "LogLoss": round(float(log_loss(y_true, np.clip(proba, 1e-4, 1 - 1e-4))), 4),
        "Threshold": round(float(threshold), 2),
    }


def top3_hit_rate(frame, proba_col):
    """Доля реальных подиумов, найденных, если в каждой гонке взять топ-3 по вероятности."""
    hits, total = 0, 0
    for _, race in frame.groupby("raceId"):
        top3 = race.nlargest(3, proba_col)
        hits += int(top3["podium"].sum())
        total += int(race["podium"].sum())
    return round(hits / total, 4) if total else 0.0


def cross_validate_model(model, X, y, groups, cv):
    """5-fold CV с группами по гонкам. Возвращает OOF вероятности и метрики по фолдам."""
    oof = np.zeros(len(y))
    fold_data = []
    for tr, va in cv.split(X, y, groups):
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[va])[:, 1]
        oof[va] = p
        fold_data.append((va, p))

    threshold = best_threshold(y, oof)
    f1s, aucs, aps = [], [], []
    for va, p in fold_data:
        yt = y.iloc[va]
        f1s.append(f1_score(yt, p >= threshold, zero_division=0))
        aucs.append(roc_auc_score(yt, p))
        aps.append(average_precision_score(yt, p))
    return oof, threshold, np.array(f1s), np.array(aucs), np.array(aps)


def bootstrap_ci(test_df, scores, n_boot=300, seed=RANDOM_STATE):
    """
    Доверительные интервалы (95%) бутстрепом ПО ГОНКАМ: гонки ресемплируются целиком.
    Показывает, где различие между моделями - реальное, а где - шум малого теста.
    scores: {имя: массив оценок на тесте}, обязательно есть ключ "__baseline__".
    """
    y = test_df["podium"].to_numpy()
    race_ids = test_df["raceId"].to_numpy()
    uniq = np.unique(race_ids)
    rows_by_race = [np.where(race_ids == r)[0] for r in uniq]
    pods = np.array([y[i].sum() for i in rows_by_race])

    hits = {}
    for name, s in scores.items():
        s = np.asarray(s, dtype=float)
        hits[name] = np.array([
            int(y[i][np.argsort(-s[i], kind="stable")[:3]].sum())
            for i in rows_by_race
        ])

    rng = np.random.default_rng(seed)
    boot = {n: {"top3": [], "pr": []} for n in scores}
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        rows = np.concatenate([rows_by_race[k] for k in pick])
        total = pods[pick].sum()
        if total == 0 or y[rows].sum() == 0:
            continue
        for n, s in scores.items():
            boot[n]["top3"].append(hits[n][pick].sum() / total)
            boot[n]["pr"].append(average_precision_score(y[rows], np.asarray(s)[rows]))

    def ci(values):
        lo, hi = np.percentile(values, [2.5, 97.5])
        return [round(float(lo), 4), round(float(hi), 4)]

    base_t = np.array(boot["__baseline__"]["top3"])
    base_p = np.array(boot["__baseline__"]["pr"])
    out = {}
    for n in scores:
        t = np.array(boot[n]["top3"])
        pr = np.array(boot[n]["pr"])
        out[n] = {
            "top3": ci(t),
            "pr_auc": ci(pr),
            "delta_top3": ci(t - base_t),
            "delta_pr_auc": ci(pr - base_p),
        }
    return out, int(len(uniq))


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    return obj


# ============================================================
# PLOTS: графики по КАЖДОЙ модели + сводные графики
# ============================================================
# Используется объектный API matplotlib (Figure, без pyplot): не зависит от
# бэкенда, не открывает окон и безопасно работает в сервере/потоке.

PODIUM_COLOR = "#d62728"
BASE_COLOR = "#7f7f7f"


def _slug(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()


def _new_fig(figsize):
    from matplotlib.figure import Figure
    return Figure(figsize=figsize, layout="constrained")


def _save(fig, path):
    fig.savefig(path, dpi=130)
    return path


def plot_model_report(name, family, y_true, proba, threshold, base_score, baseline,
                      test_metrics, f1_folds, pr_folds, out_path, test_years=""):
    """Одна страница на модель: ROC, PR, матрица ошибок, распределение вероятностей,
    стабильность по фолдам CV и метрики против baseline."""
    y = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    color = "#1f77b4"

    fig = _new_fig((17, 9.5))
    ax = fig.subplots(2, 3)

    # 1) ROC
    a = ax[0, 0]
    fpr, tpr, _ = roc_curve(y, proba)
    bfpr, btpr, _ = roc_curve(y, base_score)
    a.plot(fpr, tpr, color=color, lw=2.2,
           label=f"{name} (AUC = {test_metrics['ROC_AUC']:.3f})")
    a.plot(bfpr, btpr, color=BASE_COLOR, ls="--", lw=1.5,
           label=f"Grid-position baseline (AUC = {baseline['ROC_AUC']:.3f})")
    a.plot([0, 1], [0, 1], color="lightgray", lw=1)
    a.scatter([fp / max(fp + tn, 1)], [tp / max(tp + fn, 1)], color=PODIUM_COLOR,
              zorder=5, s=60, label=f"decision threshold {threshold:.2f}")
    a.set(title="ROC curve (test)", xlabel="False Positive Rate",
          ylabel="True Positive Rate", xlim=(0, 1), ylim=(0, 1.02))
    a.legend(loc="lower right", fontsize=9)
    a.grid(alpha=0.3)

    # 2) Precision-Recall
    a = ax[0, 1]
    prec, rec, _ = precision_recall_curve(y, proba)
    bprec, brec, _ = precision_recall_curve(y, base_score)
    a.plot(rec, prec, color=color, lw=2.2,
           label=f"{name} (PR-AUC = {test_metrics['PR_AUC']:.3f})")
    a.plot(brec, bprec, color=BASE_COLOR, ls="--", lw=1.5,
           label=f"Grid-position baseline (PR-AUC = {baseline['PR_AUC']:.3f})")
    a.axhline(y.mean(), color="lightgray", lw=1, label=f"podium share = {y.mean():.2f}")
    a.scatter([tp / max(tp + fn, 1)], [tp / max(tp + fp, 1)], color=PODIUM_COLOR,
              zorder=5, s=60, label=f"decision threshold {threshold:.2f}")
    a.set(title="Precision-Recall curve (test)", xlabel="Recall",
          ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
    a.legend(loc="upper right", fontsize=9)
    a.grid(alpha=0.3)

    # 3) Матрица ошибок
    a = ax[0, 2]
    a.imshow(cm, cmap="Blues")
    row_sum = cm.sum(axis=1, keepdims=True).clip(min=1)
    for i in range(2):
        for j in range(2):
            a.text(j, i, f"{cm[i, j]}\n({cm[i, j] / row_sum[i, 0]:.0%} of row)",
                   ha="center", va="center", fontsize=12,
                   color="white" if cm[i, j] > cm.max() / 2 else "black")
    a.set(title=f"Confusion matrix (threshold {threshold:.2f})",
          xticks=[0, 1], yticks=[0, 1],
          xticklabels=["no podium", "podium"], yticklabels=["no podium", "podium"],
          xlabel="Predicted", ylabel="Actual")

    # 4) Распределение предсказанных вероятностей
    a = ax[1, 0]
    bins = np.linspace(0, 1, 41)
    a.hist(proba[y == 0], bins=bins, density=True, alpha=0.6, color=BASE_COLOR,
           label="no podium")
    a.hist(proba[y == 1], bins=bins, density=True, alpha=0.6, color=PODIUM_COLOR,
           label="podium")
    a.axvline(threshold, color="black", ls="--", lw=1.5, label=f"threshold {threshold:.2f}")
    a.set(title="Predicted podium probability (test)",
          xlabel="Predicted probability", ylabel="Density")
    a.legend(fontsize=9)
    a.grid(alpha=0.3)

    # 5) Стабильность по фолдам кросс-валидации
    a = ax[1, 1]
    f1_folds = np.asarray(f1_folds, dtype=float)
    pr_folds = np.asarray(pr_folds, dtype=float)
    x = np.arange(1, len(f1_folds) + 1)
    w = 0.38
    a.bar(x - w / 2, f1_folds, w, color="#2ca02c",
          label=f"F1 (mean {f1_folds.mean():.3f} ± {f1_folds.std():.3f})")
    a.bar(x + w / 2, pr_folds, w, color="#ff7f0e",
          label=f"PR-AUC (mean {pr_folds.mean():.3f} ± {pr_folds.std():.3f})")
    a.set(title="Cross-validation by fold (train)", xlabel="Fold", ylabel="Value",
          xticks=x, ylim=(0, 1))
    a.legend(loc="lower right", fontsize=9)
    a.grid(alpha=0.3, axis="y")

    # 6) Метрики модели против baseline
    a = ax[1, 2]
    keys = ["Precision", "Recall", "F1", "ROC_AUC", "PR_AUC", "Top3_Hit"]
    mv = [float(test_metrics.get(k) or 0) for k in keys]
    bv = [float(baseline.get(k) or 0) for k in keys]
    x = np.arange(len(keys))
    a.bar(x - w / 2, mv, w, color=color, label=name)
    a.bar(x + w / 2, bv, w, color=BASE_COLOR, label="Baseline (grid ≤ 3)")
    for xi, v in zip(x - w / 2, mv):
        a.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    for xi, v in zip(x + w / 2, bv):
        a.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    a.set(title="Test metrics vs baseline", ylim=(0, 1.1), xticks=x)
    a.set_xticklabels(keys, rotation=20)
    a.legend(loc="upper left", fontsize=9, ncols=2)
    a.grid(alpha=0.3, axis="y")

    fig.suptitle(f"{name}  |  family: {family}  |  test: {test_years}",
                 fontsize=15, fontweight="bold")
    return _save(fig, out_path)


def plot_summary_metrics(zoo, baseline, baseline_ci, out_path):
    """PR-AUC и Top-3 hit на тесте для всех моделей с 95% бутстреп-интервалами."""
    zoo = sorted(zoo, key=lambda z: z["test"]["PR_AUC"])
    names = [z["Model"] for z in zoo]
    y = np.arange(len(names))
    fig = _new_fig((16, 0.55 * len(names) + 3.5))
    axes = fig.subplots(1, 2, sharey=True)

    for a, key, ci_key, base_val, base_ci, title in [
        (axes[0], "PR_AUC", "pr_auc", baseline["PR_AUC"], baseline_ci["pr_auc"],
         "Test PR-AUC (95% CI over races)"),
        (axes[1], "Top3_Hit", "top3", baseline["Top3_Hit"], baseline_ci["top3"],
         "Test Top-3 hit rate (95% CI over races)"),
    ]:
        vals = np.array([float(z["test"][key]) for z in zoo])
        lo = np.array([z["ci"][ci_key][0] for z in zoo])
        hi = np.array([z["ci"][ci_key][1] for z in zoo])
        colors = [PODIUM_COLOR if z["tuned"] else "#1f77b4" for z in zoo]
        a.barh(y, vals, color=colors, alpha=0.85)
        a.errorbar(vals, y, xerr=[(vals - lo).clip(min=0), (hi - vals).clip(min=0)],
                   fmt="none", ecolor="black", capsize=3, lw=1.2)
        a.axvspan(base_ci[0], base_ci[1], color=BASE_COLOR, alpha=0.2)
        a.axvline(base_val, color=BASE_COLOR, ls="--", lw=2,
                  label=f"Baseline = {base_val:.3f}")
        for yi, v in zip(y, vals):
            a.text(v + 0.003, yi + 0.28, f"{v:.3f}", fontsize=8, va="center")
        a.set(title=title, xlabel=key.replace("_", "-"))
        lo_lim = min(lo.min(), base_ci[0]) - 0.03
        a.set_xlim(max(lo_lim, 0), min(max(hi.max(), base_ci[1]) + 0.05, 1.02))
        a.legend(loc="lower right", fontsize=9)
        a.grid(alpha=0.3, axis="x")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names)
    fig.suptitle("All models compared (red = tuned best model)",
                 fontsize=14, fontweight="bold")
    return _save(fig, out_path)


def plot_summary_curves(y_true, probas, base_score, tuned_name, out_path):
    """ROC и PR всех моделей на одном графике."""
    import matplotlib
    y = np.asarray(y_true).astype(int)
    cmap = matplotlib.colormaps["tab20"]
    fig = _new_fig((16, 7))
    a_roc, a_pr = fig.subplots(1, 2)
    for i, (name, p) in enumerate(probas.items()):
        tuned = name == tuned_name
        kw = dict(color=PODIUM_COLOR if tuned else cmap(i % 20),
                  lw=3 if tuned else 1.5, label=name)
        fpr, tpr, _ = roc_curve(y, p)
        a_roc.plot(fpr, tpr, **kw)
        pr, rc, _ = precision_recall_curve(y, p)
        a_pr.plot(rc, pr, **kw)
    fpr, tpr, _ = roc_curve(y, base_score)
    a_roc.plot(fpr, tpr, color="black", ls="--", lw=1.5, label="Baseline (grid)")
    pr, rc, _ = precision_recall_curve(y, base_score)
    a_pr.plot(rc, pr, color="black", ls="--", lw=1.5, label="Baseline (grid)")
    a_roc.plot([0, 1], [0, 1], color="lightgray", lw=1)
    a_pr.axhline(y.mean(), color="lightgray", lw=1)
    a_roc.set(title="ROC curves (test)", xlabel="False Positive Rate",
              ylabel="True Positive Rate")
    a_pr.set(title="Precision-Recall curves (test)", xlabel="Recall", ylabel="Precision")
    a_pr.legend(fontsize=8, loc="upper right")
    for a in (a_roc, a_pr):
        a.grid(alpha=0.3)
    return _save(fig, out_path)


def plot_cv_vs_test(zoo, out_path):
    """PR-AUC: среднее по CV (± std) рядом с тестом — видно переобучение/стабильность."""
    zoo = sorted(zoo, key=lambda z: z["cv"]["PR_AUC"])
    names = [z["Model"] for z in zoo]
    y = np.arange(len(names))
    h = 0.38
    fig = _new_fig((13, 0.55 * len(names) + 3))
    a = fig.subplots()
    cv_m = [z["cv"]["PR_AUC"] for z in zoo]
    cv_s = [z["cv"]["PR_AUC_std"] for z in zoo]
    te = [z["test"]["PR_AUC"] for z in zoo]
    a.barh(y + h / 2, cv_m, h, xerr=cv_s, capsize=3, color="#ff7f0e",
           label="CV (train, mean ± std)")
    a.barh(y - h / 2, te, h, color="#1f77b4", label="Test")
    a.set_yticks(y)
    a.set_yticklabels(names)
    a.set(title="PR-AUC: cross-validation vs test", xlabel="PR-AUC")
    a.legend(loc="lower right")
    a.grid(alpha=0.3, axis="x")
    return _save(fig, out_path)


def plot_feature_importance(feature_importance, model_name, out_path, top=15, title=None):
    items = feature_importance[:top][::-1]
    fig = _new_fig((11, 0.4 * len(items) + 2.5))
    a = fig.subplots()
    a.barh([i["Feature"] for i in items], [i["Importance"] for i in items],
           color="#2ca02c")
    a.set(title=title or f"Permutation importance — {model_name} (drop in test PR-AUC)",
          xlabel="Importance")
    a.grid(alpha=0.3, axis="x")
    return _save(fig, out_path)


def plot_pca(pca_info, out_path):
    """Scree-plot: доля дисперсии по главным компонентам и накопленная доля (Patil et al.)."""
    evr = np.array(pca_info["explained_ratio"])
    cum = np.array(pca_info["cumulative"])
    x = np.arange(1, len(evr) + 1)
    fig = _new_fig((11, 5.5))
    a = fig.subplots()
    a.bar(x, evr, color="#1f77b4", label="explained variance")
    a.set_xlabel("Principal component")
    a.set_ylabel("Explained variance ratio")
    b = a.twinx()
    b.plot(x, cum, color=PODIUM_COLOR, marker="o", lw=2, label="cumulative")
    for thr, key in ((0.70, "n_components_70"), (0.90, "n_components_90")):
        b.axhline(thr, color="gray", ls="--", lw=1)
        b.axvline(pca_info[key], color="gray", ls=":", lw=1)
        b.text(len(x), thr + 0.01, f"{int(thr * 100)}% -> {pca_info[key]} PCs",
               ha="right", fontsize=9)
    b.set_ylim(0, 1.05)
    b.set_ylabel("Cumulative variance")
    a.set_title(f"PCA of {pca_info['n_features']} continuous features (train)")
    a.grid(alpha=0.3)
    return _save(fig, out_path)


def plot_calibration(y_true, probas, tuned_name, zoo, out_path):
    """Reliability diagram всех моделей + Brier score и log-loss (Yathishkumar)."""
    import matplotlib
    y = np.asarray(y_true).astype(int)
    cmap = matplotlib.colormaps["tab20"]
    fig = _new_fig((17, 6.5))
    a, b, c = fig.subplots(1, 3)
    for i, (name, p) in enumerate(probas.items()):
        pts = np.array(adv.calibration_points(y, p))
        hot = name == tuned_name or "isotonic" in name
        a.plot(pts[:, 0], pts[:, 1], marker="o", ms=3,
               lw=3 if name == tuned_name else 1.3,
               ls="-" if not hot or name == tuned_name else "--",
               color=PODIUM_COLOR if name == tuned_name else cmap(i % 20), label=name)
    a.plot([0, 1], [0, 1], color="black", lw=1, ls=":")
    a.set(title="Reliability diagram (test)", xlabel="Mean predicted probability",
          ylabel="Observed podium frequency")
    a.legend(fontsize=7, loc="upper left")
    a.grid(alpha=0.3)
    for ax, key, title in ((b, "Brier", "Brier score (lower is better)"),
                           (c, "LogLoss", "Log-loss (lower is better)")):
        rows = sorted(zoo, key=lambda z: z["test"][key])
        names = [z["Model"] for z in rows]
        vals = [z["test"][key] for z in rows]
        colors = [PODIUM_COLOR if z["tuned"] else "#1f77b4" for z in rows]
        ax.barh(np.arange(len(rows)), vals, color=colors, alpha=0.85)
        ax.set_yticks(np.arange(len(rows)))
        ax.set_yticklabels(names, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlim(min(vals) * 0.9, max(vals) * 1.03)
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="x")
    return _save(fig, out_path)


def plot_mcnemar(names, pmat, out_path):
    """Матрица p-value точного теста Макнемара между моделями (на пороговых предсказаниях)."""
    n = len(names)
    fig = _new_fig((max(8, 0.55 * n + 4), max(7, 0.5 * n + 3)))
    a = fig.subplots()
    im = a.imshow(np.array(pmat), cmap="RdYlGn", vmin=0, vmax=0.2)
    a.set_xticks(range(n))
    a.set_yticks(range(n))
    a.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    a.set_yticklabels(names, fontsize=7)
    for i in range(n):
        for j in range(n):
            if i != j:
                a.text(j, i, f"{pmat[i][j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=a, label="p-value (green = p < 0.05 -> different errors)")
    a.set_title("McNemar exact test between models (test set)")
    return _save(fig, out_path)


def make_plots(out_dir, y_test, probas, thresholds, base_score, baseline, model_zoo,
               ci_all, cv_folds, family_of, feature_importance, tuned_name,
               test_years="", show=False, pca_info=None, shap_items=None,
               mcnemar_matrix=None):
    """Сохраняет все графики в out_dir и возвращает {'models': {...}, 'summary': {...}}."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {"models": {}, "summary": {}}

    for name, proba in probas.items():
        z = next(z for z in model_zoo if z["Model"] == name)
        f1_f, pr_f = cv_folds[name]
        path = os.path.join(out_dir, f"model_{_slug(name)}.png")
        plot_model_report(name, family_of(name), y_test, proba, thresholds[name],
                          base_score, baseline, z["test"], f1_f, pr_f, path,
                          test_years=test_years)
        paths["models"][name] = path

    s = paths["summary"]
    s["metrics_with_ci"] = plot_summary_metrics(
        model_zoo, baseline, ci_all["__baseline__"],
        os.path.join(out_dir, "summary_metrics_ci.png"))
    s["roc_pr_all_models"] = plot_summary_curves(
        y_test, probas, base_score, tuned_name,
        os.path.join(out_dir, "summary_roc_pr_all_models.png"))
    s["cv_vs_test"] = plot_cv_vs_test(
        model_zoo, os.path.join(out_dir, "summary_cv_vs_test.png"))
    s["feature_importance"] = plot_feature_importance(
        feature_importance, tuned_name,
        os.path.join(out_dir, "feature_importance_tuned.png"))
    s["calibration"] = plot_calibration(
        y_test, probas, tuned_name, model_zoo, os.path.join(out_dir, "summary_calibration.png"))
    if pca_info:
        s["pca"] = plot_pca(pca_info, os.path.join(out_dir, "summary_pca.png"))
    if shap_items:
        s["shap"] = plot_feature_importance(
            shap_items, tuned_name, os.path.join(out_dir, "shap_tuned.png"),
            title=f"SHAP — mean |value|, {tuned_name}")
    if mcnemar_matrix:
        s["mcnemar"] = plot_mcnemar(
            mcnemar_matrix["models"], mcnemar_matrix["p"],
            os.path.join(out_dir, "summary_mcnemar.png"))

    if show:  # в Jupyter/Colab картинки выводятся прямо в ячейку
        try:
            from IPython.display import Image, display
            for p in list(paths["summary"].values()) + list(paths["models"].values()):
                display(Image(filename=p))
        except ImportError:
            pass
    return paths


# ============================================================
# WHAT-IF: предсказание для одного пилота (используется приложением)
# ============================================================

def apply_whatif(bundle, features, teammate_grid=None):
    """Пересчитывает производные стартовые признаки по новой позиции на решётке
    и возвращает {модель: вероятность подиума} для всех моделей bundle."""
    row = dict(bundle["defaults"])
    row.update({k: v for k, v in features.items() if k in bundle["features"]})
    grid = float(row["grid_clean"])
    field = float(features.get("field_size") or 20)
    row["grid_clean"] = grid
    row["grid_pct"] = grid / field
    row["front_row"] = int(grid <= 2)
    row["top6_start"] = int(grid <= 6)
    row["top10_start"] = int(grid <= 10)
    row["pit_lane_start"] = int(features.get("pit_lane_start", 0))
    if row.get("qualifying_position") is not None:
        row["grid_vs_quali"] = float(row["qualifying_position"]) - grid
    if teammate_grid is not None:
        row["grid_minus_team_avg"] = (grid - float(teammate_grid)) / 2.0
    row["grid_zone"] = (
        "pit_lane" if row["pit_lane_start"] else
        "pole" if grid == 1 else "front_row" if grid == 2 else "top6" if grid <= 6 else
        "top10" if grid <= 10 else "midfield" if grid <= 15 else "back")
    frame = pd.DataFrame([{f: row.get(f) for f in bundle["features"]}])
    out = {}
    for name, model in bundle["models"].items():
        out[name] = round(float(model.predict_proba(frame)[:, 1][0]), 4)
    out["Baseline (grid <= 3)"] = 1.0 if grid <= 3 else 0.0
    return out, row


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_ml(data_path, save_model_path=None, plots_dir="plots", show_plots=False,
           include=None, quick=False, progress=None):
    """
    plots_dir  - папка для PNG-графиков (None = не строить)
    include    - список имён моделей (None = все); quick - пропустить медленные модели
    progress   - callback(percent, message) для индикатора выполнения в приложении
    """
    def say(pct, msg):
        if progress:
            progress(int(pct), msg)

    # ---------------- LOAD / MERGE / PREPARE ----------------
    say(1, "Loading and merging CSV files")
    results, races, qualifying, drivers, constructors = load_data(data_path)
    merged = merge_data(results, races, qualifying, drivers, constructors)
    say(3, "Feature engineering (driver, team, start, sequences)")
    dataset, features = prepare_dataset(merged)

    # ---------------- TRAIN / TEST (по времени) ----------------
    train_df, test_df = time_split(dataset)

    # ---------------- Hettmann: профили пилотов (KMeans, fit только на train) ----------------
    train_df, test_df, clusters = adv.driver_profile_clusters(train_df, test_df)

    # ---------------- TASK 4: feature selection ----------------
    continuous, dropped, target_corr = select_features(
        train_df, train_df["podium"], CONTINUOUS_FEATURES
    )
    binary, categorical = BINARY_FEATURES, CATEGORICAL_FEATURES + ["driver_profile"]
    used_features = continuous + binary + categorical
    model_cols = list(dict.fromkeys(used_features + CONTEXT_COLUMNS + SEQ_COLUMNS))

    X_train, y_train = train_df[model_cols], train_df["podium"]
    X_test, y_test = test_df[model_cols], test_df["podium"]
    groups_train = train_df["raceId"]

    say(5, "PCA analysis")
    pca_info = adv.pca_analysis(X_train, continuous, y_train)

    all_models = build_models(continuous, binary, categorical)
    models = {n: m for n, m in all_models.items()
              if (include is None or n in include) and not (quick and n in SLOW_MODELS)}
    if not any(n in PARAM_GRIDS for n in models):
        raise ValueError("Select at least one model that supports hyperparameter tuning "
                         "(e.g. Logistic Regression, Random Forest, Gradient Boosting).")
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # ---------------- BASELINE ----------------
    baseline_pred = (test_df["grid_clean"] <= 3).astype(int)
    baseline = {
        "Model": "Baseline (grid <= 3)",
        "Accuracy": round(accuracy_score(y_test, baseline_pred), 4),
        "Precision": round(precision_score(y_test, baseline_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, baseline_pred, zero_division=0), 4),
        "F1": round(f1_score(y_test, baseline_pred, zero_division=0), 4),
        "ROC_AUC": round(roc_auc_score(y_test, -test_df["grid_clean"]), 4),
        "PR_AUC": round(average_precision_score(y_test, -test_df["grid_clean"]), 4),
        "Top3_Hit": top3_hit_rate(
            test_df.assign(_neg=-test_df["grid_clean"]), "_neg"),
    }

    # ---------------- TASK 5 + TASK 6: TRAIN + 5-FOLD CV ----------------
    model_results, cv_results, trained, thresholds = [], [], {}, {}
    test_probas, fit_times, skipped = {}, {}, []

    for i, (name, model) in enumerate(models.items()):
        say(6 + 60 * i / len(models), f"Training {name} ({i + 1}/{len(models)})")
        t0 = time.time()
        try:
            oof, threshold, f1s, aucs, aps = cross_validate_model(
                model, X_train, y_train, groups_train, cv
            )
            fitted = clone(model).fit(X_train, y_train)
            proba = fitted.predict_proba(X_test)[:, 1]
        except Exception as exc:  # одна упавшая модель не должна ронять весь пайплайн
            skipped.append({"model": name, "reason": f"{type(exc).__name__}: {exc}"[:300]})
            continue
        thresholds[name] = threshold
        fit_times[name] = round(time.time() - t0, 1)
        trained[name] = fitted
        test_probas[name] = proba
        model_results.append(compute_metrics(y_test, proba, threshold, name, test_df))

        cv_results.append({
            "Model": name,
            "Family": MODEL_FAMILY.get(name, "Other"),
            "Mean F1": round(f1s.mean(), 4),
            "Std F1": round(f1s.std(), 4),
            "Mean ROC_AUC": round(aucs.mean(), 4),
            "Std ROC_AUC": round(aucs.std(), 4),
            "Mean PR_AUC": round(aps.mean(), 4),
            "Std PR_AUC": round(aps.std(), 4),
            "CV_Top3_Hit": top3_hit_rate(train_df.assign(_p=oof), "_p"),
            "Threshold": round(threshold, 2),
            "folds": [round(float(s), 4) for s in f1s],
            "pr_folds": [round(float(s), 4) for s in aps],
        })

    cv_df = pd.DataFrame(cv_results)
    overall_best_cv = cv_df.loc[cv_df["Mean PR_AUC"].idxmax(), "Model"]
    tunable = cv_df[cv_df["Model"].isin(PARAM_GRIDS)]
    best_model_name = tunable.loc[tunable["Mean PR_AUC"].idxmax(), "Model"]
    before_tuning = next(r for r in model_results if r["Model"] == best_model_name)

    # ---------------- TASK 7: HYPERPARAMETER TUNING ----------------
    say(68, f"GridSearchCV for {best_model_name}")
    grid = GridSearchCV(
        models[best_model_name],
        PARAM_GRIDS[best_model_name],
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train, y_train, groups=groups_train)
    tuned_model = grid.best_estimator_

    _, tuned_threshold, t_f1s, t_aucs, t_aps = cross_validate_model(
        tuned_model, X_train, y_train, groups_train, cv
    )
    tuned_proba = tuned_model.predict_proba(X_test)[:, 1]
    tuned_name = f"{best_model_name} Tuned"
    tuned_metrics = compute_metrics(
        y_test, tuned_proba, tuned_threshold, tuned_name, test_df)

    tuned_pred = (tuned_proba >= tuned_threshold).astype(int)
    cm = confusion_matrix(y_test, tuned_pred)

    # ---------------- Yathishkumar: изотоническая калибровка настроенной модели ----------------
    say(74, "Isotonic calibration of the tuned model")
    cal_name = f"{best_model_name} Tuned (isotonic)"
    cal_model = adv.IsotonicCalibratedClassifier(base=tuned_model, n_splits=3)
    cal_oof, cal_thr, c_f1s, c_aucs, c_aps = cross_validate_model(
        cal_model, X_train, y_train, groups_train, cv)
    cal_model.fit(X_train, y_train)
    cal_proba = cal_model.predict_proba(X_test)[:, 1]
    cal_metrics = compute_metrics(y_test, cal_proba, cal_thr, cal_name, test_df)

    # ---------------- RACE-LEVEL EVALUATION ----------------
    test_eval = test_df.copy()
    test_eval["proba"] = tuned_proba
    test_eval["pred"] = tuned_pred
    race_level = {
        "model_top3_hit_rate": top3_hit_rate(test_eval, "proba"),
        "baseline_top3_hit_rate": top3_hit_rate(
            test_eval.assign(neg_grid=-test_eval["grid_clean"]), "neg_grid"
        ),
    }

    # ---------------- MODEL COMPARISON: bootstrap + McNemar ----------------
    say(78, "Bootstrap confidence intervals and McNemar tests")
    scores = {"__baseline__": -test_df["grid_clean"].to_numpy()}
    scores.update(test_probas)
    scores[tuned_name] = tuned_proba
    scores[cal_name] = cal_proba
    ci_all, n_test_races = bootstrap_ci(test_df, scores)

    all_probas = dict(test_probas)
    all_probas[tuned_name] = tuned_proba
    all_probas[cal_name] = cal_proba
    all_thr = {**thresholds, tuned_name: tuned_threshold, cal_name: cal_thr}
    yt = y_test.to_numpy()
    preds = {n: (p >= all_thr[n]).astype(int) for n, p in all_probas.items()}
    base_pred_arr = baseline_pred.to_numpy()
    mc_vs_base = {n: adv.mcnemar_exact(yt, p, base_pred_arr) for n, p in preds.items()}
    names_mc = list(preds)
    pmat = [[1.0 if a == b else adv.mcnemar_exact(yt, preds[a], preds[b])["p"]
             for b in names_mc] for a in names_mc]
    mcnemar_matrix = {"models": names_mc, "p": pmat}

    model_zoo = []
    for r in cv_results:
        test_m = next(m for m in model_results if m["Model"] == r["Model"])
        model_zoo.append({
            "Model": r["Model"],
            "Family": r["Family"],
            "tuned": False,
            "calibrated": False,
            "cv": {
                "F1": r["Mean F1"], "F1_std": r["Std F1"],
                "ROC_AUC": r["Mean ROC_AUC"], "ROC_AUC_std": r["Std ROC_AUC"],
                "PR_AUC": r["Mean PR_AUC"], "PR_AUC_std": r["Std PR_AUC"],
                "Top3_Hit": r["CV_Top3_Hit"],
            },
            "test": test_m,
            "ci": ci_all[r["Model"]],
            "mcnemar_vs_baseline": mc_vs_base[r["Model"]],
            "fit_seconds": fit_times[r["Model"]],
        })
    for nm, mt, fl, au, ap, is_tuned in (
            (tuned_name, tuned_metrics, t_f1s, t_aucs, t_aps, True),
            (cal_name, cal_metrics, c_f1s, c_aucs, c_aps, False)):
        model_zoo.append({
            "Model": nm,
            "Family": family_of(nm),
            "tuned": is_tuned,
            "calibrated": not is_tuned,
            "cv": {
                "F1": round(fl.mean(), 4), "F1_std": round(fl.std(), 4),
                "ROC_AUC": round(au.mean(), 4), "ROC_AUC_std": round(au.std(), 4),
                "PR_AUC": round(ap.mean(), 4), "PR_AUC_std": round(ap.std(), 4),
                "Top3_Hit": None,
            },
            "test": mt,
            "ci": ci_all[nm],
            "mcnemar_vs_baseline": mc_vs_base[nm],
            "fit_seconds": None,
        })
    model_zoo.sort(key=lambda z: z["cv"]["PR_AUC"], reverse=True)
    for i, z in enumerate(model_zoo, 1):
        z["rank"] = i

    comparison = {
        "n_models": len(trained),
        "n_test_races": n_test_races,
        "n_bootstrap": 300,
        "overall_best_cv": overall_best_cv,
        "ranking_metric": "5-fold grouped CV PR-AUC",
        "baseline_ci": ci_all["__baseline__"],
        "optional_libraries": {
            "XGBoost": XGBClassifier is not None,
            "LightGBM": LGBMClassifier is not None,
            "PyTorch": adv.TORCH_OK,
            "SHAP": adv.shap is not None,
        },
    }

    # ---------------- FEATURE IMPORTANCE (permutation, на тесте) + SHAP ----------------
    say(82, "Permutation importance and SHAP")
    feature_importance = adv.permutation_importance_cols(
        tuned_model, X_test, y_test, used_features, n_repeats=5)
    shap_items = None
    if best_model_name not in CONTEXT_MODELS:
        shap_items = adv.shap_importance(
            tuned_model, used_features, X_train[used_features], X_test[used_features])

    # ---------------- Multilevel: дисперсия по уровням (van Kesteren) ----------------
    multilevel = None
    ml_model = trained.get("Multilevel Rank-Logit")
    if ml_model is not None:
        est = ml_model.named_steps["model"]
        multilevel = {"variance_share": est.variance_share_, **est.top_effects()}

    # ---------------- ПРИМЕРЫ ПРЕДСКАЗАНИЙ (последние 3 гонки теста) ----------------
    sample_predictions = []
    last_races = (
        test_eval[["raceId", "race_order", "year", "round", "race_name"]]
        .drop_duplicates("raceId").nlargest(3, "race_order")
    )
    for _, r in last_races.iterrows():
        race = test_eval[test_eval["raceId"] == r["raceId"]].nlargest(5, "proba")
        sample_predictions.append({
            "race": f"{int(r['year'])} {r['race_name']}",
            "top_predicted": [
                {
                    "driver": row["driverRef"],
                    "team": row["constructorRef"],
                    "grid": None if pd.isna(row["grid"]) else int(row["grid"]),
                    "podium_probability": round(float(row["proba"]), 3),
                    "actual_position": None if pd.isna(row["position"])
                    else int(row["position"]),
                }
                for _, row in race.iterrows()
            ],
        })

    # ---------------- RACE EXPLORER: все гонки теста, вероятности ВСЕХ моделей ----------------
    say(86, "Building race explorer data")
    explorer = []
    order = test_df.drop_duplicates("raceId").sort_values("race_order", ascending=False)
    pos_of = {i: k for k, i in enumerate(test_df.index)}
    for _, r in order.iterrows():
        part = test_df[test_df["raceId"] == r["raceId"]]
        drivers_out = []
        for idx, row in part.iterrows():
            k = pos_of[idx]
            gmt = row["grid_minus_team_avg"] if "grid_minus_team_avg" in row else 0
            drivers_out.append({
                "driver": row["driverRef"],
                "team": row["constructorRef"],
                "grid": None if pd.isna(row["grid"]) else int(row["grid"]),
                "actual": None if pd.isna(row["position"]) else int(row["position"]),
                "teammate_grid": None if pd.isna(gmt) else round(
                    float(row["grid_clean"] - 2 * gmt), 1),
                "field_size": int(len(part)),
                "probs": {n: round(float(p[k]), 4) for n, p in all_probas.items()},
                "features": {f: (None if pd.isna(row[f]) else
                                 (row[f] if isinstance(row[f], str) else round(float(row[f]), 4)))
                             for f in used_features},
            })
        explorer.append({
            "race_id": int(r["raceId"]),
            "label": f"{int(r['year'])} R{int(r['round'])} · {r['race_name']}",
            "drivers": sorted(drivers_out, key=lambda d: d["grid"] if d["grid"] else 99),
        })

    # ---------------- WHAT-IF bundle (для /api/predict) ----------------
    bundle = None
    if save_model_path:
        import joblib
        whatif_models = {n: m for n, m in trained.items() if n not in CONTEXT_MODELS}
        if best_model_name not in CONTEXT_MODELS:
            whatif_models[tuned_name] = tuned_model
        defaults = {}
        ranges = {}
        for f in continuous:
            defaults[f] = float(X_train[f].median())
            ranges[f] = [round(float(X_train[f].quantile(0.01)), 3),
                         round(float(X_train[f].quantile(0.99)), 3)]
        for f in binary:
            defaults[f] = 0
        for f in categorical:
            defaults[f] = str(X_train[f].mode().iloc[0])
        bundle = {"models": whatif_models, "features": used_features, "defaults": defaults,
                  "ranges": ranges, "thresholds": {n: all_thr[n] for n in whatif_models
                                                   if n in all_thr},
                  "tuned_name": tuned_name}
        os.makedirs(os.path.dirname(os.path.abspath(save_model_path)), exist_ok=True)
        joblib.dump(bundle, save_model_path)

    # ---------------- FINAL TABLE ----------------
    final_results = model_results + [tuned_metrics, cal_metrics]

    # ---------------- ГРАФИКИ ----------------
    plot_paths = None
    if plots_dir:
        say(90, "Drawing charts")
        thr_by_model = {m["Model"]: m["Threshold"] for m in final_results}
        cv_folds = {r["Model"]: (r["folds"], r["pr_folds"]) for r in cv_results}
        cv_folds[tuned_name] = (t_f1s, t_aps)
        cv_folds[cal_name] = (c_f1s, c_aps)
        plot_paths = make_plots(
            plots_dir, y_test, all_probas, thr_by_model,
            -test_df["grid_clean"].to_numpy(), baseline, model_zoo, ci_all, cv_folds,
            family_of, feature_importance, tuned_name,
            test_years=f"{int(test_df['year'].min())}-{int(test_df['year'].max())}",
            show=show_plots, pca_info=pca_info, shap_items=shap_items,
            mcnemar_matrix=mcnemar_matrix,
        )

    calibration = {n: adv.calibration_points(y_test, p) for n, p in all_probas.items()}
    say(98, "Packaging results")

    return _jsonable({
        "dataset": {
            "rows": len(dataset),
            "columns": len(used_features) + 1,
            "train": len(X_train),
            "test": len(X_test),
            "train_years": f"{int(train_df['year'].min())}-{int(train_df['year'].max())}",
            "test_years": f"{int(test_df['year'].min())}-{int(test_df['year'].max())}",
            "split": "chronological (train = earlier races, test = latest races)",
        },
        "target": {
            "podium": int(dataset["podium"].sum()),
            "non_podium": int((dataset["podium"] == 0).sum()),
        },
        "feature_selection": {
            "used_features": used_features,
            "dropped_for_collinearity": dropped,
            "target_correlation": {
                k: round(float(v), 4) for k, v in target_corr.head(10).items()
            },
        },
        "pca": pca_info,
        "driver_profiles": clusters,
        "baseline": baseline,
        "models": final_results,
        "cross_validation": cv_results,
        "best_model": best_model_name,
        "tuning": {
            "best_parameters": grid.best_params_,
            "best_cv_pr_auc": round(grid.best_score_, 4),
            "cv_mean_f1": round(t_f1s.mean(), 4),
            "cv_std_f1": round(t_f1s.std(), 4),
            "before_tuning": before_tuning,
            "test_metrics": tuned_metrics,
        },
        "calibration": {"tuned": tuned_name, "calibrated": cal_name, "curves": calibration},
        "confusion_matrix": cm,
        "race_level": race_level,
        "feature_importance": feature_importance[:20],
        "shap_importance": shap_items[:20] if shap_items else None,
        "multilevel": multilevel,
        "mcnemar": {"vs_baseline": mc_vs_base, "matrix": mcnemar_matrix},
        "sample_predictions": sample_predictions,
        "explorer": explorer,
        "model_zoo": model_zoo,
        "comparison": comparison,
        "skipped_models": skipped,
        "run_config": {"quick": bool(quick), "models": list(trained)},
        "has_whatif": bundle is not None,
        "plots": plot_paths,
    })


if __name__ == "__main__":
    import json
    import sys as _sys

    path = _sys.argv[1] if len(_sys.argv) > 1 else "."
    out = _sys.argv[2] if len(_sys.argv) > 2 else "plots"
    res = run_ml(path, plots_dir=out, quick="--quick" in _sys.argv,
                 progress=lambda p, m: print(f"[{p:3d}%] {m}", file=_sys.stderr))
    res.pop("explorer", None)
    print(json.dumps(res, indent=2, ensure_ascii=False))