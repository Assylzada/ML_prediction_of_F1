"""
Методы из литературного обзора, которых не было в базовом пайплайне.

  Patil et al. (2023)              -> pca_analysis(), PCA + Logistic Regression
  Fatima & Johrendt (2023)         -> EDNN-style: две связанные нейросети (build_ednn)
  Surendran (2023), Mistry (2025)  -> LSTM / GRU по последним гонкам пилота (TorchSeqClassifier)
  Sasikumar et al. (2025)          -> Bi-LSTM, TCN, CNN-BiLSTM, GRU + McNemar + bootstrap PR-AUC
  van Kesteren & Bergkamp (2023)   -> MultilevelRankLogit (иерархический rank-ordered logit, MAP)
  NCI thesis (2021), ranking       -> PairwiseRankClassifier (learning-to-rank, RankSVM-стиль)
  Hettmann (2024)                  -> driver_profile_clusters() (KMeans по профилю пилота)
  Yathishkumar (2025)              -> class_weight, IsotonicCalibratedClassifier, Brier/log-loss, SHAP

PyTorch и SHAP - необязательные библиотеки: если не установлены, соответствующие
модели/графики просто пропускаются (как XGBoost/LightGBM в pipeline.py).
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.common import (
    RANDOM_STATE, SEQ_COLUMNS, SEQ_FEATURES, SEQ_LEN, Winsorizer, create_preprocessor,
)

try:
    import torch
    from torch import nn
    TORCH_OK = True
except Exception:  # PyTorch не установлен
    torch, nn, TORCH_OK = None, None, False

try:
    import shap
except Exception:
    shap = None


def _dense(a):
    return a.toarray() if hasattr(a, "toarray") else np.asarray(a)


def _race_layout(race_ids):
    """Раскладывает строки по гонкам: (уникальные гонки, номер гонки у строки, слот внутри гонки, маска R x N)."""
    uniq, inv = np.unique(np.asarray(race_ids), return_inverse=True)
    counts = np.bincount(inv)
    order = np.argsort(inv, kind="stable")
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    slot = np.empty(len(inv), dtype=int)
    slot[order] = np.arange(len(inv)) - np.repeat(starts, counts)
    mask = np.zeros((len(uniq), counts.max()), dtype=bool)
    mask[inv, slot] = True
    return uniq, inv, slot, mask


# ============================================================
# 1) van Kesteren & Bergkamp: multilevel rank-ordered logit
# ============================================================

class MultilevelRankLogit(BaseEstimator, ClassifierMixin):
    """
    Иерархическая модель Plackett-Luce (rank-ordered logit) для топ-3 гонки:

        score_i = x_i * beta + u[пилот_i] + v[команда_i]
        u ~ N(0, sigma_driver^2),  v ~ N(0, sigma_team^2)     (random effects)

    Максимизируется апостериорная плотность (MAP) - то есть это байесовская модель
    с гауссовыми априорами, но без полного MCMC. Правдоподобие гонки: вероятность,
    что пилоты P1, P2, P3 выбираются последовательно (softmax среди оставшихся).
    Позиция (колонка `position`) используется ТОЛЬКО как метка при обучении.
    Вероятность подиума на предсказании - по симуляции Plackett-Luce (трюк Гумбеля).
    """

    def __init__(self, continuous=(), binary=(), categorical=(), sigma_driver=1.0,
                 sigma_team=1.0, l2=1.0, n_sim=300, top_k=3, random_state=RANDOM_STATE):
        self.continuous = continuous
        self.binary = binary
        self.categorical = categorical
        self.sigma_driver = sigma_driver
        self.sigma_team = sigma_team
        self.l2 = l2
        self.n_sim = n_sim
        self.top_k = top_k
        self.random_state = random_state

    def _design(self, X, fit=False, y=None):
        if fit:
            self.pre_ = create_preprocessor(list(self.continuous), list(self.binary),
                                            list(self.categorical))
            Z = self.pre_.fit_transform(X, y)
        else:
            Z = self.pre_.transform(X)
        return _dense(Z)

    def fit(self, X, y=None):
        Z = self._design(X, fit=True, y=y)
        n, p = Z.shape
        drv = X["driverRef"].astype(str).to_numpy()
        con = X["constructorRef"].astype(str).to_numpy()
        pos = pd.to_numeric(X["position"], errors="coerce").to_numpy()
        self.drv_index_ = {d: i for i, d in enumerate(pd.unique(drv))}
        self.con_index_ = {c: i for i, c in enumerate(pd.unique(con))}
        di = np.array([self.drv_index_[d] for d in drv])
        ci = np.array([self.con_index_[c] for c in con])
        nd, nc = len(self.drv_index_), len(self.con_index_)

        _, inv, slot, mask = _race_layout(X["raceId"].to_numpy())
        R, N = mask.shape
        K = self.top_k
        winners = -np.ones((R, K), dtype=int)       # слот пилота, занявшего место k+1
        for k in range(1, K + 1):
            sel = np.where(pos == k)[0]
            winners[inv[sel], k - 1] = slot[sel]
        rows = np.arange(R)
        sd2, sc2 = self.sigma_driver ** 2, self.sigma_team ** 2

        def objective(theta):
            b, u, v = theta[:p], theta[p:p + nd], theta[p + nd:]
            s = Z @ b + u[di] + v[ci]
            S = np.full((R, N), -1e30)
            S[inv, slot] = s
            alive = mask.copy()
            ll = 0.0
            gS = np.zeros((R, N))
            for k in range(K):
                ok = winners[:, k] >= 0
                if not ok.any():
                    break
                Sm = np.where(alive, S, -1e30)
                lse = logsumexp(Sm, axis=1)
                prob = np.exp(Sm - lse[:, None])
                r = rows[ok]
                w = winners[ok, k]
                ll += (S[r, w] - lse[ok]).sum()
                gS[ok] -= prob[ok]
                gS[r, w] += 1.0
                alive[r, w] = False
            gs = gS[inv, slot]                       # d loglik / d s_i
            nll = -ll + 0.5 * self.l2 * b @ b + 0.5 * u @ u / sd2 + 0.5 * v @ v / sc2
            grad = np.concatenate([
                -(Z.T @ gs) + self.l2 * b,
                -np.bincount(di, weights=gs, minlength=nd) + u / sd2,
                -np.bincount(ci, weights=gs, minlength=nc) + v / sc2,
            ])
            return nll, grad

        theta0 = np.zeros(p + nd + nc)
        res = minimize(objective, theta0, jac=True, method="L-BFGS-B",
                       options={"maxiter": 300})
        self.beta_, self.u_, self.v_ = res.x[:p], res.x[p:p + nd], res.x[p + nd:]
        self.classes_ = np.array([0, 1])

        # доля дисперсии "силы" по уровням иерархии (аналог вывода "88% дисперсии - команда")
        parts = np.array([np.var(Z @ self.beta_), np.var(self.u_[di]), np.var(self.v_[ci])])
        parts = parts / max(parts.sum(), 1e-12)
        self.variance_share_ = {"features": float(parts[0]), "driver": float(parts[1]),
                                "constructor": float(parts[2])}
        return self

    def _scores(self, X):
        Z = self._design(X)
        drv = X["driverRef"].astype(str).to_numpy()
        con = X["constructorRef"].astype(str).to_numpy()
        u = np.array([self.u_[self.drv_index_[d]] if d in self.drv_index_ else 0.0 for d in drv])
        v = np.array([self.v_[self.con_index_[c]] if c in self.con_index_ else 0.0 for c in con])
        return Z @ self.beta_ + u + v

    def predict_proba(self, X):
        s = self._scores(X)
        _, inv, slot, mask = _race_layout(X["raceId"].to_numpy())
        R, N = mask.shape
        S = np.full((R, N), -np.inf)
        S[inv, slot] = s
        rng = np.random.default_rng(self.random_state)
        top = np.zeros((R, N))
        chunk = max(1, 2_000_000 // (N * self.n_sim))
        for a in range(0, R, chunk):
            Sc = S[a:a + chunk]
            G = rng.gumbel(size=(Sc.shape[0], N, self.n_sim))
            T = np.where(np.isfinite(Sc)[:, :, None], Sc[:, :, None] + G, -np.inf)
            best = np.argsort(-T, axis=1)[:, :self.top_k, :]       # (r, K, sim)
            for k in range(self.top_k):
                np.add.at(top[a:a + chunk],
                          (np.arange(Sc.shape[0])[:, None], best[:, k, :]), 1)
        p = np.clip(top[inv, slot] / self.n_sim, 1e-4, 1 - 1e-4)
        return np.c_[1 - p, p]

    def top_effects(self, n=8):
        inv_c = {i: c for c, i in self.con_index_.items()}
        inv_d = {i: d for d, i in self.drv_index_.items()}
        co = sorted(((inv_c[i], float(self.v_[i])) for i in inv_c), key=lambda x: -x[1])[:n]
        do = sorted(((inv_d[i], float(self.u_[i])) for i in inv_d), key=lambda x: -x[1])[:n]
        return {"constructors": co, "drivers": do}


# ============================================================
# 2) Learning to rank (pairwise, RankSVM-стиль)
# ============================================================

class PairwiseRankClassifier(BaseEstimator, ClassifierMixin):
    """
    Learning-to-rank: внутри гонки берутся пары (подиумник, не-подиумник),
    логистическая регрессия учится по разности признаков отличать "лучший" от "худшего".
    Оценка = w * x; вероятность подиума = softmax внутри гонки * 3 (ожидаемо 3 подиумника).
    """

    def __init__(self, continuous=(), binary=(), categorical=(), C=0.3, max_pairs=60,
                 random_state=RANDOM_STATE):
        self.continuous = continuous
        self.binary = binary
        self.categorical = categorical
        self.C = C
        self.max_pairs = max_pairs
        self.random_state = random_state

    def fit(self, X, y):
        self.pre_ = create_preprocessor(list(self.continuous), list(self.binary),
                                        list(self.categorical))
        Z = _dense(self.pre_.fit_transform(X, y))
        y = np.asarray(y)
        rng = np.random.default_rng(self.random_state)
        races = X["raceId"].to_numpy()
        diffs, labels = [], []
        for r in np.unique(races):
            idx = np.where(races == r)[0]
            pos_i, neg_i = idx[y[idx] == 1], idx[y[idx] == 0]
            if len(pos_i) == 0 or len(neg_i) == 0:
                continue
            pairs = [(a, b) for a in pos_i for b in neg_i]
            if len(pairs) > self.max_pairs:
                pairs = [pairs[i] for i in rng.choice(len(pairs), self.max_pairs, replace=False)]
            for a, b in pairs:
                if rng.random() < 0.5:
                    diffs.append(Z[a] - Z[b]); labels.append(1)
                else:
                    diffs.append(Z[b] - Z[a]); labels.append(0)
        self.lr_ = LogisticRegression(C=self.C, fit_intercept=False, max_iter=2000)
        self.lr_.fit(np.array(diffs), np.array(labels))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        Z = _dense(self.pre_.transform(X))
        s = Z @ self.lr_.coef_.ravel()
        _, inv, slot, mask = _race_layout(X["raceId"].to_numpy())
        S = np.full(mask.shape, -1e30)
        S[inv, slot] = s
        P = np.exp(S - logsumexp(S, axis=1, keepdims=True)) * 3.0
        p = np.clip(P[inv, slot], 1e-4, 1 - 1e-4)
        return np.c_[1 - p, p]


# ============================================================
# 3) Recurrent / convolutional sequence models (PyTorch, optional)
# ============================================================

SEQ_ARCHS = {
    "lstm": "LSTM",
    "gru": "GRU",
    "bilstm": "Bi-LSTM",
    "tcn": "TCN",
    "cnn_bilstm": "CNN-BiLSTM",
}

if TORCH_OK:
    class _Encoder(nn.Module):
        def __init__(self, arch, in_dim, h):
            super().__init__()
            self.arch = arch
            if arch == "lstm":
                self.rnn, self.out_dim = nn.LSTM(in_dim, h, batch_first=True), h
            elif arch == "gru":
                self.rnn, self.out_dim = nn.GRU(in_dim, h, batch_first=True), h
            elif arch == "bilstm":
                self.rnn = nn.LSTM(in_dim, h, batch_first=True, bidirectional=True)
                self.out_dim = 2 * h
            elif arch == "tcn":
                layers, c = [], in_dim
                for d in (1, 2, 4):                  # причинные dilated-свёртки
                    layers += [nn.ConstantPad1d((d, 0), 0.0),
                               nn.Conv1d(c, h, kernel_size=2, dilation=d), nn.ReLU()]
                    c = h
                self.conv, self.out_dim = nn.Sequential(*layers), h
            elif arch == "cnn_bilstm":
                self.conv = nn.Sequential(nn.Conv1d(in_dim, h, 3, padding=1), nn.ReLU())
                self.rnn = nn.LSTM(h, h, batch_first=True, bidirectional=True)
                self.out_dim = 2 * h
            else:
                raise ValueError(arch)

        def forward(self, x):                        # x: (batch, time, features), время: старое -> новое
            a = self.arch
            if a in ("lstm", "gru", "bilstm"):
                out, hid = self.rnn(x)
                if a == "bilstm":
                    return torch.cat([hid[0][-2], hid[0][-1]], dim=1)
                return hid[0][-1] if a == "lstm" else hid[-1]
            if a == "tcn":
                return self.conv(x.transpose(1, 2))[:, :, -1]
            out, hid = self.rnn(self.conv(x.transpose(1, 2)).transpose(1, 2))
            return torch.cat([hid[0][-2], hid[0][-1]], dim=1)

    class _SeqNet(nn.Module):
        def __init__(self, arch, seq_dim, static_dim, h):
            super().__init__()
            self.enc = _Encoder(arch, seq_dim, h)
            self.static = nn.Sequential(nn.Linear(static_dim, h), nn.ReLU())
            self.head = nn.Sequential(nn.Linear(self.enc.out_dim + h, 32), nn.ReLU(),
                                      nn.Dropout(0.2), nn.Linear(32, 1))

        def forward(self, seq, static):
            return self.head(torch.cat([self.enc(seq), self.static(static)], dim=1)).squeeze(1)


class TorchSeqClassifier(BaseEstimator, ClassifierMixin):
    """Последовательность последних SEQ_LEN гонок пилота + статические признаки гонки -> P(подиум)."""

    def __init__(self, arch="gru", continuous=(), binary=(), categorical=(), hidden=32,
                 epochs=40, lr=3e-3, batch_size=256, patience=6, random_state=RANDOM_STATE):
        self.arch = arch
        self.continuous = continuous
        self.binary = binary
        self.categorical = categorical
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state

    def _seq(self, X, fit=False):
        a = X[SEQ_COLUMNS].to_numpy(dtype=float).reshape(len(X), SEQ_LEN, len(SEQ_FEATURES))
        a = a[:, ::-1, :]                                       # старое -> новое
        valid = ~np.isnan(a).any(axis=2, keepdims=True)         # есть ли реальная гонка в этом шаге
        if fit:
            self.mu_ = np.nanmean(a.reshape(-1, a.shape[2]), axis=0)
            self.sd_ = np.nanstd(a.reshape(-1, a.shape[2]), axis=0) + 1e-6
        a = np.nan_to_num((a - self.mu_) / self.sd_)
        return np.concatenate([a, valid.astype(float)], axis=2).astype("float32")

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        self.pre_ = create_preprocessor(list(self.continuous), list(self.binary),
                                        list(self.categorical))
        S = _dense(self.pre_.fit_transform(X, y)).astype("float32")
        Q = self._seq(X, fit=True)
        yy = np.asarray(y, dtype="float32")
        races = X["raceId"].to_numpy()
        rng = np.random.default_rng(self.random_state)
        ur = np.unique(races)
        val_races = set(rng.choice(ur, max(1, int(0.15 * len(ur))), replace=False))
        va = np.array([r in val_races for r in races])
        tr = ~va

        self.net_ = _SeqNet(self.arch, Q.shape[2], S.shape[1], self.hidden)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=1e-4)
        loss_fn = nn.BCEWithLogitsLoss()
        t = lambda a: torch.from_numpy(a)
        Qt, St, yt = t(Q), t(S), t(yy)
        tr_idx = np.where(tr)[0]
        best, best_state, bad = np.inf, None, 0
        for _ in range(self.epochs):
            self.net_.train()
            rng.shuffle(tr_idx)
            for i in range(0, len(tr_idx), self.batch_size):
                b = tr_idx[i:i + self.batch_size]
                opt.zero_grad()
                loss_fn(self.net_(Qt[b], St[b]), yt[b]).backward()
                opt.step()
            self.net_.eval()
            with torch.no_grad():
                vl = loss_fn(self.net_(Qt[va], St[va]), yt[va]).item()
            if vl < best - 1e-4:
                best, bad = vl, 0
                best_state = {k: v.clone() for k, v in self.net_.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.net_.load_state_dict(best_state)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        S = _dense(self.pre_.transform(X)).astype("float32")
        Q = self._seq(X)
        self.net_.eval()
        with torch.no_grad():
            p = torch.sigmoid(self.net_(torch.from_numpy(Q), torch.from_numpy(S))).numpy()
        return np.c_[1 - p, p]


# ============================================================
# 4) EDNN-style (Fatima & Johrendt): две связанные нейросети
# ============================================================

START_CONTINUOUS = {"grid_clean", "grid_pct", "qualifying_position", "grid_vs_quali",
                    "grid_minus_team_avg"}
START_BINARY = {"pit_lane_start", "front_row", "top6_start", "top10_start", "quali_missing"}
START_CATEGORICAL = {"grid_zone"}


def _subset_pre(cont, binr, cat):
    parts = []
    if cont:
        parts.append(("cont", Pipeline([("imp", SimpleImputer(strategy="median")),
                                        ("win", Winsorizer(0.01, 0.99)),
                                        ("sc", StandardScaler())]), cont))
    if binr:
        parts.append(("bin", SimpleImputer(strategy="constant", fill_value=0), binr))
    if cat:
        parts.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                       ("enc", OneHotEncoder(handle_unknown="ignore"))]), cat))
    return ColumnTransformer(parts)


def build_ednn(continuous, binary, categorical):
    """Сеть 1 смотрит на старт/квалификацию, сеть 2 - на форму пилота и команды;
    их выходы подаются во вторую (финальную) нейросеть."""
    c_start = [c for c in continuous if c in START_CONTINUOUS]
    c_form = [c for c in continuous if c not in START_CONTINUOUS]
    b_start = [c for c in binary if c in START_BINARY]
    k_start = [c for c in categorical if c in START_CATEGORICAL]
    k_form = [c for c in categorical if c not in START_CATEGORICAL]
    mlp = lambda layers, seed: MLPClassifier(
        hidden_layer_sizes=layers, alpha=1e-2, max_iter=400, early_stopping=True,
        random_state=seed)
    return StackingClassifier(
        estimators=[
            ("net_start", Pipeline([("pre", _subset_pre(c_start, b_start, k_start)),
                                    ("mlp", mlp((16, 8), 1))])),
            ("net_form", Pipeline([("pre", _subset_pre(c_form, [], k_form)),
                                   ("mlp", mlp((32, 16), 2))])),
        ],
        final_estimator=MLPClassifier(hidden_layer_sizes=(8,), alpha=1e-2, max_iter=400,
                                      early_stopping=True, random_state=3),
        cv=3,
    )


# ============================================================
# 5) Isotonic calibration (Yathishkumar), Brier / log-loss
# ============================================================

class IsotonicCalibratedClassifier(BaseEstimator, ClassifierMixin):
    """Калибровка вероятностей изотонической регрессией; OOF-предсказания считаются
    группами по гонкам (строки одной гонки не попадают в train и calibration одновременно)."""

    def __init__(self, base=None, n_splits=3):
        self.base = base
        self.n_splits = n_splits

    def fit(self, X, y):
        y = np.asarray(y)
        groups = X["raceId"].to_numpy()
        oof = np.zeros(len(y))
        for tr, va in GroupKFold(self.n_splits).split(X, y, groups):
            m = clone(self.base).fit(X.iloc[tr], y[tr])
            oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        self.iso_ = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(oof, y)
        self.base_ = clone(self.base).fit(X, y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = np.clip(self.iso_.predict(self.base_.predict_proba(X)[:, 1]), 1e-4, 1 - 1e-4)
        return np.c_[1 - p, p]


def calibration_points(y, proba, n_bins=10):
    """Точки reliability diagram (квантильные корзины)."""
    from sklearn.calibration import calibration_curve
    frac, mean_pred = calibration_curve(y, proba, n_bins=n_bins, strategy="quantile")
    return [[round(float(a), 4), round(float(b), 4)] for a, b in zip(mean_pred, frac)]


# ============================================================
# 6) McNemar (Sasikumar): попарное сравнение моделей
# ============================================================

def mcnemar_exact(y_true, pred_a, pred_b):
    """Точный тест Макнемара. b = A прав, B ошибся; c = A ошибся, B прав."""
    ok_a, ok_b = (pred_a == y_true), (pred_b == y_true)
    b, c = int((ok_a & ~ok_b).sum()), int((~ok_a & ok_b).sum())
    n = b + c
    p = 1.0 if n == 0 else float(stats.binomtest(min(b, c), n, 0.5).pvalue)
    return {"a_only": b, "b_only": c, "p": round(p, 4)}


# ============================================================
# 7) PCA (Patil et al.)
# ============================================================

def pca_analysis(X_train, continuous, y_train, top_loadings=4):
    pre = Pipeline([("imp", SimpleImputer(strategy="median")),
                    ("win", Winsorizer(0.01, 0.99)), ("sc", StandardScaler())])
    Z = pre.fit_transform(X_train[continuous])
    pca = PCA(random_state=RANDOM_STATE).fit(Z)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    scores = pca.transform(Z)
    y = np.asarray(y_train)
    comps = []
    for i in range(min(6, len(evr))):
        load = pca.components_[i]
        top = np.argsort(-np.abs(load))[:top_loadings]
        comps.append({
            "pc": i + 1,
            "explained": round(float(evr[i]), 4),
            "target_corr": round(float(np.corrcoef(scores[:, i], y)[0, 1]), 4),
            "top_features": [{"feature": continuous[j], "loading": round(float(load[j]), 3)}
                             for j in top],
        })
    return {
        "n_features": len(continuous),
        "n_components_70": int(np.argmax(cum >= 0.70) + 1),
        "n_components_90": int(np.argmax(cum >= 0.90) + 1),
        "explained_ratio": [round(float(v), 4) for v in evr],
        "cumulative": [round(float(v), 4) for v in cum],
        "components": comps,
    }


def build_pca_logreg(continuous, binary, categorical):
    return Pipeline([
        ("preprocessor", create_preprocessor(continuous, binary, categorical)),
        ("pca", PCA(n_components=0.90, random_state=RANDOM_STATE)),
        ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ])


# ============================================================
# 8) Hettmann: профили пилотов (кластеризация)
# ============================================================

PROFILE_FEATURES = ["drv_form_finish5", "drv_podium_rate10", "drv_dnf_rate10",
                    "drv_pts_share", "drv_circuit_avg_finish"]
PROFILE_NAMES = ["Front-runner", "Contender", "Midfield", "Backmarker"]


def driver_profile_clusters(train_df, test_df, k=4):
    """KMeans по профилю формы пилота (только train, без утечки). Добавляет колонку
    driver_profile в обе выборки и возвращает описание кластеров."""
    feats = PROFILE_FEATURES
    pre = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    Z = pre.fit_transform(train_df[feats])

    sil = {}
    sample = np.random.default_rng(RANDOM_STATE).choice(len(Z), min(3000, len(Z)), replace=False)
    for kk in range(2, 7):
        lab = KMeans(kk, n_init=5, random_state=RANDOM_STATE).fit_predict(Z)
        sil[kk] = round(float(silhouette_score(Z[sample], lab[sample])), 4)

    km = KMeans(k, n_init=10, random_state=RANDOM_STATE).fit(Z)
    raw = train_df[feats].copy()
    raw["_c"] = km.labels_
    by = raw.groupby("_c")["drv_podium_rate10"].mean().sort_values(ascending=False)
    names = {c: PROFILE_NAMES[i] if i < len(PROFILE_NAMES) else f"Profile {i + 1}"
             for i, c in enumerate(by.index)}

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["driver_profile"] = [names[c] for c in km.labels_]
    test_df["driver_profile"] = [names[c] for c in km.predict(pre.transform(test_df[feats]))]

    summary = []
    for c in by.index:
        part = raw[raw["_c"] == c]
        summary.append({
            "profile": names[c],
            "rows": int(len(part)),
            "avg_finish_last5": round(float(part["drv_form_finish5"].mean()), 2),
            "podium_rate_last10": round(float(part["drv_podium_rate10"].mean()), 3),
            "dnf_rate_last10": round(float(part["drv_dnf_rate10"].mean()), 3),
            "points_share": round(float(part["drv_pts_share"].mean()), 3),
        })
    return train_df, test_df, {"k": k, "silhouette_by_k": sil, "profiles": summary}


# ============================================================
# 9) SHAP (Yathishkumar), необязательная библиотека
# ============================================================

def shap_importance(model, features, X_background, X_explain, max_rows=120):
    """Среднее |SHAP| по признакам для модели, которой нужны только `features`.
    Возвращает None, если shap не установлен или что-то пошло не так."""
    if shap is None:
        return None
    try:
        bg = X_background[features].sample(min(60, len(X_background)),
                                           random_state=RANDOM_STATE).to_numpy()
        ex = X_explain[features].sample(min(max_rows, len(X_explain)),
                                        random_state=RANDOM_STATE).to_numpy()
        f = lambda a: model.predict_proba(pd.DataFrame(a, columns=features))[:, 1]
        expl = shap.Explainer(f, bg, algorithm="permutation")
        vals = np.abs(expl(ex, max_evals=2 * len(features) + 1).values).mean(axis=0)
        out = sorted(({"Feature": n, "Importance": round(float(v), 5)}
                      for n, v in zip(features, vals)), key=lambda d: -d["Importance"])
        return out
    except Exception:
        return None


def permutation_importance_cols(model, X, y, features, n_repeats=5, seed=RANDOM_STATE):
    """Permutation importance (падение PR-AUC) только по `features`; контекстные колонки не трогаем."""
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(seed)
    base = average_precision_score(y, model.predict_proba(X)[:, 1])
    out = []
    for f in features:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[f] = rng.permutation(Xp[f].to_numpy())
            drops.append(base - average_precision_score(y, model.predict_proba(Xp)[:, 1]))
        out.append({"Feature": f, "Importance": round(float(np.mean(drops)), 5)})
    return sorted(out, key=lambda d: -d["Importance"])