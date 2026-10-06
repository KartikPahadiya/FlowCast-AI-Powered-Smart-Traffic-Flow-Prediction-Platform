"""M5 — Classical ML engine.

Trains, tunes and compares classical models on both task types:
  * regression: next-window traffic_volume and travel_time
  * classification: 4-class congestion_level, binary accident_risk

Includes LinearRegressionFromScratch — plain NumPy gradient descent on
MSE, satisfying the PRD's build-it-once rule (Section 11, Table 17).
"""
import json
import logging
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score,
                             roc_auc_score)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from . import config
from .features import FEATURE_COLS

log = logging.getLogger("flowcast.models")


# ------------------------------------------------- from-scratch model
class LinearRegressionFromScratch:
    """Linear regression trained by batch gradient descent in pure NumPy.

    Prediction is Xw (matrix-vector product); the loss is MSE; gradients
    are derived analytically: dL/dw = (2/n) X^T (Xw - y).
    """

    def __init__(self, lr=0.01, n_iters=2000):
        self.lr, self.n_iters = lr, n_iters
        self.loss_curve_ = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.mu_, self.sigma_ = X.mean(0), X.std(0) + 1e-12
        Xs = (X - self.mu_) / self.sigma_
        Xb = np.hstack([Xs, np.ones((len(Xs), 1))])
        w = np.zeros(Xb.shape[1])
        n = len(y)
        for i in range(self.n_iters):
            err = Xb @ w - y
            grad = (2 / n) * Xb.T @ err          # d(MSE)/dw
            w -= self.lr * grad                  # gradient-descent step
            self.loss_curve_.append(float((err ** 2).mean()))
            if i and abs(self.loss_curve_[-2] - self.loss_curve_[-1]) < 1e-10:
                break
        self.w_ = w
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        Xs = (X - self.mu_) / self.sigma_
        Xb = np.hstack([Xs, np.ones((len(Xs), 1))])
        return Xb @ self.w_


# ------------------------------------------------------- metric helpers
def reg_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    mask = y_true != 0  # MAPE undefined at y=0
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE": float(np.mean(np.abs((y_true[mask] - y_pred[mask])
                                     / y_true[mask])) * 100),
        "R2": float(r2_score(y_true, y_pred)),
    }


def clf_metrics(y_true, y_pred, labels):
    return {
        "macro_F1": float(f1_score(y_true, y_pred, average="macro")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred,
                                                 average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred,
                                           average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred,
                                             labels=labels).tolist(),
    }


# ------------------------------------------------------------ training
class Trainer:
    """Holds the splits and trains every classical model for each target."""

    def __init__(self, train, val, test):
        self.train, self.val, self.test = train, val, test
        self.results = {}          # scoreboard: model -> target -> metrics
        self.fitted = {}           # name -> fitted estimator
        self.scalers = {}
        self.gd_loss_curve = None
        self._X = {}

    def __setstate__(self, state):
        """Older pickles may predate the per-target feature cache."""
        state.setdefault("_X", {})
        self.__dict__.update(state)

    def _arrays(self, cols):
        """Feature matrices are cached per feature list (train/val/test)."""
        key = tuple(cols)
        if key not in self._X:
            self._X[key] = (
                self.train[list(cols)].to_numpy(np.float32),
                self.val[list(cols)].to_numpy(np.float32),
                self.test[list(cols)].to_numpy(np.float32))
        return self._X[key]

    # ---------------------------------------------------- regression
    def _reg_candidates(self, target):
        pos = max(1, int((self.train[target] > 0).sum()))
        ratio = (len(self.train) - pos) / pos
        return {
            "LinearRegression_sklearn": LinearRegression(),
            "LinearRegression_from_scratch": LinearRegressionFromScratch(),
            "DecisionTree": DecisionTreeRegressor(
                max_depth=12, min_samples_leaf=20, random_state=config.RANDOM_SEED),
            "RandomForest": RandomForestRegressor(
                n_estimators=150, max_depth=20, min_samples_leaf=2, n_jobs=-1,
                random_state=config.RANDOM_SEED),
            "XGBoost": XGBRegressor(
                n_estimators=1500, learning_rate=0.05, max_depth=8,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                tree_method="hist", early_stopping_rounds=50,
                random_state=config.RANDOM_SEED, n_jobs=-1),
        }

    def train_regressors(self, target, feature_cols=FEATURE_COLS):
        log.info("training volume regressors for target=%s", target)
        Xtr, Xva, Xte = self._arrays(feature_cols)
        self.results.setdefault(target, {})
        for name, model in self._reg_candidates(target).items():
            t0 = time.time()
            if isinstance(model, XGBRegressor):
                model.fit(Xtr, self.train[target],
                          eval_set=[(Xva, self.val[target])], verbose=False)
            else:
                model.fit(Xtr, self.train[target])
            pred = model.predict(Xte)
            m = reg_metrics(self.test[target], pred)
            m["train_seconds"] = round(time.time() - t0, 1)
            if isinstance(model, LinearRegressionFromScratch):
                self.gd_loss_curve = model.loss_curve_
                m["gd_final_train_mse"] = model.loss_curve_[-1]
            self.results[target][name] = m
            self.fitted[f"{target}__{name}"] = model
            log.info("  %-32s RMSE=%.1f MAPE=%.2f%% (%.0fs)", name,
                     m["RMSE"], m["MAPE"], m["train_seconds"])
        return self

    # ------------------------------------------------- classification
    def train_classifiers(self, target, y_col, labels, pos_weight=None,
                          feature_cols=FEATURE_COLS):
        log.info("training classifiers for target=%s", target)
        Xtr, Xva, Xte = self._arrays(feature_cols)
        ytr, yva, yte = (self.train[y_col], self.val[y_col], self.test[y_col])
        self.results.setdefault(target, {})
        cands = {
            "DecisionTree": DecisionTreeClassifier(
                max_depth=12, min_samples_leaf=20, random_state=config.RANDOM_SEED),
            "RandomForest": RandomForestClassifier(
                n_estimators=150, max_depth=20, n_jobs=-1,
                random_state=config.RANDOM_SEED),
            "XGBoost": XGBClassifier(
                n_estimators=800, learning_rate=0.08, max_depth=8,
                subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                scale_pos_weight=pos_weight or 1,
                eval_metric="mlogloss" if len(labels) > 2 else "logloss",
                random_state=config.RANDOM_SEED, n_jobs=-1),
        }
        for name, model in cands.items():
            t0 = time.time()
            model.fit(Xtr, ytr)
            pred = model.predict(Xte)
            m = clf_metrics(yte, pred, labels=list(range(len(labels))))
            m["ROC_AUC"] = self._auc(model, yte, labels, X=Xte)
            m["train_seconds"] = round(time.time() - t0, 1)
            self.results[target][name] = m
            self.fitted[f"{target}__{name}"] = model
            log.info("  %-32s macroF1=%.3f AUC=%.3f (%.0fs)", name,
                     m["macro_F1"], m["ROC_AUC"], m["train_seconds"])

        # SVM baseline on scaled features, seeded subsample (PRD-configured)
        log.info("training SVM baseline on %d-row subsample ...",
                 config.SVM_TRAIN_SUBSAMPLE)
        t0 = time.time()
        scaler = StandardScaler().fit(Xtr)
        self.scalers[target] = scaler
        idx = np.random.RandomState(config.RANDOM_SEED).choice(
            len(Xtr), size=min(config.SVM_TRAIN_SUBSAMPLE, len(Xtr)),
            replace=False)
        if len(labels) > 2:
            svm = SVC(kernel="rbf", C=10, gamma="scale",
                      random_state=config.RANDOM_SEED)
        else:
            svm = SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced",
                      random_state=config.RANDOM_SEED)
        svm.fit(scaler.transform(Xtr[idx]), ytr.iloc[idx])
        pred = svm.predict(scaler.transform(Xte))
        m = clf_metrics(yte, pred, labels=list(range(len(labels))))
        m["ROC_AUC"] = self._auc(svm, yte, labels, X=scaler.transform(Xte))
        m["train_seconds"] = round(time.time() - t0, 1)
        m["note"] = (f"trained on a seeded {len(idx):,}-row subsample "
                     "of the train window (SVM cubic cost); features z-scored")
        self.results[target]["SVM"] = m
        self.fitted[f"{target}__SVM"] = svm
        log.info("  %-32s macroF1=%.3f AUC=%.3f (%.0fs)", "SVM",
                 m["macro_F1"], m["ROC_AUC"], m["train_seconds"])
        return self

    def _auc(self, model, yte, labels, X=None):
        X = self.Xte if X is None else X
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X)
        else:
            p = model.decision_function(X)
        if len(labels) > 2:
            # per-class binarized AUC also accepts ranking scores such as
            # the SVM decision function (sklearn's multiclass path requires
            # probabilities, which an SVC without Platt scaling has not)
            y_bin = label_binarize(np.asarray(yte), classes=list(range(len(labels))))
            return float(roc_auc_score(y_bin, np.asarray(p), average="macro"))
        return float(roc_auc_score(yte, np.asarray(p)[:, 1]
                                   if np.ndim(p) == 2 else p))


# ------------------------------------------------------ winners & I/O
def pick_winner(results: dict, target: str, metric: str):
    return max(results[target], key=lambda k: results[target][k][metric])


def save_artifacts(trainer: Trainer):
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for key, model in trainer.fitted.items():
        joblib.dump(model, config.MODELS_DIR / f"{key}.joblib")
    with open(config.SCOREBOARD_JSON, "w") as f:
        json.dump(trainer.results, f, indent=2)
    if trainer.gd_loss_curve:
        json.dump(trainer.gd_loss_curve,
                  open(config.MODELS_DIR / "gd_loss_curve.json", "w"))


def load_scoreboard():
    with open(config.SCOREBOARD_JSON) as f:
        return json.load(f)
