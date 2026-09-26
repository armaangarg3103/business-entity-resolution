"""Gradient-boosted trees with one interface: XGBoost on the GPU when available, else LightGBM on CPU.

GBM_BACKEND=auto (default) | xgb | lgb    choose the library
GBM_DEVICE=cuda (default) | cpu           device for XGBoost
If XGBoost fails for any reason, training falls back to LightGBM so the pipeline never stalls.
"""
import os
import time

import numpy as np
import pandas as pd

from common import N_THREADS


def _gpu_visible():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


class GBM:
    def __init__(self, kind, booster, best_iteration):
        self.kind, self.booster, self.best_iteration = kind, booster, best_iteration

    def predict(self, X):
        if self.kind == "xgb":
            return self.booster.inplace_predict(X, iteration_range=(0, self.best_iteration + 1)).astype(np.float32)
        return self.booster.predict(X, num_iteration=self.best_iteration, num_threads=N_THREADS).astype(np.float32)

    def importance(self, cols):
        if self.kind == "xgb":
            g = self.booster.get_score(importance_type="total_gain")
            s = pd.Series([g.get(c, 0.0) for c in cols], index=cols)
        else:
            s = pd.Series(self.booster.feature_importance("gain"), index=cols)
        return (s / max(s.sum(), 1e-12)).sort_values(ascending=False)

    def save(self, path_no_ext):
        self.booster.save_model(path_no_ext + (".json" if self.kind == "xgb" else ".txt"))


def _train_xgb(Xt, yt, Xv, yv, wt, wv, leaves, lr, rounds, seed, name):
    import xgboost as xgb
    device = os.environ.get("GBM_DEVICE", "cuda")
    params = {"objective": "binary:logistic", "eval_metric": "logloss", "device": device, "tree_method": "hist",
              "eta": lr, "max_depth": 0, "grow_policy": "lossguide", "max_leaves": leaves, "min_child_weight": 10,
              "subsample": 0.8, "colsample_bytree": 0.8, "lambda": 1.0, "seed": seed, "nthread": N_THREADS}
    dt = xgb.QuantileDMatrix(Xt, yt, weight=wt, max_bin=256)
    dv = xgb.QuantileDMatrix(Xv, yv, weight=wv, ref=dt)
    print(f"[{name}] XGBoost on {device}", flush=True)
    bst = xgb.train(params, dt, rounds, evals=[(dv, "valid")], early_stopping_rounds=100, verbose_eval=50)
    return GBM("xgb", bst, bst.best_iteration)


def _train_lgb(Xt, yt, Xv, yv, wt, wv, leaves, lr, rounds, seed, name):
    import lightgbm as lgb
    params = dict(objective="binary", learning_rate=lr, num_leaves=leaves, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=N_THREADS, force_col_wise=True, verbose=-1, seed=seed)
    print(f"[{name}] LightGBM on CPU with {N_THREADS} threads", flush=True)
    dt = lgb.Dataset(Xt, yt, weight=wt)
    dv = lgb.Dataset(Xv, yv, weight=wv, reference=dt)
    m = lgb.train(params, dt, rounds, valid_sets=[dv], valid_names=["valid"],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)])
    return GBM("lgb", m, m.best_iteration)


def train_gbm(Xt, yt, Xv, yv, wt=None, wv=None, leaves=255, lr=0.08, rounds=3000, seed=42, name="model"):
    backend = os.environ.get("GBM_BACKEND", "auto")
    if backend == "auto":
        backend = "xgb" if _gpu_visible() else "lgb"
    t0 = time.time()
    m = None
    if backend == "xgb":
        try:
            m = _train_xgb(Xt, yt, Xv, yv, wt, wv, leaves, lr, rounds, seed, name)
        except Exception as e:  # never let the GPU path stall the pipeline
            print(f"[{name}] XGBoost failed ({type(e).__name__}: {e}); falling back to LightGBM", flush=True)
    if m is None:
        m = _train_lgb(Xt, yt, Xv, yv, wt, wv, leaves, lr, rounds, seed, name)
    print(f"[{name}] {m.kind} trained in {time.time() - t0:.0f}s, best iteration {m.best_iteration}", flush=True)
    return m
