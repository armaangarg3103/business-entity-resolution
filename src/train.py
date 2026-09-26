"""Stage 5: first-stage pair classifier, cross-fitted over the two train halves.

  model_A is trained on half A (early stopping on B); model_B on half B (early stopping on A).
  Out-of-fold probabilities p1 are saved for A (from model_B) and B (from model_A), so the
  second stage (refine.py) learns from honest, never-seen-in-training probabilities.
  The test probability is the average of both models, so all training data is used.

Output: <work>/model_A.txt, model_B.txt, stage1.json and <work>/<universe>/p1.npy
"""
import argparse
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import N_THREADS, macro_f05_vec

NOT_FEATURES = {"q", "s1", "label"}


def feature_cols(df):
    return [c for c in df.columns if c not in NOT_FEATURES]


def decide(pairs, prob, threshold):
    """Threshold rule: best S1 per q, kept if its probability >= threshold."""
    d = pd.DataFrame({"q": pairs.q.to_numpy(), "s1": pairs.s1.to_numpy(), "p": prob})
    best = d.sort_values("p", ascending=False, kind="stable").drop_duplicates("q")
    return best[best.p >= threshold]


def load_eval(work, universe):
    recs = pd.read_parquet(os.path.join(work, universe, "records.parquet"), columns=["entity_id", "src", "country"])
    ids = recs.entity_id.to_numpy()
    truth = pd.read_parquet(os.path.join(work, universe, "truth.parquet"))
    s1_all = ids[recs.src.to_numpy() == 1]
    country = pd.Series(recs.country.to_numpy(), index=ids)
    return ids, truth, s1_all, country


def score_kept(ev, kept):
    ids, truth, s1_all, _ = ev
    pred = pd.DataFrame({"s1": ids[kept.s1.to_numpy()], "q": ids[kept.q.to_numpy()]})
    return macro_f05_vec(s1_all, truth, pred)


def score_by_country(ev, kept):
    ids, truth, s1_all, country = ev
    pred = pd.DataFrame({"s1": ids[kept.s1.to_numpy()], "q": ids[kept.q.to_numpy()]})
    out = {}
    for c in country.unique():
        s1_c = s1_all[country.loc[s1_all].to_numpy() == c]
        keep = set(s1_c)
        out[c] = round(macro_f05_vec(s1_c, truth[truth.s1.isin(keep)], pred[pred.s1.isin(keep)]), 4)
    return out


def threshold_sweep(ev, pairs, prob, grid=None):
    grid = grid or [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]
    res = {t: score_kept(ev, decide(pairs, prob, t)) for t in grid}
    best = max(res, key=res.get)
    return best, res


def fit(train, valid, cols, args, name):
    params = dict(objective="binary", learning_rate=args.lr, num_leaves=args.leaves, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=args.threads, force_col_wise=True, verbose=-1, seed=42)
    print(f"[{name}] training on {len(train):,} rows with {args.threads} threads (progress every 50 rounds)", flush=True)
    t0 = time.time()
    dt = lgb.Dataset(train[cols], train.label)
    dv = lgb.Dataset(valid[cols], valid.label, reference=dt)
    m = lgb.train(params, dt, args.rounds, valid_sets=[dv], valid_names=["valid"],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)])
    print(f"[{name}] done in {time.time() - t0:.0f}s, best iteration {m.best_iteration}", flush=True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.08)
    ap.add_argument("--leaves", type=int, default=255)
    ap.add_argument("--threads", type=int, default=N_THREADS)
    args = ap.parse_args()
    w = args.work_dir
    pred_kw = dict(num_threads=args.threads)

    A = pd.read_parquet(os.path.join(w, "trainA", "features.parquet"))
    B = pd.read_parquet(os.path.join(w, "trainB", "features.parquet"))
    cols = feature_cols(A)
    print(f"A rows {len(A):,} | B rows {len(B):,} | {len(cols)} features")

    mA = fit(A, B, cols, args, "model_A")
    mA.save_model(os.path.join(w, "model_A.txt"))
    mB = fit(B, A, cols, args, "model_B")
    mB.save_model(os.path.join(w, "model_B.txt"))

    summary = {"features": cols}
    for u, df, m in [("trainA", A, mB), ("trainB", B, mA)]:
        p = m.predict(df[cols], num_iteration=m.best_iteration, **pred_kw).astype(np.float32)
        np.save(os.path.join(w, u, "p1.npy"), p)
        ev = load_eval(w, u)
        t, res = threshold_sweep(ev, df, p)
        summary[u] = {"threshold": t, "f05": res[t], "by_country": score_by_country(ev, decide(df, p, t))}
        print(f"[stage 1] {u} out-of-fold: best threshold {t} -> macro F0.5 {res[t]:.4f} {summary[u]['by_country']}")
    imp = pd.Series(mA.feature_importance("gain"), index=cols).sort_values(ascending=False)
    print("top features:", (imp / imp.sum()).round(3).head(15).to_dict())
    del A, B

    T = pd.read_parquet(os.path.join(w, "test", "features.parquet"), columns=cols)
    p = 0.5 * (mA.predict(T, num_iteration=mA.best_iteration, **pred_kw)
               + mB.predict(T, num_iteration=mB.best_iteration, **pred_kw))
    np.save(os.path.join(w, "test", "p1.npy"), p.astype(np.float32))
    print(f"[stage 1] test probabilities written ({len(p):,} pairs)")
    with open(os.path.join(w, "stage1.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
