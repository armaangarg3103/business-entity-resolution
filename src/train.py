"""Stage 4: train the LightGBM pair classifier on universe A, tune the decision on universe B.

Decision rule (see decide()):
  1. each S2/S3 record is given only to its highest-probability S1 candidate
     (ground truth never assigns one record to two S1 entities);
  2. that link is kept only if its probability >= threshold.
The threshold is chosen to maximise the challenge metric (macro F0.5 over all S1 entities,
singletons included) on universe B, which the model never trained on.

Output: <work>/model.txt, <work>/decision.json
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
    """Return the kept (q, s1) rows: best S1 per q, above threshold."""
    d = pd.DataFrame({"q": pairs.q.to_numpy(), "s1": pairs.s1.to_numpy(), "p": prob})
    best = d.sort_values("p", ascending=False, kind="stable").drop_duplicates("q")
    return best[best.p >= threshold]


def evaluate(work, universe, feats, prob, thresholds):
    """Macro F0.5 on a train universe for each threshold; also per country at the best one."""
    recs = pd.read_parquet(os.path.join(work, universe, "records.parquet"), columns=["entity_id", "src", "country"])
    ids = recs.entity_id.to_numpy()
    truth = pd.read_parquet(os.path.join(work, universe, "truth.parquet"))
    s1_all = ids[recs.src.to_numpy() == 1]
    res = {}
    for t in thresholds:
        kept = decide(feats, prob, t)
        pred = pd.DataFrame({"s1": ids[kept.s1.to_numpy()], "q": ids[kept.q.to_numpy()]})
        res[t] = macro_f05_vec(s1_all, truth, pred)
    best_t = max(res, key=res.get)
    kept = decide(feats, prob, best_t)
    pred = pd.DataFrame({"s1": ids[kept.s1.to_numpy()], "q": ids[kept.q.to_numpy()]})
    country = pd.Series(recs.country.to_numpy(), index=ids)
    by_c = {}
    for c in country.unique():
        s1_c = s1_all[country.loc[s1_all].to_numpy() == c]
        keep = set(s1_c)
        by_c[c] = macro_f05_vec(s1_c, truth[truth.s1.isin(keep)], pred[pred.s1.isin(keep)])
    return res, best_t, by_c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--leaves", type=int, default=255)
    ap.add_argument("--threads", type=int, default=N_THREADS)
    args = ap.parse_args()
    w = args.work_dir

    t0 = time.time()
    A = pd.read_parquet(os.path.join(w, "trainA", "features.parquet"))
    B = pd.read_parquet(os.path.join(w, "trainB", "features.parquet"))
    cols = feature_cols(A)
    print(f"train rows {len(A):,} | valid rows {len(B):,} | {len(cols)} features")

    params = dict(objective="binary", learning_rate=args.lr, num_leaves=args.leaves, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=args.threads, verbose=-1, seed=42)
    dA = lgb.Dataset(A[cols], A.label, free_raw_data=True)
    dB = lgb.Dataset(B[cols], B.label, reference=dA)
    model = lgb.train(params, dA, args.rounds, valid_sets=[dB], valid_names=["B"],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)])
    model.save_model(os.path.join(w, "model.txt"))
    print(f"trained in {time.time() - t0:.0f}s, best iteration {model.best_iteration}")

    prob = model.predict(B[cols], num_iteration=model.best_iteration)
    grid = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]
    res, best_t, by_c = evaluate(w, "trainB", B, prob, grid)
    for t in grid:
        print(f"  threshold {t:.2f}  macro F0.5 {res[t]:.4f}" + ("   <- best" if t == best_t else ""))
    print("  per country at best threshold:", {k: round(v, 4) for k, v in by_c.items()})

    imp = pd.Series(model.feature_importance("gain"), index=cols).sort_values(ascending=False)
    print("top features:", (imp / imp.sum()).round(3).head(15).to_dict())
    with open(os.path.join(w, "decision.json"), "w") as f:
        json.dump({"threshold": best_t, "valid_f05": res[best_t], "valid_f05_by_country": by_c,
                   "best_iteration": model.best_iteration, "features": cols}, f, indent=2)


if __name__ == "__main__":
    main()
