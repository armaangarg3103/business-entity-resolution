"""Stage 5: first-stage pair classifier, cross-fitted over the two train halves.

  model_A is trained on half A (early stopping on B); model_B on half B (early stopping on A).
  Out-of-fold probabilities p1 are saved for A (from model_B) and B (from model_A), so the
  second stage (refine.py) learns from honest, never-seen-in-training probabilities.
  The test probability is the average of both models, so all training data is used.

Memory: a half has ~60M candidate pairs. Training keeps every positive and a random share of
negatives (--neg-rate), weighted by 1/rate so probabilities stay calibrated; prediction streams
the parquet files in batches. Peak RAM is a few GB instead of tens.

Output: <work>/model_A.*, model_B.*, stage1.json and <work>/<universe>/p1.npy
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common import macro_f05_vec
from gbm import train_gbm

NOT_FEATURES = {"q", "s1", "label"}
BATCH = int(os.environ.get("FEATURE_BATCH", 4_000_000))   # rows per streamed batch


# ----------------------------------------------------------------------------- streaming helpers

def feature_names(path):
    return [c for c in pq.ParquetFile(path).schema_arrow.names if c not in NOT_FEATURES]


def read_cols(path, cols):
    return pq.read_table(path, columns=cols).to_pandas()


def iter_batches(path, cols):
    for b in pq.ParquetFile(path).iter_batches(batch_size=BATCH, columns=cols):
        yield b.to_pandas()


def sample_mask(labels, rate, seed):
    """All positives plus a random `rate` share of negatives."""
    return (labels == 1) | (np.random.default_rng(seed).random(len(labels)) < rate)


def load_rows(path, cols, mask, extra=None):
    """Rows of the parquet where mask is True (optionally joined with row-aligned extra columns)."""
    parts, off = [], 0
    for b in iter_batches(path, cols):
        n = len(b)
        m = mask[off:off + n]
        part = b[m].reset_index(drop=True)
        if extra is not None:
            part = pd.concat([part, extra.iloc[off:off + n][m].reset_index(drop=True)], axis=1)
        parts.append(part)
        off += n
    return pd.concat(parts, ignore_index=True)


def predict_stream(models, path, cols, extra=None, use_cols=None):
    """Average prediction of `models` over all rows of the parquet, batch by batch."""
    out, off = [], 0
    for b in iter_batches(path, cols):
        n = len(b)
        if extra is not None:
            b = pd.concat([b.reset_index(drop=True), extra.iloc[off:off + n].reset_index(drop=True)], axis=1)
        X = b[use_cols] if use_cols else b
        out.append(np.mean([m.predict(X) for m in models], axis=0).astype(np.float32))
        off += n
    return np.concatenate(out)


def training_set(path, cols, rate, seed, extra=None):
    y = read_cols(path, ["label"]).label.to_numpy()
    mask = sample_mask(y, rate, seed)
    X = load_rows(path, cols, mask, extra)
    y = y[mask]
    w = np.where(y == 1, 1.0, 1.0 / rate).astype(np.float32)
    print(f"   {os.path.basename(os.path.dirname(path))}: {mask.sum():,} of {len(mask):,} rows "
          f"({(y == 1).sum():,} positives)", flush=True)
    return X, y, w


# ----------------------------------------------------------------------------- decision + metric

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
    grid = grid or [round(float(x), 2) for x in np.arange(0.05, 0.96, 0.05)]
    best_per_q = decide(pairs, prob, -1.0)          # sort once, then only filter per threshold
    res = {t: score_kept(ev, best_per_q[best_per_q.p >= t]) for t in grid}
    best = max(res, key=res.get)
    return best, res


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.08)
    ap.add_argument("--leaves", type=int, default=255)
    ap.add_argument("--neg-rate", type=float, default=0.3, help="share of negative pairs used for training")
    ap.add_argument("--pseudo", action="store_true", help="add <work>/pseudo.parquet (see pseudo.py) to training")
    args = ap.parse_args()
    w = args.work_dir
    path = {u: os.path.join(w, u, "features.parquet") for u in ("trainA", "trainB", "test")}
    cols = feature_names(path["trainA"])
    print(f"{len(cols)} features; sampling negatives at {args.neg_rate}")

    XA, yA, wA = training_set(path["trainA"], cols, args.neg_rate, seed=1)
    XB, yB, wB = training_set(path["trainB"], cols, args.neg_rate, seed=2)
    kw = dict(leaves=args.leaves, lr=args.lr, rounds=args.rounds)
    tA, tB = (XA, yA, wA), (XB, yB, wB)
    if args.pseudo:  # pseudo-labelled test rows go into training only, never into validation
        P = pd.read_parquet(os.path.join(w, "pseudo.parquet"))
        yP, wP = P.label.to_numpy(), np.ones(len(P), np.float32)
        print(f"adding {len(P):,} pseudo-labelled rows ({int(yP.sum()):,} positives) to both training sets")
        tA = (pd.concat([XA, P[cols]], ignore_index=True), np.r_[yA, yP], np.r_[wA, wP])
        tB = (pd.concat([XB, P[cols]], ignore_index=True), np.r_[yB, yP], np.r_[wB, wP])
        del P
    mA = train_gbm(*tA[:2], XB, yB, tA[2], wB, seed=42, name="model_A", **kw)
    mA.save(os.path.join(w, "model_A"))
    mB = train_gbm(*tB[:2], XA, yA, tB[2], wA, seed=43, name="model_B", **kw)
    mB.save(os.path.join(w, "model_B"))
    del XA, XB, tA, tB
    print("top features:", mA.importance(cols).round(3).head(15).to_dict())

    summary = {"features": cols}
    for u, m in [("trainA", mB), ("trainB", mA)]:
        p = predict_stream([m], path[u], cols)
        np.save(os.path.join(w, u, "p1.npy"), p)
        K = read_cols(path[u], ["q", "s1"])
        ev = load_eval(w, u)
        t, res = threshold_sweep(ev, K, p)
        summary[u] = {"threshold": t, "f05": res[t], "by_country": score_by_country(ev, decide(K, p, t))}
        print(f"[stage 1] {u} out-of-fold: best threshold {t} -> macro F0.5 {res[t]:.4f} {summary[u]['by_country']}",
              flush=True)

    p = predict_stream([mA, mB], path["test"], cols)
    np.save(os.path.join(w, "test", "p1.npy"), p)
    print(f"[stage 1] test probabilities written ({len(p):,} pairs)")
    with open(os.path.join(w, "stage1.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
