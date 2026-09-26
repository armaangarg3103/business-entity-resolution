"""Pseudo-labels for a country that has no training labels (France), from confident test predictions.

France only exists in test. Its names come from a small generic vocabulary (SARL, Club, Ecole,
Amicale ...) and its records cluster in a few cities, so look-alike businesses are common and
a model trained on US/India over-merges them. Self-training adapts the model:
  positives       the record's best S1 candidate when stage 1 is very sure (p >= --pos, and it
                  beats every other candidate of that record by >= --margin)
  hard negatives  all OTHER candidates of those confidently assigned records: French look-alikes
  easy negatives  a small random share of very unlikely pairs (p <= --neg)
Nothing external is used: only the provided test records and our own model's probabilities.

Output: <work>/pseudo.parquet (same feature columns as training + label)
"""
import argparse
import os

import numpy as np
import pandas as pd

from features import margin_over_others
from train import feature_names, iter_batches, read_cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--countries", default="France", help="comma-separated countries to pseudo-label")
    ap.add_argument("--pos", type=float, default=0.97)
    ap.add_argument("--margin", type=float, default=0.8)
    ap.add_argument("--neg", type=float, default=0.02)
    ap.add_argument("--easy-rate", type=float, default=0.05)
    args = ap.parse_args()
    w = args.work_dir
    path = os.path.join(w, "test", "features.parquet")
    cols = feature_names(os.path.join(w, "trainA", "features.parquet"))

    K = read_cols(path, ["q", "s1"])
    p = np.load(os.path.join(w, "test", "p1.npy"))
    country = pd.read_parquet(os.path.join(w, "test", "records.parquet"), columns=["country"]).country.to_numpy()
    in_c = np.isin(country[K.s1.to_numpy()], args.countries.split(","))
    margin = margin_over_others(K.q.to_numpy(), p)

    pos = in_c & (p >= args.pos) & (margin >= args.margin)
    sure_q = np.zeros(int(K.q.max()) + 1, bool)
    sure_q[K.q.to_numpy()[pos]] = True
    hard = in_c & ~pos & sure_q[K.q.to_numpy()]
    easy = in_c & (p <= args.neg) & ~hard & (np.random.default_rng(5).random(len(p)) < args.easy_rate)
    keep = pos | hard | easy
    label = pos.astype(np.int8)
    print(f"pseudo-labels for {args.countries}: {pos.sum():,} positives, {hard.sum():,} hard negatives, "
          f"{easy.sum():,} easy negatives (of {in_c.sum():,} pairs)")

    parts, off = [], 0
    for b in iter_batches(path, cols):
        n = len(b)
        m = keep[off:off + n]
        part = b[m].reset_index(drop=True)
        part["label"] = label[off:off + n][m]
        parts.append(part)
        off += n
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(os.path.join(w, "pseudo.parquet"), index=False)
    print(f"written {len(out):,} rows to {os.path.join(w, 'pseudo.parquet')}")


if __name__ == "__main__":
    main()
