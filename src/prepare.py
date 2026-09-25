"""Stage 1: load raw TSVs, split train into two independent 'universes', normalize text.

Train is split by Source 1 entity into halves A and B. Each half keeps its S1 records, every
S2/S3 record matched to them, and a random half of the unmatched S2/S3 records. Each half is
roughly the size of the test set, so blocking and thresholds tuned on B behave like on test.

Output (per universe in {trainA, trainB, test}):
    <work>/<universe>/records.parquet  - all records with normalized fields
    <work>/<universe>/truth.parquet    - (s1, q) true pairs, train universes only
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from common import add_core_names, ensure_dir, normalize_frame, read_tsv


def load_split(data_dir, split):
    frames = []
    for i in (1, 2, 3):
        df = read_tsv(os.path.join(data_dir, split, f"{split}_source{i}.tsv"))
        df["src"] = np.int8(i)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_universes(recs, gt, sample, seed):
    """Split train records + ground truth into universes A/B (optionally subsampled)."""
    rng = np.random.default_rng(seed)
    pairs = gt.assign(q=gt.matched_entity_ids.str.split(",")).explode("q")
    pairs = pairs[pairs.q.notna() & (pairs.q != "")].rename(columns={"source1_entity_id": "s1"})[["s1", "q"]]

    s1_ids = recs.loc[recs.src == 1, "entity_id"].to_numpy()
    s1_uni = pd.Series(rng.integers(0, 2, len(s1_ids)), index=s1_ids)
    s1_keep = pd.Series(rng.random(len(s1_ids)) < sample, index=s1_ids)

    other = recs.loc[recs.src != 1, "entity_id"]
    q_uni = pd.Series(rng.integers(0, 2, len(other)), index=other.to_numpy())
    q_keep = pd.Series(rng.random(len(other)) < sample, index=other.to_numpy())
    # matched records follow their S1 entity
    q_uni.loc[pairs.q.to_numpy()] = s1_uni.loc[pairs.s1.to_numpy()].to_numpy()
    q_keep.loc[pairs.q.to_numpy()] = s1_keep.loc[pairs.s1.to_numpy()].to_numpy()

    uni = pd.concat([s1_uni, q_uni])
    keep = pd.concat([s1_keep, q_keep])
    recs = recs.assign(uni=recs.entity_id.map(uni).to_numpy(), keep=recs.entity_id.map(keep).to_numpy())
    recs = recs[recs.keep].drop(columns="keep")
    out = {}
    for u, name in [(0, "trainA"), (1, "trainB")]:
        r = recs[recs.uni == u].drop(columns="uni").reset_index(drop=True)
        t = pairs[pairs.s1.isin(set(r.entity_id[r.src == 1]))].reset_index(drop=True)
        out[name] = (r, t)
    return out


def finish(recs, work, name, truth=None):
    t0 = time.time()
    recs = normalize_frame(recs)
    recs, generic = add_core_names(recs)
    d = ensure_dir(os.path.join(work, name))
    recs.to_parquet(os.path.join(d, "records.parquet"), index=False)
    if truth is not None:
        truth.to_parquet(os.path.join(d, "truth.parquet"), index=False)
    print(f"[{name}] {len(recs):,} records ({recs.src.value_counts().sort_index().to_dict()}) "
          f"normalized in {time.time() - t0:.0f}s")
    for c, g in generic.items():
        print(f"   generic name words [{c}] ({len(g)}): {sorted(g)[:40]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="folder containing train/ and test/")
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--sample", type=float, default=1.0, help="fraction of S1 entities to keep (debug)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    train = load_split(args.data_dir, "train")
    gt = read_tsv(os.path.join(args.data_dir, "train", "train_ground_truth.tsv"))
    print(f"loaded train in {time.time() - t0:.0f}s")
    for name, (r, t) in build_universes(train, gt, args.sample, args.seed).items():
        finish(r, args.work_dir, name, t)
    del train, gt

    test = load_split(args.data_dir, "test")
    if args.sample < 1.0:  # debug only: a sampled test set is NOT a valid submission
        rng = np.random.default_rng(args.seed)
        test = test[rng.random(len(test)) < args.sample].reset_index(drop=True)
    finish(test, args.work_dir, "test")


if __name__ == "__main__":
    main()
