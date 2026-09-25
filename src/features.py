"""Stage 3: pairwise features for every candidate pair (country-independent by design).

Feature groups
  name     fuzzy scores on the full, core (generic words removed) and space-free names
  alias    best match of the DBA / 'formerly' part of the S2/S3 name
  address  fuzzy scores on normalized address and on its set of numbers
  context  blocking score, rank, gap to the best candidate, and how this pair compares with
           the other candidates of the same S2/S3 record (each record has at most one owner)
  flags    source, empty address, transliterated name, domain-style name
Country itself is never a feature, so the model transfers to countries unseen in training.

Output: <work>/<universe>/features.parquet
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

from common import N_THREADS

TEXT = ["name_n", "name_core", "name_sq", "name_alt", "addr_n", "nums"]

# (feature name, text column, scorer)
PAIR_SCORES = [
    ("name_ratio", "name_n", fuzz.ratio),
    ("name_tset", "name_n", fuzz.token_set_ratio),
    ("name_tsort", "name_n", fuzz.token_sort_ratio),
    ("name_partial", "name_n", fuzz.partial_ratio),
    ("core_ratio", "name_core", fuzz.ratio),
    ("core_tset", "name_core", fuzz.token_set_ratio),
    ("core_jw", "name_core", JaroWinkler.normalized_similarity),
    ("sq_ratio", "name_sq", fuzz.ratio),
    ("sq_partial", "name_sq", fuzz.partial_ratio),
    ("addr_ratio", "addr_n", fuzz.ratio),
    ("addr_tset", "addr_n", fuzz.token_set_ratio),
    ("addr_tsort", "addr_n", fuzz.token_sort_ratio),
    ("addr_partial", "addr_n", fuzz.partial_ratio),
    ("nums_tset", "nums", fuzz.token_set_ratio),
    ("nums_ratio", "nums", fuzz.ratio),
]
GROUP_REL = ["name_tset", "core_tset", "sq_partial", "addr_tset", "nums_tset", "tfidf"]


def score(a, b, scorer):
    return cpdist(a, b, scorer=scorer, workers=N_THREADS, dtype=np.float32)


def margin_over_others(q, v):
    """v minus the best v among the OTHER candidates of the same q (0 competitors -> v itself).

    Positive means this S1 beats every competitor for that S2/S3 record."""
    v = np.nan_to_num(v.astype(np.float32))
    order = np.lexsort((-v, q))
    qs, vs = q[order], v[order]
    first = np.r_[True, qs[1:] != qs[:-1]]
    grp = np.cumsum(first) - 1
    starts = np.flatnonzero(first)
    sizes = np.diff(np.r_[starts, len(qs)])
    m1 = vs[starts]
    m2 = np.where(sizes > 1, vs[np.minimum(starts + 1, len(vs) - 1)], 0).astype(np.float32)
    others = np.where(first, m2[grp], m1[grp])
    out = np.empty_like(v)
    out[order] = vs - others
    return out


def chunk_features(recs, q, s):
    f = {}
    for name, col, scorer in PAIR_SCORES:
        arr = recs[col]
        f[name] = score(arr[q].tolist(), arr[s].tolist(), scorer)
    alt = recs["name_alt"][q]
    has_alt = alt != ""
    f["alt_tset"] = np.full(len(q), np.nan, np.float32)
    if has_alt.any():
        f["alt_tset"][has_alt] = score(alt[has_alt].tolist(), recs["name_n"][s[has_alt]].tolist(), fuzz.token_set_ratio)
    return f


def run(work, universe, chunk):
    d = os.path.join(work, universe)
    t0 = time.time()
    df = pd.read_parquet(d + "/records.parquet")
    recs = {c: df[c].to_numpy(dtype=object) for c in TEXT}
    pairs = pd.read_parquet(d + "/pairs.parquet")
    q, s = pairs.q.to_numpy(), pairs.s1.to_numpy()

    feats = {k: [] for k in [n for n, _, _ in PAIR_SCORES] + ["alt_tset"]}
    for i in range(0, len(pairs), chunk):
        cf = chunk_features(recs, q[i:i + chunk], s[i:i + chunk])
        for k, v in cf.items():
            feats[k].append(v)
        print(f"[{universe}] features {min(i + chunk, len(pairs)):,}/{len(pairs):,} ({time.time() - t0:.0f}s)", flush=True)
    for k in feats:
        pairs[k] = np.concatenate(feats[k])

    # lengths and flags
    for side, idx in [("q", q), ("s1", s)]:
        pairs[f"{side}_name_len"] = df.name_n.str.len().to_numpy()[idx].astype(np.int16)
        pairs[f"{side}_addr_len"] = df.addr_n.str.len().to_numpy()[idx].astype(np.int16)
        pairs[f"{side}_nums_cnt"] = df.nums.str.count(" ").add(df.nums.ne("")).to_numpy()[idx].astype(np.int8)
    for c in ["src", "is_domain", "name_nonascii", "addr_empty"]:
        pairs[f"q_{c}"] = df[c].to_numpy()[q]
    pairs["s1_addr_empty"] = df.addr_empty.to_numpy()[s]

    # competition among the candidates of the same S2/S3 record
    pairs["n_cand"] = pairs.groupby("q")["s1"].transform("size").astype(np.int16)
    for c in GROUP_REL:
        pairs[f"{c}_margin"] = margin_over_others(q, pairs[c].to_numpy())
    # popularity of the S1 side: how many S2/S3 records point at it (at all / at rank 1)
    gs = pairs.assign(top1=(pairs["rank"] == 1).astype(np.int32)).groupby("s1")
    pairs["s1_n_q"] = gs["q"].transform("size").astype(np.int32)
    pairs["s1_n_top1"] = gs["top1"].transform("sum").astype(np.int32)

    truth_path = d + "/truth.parquet"
    if os.path.exists(truth_path):
        ids = df.entity_id.to_numpy()
        t = pd.read_parquet(truth_path)
        key = pd.Series(1, index=pd.MultiIndex.from_arrays([t.q, t.s1]))
        pairs["label"] = key.reindex(pd.MultiIndex.from_arrays([ids[q], ids[s]])).fillna(0).to_numpy(np.int8)
        print(f"[{universe}] positives {pairs.label.mean():.3f}")
    pairs.to_parquet(d + "/features.parquet", index=False)
    print(f"[{universe}] {pairs.shape} features written in {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--universes", default="trainA,trainB,test")
    ap.add_argument("--chunk", type=int, default=2_000_000)
    args = ap.parse_args()
    for u in args.universes.split(","):
        run(args.work_dir, u, args.chunk)


if __name__ == "__main__":
    main()
