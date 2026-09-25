"""Stage 2: candidate generation (blocking) with sparse TF-IDF top-k search.

Every S2/S3 record belongs to at most one S1 entity, so we search from the S2/S3 side:
for each S2/S3 record we keep the top-K most similar S1 records of the SAME country
(country is used only as an open-set grouping key; any label works).

Each record becomes a bag of prefixed tokens:
    n_<word>   words of the core name and of the DBA / 'formerly' alias
    c_<3gram>  character trigrams of the space-free core name (typos, transliteration, domains)
    a_<word>   address words
    d_<num>    numbers in the address (house numbers, PIN / ZIP codes)
Tokens that are very common inside a country (max_df) are dropped: they add cost, not signal.

Output: <work>/<universe>/pairs.parquet with columns q, s1 (row indices into records.parquet),
        tfidf (cosine score), rank (1 = best S1 for that S2/S3 record).
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from common import N_THREADS


def record_tokens(core, alt, sq, addr, nums):
    toks = ["n_" + t for t in core.split()]
    toks += ["n_" + t for t in alt.split()]
    s = f"#{sq}#"
    toks += ["c_" + s[i:i + 3] for i in range(len(s) - 2)]
    toks += ["a_" + t for t in addr.split()]
    toks += ["d_" + t for t in nums.split()]
    return toks


def block_country(g, top_k, max_df, threads):
    docs = [record_tokens(*r) for r in zip(g.name_core, g.name_alt, g.name_sq, g.addr_n, g.nums)]
    vec = TfidfVectorizer(analyzer=lambda x: x, sublinear_tf=True, max_df=max_df, min_df=1, dtype=np.float32)
    X = vec.fit_transform(docs)
    is_s1 = (g.src == 1).to_numpy()
    s1_rows, q_rows = np.flatnonzero(is_s1), np.flatnonzero(~is_s1)
    if len(s1_rows) == 0 or len(q_rows) == 0:
        return None
    S = X[s1_rows].T.tocsr()
    Q = X[q_rows]
    C = sp_matmul_topn(Q, S, top_n=top_k, threshold=1e-6, sort=True, n_threads=threads).tocoo()
    gi = g.index.to_numpy()
    out = pd.DataFrame({"q": gi[q_rows[C.row]], "s1": gi[s1_rows[C.col]], "tfidf": C.data.astype(np.float32)})
    out = out.sort_values(["q", "tfidf"], ascending=[True, False], kind="stable")
    out["rank"] = out.groupby("q").cumcount().astype(np.int16) + 1
    return out


def run(work, universe, top_k, max_df, threads):
    d = os.path.join(work, universe)
    recs = pd.read_parquet(d + "/records.parquet",
                           columns=["entity_id", "src", "country", "name_core", "name_alt", "name_sq", "addr_n", "nums"])
    parts = []
    for country, g in recs.groupby("country"):
        t0 = time.time()
        p = block_country(g, top_k, max_df, threads)
        if p is not None:
            parts.append(p)
            print(f"[{universe}] {country}: {len(g):,} records -> {len(p):,} pairs in {time.time() - t0:.0f}s")
    pairs = pd.concat(parts, ignore_index=True)
    pairs.to_parquet(d + "/pairs.parquet", index=False)

    truth_path = d + "/truth.parquet"
    if os.path.exists(truth_path):  # blocking recall = ceiling for the matcher
        ids = recs.entity_id.to_numpy()
        cand = pd.DataFrame({"s1": ids[pairs.s1], "q": ids[pairs.q], "rank": pairs["rank"].to_numpy()})
        truth = pd.read_parquet(truth_path)
        hit = truth.merge(cand, on=["s1", "q"], how="left")
        rec = hit["rank"].notna().mean()
        print(f"[{universe}] blocking recall {rec:.4f} | recall@1 {(hit['rank'] == 1).mean():.4f} "
              f"| @3 {(hit['rank'] <= 3).mean():.4f} | pairs/query {len(pairs) / max(1, (recs.src != 1).sum()):.2f}")
        for c, g in hit.assign(country=hit.s1.map(recs.set_index('entity_id').country)).groupby("country"):
            print(f"      {c}: recall {g['rank'].notna().mean():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--universes", default="trainA,trainB,test")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--max-df", type=float, default=0.01)
    ap.add_argument("--threads", type=int, default=N_THREADS)
    args = ap.parse_args()
    for u in args.universes.split(","):
        run(args.work_dir, u, args.top_k, args.max_df, args.threads)


if __name__ == "__main__":
    main()
