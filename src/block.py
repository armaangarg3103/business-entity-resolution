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


def block_country(g, top_k, max_df, threads, emb=None, emb_k=5):
    """Top-K S1 candidates per S2/S3 record for one country.

    TF-IDF top-K always; if embeddings are given, the embedding top-emb_k is added (union).
    Every kept pair gets both scores and both ranks, so the model sees each view."""
    docs = [record_tokens(*r) for r in zip(g.name_core, g.name_alt, g.name_sq, g.addr_n, g.nums)]
    vec = TfidfVectorizer(analyzer=lambda x: x, sublinear_tf=True, max_df=max_df, min_df=1, dtype=np.float32)
    X = vec.fit_transform(docs)
    is_s1 = (g.src == 1).to_numpy()
    s1_rows, q_rows = np.flatnonzero(is_s1), np.flatnonzero(~is_s1)
    if len(s1_rows) == 0 or len(q_rows) == 0:
        return None
    C = sp_matmul_topn(X[q_rows], X[s1_rows].T.tocsr(), top_n=top_k, threshold=1e-6, sort=True,
                       n_threads=threads).tocoo()
    lq, ls = q_rows[C.row], s1_rows[C.col]          # positions inside g
    out = {"lq": lq, "ls": ls}
    if emb is not None:
        gi = g.index.to_numpy()
        eq, es = embedding_topk(emb, gi[q_rows], gi[s1_rows], emb_k)
        pairs = pd.DataFrame({"lq": np.r_[lq, q_rows[eq]], "ls": np.r_[ls, s1_rows[es]]}).drop_duplicates()
        lq, ls = pairs.lq.to_numpy(), pairs.ls.to_numpy()
        out = {"lq": lq, "ls": ls, "emb": combined_cosine(emb, gi[lq], gi[ls])}
    out["tfidf"] = rowwise_dot(X, lq, ls)
    out = pd.DataFrame(out)
    gi = g.index.to_numpy()
    out["q"], out["s1"] = gi[out.lq], gi[out.ls]
    out = out.drop(columns=["lq", "ls"]).sort_values(["q", "tfidf"], ascending=[True, False], kind="stable")
    out["rank"] = out.groupby("q").cumcount().astype(np.int16) + 1
    if emb is not None:
        out["emb_rank"] = out.groupby("q")["emb"].rank(ascending=False, method="first").astype(np.int16)
    return out


def rowwise_dot(X, a, b, chunk=2_000_000):
    """Cosine of TF-IDF rows a[i] and b[i] (rows are already L2-normalised)."""
    res = np.empty(len(a), np.float32)
    for i in range(0, len(a), chunk):
        res[i:i + chunk] = np.asarray(X[a[i:i + chunk]].multiply(X[b[i:i + chunk]]).sum(axis=1)).ravel()
    return res


def _comb(emb, rows, torch, dev, dtype):
    """Normalised (name + address) embedding for the given universe rows, as a GPU tensor."""
    n = torch.from_numpy(np.asarray(emb[0][rows], dtype=np.float32)).to(dev)
    n += torch.from_numpy(np.asarray(emb[1][rows], dtype=np.float32)).to(dev)
    return torch.nn.functional.normalize(n, dim=1).to(dtype)


def embedding_topk(emb, q_idx, s_idx, k, chunk=4096):
    """For each query row, the k most similar S1 rows by combined embedding (GPU matmul + topk).

    Returns (query position, S1 position) arrays, positions relative to q_idx / s_idx."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.float32
    S = torch.cat([_comb(emb, s_idx[i:i + 500_000], torch, dev, dtype) for i in range(0, len(s_idx), 500_000)])
    k = min(k, len(s_idx))
    qi, si = [], []
    for i in range(0, len(q_idx), chunk):
        Q = _comb(emb, q_idx[i:i + chunk], torch, dev, dtype)
        top = (Q @ S.T).topk(k, dim=1).indices.cpu().numpy()
        qi.append(np.repeat(np.arange(i, i + len(top)), k))
        si.append(top.ravel())
    del S
    return np.concatenate(qi), np.concatenate(si)


def combined_cosine(emb, a, b, chunk=250_000):
    res = np.empty(len(a), np.float32)
    for i in range(0, len(a), chunk):
        x = emb[0][a[i:i + chunk]].astype(np.float32) + emb[1][a[i:i + chunk]].astype(np.float32)
        y = emb[0][b[i:i + chunk]].astype(np.float32) + emb[1][b[i:i + chunk]].astype(np.float32)
        res[i:i + chunk] = (x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1) + 1e-6)
    return res


def load_embeddings(d):
    """(name, address) float16 arrays, memory-mapped from disk; None if the embed stage has not run."""
    paths = [os.path.join(d, "emb_name.npy"), os.path.join(d, "emb_addr.npy")]
    if not all(os.path.exists(p) for p in paths):
        return None
    return tuple(np.load(p, mmap_mode="r") for p in paths)


def run(work, universe, top_k, max_df, threads, emb_k):
    d = os.path.join(work, universe)
    recs = pd.read_parquet(d + "/records.parquet",
                           columns=["entity_id", "src", "country", "name_core", "name_alt", "name_sq", "addr_n", "nums"])
    emb = load_embeddings(d) if emb_k > 0 else None
    print(f"[{universe}] embeddings: {'yes, top-%d added' % emb_k if emb is not None else 'not used'}")
    parts = []
    for country, g in recs.groupby("country"):
        t0 = time.time()
        p = block_country(g, top_k, max_df, threads, emb, emb_k)
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
    ap.add_argument("--emb-k", type=int, default=5, help="embedding candidates per record (0 = TF-IDF only)")
    args = ap.parse_args()
    for u in args.universes.split(","):
        run(args.work_dir, u, args.top_k, args.max_df, args.threads, args.emb_k)


if __name__ == "__main__":
    main()
