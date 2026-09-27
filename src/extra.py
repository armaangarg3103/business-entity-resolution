"""Stage 4b: specificity-aware features, appended to the existing features.parquet files.

Why: how much a shared house number or city proves depends on the country. In US test data the
median number of S1 records sharing a (house number, city) is 1; in India 30; in France 164
(small house numbers, a few dense cities, names built from a small generic vocabulary).
A model that learned "same number + same city => match" from US data over-merges in France.

These features weight every token by its rarity INSIDE ITS OWN COUNTRY (IDF fitted per country
on all records, labelled or not), so a shared rare street name counts a lot and a shared
"12" / "bordeaux" / "sarl" counts almost nothing. Countries unseen in training get their own
weights automatically.

  addr_idf / name_idf / name_cidf   IDF-weighted cosine of address words, name words, name 3-grams
  *_margin                          vs the best OTHER candidate of the same S2/S3 record
  s1_same_name / q_same_name        how many S1 records in the country carry exactly that name
                                    (ambiguity, important for records with no address)

Rows stay aligned with features.parquet; the file is rewritten in streamed batches.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer

from features import margin_over_others
from train import BATCH, read_cols

NEW = ["addr_idf", "name_idf", "name_cidf", "addr_idf_margin", "name_idf_margin", "name_cidf_margin",
       "s1_same_name", "q_same_name"]


def rowwise_dot(X, a, b, chunk=2_000_000):
    out = np.empty(len(a), np.float32)
    for i in range(0, len(a), chunk):
        out[i:i + chunk] = np.asarray(X[a[i:i + chunk]].multiply(X[b[i:i + chunk]]).sum(axis=1)).ravel()
    return out


def compute(work, universe):
    d = os.path.join(work, universe)
    recs = pd.read_parquet(os.path.join(d, "records.parquet"), columns=["src", "country", "name_n", "addr_n"])
    K = read_cols(os.path.join(d, "features.parquet"), ["q", "s1"])
    q, s = K.q.to_numpy(), K.s1.to_numpy()
    country = recs.country.to_numpy()
    pair_c = country[s]
    out = {c: np.zeros(len(K), np.float32) for c in ["addr_idf", "name_idf", "name_cidf"]}

    for c in np.unique(country):
        t0 = time.time()
        rows = np.flatnonzero(country == c)
        pos = np.full(len(recs), -1, np.int64)
        pos[rows] = np.arange(len(rows))
        sel = np.flatnonzero(pair_c == c)
        a, b = pos[q[sel]], pos[s[sel]]
        vecs = {"addr_idf": (TfidfVectorizer(token_pattern=r"\S+", lowercase=False, sublinear_tf=True,
                                             dtype=np.float32), recs.addr_n.to_numpy()[rows]),
                "name_idf": (TfidfVectorizer(token_pattern=r"\S+", lowercase=False, sublinear_tf=True,
                                             dtype=np.float32), recs.name_n.to_numpy()[rows]),
                "name_cidf": (TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), lowercase=False,
                                              sublinear_tf=True, dtype=np.float32), recs.name_n.to_numpy()[rows])}
        for name, (vec, text) in vecs.items():
            X = vec.fit_transform(text)
            out[name][sel] = rowwise_dot(X, a, b)
        print(f"[{universe}] {c}: {len(sel):,} pairs in {time.time() - t0:.0f}s", flush=True)

    for c in ["addr_idf", "name_idf", "name_cidf"]:
        out[c + "_margin"] = margin_over_others(q, out[c])

    # how many S1 records in the same country carry exactly this (order-free) name
    key = pd.Series(recs.country.to_numpy() + "|" +
                    recs.name_n.map(lambda x: " ".join(sorted(x.split()))).to_numpy())
    s1_counts = key[recs.src.to_numpy() == 1].value_counts()
    n_same = key.map(s1_counts).fillna(0).to_numpy(np.float32)
    out["s1_same_name"] = n_same[s]
    out["q_same_name"] = n_same[q]
    return out


def append(work, universe, extra):
    """Rewrite features.parquet batch by batch with the new columns (replacing old copies)."""
    path = os.path.join(work, universe, "features.parquet")
    tmp = path + ".tmp"
    fh = open(path, "rb")
    pf = pq.ParquetFile(fh)
    keep = [c for c in pf.schema_arrow.names if c not in NEW]
    writer, off = None, 0
    for b in pf.iter_batches(batch_size=BATCH, columns=keep):
        df = b.to_pandas()
        n = len(df)
        for c in NEW:
            df[c] = extra[c][off:off + n]
        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(tmp, table.schema)
        writer.write_table(table, row_group_size=1_000_000)
        off += n
    writer.close()
    fh.close()
    assert off == len(extra[NEW[0]]), "row count mismatch"
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--universes", default="trainA,trainB,test")
    args = ap.parse_args()
    for u in args.universes.split(","):
        t0 = time.time()
        extra = compute(args.work_dir, u)
        append(args.work_dir, u, extra)
        print(f"[{u}] {len(NEW)} specificity features appended in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
