"""Error analysis on a labelled train half: where exactly is F0.5 lost?

For every true pair: was it a blocking miss, given to another S1 (wrong owner), or rejected by
the threshold? For every false match: was the record a distractor (matches nothing) or does it
belong to another S1? Plus the rank of true pairs inside the candidate lists (does a larger top-K
help?) and printed examples of each error type.

  python src/diagnose.py --work-dir /workspace/er/work --universe trainB
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from train import decide


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--universe", default="trainB")
    ap.add_argument("--examples", type=int, default=6)
    args = ap.parse_args()
    w, u = args.work_dir, args.universe
    d = os.path.join(w, u)

    recs = pd.read_parquet(os.path.join(d, "records.parquet"),
                           columns=["entity_id", "src", "country", "business_name", "business_address"])
    idx = pd.Series(np.arange(len(recs)), index=recs.entity_id.to_numpy())
    truth = pd.read_parquet(os.path.join(d, "truth.parquet"))
    T = pd.DataFrame({"s1": idx.loc[truth.s1].to_numpy(), "q": idx.loc[truth.q].to_numpy()})
    T["country"] = recs.country.to_numpy()[T.s1]

    fpath = os.path.join(d, "features.parquet")
    have = set(pq.ParquetFile(fpath).schema_arrow.names)
    want = [c for c in ["q", "s1", "label", "rank", "emb_rank", "tfidf", "emb", "name_tset", "addr_tset",
                        "emb_name", "emb_addr", "nums_tset"] if c in have]
    K = pq.read_table(fpath, columns=want).to_pandas()
    K["p"] = np.load(os.path.join(d, "p1.npy"))
    thr = json.load(open(os.path.join(w, "stage1.json")))[u]["threshold"]
    print(f"{u}: {len(T):,} true pairs, {len(K):,} candidate pairs, stage-1 threshold {thr}\n")

    # ---- blocking recall and rank of true pairs
    tp_rows = K[K.label == 1]
    print("blocking recall (true pairs present among candidates):")
    got = T.merge(tp_rows[["s1", "q"]], on=["s1", "q"], how="left", indicator=True)
    got["found"] = got._merge == "both"
    print(got.groupby("country").found.mean().round(4).to_string(), "\n  overall", round(got.found.mean(), 4))
    print("\nrank of true pairs by TF-IDF (share):",
          tp_rows["rank"].clip(upper=12).value_counts(normalize=True).sort_index().round(4).to_dict())
    if "emb_rank" in tp_rows:
        print("rank of true pairs by embedding (share):",
              tp_rows["emb_rank"].clip(upper=12).value_counts(normalize=True).sort_index().round(4).to_dict())

    # ---- decision errors
    kept = decide(K, K.p.to_numpy(), thr)[["q", "s1"]]
    best = decide(K, K.p.to_numpy(), -1.0)[["q", "s1", "p"]].rename(columns={"s1": "best_s1", "p": "best_p"})
    fn = got[got.found].drop(columns=["_merge", "found"]).merge(kept.assign(k=1), on=["s1", "q"], how="left")
    fn = fn[fn.k.isna()].drop(columns="k").merge(best, on="q", how="left")
    fn["type"] = np.where(fn.best_s1 != fn.s1, "given to another S1", "best S1 but below threshold")
    miss = got[~got.found].assign(type="blocking miss")
    allfn = pd.concat([miss[["s1", "q", "country", "type"]], fn[["s1", "q", "country", "type"]]])
    print(f"\nmissed true pairs: {len(allfn):,} of {len(T):,} ({len(allfn) / len(T):.2%})")
    print(allfn.groupby(["country", "type"]).size().unstack(fill_value=0).to_string())

    q_owner = pd.Series(T.s1.to_numpy(), index=T.q.to_numpy())
    fp = kept.merge(T[["s1", "q"]].assign(t=1), on=["s1", "q"], how="left")
    fp = fp[fp.t.isna()].drop(columns="t")
    fp["type"] = np.where(fp.q.isin(q_owner.index), "record belongs to another S1", "record matches nothing")
    fp["country"] = recs.country.to_numpy()[fp.s1]
    print(f"\nfalse matches: {len(fp):,} of {len(kept):,} predicted ({len(fp) / max(1, len(kept)):.2%})")
    print(fp.groupby(["country", "type"]).size().unstack(fill_value=0).to_string())

    # ---- feature profile of true pairs the model rejected
    rej = K.merge(fn[["s1", "q"]], on=["s1", "q"])
    cols = [c for c in ["p", "tfidf", "emb", "name_tset", "addr_tset", "emb_name", "emb_addr", "nums_tset"] if c in K]
    print("\nmedian features, rejected true pairs vs accepted true pairs:")
    acc = K[(K.label == 1)].merge(kept, on=["s1", "q"])
    print(pd.DataFrame({"rejected": rej[cols].median(), "accepted": acc[cols].median()}).round(3).to_string())

    # ---- examples
    name, addr = recs.business_name.to_numpy(), recs.business_address.to_numpy()

    def show(title, df, other_col=None):
        print(f"\n--- {title}")
        for _, r in df.head(args.examples).iterrows():
            print(f"S1 : {name[r.s1]} | {addr[r.s1]}")
            print(f"rec: {name[r.q]} | {addr[r.q]}")
            if other_col and not pd.isna(r[other_col]):
                o = int(r[other_col])
                print(f"got: {name[o]} | {addr[o]}")
            print()

    rng = 7
    show("blocking misses", miss.sample(min(len(miss), args.examples), random_state=rng))
    show("given to another S1 ('got' = where it went)",
         fn[fn.type == "given to another S1"].sample(frac=1, random_state=rng), "best_s1")
    show("best S1 but below threshold", fn[fn.type != "given to another S1"].sample(frac=1, random_state=rng))
    show("false matches to distractor records", fp[fp.type == "record matches nothing"].sample(frac=1, random_state=rng))


if __name__ == "__main__":
    main()
