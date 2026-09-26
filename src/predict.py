"""Stage 5: score the test candidate pairs and write the two submission files.

  <out>/candidate_pairs.tsv   every S2/S3 record the model scored for each S1 entity
  <out>/matching_results.tsv  the final matches (always a subset of the candidates)
Every test S1 entity gets exactly one row in both files, with an empty list if needed.
"""
import argparse
import json
import os

import lightgbm as lgb
import pandas as pd

from common import N_THREADS, ensure_dir, write_id_lists
from train import decide


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--threshold", type=float, default=None, help="override the tuned threshold")
    args = ap.parse_args()
    w = args.work_dir

    cfg = json.load(open(os.path.join(w, "decision.json")))
    thr = cfg["threshold"] if args.threshold is None else args.threshold
    model = lgb.Booster(model_file=os.path.join(w, "model.txt"))

    recs = pd.read_parquet(os.path.join(w, "test", "records.parquet"), columns=["entity_id", "src", "country"])
    ids = recs.entity_id.to_numpy()
    s1_ids = ids[recs.src.to_numpy() == 1]
    feats = pd.read_parquet(os.path.join(w, "test", "features.parquet"))
    prob = model.predict(feats[cfg["features"]], num_iteration=cfg["best_iteration"], num_threads=N_THREADS)

    kept = decide(feats, prob, thr)
    out = ensure_dir(args.out_dir)
    cand = pd.DataFrame({"s1": ids[feats.s1.to_numpy()], "q": ids[feats.q.to_numpy()]})
    match = pd.DataFrame({"s1": ids[kept.s1.to_numpy()], "q": ids[kept.q.to_numpy()]})
    write_id_lists(os.path.join(out, "candidate_pairs.tsv"), s1_ids, cand, "candidate_entity_ids")
    write_id_lists(os.path.join(out, "matching_results.tsv"), s1_ids, match, "matched_entity_ids")

    n = match.groupby("s1").size()
    country = pd.Series(recs.country.to_numpy(), index=ids).loc[s1_ids]
    stats = pd.DataFrame({"country": country.to_numpy(), "n": n.reindex(s1_ids).fillna(0).to_numpy()})
    print(f"threshold {thr} | {len(s1_ids):,} S1 entities | {len(match):,} matches")
    print(stats.groupby("country").n.agg(avg_matches="mean", empty_share=lambda x: (x == 0).mean()).round(3))


if __name__ == "__main__":
    main()
