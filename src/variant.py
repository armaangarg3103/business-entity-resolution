"""Rebuild the submission from saved stage-2 test probabilities, optionally with a different
decision rule for chosen countries. Takes seconds, no retraining.

Used to probe a country without labels (France) on the leaderboard: two submissions that
differ only in that country's rows isolate its effect on the score.

  python src/variant.py --work-dir W --out-dir O                                  # same as refine
  python src/variant.py --work-dir W --out-dir O --override France=threshold:0.9  # stricter France
  python src/variant.py --work-dir W --out-dir O --override France=expected:2.0
  python src/variant.py --work-dir W --out-dir O --stage1 --rule threshold:0.75   # stage 1 only

Rules: threshold:<t> keeps a record's best S1 if p >= t;
       expected:<alpha> keeps the per-entity set maximising expected F0.5 of p**alpha
       (alpha > 1 = more conservative, < 1 = more generous).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from common import ensure_dir, write_id_lists
from refine import expected_f_decide
from train import decide


def apply(rule, value, K, p):
    return decide(K, p, value) if rule == "threshold" else expected_f_decide(K, p, value)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--override", action="append", default=[], help="COUNTRY=rule:value, repeatable")
    ap.add_argument("--stage1", action="store_true", help="use stage-1 test probabilities (p1.npy)")
    ap.add_argument("--rule", default=None, help="default rule for all countries, e.g. threshold:0.75")
    args = ap.parse_args()
    w = args.work_dir

    if args.stage1:
        P = pd.read_parquet(os.path.join(w, "test", "features.parquet"), columns=["q", "s1"])
        P["p"] = np.load(os.path.join(w, "test", "p1.npy"))
    else:
        P = pd.read_parquet(os.path.join(w, "test", "p2.parquet"))
    if args.rule:
        r, v = args.rule.split(":")
        default = (r, float(v))
    else:
        default = json.load(open(os.path.join(w, "stage2.json")))["rule"]
    recs = pd.read_parquet(os.path.join(w, "test", "records.parquet"), columns=["entity_id", "src", "country"])
    ids, country = recs.entity_id.to_numpy(), recs.country.to_numpy()
    s1_ids = ids[recs.src.to_numpy() == 1]
    P["country"] = country[P.s1.to_numpy()]

    rules = {}
    for o in args.override:
        c, rv = o.split("=", 1)
        r, v = rv.split(":")
        rules[c] = (r, float(v))
    kept = []
    for c, g in P.groupby("country"):
        r, v = rules.get(c, tuple(default))
        k = apply(r, v, g, g.p.to_numpy())
        kept.append(k[["q", "s1"]])
        n = k.groupby("s1").size()
        s1_c = (country[recs.src.to_numpy() == 1] == c).sum()
        print(f"{c:>8}: rule {r}:{v} -> {len(k):,} matches, avg {len(k) / s1_c:.3f} per entity, "
              f"empty share {1 - len(n) / s1_c:.3f}")
    kept = pd.concat(kept)

    out = ensure_dir(args.out_dir)
    cand = pd.DataFrame({"s1": ids[P.s1.to_numpy()], "q": ids[P.q.to_numpy()]})
    match = pd.DataFrame({"s1": ids[kept.s1.to_numpy()], "q": ids[kept.q.to_numpy()]})
    write_id_lists(os.path.join(out, "candidate_pairs.tsv"), s1_ids, cand, "candidate_entity_ids")
    write_id_lists(os.path.join(out, "matching_results.tsv"), s1_ids, match, "matched_entity_ids")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
