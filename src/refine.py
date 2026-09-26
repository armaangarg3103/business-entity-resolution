"""Stage 6: second-stage model over entity groups, per-entity decision, submission files.

Stage 1 scores each pair in isolation. Here every pair also sees its neighbourhood:
  q-side    how its stage-1 probability compares with the record's other S1 candidates
  S1-side   how many strong candidates the S1 entity has, this pair's rank among them,
            the expected cluster size (sum of probabilities)
  coherence how similar the record is to the entity's strongest OTHER candidate (anchor):
            true members of one business agree with each other, look-alikes do not
The stage-2 model is trained on half A (out-of-fold stage-1 probabilities) and tuned on B.
Like stage 1 it trains on all positives plus a weighted share of negatives and predicts in
streamed batches, so memory stays bounded on ~100M-pair universes.

Two decision rules are compared on B and the better one is used for test:
  threshold  each record goes to its best S1 if p >= t
  expected   each record goes to its best S1; then per S1 entity the top-k set that maximises
             the expected F0.5 (including k = 0, i.e. predicting a singleton) is kept

Output: <out>/matching_results.tsv, <out>/candidate_pairs.tsv, <work>/stage2.json,
        <work>/test/p2.parquet (for variant.py)
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from common import N_THREADS, ensure_dir, write_id_lists
from features import margin_over_others
from gbm import train_gbm
from train import (decide, load_eval, predict_stream, read_cols, score_by_country, score_kept,
                   threshold_sweep, training_set)

F32 = np.float32


def top2_rows(group, value):
    """For every row: index of the best and second-best row of its group by value (-1 if none)."""
    order = np.lexsort((-value, group))
    gs = group[order]
    first = np.r_[True, gs[1:] != gs[:-1]]
    starts = np.flatnonzero(first)
    sizes = np.diff(np.r_[starts, len(gs)])
    gid = np.cumsum(first) - 1
    best = order[starts]
    second = np.where(sizes > 1, order[np.minimum(starts + 1, len(order) - 1)], -1)
    b = np.empty(len(group), np.int64)
    s = np.empty(len(group), np.int64)
    b[order], s[order] = best[gid], second[gid]
    return b, s


def graph_features(work, universe, K, p):
    """Neighbourhood features from stage-1 probabilities p (row-aligned with K: q, s1, q_src)."""
    q, s1, src = K.q.to_numpy(), K.s1.to_numpy(), K.q_src.to_numpy()
    G = pd.DataFrame({"p1": p.astype(F32)})
    G["p1_q_margin"] = margin_over_others(q, p)
    tmp = pd.DataFrame({"q": q, "s1": s1, "p": p})
    G["p1_q_sum"] = tmp.groupby("q").p.transform("sum").astype(F32).to_numpy()
    G["p1_q_rank"] = tmp.groupby("q").p.rank(ascending=False, method="first").astype(F32).to_numpy()
    tmp["own"] = np.where(G.p1_q_margin.to_numpy() >= 0, p, 0).astype(F32)
    tmp["hi"] = (p > 0.5).astype(np.int32)
    for sv in (2, 3):
        tmp[f"own{sv}"] = np.where(src == sv, tmp.own, 0).astype(F32)
    gs = tmp.groupby("s1")
    G["s1_psum"] = gs.p.transform("sum").astype(F32).to_numpy()
    G["s1_own_sum"] = gs.own.transform("sum").astype(F32).to_numpy()
    G["s1_pmax"] = gs.p.transform("max").astype(F32).to_numpy()
    G["s1_n_hi"] = gs.hi.transform("sum").astype(F32).to_numpy()
    G["s1_prank"] = gs.p.rank(ascending=False, method="first").astype(F32).to_numpy()
    for sv in (2, 3):
        G[f"s1_own_src{sv}"] = gs[f"own{sv}"].transform("sum").astype(F32).to_numpy()
    G["p1_rel_s1max"] = (p / np.maximum(G.s1_pmax.to_numpy(), 1e-6)).astype(F32)
    del tmp, gs

    # coherence with the entity's strongest other candidate
    best, second = top2_rows(s1, p)
    rows = np.arange(len(G))
    anchor = np.where(best == rows, second, best)
    has = np.flatnonzero(anchor >= 0)
    recs = pd.read_parquet(os.path.join(work, universe, "records.parquet"), columns=["name_n", "addr_n"])
    for col, arr in [("anchor_name_tset", recs.name_n.to_numpy(dtype=object)),
                     ("anchor_addr_tset", recs.addr_n.to_numpy(dtype=object))]:
        v = np.full(len(G), np.nan, F32)
        for i in range(0, len(has), 10_000_000):
            h = has[i:i + 10_000_000]
            v[h] = cpdist(arr[q[h]].tolist(), arr[q[anchor[h]]].tolist(), scorer=fuzz.token_set_ratio,
                          workers=N_THREADS, dtype=F32)
        G[col] = v
    G["anchor_p1"] = np.where(anchor >= 0, p[np.maximum(anchor, 0)], np.nan).astype(F32)
    return G


def expected_f_decide(pairs, prob, alpha=1.0):
    """Per S1 entity, keep the top-k assigned records that maximise the expected F0.5."""
    p = np.clip(prob, 1e-6, 1 - 1e-6) ** alpha
    d = pd.DataFrame({"q": pairs.q.to_numpy(), "s1": pairs.s1.to_numpy(), "p": p})
    e_true = d.groupby("s1").p.sum()                              # expected number of true matches
    p_none = np.exp(np.log1p(-d.p).groupby(d.s1).sum())         # P(entity is a singleton)
    a = d.sort_values("p", ascending=False, kind="stable").drop_duplicates("q")
    a = a.sort_values(["s1", "p"], ascending=[True, False], kind="stable")
    a["k"] = a.groupby("s1").cumcount() + 1
    a["tp"] = a.groupby("s1").p.cumsum()
    a["ef"] = 1.25 * a.tp / (0.25 * a.s1.map(e_true) + a.k)
    best = a.loc[a.groupby("s1").ef.idxmax(), ["s1", "k", "ef"]].set_index("s1")
    best = best[best.ef > p_none.reindex(best.index).fillna(0)]
    a = a[a.s1.isin(best.index)]
    return a[a.k <= a.s1.map(best.k)]


def error_report(ev, kept):
    """Where the F0.5 is lost, per country."""
    ids, truth, s1_all, country = ev
    pred = pd.DataFrame({"s1": ids[kept.s1.to_numpy()], "q": ids[kept.q.to_numpy()]})
    t = truth.groupby("s1").size().reindex(s1_all).fillna(0)
    pn = pred.groupby("s1").size().reindex(s1_all).fillna(0)
    tp = pred.merge(truth, on=["s1", "q"]).groupby("s1").size().reindex(s1_all).fillna(0)
    df = pd.DataFrame({"country": country.loc[s1_all].to_numpy(), "t": t.values, "p": pn.values, "tp": tp.values})
    rows = []
    for c, g in df.groupby("country"):
        single = g[g.t == 0]
        multi = g[g.t > 0]
        rows.append({"country": c,
                     "singleton_share": round(len(single) / len(g), 3),
                     "singletons_wrong": round((single.p > 0).mean(), 3),
                     "matched_pred_empty": round((multi.p == 0).mean(), 3),
                     "precision": round((multi.tp / multi.p.clip(lower=1)).mean(), 4),
                     "recall": round((multi.tp / multi.t).mean(), 4)})
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--leaves", type=int, default=127)
    ap.add_argument("--neg-rate", type=float, default=0.3)
    args = ap.parse_args()
    w = args.work_dir
    cols1 = json.load(open(os.path.join(w, "stage1.json")))["features"]
    path = {u: os.path.join(w, u, "features.parquet") for u in ("trainA", "trainB", "test")}

    def graph(u):
        t0 = time.time()
        K = read_cols(path[u], ["q", "s1", "q_src"])
        p = np.load(os.path.join(w, u, "p1.npy"))
        G = graph_features(w, u, K, p)
        print(f"[{u}] {G.shape[1]} group features for {len(G):,} pairs in {time.time() - t0:.0f}s", flush=True)
        return K, G, p

    _, GA, _ = graph("trainA")
    XA, yA, wA = training_set(path["trainA"], cols1, args.neg_rate, seed=11, extra=GA)
    del GA
    KB, GB, pB1 = graph("trainB")
    XB, yB, wB = training_set(path["trainB"], cols1, args.neg_rate, seed=12, extra=GB)
    cols2 = list(XA.columns)
    m = train_gbm(XA, yA, XB, yB, wA, wB, leaves=args.leaves, lr=args.lr, rounds=args.rounds, seed=7,
                  name="stage2")
    m.save(os.path.join(w, "model_stage2"))
    del XA, XB
    pB = predict_stream([m], path["trainB"], cols1, extra=GB, use_cols=cols2)
    del GB

    ev = load_eval(w, "trainB")
    t1, r1 = threshold_sweep(ev, KB, pB1)
    t2, r2 = threshold_sweep(ev, KB, pB)
    print(f"[B] stage 1 threshold {t1}: {r1[t1]:.4f} | stage 2 threshold {t2}: {r2[t2]:.4f}")
    cands = {("threshold", t2): r2[t2]}
    for alpha in (0.5, 0.75, 1.0, 1.5, 2.0):
        cands[("expected", alpha)] = score_kept(ev, expected_f_decide(KB, pB, alpha))
        print(f"[B] stage 2 expected-F rule, alpha {alpha}: {cands[('expected', alpha)]:.4f}")
    rule = max(cands, key=cands.get)

    def apply(K, p):
        return decide(K, p, rule[1]) if rule[0] == "threshold" else expected_f_decide(K, p, rule[1])

    keptB = apply(KB, pB)
    print(f"[B] chosen rule {rule}: macro F0.5 {cands[rule]:.4f} by country {score_by_country(ev, keptB)}")
    error_report(ev, keptB)
    print("stage-2 top features:", m.importance(cols2).round(3).head(12).to_dict())
    del KB, keptB

    KT, GT, _ = graph("test")
    pT = predict_stream([m], path["test"], cols1, extra=GT, use_cols=cols2)
    del GT
    pd.DataFrame({"q": KT.q.to_numpy(), "s1": KT.s1.to_numpy(), "p": pT}) \
        .to_parquet(os.path.join(w, "test", "p2.parquet"), index=False)   # for variant.py
    kept = apply(KT, pT)
    recs = pd.read_parquet(os.path.join(w, "test", "records.parquet"), columns=["entity_id", "src", "country"])
    ids = recs.entity_id.to_numpy()
    s1_ids = ids[recs.src.to_numpy() == 1]
    out = ensure_dir(args.out_dir)
    cand = pd.DataFrame({"s1": ids[KT.s1.to_numpy()], "q": ids[KT.q.to_numpy()]})
    match = pd.DataFrame({"s1": ids[kept.s1.to_numpy()], "q": ids[kept.q.to_numpy()]})
    write_id_lists(os.path.join(out, "candidate_pairs.tsv"), s1_ids, cand, "candidate_entity_ids")
    write_id_lists(os.path.join(out, "matching_results.tsv"), s1_ids, match, "matched_entity_ids")

    n = match.groupby("s1").size().reindex(s1_ids).fillna(0)
    stats = pd.DataFrame({"country": pd.Series(recs.country.to_numpy(), index=ids).loc[s1_ids].to_numpy(),
                          "n": n.to_numpy()})
    print(f"[test] rule {rule} | {len(s1_ids):,} S1 entities | {len(match):,} matches")
    print(stats.groupby("country").n.agg(avg_matches="mean", empty_share=lambda x: (x == 0).mean()).round(3))
    with open(os.path.join(w, "stage2.json"), "w") as f:
        json.dump({"rule": [rule[0], float(rule[1])], "valid_f05": cands[rule], "best_iteration": m.best_iteration,
                   "features": cols2}, f, indent=2)


if __name__ == "__main__":
    main()
