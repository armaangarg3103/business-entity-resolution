"""Stage 6: second-stage model over entity groups, per-entity decision, submission files.

Stage 1 scores each pair in isolation. Here every pair also sees its neighbourhood:
  q-side    how its stage-1 probability compares with the record's other S1 candidates
  S1-side   how many strong candidates the S1 entity has, this pair's rank among them,
            the expected cluster size (sum of probabilities)
  coherence how similar the record is to the entity's strongest OTHER candidate (anchor):
            true members of one business agree with each other, look-alikes do not
The stage-2 LightGBM is trained on half A (out-of-fold stage-1 probabilities) and tuned on B.

Two decision rules are compared on B and the better one is used for test:
  threshold  each record goes to its best S1 if p >= t
  expected   each record goes to its best S1; then per S1 entity the top-k set that maximises
             the expected F0.5 (including k = 0, i.e. predicting a singleton) is kept

Output: <out>/matching_results.tsv, <out>/candidate_pairs.tsv, <work>/stage2.json
"""
import argparse
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from common import N_THREADS, ensure_dir, write_id_lists
from features import margin_over_others
from train import decide, load_eval, score_by_country, score_kept, threshold_sweep


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


def graph_features(work, universe, F, p):
    """Neighbourhood features from stage-1 probabilities p (aligned with rows of F)."""
    q, s1 = F.q.to_numpy(), F.s1.to_numpy()
    G = pd.DataFrame({"q": q, "s1": s1, "p1": p})
    G["p1_q_margin"] = margin_over_others(q, p)
    G["p1_q_sum"] = G.groupby("q").p1.transform("sum")
    G["p1_q_rank"] = G.groupby("q").p1.rank(ascending=False, method="first").astype(np.float32)
    is_best = G.p1_q_margin >= 0
    G["own"] = np.where(is_best, p, 0).astype(np.float32)
    gs = G.groupby("s1")
    G["s1_psum"] = gs.p1.transform("sum")
    G["s1_own_sum"] = gs.own.transform("sum")
    G["s1_pmax"] = gs.p1.transform("max")
    G["s1_n_hi"] = G.assign(h=(G.p1 > 0.5).astype(np.int32)).groupby("s1").h.transform("sum")
    G["s1_prank"] = gs.p1.rank(ascending=False, method="first").astype(np.float32)
    G["p1_rel_s1max"] = (G.p1 / G.s1_pmax.clip(lower=1e-6)).astype(np.float32)
    src = F.q_src.to_numpy()
    for sv in (2, 3):
        G[f"s1_own_src{sv}"] = G.assign(o=np.where(src == sv, G.own, 0)).groupby("s1").o.transform("sum")

    # coherence with the entity's strongest other candidate
    best, second = top2_rows(s1, p)
    rows = np.arange(len(G))
    anchor = np.where(best == rows, second, best)
    has = anchor >= 0
    recs = pd.read_parquet(os.path.join(work, universe, "records.parquet"), columns=["name_n", "addr_n"])
    name = recs.name_n.to_numpy(dtype=object)
    addr = recs.addr_n.to_numpy(dtype=object)
    qa = q[anchor[has]]
    qq = q[has]
    for col, arr in [("anchor_name_tset", name), ("anchor_addr_tset", addr)]:
        v = np.full(len(G), np.nan, np.float32)
        v[has] = cpdist(arr[qq].tolist(), arr[qa].tolist(), scorer=fuzz.token_set_ratio,
                        workers=N_THREADS, dtype=np.float32)
        G[col] = v
    G["anchor_p1"] = np.where(has, p[np.maximum(anchor, 0)], np.nan).astype(np.float32)
    return G.drop(columns=["q", "s1", "own"])


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
    ap.add_argument("--threads", type=int, default=N_THREADS)
    args = ap.parse_args()
    w = args.work_dir
    cols1 = json.load(open(os.path.join(w, "stage1.json")))["features"]

    def build(u):
        t0 = time.time()
        F = pd.read_parquet(os.path.join(w, u, "features.parquet"))
        p = np.load(os.path.join(w, u, "p1.npy"))
        G = graph_features(w, u, F, p)
        X = pd.concat([F[cols1].reset_index(drop=True), G.reset_index(drop=True)], axis=1)
        print(f"[{u}] stage-2 features {X.shape} in {time.time() - t0:.0f}s", flush=True)
        return F[["q", "s1"] + (["label"] if "label" in F else [])], X, p

    KA, XA, _ = build("trainA")
    KB, XB, pB1 = build("trainB")
    cols2 = list(XA.columns)
    params = dict(objective="binary", learning_rate=args.lr, num_leaves=args.leaves, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=args.threads, force_col_wise=True, verbose=-1, seed=7)
    dA = lgb.Dataset(XA, KA.label)
    dB = lgb.Dataset(XB, KB.label, reference=dA)
    m = lgb.train(params, dA, args.rounds, valid_sets=[dB], valid_names=["B"],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)])
    m.save_model(os.path.join(w, "model_stage2.txt"))
    pB = m.predict(XB, num_iteration=m.best_iteration, num_threads=args.threads)
    del XA, dA, dB

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
    imp = pd.Series(m.feature_importance("gain"), index=cols2).sort_values(ascending=False)
    print("stage-2 top features:", (imp / imp.sum()).round(3).head(12).to_dict())
    del XB

    KT, XT, _ = build("test")
    pT = m.predict(XT[cols2], num_iteration=m.best_iteration, num_threads=args.threads)
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
        json.dump({"rule": list(rule), "valid_f05": cands[rule], "best_iteration": m.best_iteration,
                   "features": cols2}, f, indent=2)


if __name__ == "__main__":
    main()
