"""Stage 5b (GPU): cross-encoder that reads both records side by side, for the uncertain pairs.

Backbone: intfloat/multilingual-e5-small (MIT, 118M parameters) fine-tuned as a pair classifier
on "name ; address" vs "name ; address". Unlike the hand-made features it can learn typos,
transliterations, DBA rewrites and name-only records directly from text, in any script.

Only pairs whose stage-1 probability is uncertain (between --lo and --hi) are scored; the rest
are already decided. The selection uses a fixed probability file (--sel, default p1_sel.npy,
a frozen copy of p1) so this stage can run while stage 1 is being retrained.

Cross-fitted like stage 1: model A (trained on half A) scores half B and test, model B scores
half A and test; test gets the average. Unscored pairs are NaN.

Output: <work>/<universe>/ce.npy (float32, row-aligned with features.parquet)
"""
import argparse
import math
import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from train import read_cols

BACKBONE = "intfloat/multilingual-e5-small"


def record_text(work, universe):
    r = pd.read_parquet(os.path.join(work, universe, "records.parquet"),
                        columns=["business_name", "business_address"])
    return (r.business_name + " ; " + r.business_address).to_numpy(dtype=object)


def selected(work, universe, sel_name, lo, hi, limit=None):
    """Rows of features.parquet to score, plus their (q, s1, label). Only those rows stay in memory."""
    K = read_cols(os.path.join(work, universe, "features.parquet"),
                  ["q", "s1"] + (["label"] if universe != "test" else []))
    p = np.load(os.path.join(work, universe, sel_name))
    rows = np.flatnonzero((p >= lo) & (p <= hi))
    if limit:
        rows = rows[:limit]
    n = len(K)
    K = K.iloc[rows].reset_index(drop=True)
    return n, K, rows


def batches(text, q, s, y, bs, tok, max_len, shuffle, seed=0):
    order = np.random.default_rng(seed).permutation(len(q)) if shuffle else np.arange(len(q))
    for i in range(0, len(order), bs):
        j = order[i:i + bs]
        enc = tok(text[s[j]].tolist(), text[q[j]].tolist(), truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt")
        yield j, enc, (torch.tensor(y[j], dtype=torch.float32) if y is not None else None)


def train_model(text, q, s, y, args, dev, name):
    tok = AutoTokenizer.from_pretrained(args.backbone)
    model = AutoModelForSequenceClassification.from_pretrained(args.backbone, num_labels=1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = math.ceil(len(q) / args.batch) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    t0 = time.time()
    stream = (b for ep in range(args.epochs)
              for b in batches(text, q, s, y, args.batch, tok, args.max_len, True, seed=ep))
    for k, (_, enc, yb) in enumerate(stream):
        enc = {a: b.to(dev) for a, b in enc.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            logit = model(**enc).logits.squeeze(-1)
        loss = lossf(logit.float(), yb.to(dev))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if k % 200 == 0 or k == steps - 1:
            done = (k + 1) * args.batch
            print(f"   [{name}] step {k + 1}/{steps} loss {loss.item():.4f} "
                  f"({done / (time.time() - t0):,.0f} pairs/s)", flush=True)
    model.eval()
    return tok, model


@torch.inference_mode()
def score(tok, model, text, q, s, args, dev, name):
    out = np.empty(len(q), np.float32)
    t0 = time.time()
    bs = args.batch * 4
    for k, (j, enc, _) in enumerate(batches(text, q, s, None, bs, tok, args.max_len, False)):
        enc = {a: b.to(dev) for a, b in enc.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            logit = model(**enc).logits.squeeze(-1)
        out[j] = torch.sigmoid(logit.float()).cpu().numpy()
        if k % 200 == 0:
            done = min((k + 1) * bs, len(q))
            print(f"   [{name}] scored {done:,}/{len(q):,} ({done / (time.time() - t0):,.0f} pairs/s)", flush=True)
    return out


def save(work, universe, n_rows, rows, values):
    arr = np.full(n_rows, np.nan, np.float32)
    arr[rows] = values
    path = os.path.join(work, universe, "ce.npy")
    np.save(path + ".tmp.npy", arr)
    os.replace(path + ".tmp.npy", path)   # atomic: refine never sees a half-written file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--sel", default="p1_sel.npy", help="probability file used to pick uncertain pairs")
    ap.add_argument("--lo", type=float, default=0.01)
    ap.add_argument("--hi", type=float, default=0.99)
    ap.add_argument("--max-train", type=int, default=1_500_000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--debug-limit", type=int, default=None, help="score only the first N selected pairs")
    ap.add_argument("--backbone", default=BACKBONE, help="e.g. intfloat/multilingual-e5-base (MIT)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--pseudo-file", default=None, help="test pairs with q, s1, label (pseudo.py --out ...)")
    ap.add_argument("--pseudo-max", type=int, default=400_000)
    args = ap.parse_args()
    w = args.work_dir
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    data = {}
    for u in ("trainA", "trainB", "test"):
        n, K, rows = selected(w, u, args.sel, args.lo, args.hi, args.debug_limit)
        data[u] = (n, K, rows, record_text(w, u))
        print(f"[{u}] {len(rows):,} uncertain pairs of {n:,} to score", flush=True)

    pseudo = None
    if args.pseudo_file:  # confident French test pairs: lets the text model see the unseen country
        P = pd.read_parquet(os.path.join(w, args.pseudo_file), columns=["q", "s1", "label"])
        if len(P) > args.pseudo_max:
            P = P.sample(args.pseudo_max, random_state=3)
        pseudo = (P.q.to_numpy(), P.s1.to_numpy(), P.label.to_numpy().astype(np.float32))
        print(f"adding {len(P):,} pseudo-labelled test pairs ({pseudo[2].mean():.1%} positive) to training")
    test_scores = []
    for train_u, other_u in (("trainA", "trainB"), ("trainB", "trainA")):
        _, K, rows, text = data[train_u]
        rng = np.random.default_rng(1)
        tr = np.arange(len(K)) if len(K) <= args.max_train else np.sort(rng.choice(len(K), args.max_train, replace=False))
        q, s, y = K.q.to_numpy()[tr], K.s1.to_numpy()[tr], K.label.to_numpy()[tr].astype(np.float32)
        if pseudo is not None:  # test texts are appended after the train texts, indices offset
            text = np.concatenate([text, data["test"][3]])
            off = len(data[train_u][3])
            q, s, y = np.r_[q, pseudo[0] + off], np.r_[s, pseudo[1] + off], np.r_[y, pseudo[2]]
        print(f"[ce_{train_u}] training on {len(q):,} pairs ({y.mean():.1%} positive) on {dev}, "
              f"{args.backbone}, {args.epochs} epoch(s)", flush=True)
        tok, model = train_model(text, q, s, y, args, dev, f"ce_{train_u}")

        n2, K2, rows2, text2 = data[other_u]
        v = score(tok, model, text2, K2.q.to_numpy(), K2.s1.to_numpy(), args, dev, f"ce_{train_u}->{other_u}")
        yv = K2.label.to_numpy()
        pv = np.load(os.path.join(w, other_u, args.sel))[rows2]
        ll = lambda p: float(-np.mean(yv * np.log(np.clip(p, 1e-6, 1)) + (1 - yv) * np.log(np.clip(1 - p, 1e-6, 1))))
        print(f"[ce_{train_u}] on {other_u} uncertain pairs: logloss cross-encoder {ll(v):.4f} vs stage-1 {ll(pv):.4f}",
              flush=True)
        save(w, other_u, n2, rows2, v)

        _, KT, _, textT = data["test"]
        test_scores.append(score(tok, model, textT, KT.q.to_numpy(), KT.s1.to_numpy(), args, dev,
                                 f"ce_{train_u}->test"))
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()

    nT, _, rowsT, _ = data["test"]
    save(w, "test", nT, rowsT, np.mean(test_scores, axis=0))
    print("cross-encoder scores written for trainA, trainB, test")


if __name__ == "__main__":
    main()
