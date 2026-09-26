"""Stage 1b (GPU): multilingual sentence embeddings of every name and address.

Model: intfloat/multilingual-e5-small (MIT licence, 118M parameters). It reads Hindi, Telugu,
Malayalam, French, ... directly, so the RAW text is encoded (no transliteration needed), which
gives a second, script-independent view next to the TF-IDF / fuzzy features.

Output per universe (row order = records.parquet):
    <work>/<universe>/emb_name.npy, emb_addr.npy   float16, L2-normalised, shape (n, 384)
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

MODEL = "intfloat/multilingual-e5-small"


def encode(model, texts, batch, chunk, out_path):
    """Encode in chunks straight into a .npy memmap so memory stays flat."""
    dim = model.get_sentence_embedding_dimension()
    out = np.lib.format.open_memmap(out_path + ".tmp.npy", mode="w+", dtype=np.float16, shape=(len(texts), dim))
    t0 = time.time()
    for i in range(0, len(texts), chunk):
        part = ["query: " + t if t else "query: -" for t in texts[i:i + chunk]]
        with torch.inference_mode():
            e = model.encode(part, batch_size=batch, convert_to_numpy=True, normalize_embeddings=True,
                             show_progress_bar=False)
        out[i:i + len(part)] = e.astype(np.float16)
        done = i + len(part)
        rate = done / (time.time() - t0)
        print(f"   {os.path.basename(out_path)} {done:,}/{len(texts):,} ({rate:,.0f}/s, "
              f"eta {(len(texts) - done) / rate / 60:.1f} min)", flush=True)
    out.flush()
    del out
    os.replace(out_path + ".tmp.npy", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--universes", default="trainA,trainB,test")
    ap.add_argument("--model", default=MODEL, help="model name or path to a fine-tuned copy")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--chunk", type=int, default=500_000)
    ap.add_argument("--max-len", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.model, device=device)
    model.max_seq_length = args.max_len
    if device == "cuda":
        model.half()
    print(f"encoding with {args.model} on {device}")

    for u in args.universes.split(","):
        d = os.path.join(args.work_dir, u)
        recs = pd.read_parquet(os.path.join(d, "records.parquet"), columns=["business_name", "business_address"])
        for col, name in [("business_name", "emb_name.npy"), ("business_address", "emb_addr.npy")]:
            path = os.path.join(d, name)
            if os.path.exists(path) and np.load(path, mmap_mode="r").shape[0] == len(recs):
                print(f"[{u}] {name} exists, skipping")
                continue
            print(f"[{u}] encoding {col} of {len(recs):,} records")
            encode(model, recs[col].tolist(), args.batch, args.chunk, path)


if __name__ == "__main__":
    main()
