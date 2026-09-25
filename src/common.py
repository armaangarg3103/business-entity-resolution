"""Shared helpers: file IO, text normalization, and the challenge metric (macro F0.5)."""
import os
import re
from multiprocessing import Pool

import numpy as np
import pandas as pd
from anyascii import anyascii

# Worker threads/processes. Inside Kubernetes os.cpu_count() reports the whole host,
# so set NUM_THREADS to the pod's real CPU limit.
N_THREADS = int(os.environ.get("NUM_THREADS", os.cpu_count()))

# ----------------------------------------------------------------------------- IO

def read_tsv(path):
    """Read a challenge TSV as plain strings (empty cells stay '' instead of NaN)."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


# ----------------------------------------------------------------------------- normalization
# Generic, country-independent rewrite rules. Everything maps to one canonical short form,
# applied identically to both sides of a pair, so direction does not matter.
ABBREV = {
    # street types (English + French)
    "street": "st", "saint": "st", "road": "rd", "avenue": "ave", "av": "ave", "drive": "dr",
    "boulevard": "blvd", "bd": "blvd", "bld": "blvd", "lane": "ln", "court": "ct", "place": "pl",
    "square": "sq", "highway": "hwy", "parkway": "pkwy", "circle": "cir", "terrace": "ter",
    "trail": "trl", "suite": "ste", "apartment": "apt", "floor": "fl", "building": "bldg",
    "north": "n", "south": "s", "east": "e", "west": "w", "rue": "r", "route": "rte",
    "chemin": "ch", "allee": "all", "impasse": "imp", "nagar": "ngr", "marg": "mg",
    # legal forms
    "private": "pvt", "limited": "ltd", "corporation": "corp", "incorporated": "inc",
    "company": "co", "cie": "co", "and": "&", "et": "&",
}
# address filler tokens that carry no identity
ADDR_DROP = {"no", "number", "door", "h", "house", "hno", "flat", "unit", "near", "opp", "po", "box", "cdp"}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
    "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "tennessee": "tn",
    "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa", "wisconsin": "wi",
    "wyoming": "wy",
}
TOKEN_MAP = {**US_STATES, **ABBREV}

_NON_ALNUM = re.compile(r"[^a-z0-9&]+")
_DIGITS = re.compile(r"\d+")
_DBA = re.compile(r"\b(?:doing business as|d/?b/?a|formerly(?: known as)?|f/?k/?a|a/?k/?a|trading as|t/a)\b|\|")
_DOMAIN = re.compile(r"^\s*(?:www\.)?([a-z0-9\-]+)\.(?:com|in|net|org|co|fr|io|biz|us|info)\b")


def _ascii_lower(s):
    if not s.isascii():
        s = anyascii(s)
    return s.lower()


def norm_tokens(s, drop=()):
    """ascii-fold, lowercase, strip punctuation, canonicalize abbreviations."""
    s = _ascii_lower(s).replace("&", " & ")
    out = []
    for t in _NON_ALNUM.sub(" ", s).split():
        if t.isdigit():
            t = t.lstrip("0") or "0"
        t = TOKEN_MAP.get(t, t)
        if t not in drop:
            out.append(t)
    return out


def normalize_record(name, addr):
    """Return the derived text fields for one record (pure function, used in a process pool)."""
    low = _ascii_lower(name)
    ntoks = norm_tokens(name)
    parts = [p for p in _DBA.split(low) if p and p.strip()]
    alt = " ".join(norm_tokens(parts[-1])) if len(parts) > 1 else ""
    m = _DOMAIN.match(low)
    atoks = norm_tokens(addr, ADDR_DROP)
    nums = sorted({t.lstrip("0") or "0" for t in _DIGITS.findall(addr)})
    return (" ".join(ntoks), alt, int(m is not None), " ".join(atoks), " ".join(nums),
            int(not name.isascii()), int(addr.strip() == ""))


def _norm_chunk(rows):
    return [normalize_record(n, a) for n, a in rows]


def normalize_frame(df, workers=None):
    """Add normalized columns to a records frame, in parallel."""
    rows = list(zip(df.business_name.tolist(), df.business_address.tolist()))
    workers = workers or N_THREADS
    size = max(1, len(rows) // (workers * 8) + 1)
    chunks = [rows[i:i + size] for i in range(0, len(rows), size)]
    with Pool(workers) as pool:
        res = [r for part in pool.map(_norm_chunk, chunks) for r in part]
    cols = ["name_n", "name_alt", "is_domain", "addr_n", "nums", "name_nonascii", "addr_empty"]
    out = pd.DataFrame(res, columns=cols, index=df.index)
    for c in ["is_domain", "name_nonascii", "addr_empty"]:
        out[c] = out[c].astype("int8")
    return pd.concat([df, out], axis=1)


def add_core_names(df, df_frac=0.005):
    """name_core = name tokens minus the generic words of that country.

    Generic words (legal forms like inc/pvt/sarl, filler like services/group) are discovered
    automatically as the tokens appearing in more than `df_frac` of that country's names,
    so a new country (e.g. France) gets its own list without any hand-written rules.
    """
    core = pd.Series("", index=df.index, dtype=object)
    generic_by_country = {}
    for country, g in df.groupby("country"):
        toks = g.name_n.str.split()
        dfreq = toks.apply(lambda t: list(set(t))).explode().value_counts()
        generic = set(dfreq[dfreq > df_frac * len(g)].index) | {"&"}
        generic_by_country[country] = generic
        core.loc[g.index] = [" ".join([t for t in ts if t not in generic]) or " ".join(ts) for ts in toks]
    df["name_core"] = core
    df["name_sq"] = df.name_core.str.replace(" ", "", regex=False)
    return df, generic_by_country


# ----------------------------------------------------------------------------- metric

def macro_f05(truth, pred, s1_ids):
    """Challenge metric. truth/pred: dict s1_id -> set of matched ids. Averaged over s1_ids."""
    tot = 0.0
    for s in s1_ids:
        t = truth.get(s, set())
        p = pred.get(s, set())
        if not t and not p:
            tot += 1.0
            continue
        tp = len(t & p)
        if tp == 0:
            continue
        pr, rc = tp / len(p), tp / len(t)
        tot += 1.25 * pr * rc / (0.25 * pr + rc)
    return tot / max(1, len(s1_ids))


def macro_f05_vec(s1_all, truth_pairs, pred_pairs):
    """Vectorized macro F0.5. *_pairs: DataFrames with columns s1, q."""
    n_true = truth_pairs.groupby("s1").size()
    n_pred = pred_pairs.groupby("s1").size()
    tp = pred_pairs.merge(truth_pairs, on=["s1", "q"]).groupby("s1").size()
    df = pd.DataFrame(index=pd.Index(s1_all, name="s1"))
    df["t"] = n_true.reindex(df.index).fillna(0).values
    df["p"] = n_pred.reindex(df.index).fillna(0).values
    df["tp"] = tp.reindex(df.index).fillna(0).values
    pr = np.divide(df.tp, df.p, out=np.zeros(len(df)), where=df.p > 0)
    rc = np.divide(df.tp, df.t, out=np.zeros(len(df)), where=df.t > 0)
    den = 0.25 * pr + rc
    f = np.divide(1.25 * pr * rc, den, out=np.zeros(len(df)), where=den > 0)
    f[(df.t == 0) & (df.p == 0)] = 1.0
    return float(f.mean())


# ----------------------------------------------------------------------------- output

def write_id_lists(path, s1_ids, pairs, col_name):
    """Write one row per S1 id with a comma-joined, de-duplicated list (possibly empty)."""
    lists = pairs.groupby("s1")["q"].apply(lambda x: ",".join(dict.fromkeys(x)))
    out = pd.DataFrame({"source1_entity_id": s1_ids})
    out[col_name] = out.source1_entity_id.map(lists).fillna("")
    out.to_csv(path, sep="\t", index=False)
