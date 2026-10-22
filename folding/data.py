"""CATH 4.2 (Ingraham et al. 2019) chains -> sequences and CA traces, split by CATH topology.

Download (517 MB + 0.8 MB) into data/cath/:
  http://people.csail.mit.edu/ingraham/graph-protein-design/data/cath/chain_set.jsonl
  http://people.csail.mit.edu/ingraham/graph-protein-design/data/cath/chain_set_splits.json
"""
import json
import os
import pickle

import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AA)}
AA_X = len(AA)          # unknown / non-standard residue
AA_PAD = len(AA) + 1    # prefix padding
AA_VOCAB = len(AA) + 2

MAX_CA_GAP = 4.2        # consecutive CA farther apart than this = chain break


def encode_seq(seq):
    return np.array([AA_IDX.get(a, AA_X) for a in seq], dtype=np.int64)


def segments(seq, CA, min_len):
    """Contiguous runs with resolved CA and no chain break, as (seq_ids, CA) pairs."""
    ok = ~np.isnan(CA).any(-1)
    out, start = [], None
    for i in range(len(CA) + 1):
        good = i < len(CA) and ok[i] and (start is None or np.linalg.norm(CA[i] - CA[i - 1]) < MAX_CA_GAP)
        if good and start is None:
            start = i
        elif not good and start is not None:
            if i - start >= min_len:
                out.append((encode_seq(seq[start:i]), CA[start:i]))
            start = i if (i < len(CA) and ok[i]) else None
    return out


def load_cath(root="data/cath", min_len=30, cache=True):
    """{'train'|'validation'|'test': [dict(name, seq, ca, full)]}, one entry per resolved segment.

    full=True marks a segment that covers its whole chain (no missing or broken residues).
    """
    cache_path = os.path.join(root, f"cath_ca_min{min_len}.pkl")
    if cache and os.path.exists(cache_path):
        return pickle.load(open(cache_path, "rb"))
    splits = json.load(open(os.path.join(root, "chain_set_splits.json")))
    where = {n: k for k in ("train", "validation", "test") for n in splits[k]}
    out = {k: [] for k in ("train", "validation", "test")}
    with open(os.path.join(root, "chain_set.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            split = where.get(r["name"])
            if split is None:
                continue
            CA = np.array(r["coords"]["CA"], dtype=np.float64)
            segs = segments(r["seq"], CA, min_len)
            for s, ca in segs:
                out[split].append(dict(name=r["name"], seq=s, ca=ca, full=len(segs) == 1 and len(s) == len(r["seq"])))
    if cache:
        pickle.dump(out, open(cache_path, "wb"))
    return out
