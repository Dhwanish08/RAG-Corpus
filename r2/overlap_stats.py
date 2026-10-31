"""Overlap measurements behind the near-duplicate threshold (Table 7.9 of the report).
Run from r2/:  python -X utf8 overlap_stats.py   -> ../figures/overlap_stats.md

Measures 5-gram containment (|A & B| / min(|A|, |B|)) and Jaccard between chunks:
genuine chunks of one article, genuine chunks of different articles (nearest 10 by embedding),
and each attacker's text against the best-matching genuine chunk of its own article.
"""
import os
from collections import defaultdict

import numpy as np

from attack import DATA, load_jsonl, load_poison, pad_poison, tamper
from defense import NEARDUP, _shingles
from rag import Index, load_corpus

jac = lambda a, b: len(a & b) / len(a | b)
con = lambda a, b: len(a & b) / min(len(a), len(b))


def stats() -> dict:
    docs = load_corpus(f"{DATA}/corpus.jsonl")
    sh = [_shingles(d["text"]) for d in docs]
    by = defaultdict(list)
    for i, d in enumerate(docs):
        by[d["source"]].append(i)
    same = [(con(sh[i], sh[j]), jac(sh[i], sh[j])) for ids in by.values() for a, i in enumerate(ids) for j in ids[a + 1:]]
    idx = Index.build(docs)
    sim = idx.vecs @ idx.vecs.T
    np.fill_diagonal(sim, -1)
    near = np.argsort(-sim, axis=1)[:, :10]
    cross = [(con(sh[i], sh[j]), jac(sh[i], sh[j]), i, j) for i in range(len(docs)) for j in near[i] if docs[i]["source"] != docs[j]["source"]]
    worst = max(cross)
    out = {
        "same_pairs": len(same), "same_con_max": max(c for c, _ in same), "same_jac_max": max(j for _, j in same),
        "cross_con_max": worst[0], "cross_jac_max": max(c[1] for c in cross),
        "cross_pair": (docs[worst[2]]["source"], docs[worst[3]]["source"]), "threshold": NEARDUP,
    }
    gold, poison = load_jsonl(f"{DATA}/gold.jsonl"), load_poison()
    for name in ("tampered", "padded", "stub"):
        con_v, jac_v = [], []
        for g in gold:
            stub = poison[g["id"]][0]["text"]
            text = {"tampered": lambda: tamper(g, docs, 0), "padded": lambda: pad_poison(stub, g, docs, 0), "stub": lambda: stub}[name]()
            s = _shingles(text)
            own = [sh[k] for k in by[g["source_title"]]]
            con_v.append(max(con(s, o) for o in own))
            jac_v.append(max(jac(s, o) for o in own))
        out[name] = {"con": (min(con_v), float(np.median(con_v)), max(con_v)), "jac": (min(jac_v), float(np.median(jac_v)), max(jac_v))}
    return out


def table(o: dict) -> str:
    f = lambda t: f"{t[0]:.3f} / {t[1]:.3f} / {t[2]:.3f}"
    return "\n".join([
        "| texts compared | containment | Jaccard |", "|---|---|---|",
        f"| genuine vs genuine, same article ({o['same_pairs']:,} pairs) | max {o['same_con_max']:.3f} | max {o['same_jac_max']:.3f} |",
        f"| genuine vs genuine, different articles (nearest 10; worst {o['cross_pair'][0]} / {o['cross_pair'][1]}) | max {o['cross_con_max']:.3f} | max {o['cross_jac_max']:.3f} |",
        f"| tampered copy vs best genuine chunk of its article, min / median / max | {f(o['tampered']['con'])} | {f(o['tampered']['jac'])} |",
        f"| padded stub (60 words) vs best genuine chunk, min / median / max | {f(o['padded']['con'])} | {f(o['padded']['jac'])} |",
        f"| raw stub vs best genuine chunk, min / median / max | {f(o['stub']['con'])} | {f(o['stub']['jac'])} |",
        f"\nNear-duplicate threshold: containment >= {o['threshold']}, same source only. Attackers are variant 0 for each of the 50 questions.",
    ]) + "\n"


if __name__ == "__main__":
    o = stats()
    assert o["same_con_max"] < NEARDUP < o["tampered"]["con"][0], "threshold no longer separates genuine pairs from tampered copies"
    os.makedirs("../figures", exist_ok=True)
    with open("../figures/overlap_stats.md", "w", encoding="utf-8") as fh:
        fh.write(table(o))
    print(table(o))
