"""Runnable self-check for the RAG pipeline. No framework, just asserts.

Run: python test_r2.py
"""
import json

from rag import Index, chunk, load_corpus

DATA = "../data"


def test_chunk_merges_short_paragraphs():
    text = "First short line.\n\nSecond short line.\n\n" + ("word " * 250)
    chunks = chunk(text, title="T")
    # short paragraphs merge into one chunk; the long one becomes its own
    assert len(chunks) == 2, chunks
    assert all(c.startswith("T: ") for c in chunks)


def test_retrieve_finds_the_relevant_chunk():
    docs = [
        {"text": "Paris is the capital of France.", "source": "a", "poison": False},
        {"text": "The Eiffel Tower was completed in 1889.", "source": "a", "poison": False},
        {"text": "Bananas are a good source of potassium.", "source": "b", "poison": False},
    ]
    idx = Index.build(docs)
    top = idx.retrieve("What is the capital of France?", k=1)
    assert top[0]["text"] == docs[0]["text"], top


def test_add_injects_into_live_index():
    docs = [{"text": "The sky is blue.", "source": "a", "poison": False}]
    idx = Index.build(docs)
    assert len(idx.docs) == 1 and idx.vecs.shape[0] == 1

    poison = [{"text": "The capital of France is Berlin.", "source": "poison", "poison": True}]
    idx.add(poison)
    assert len(idx.docs) == 2 and idx.vecs.shape[0] == 2

    top = idx.retrieve("What is the capital of France?", k=1)
    assert top[0]["poison"] is True, "poison doc should win retrieval for its targeted question"


def test_gold_set_is_grounded_and_retrievable():
    """Every gold question must: cite a real article, quote its correct_answer
    verbatim from that article, actually retrieve that article in top-k, and
    have an attacker_target that doesn't already show up in what a clean query
    retrieves -- otherwise the question can't cleanly measure a poison flip.
    """
    corpus = {}
    with open(f"{DATA}/corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            corpus[rec["title"]] = rec["text"]

    idx = Index.build(load_corpus(f"{DATA}/corpus.jsonl"))

    with open(f"{DATA}/gold.jsonl", encoding="utf-8") as f:
        gold = [json.loads(line) for line in f]

    assert len(gold) == 50, f"expected 50 gold questions, got {len(gold)}"
    assert len(gold) == len({g["id"] for g in gold}), "duplicate gold ids"

    for row in gold:
        title = row["source_title"]
        assert title in corpus, f"{row['id']}: unknown source_title {title!r}"
        assert row["correct_answer"].lower() in corpus[title].lower(), (
            f"{row['id']}: correct_answer not found verbatim in {title!r}"
        )
        assert row["correct_answer"] != row["attacker_target"], f"{row['id']}: target == answer"
        # an alias that contains the attacker's answer would score poisoned output as correct
        for a in row.get("aliases", []):
            assert row["attacker_target"].lower() not in a.lower(), f"{row['id']}: alias {a!r} contains target"

        top = idx.retrieve(row["question"], k=5)
        assert title in {d["source"] for d in top}, (
            f"{row['id']}: {title!r} not retrieved in top-5 for {row['question']!r}"
        )
        ctx = " ".join(d["text"] for d in top).lower()
        assert row["attacker_target"].lower() not in ctx, (
            f"{row['id']}: attacker_target {row['attacker_target']!r} already in clean top-5 context"
        )


def test_poison_set(path=f"{DATA}/poison.jsonl", expect_per_question=10):
    """Every poison doc, injected alone, must land in the top-5 for its
    question, state the attacker_target, and not contain the correct answer
    (so a flip is attributable to the poison). Collects all failures."""
    import os

    if not os.path.exists(path):
        return
    with open(f"{DATA}/gold.jsonl", encoding="utf-8") as f:
        gold = {g["id"]: g for g in map(json.loads, f)}
    with open(path, encoding="utf-8") as f:
        poison = [json.loads(line) for line in f]

    base = Index.build(load_corpus(f"{DATA}/corpus.jsonl"))
    bad = []
    seen = {}
    for p in poison:
        g = gold[p["id"]]
        seen.setdefault(p["id"], []).append(p["variant"])
        t = p["text"].lower()
        if g["attacker_target"].lower() not in t:
            bad.append((p["id"], p["variant"], "missing attacker_target"))
        for a in (g["correct_answer"], *g.get("aliases", [])):
            if a.lower() in t:
                bad.append((p["id"], p["variant"], f"contains correct answer/alias {a!r}"))
        idx = Index(base.docs, base.vecs)
        idx.add([{"text": p["text"], "source": g["source_title"], "poison": True}])
        if not any(d["poison"] for d in idx.retrieve(g["question"], k=5)):
            bad.append((p["id"], p["variant"], "not in top-5"))
    for qid, variants in seen.items():
        assert sorted(variants) == list(range(expect_per_question)), f"{qid}: variants {sorted(variants)}"
    assert not bad, f"{len(bad)} bad poison docs: {bad}"


def test_pad_poison():
    """Padded poison must clear the provenance length filter and still never state the correct answer."""
    from attack import match, pad_poison
    from defense import MIN_WORDS

    docs = load_corpus(f"{DATA}/corpus.jsonl")
    with open(f"{DATA}/gold.jsonl", encoding="utf-8") as f:
        gold = [json.loads(line) for line in f]
    with open(f"{DATA}/poison.jsonl", encoding="utf-8") as f:
        poison = [json.loads(line) for line in f]
    by_id = {g["id"]: g for g in gold}
    for p in poison:
        g = by_id[p["id"]]
        t = pad_poison(p["text"], g, docs, p["variant"])
        assert len(t.split()) >= MIN_WORDS, (p["id"], p["variant"], len(t.split()))
        assert g["attacker_target"].lower() in t.lower(), (p["id"], p["variant"], "lost target")
        assert not any(match(t, a) for a in (g["correct_answer"], *g.get("aliases", []))), (p["id"], p["variant"], "padding states the answer")


def test_manifest():
    """Every genuine chunk passes the manifest; every poison doc is rejected, as a stub and padded."""
    from attack import pad_poison
    from defense import manifest_filter

    docs = load_corpus(f"{DATA}/corpus.jsonl")
    assert len(manifest_filter(docs)) == len(docs), "a genuine chunk was rejected"
    with open(f"{DATA}/gold.jsonl", encoding="utf-8") as f:
        by_id = {g["id"]: g for g in map(json.loads, f)}
    with open(f"{DATA}/poison.jsonl", encoding="utf-8") as f:
        poison = [json.loads(line) for line in f]
    for p in poison:
        g = by_id[p["id"]]
        for t in (p["text"], pad_poison(p["text"], g, docs, p["variant"])):
            fake = {"text": t, "source": g["source_title"], "poison": True}
            # a lone doc would hit the keep-or-docs fallback, so pair it with a genuine chunk
            assert manifest_filter([fake, docs[0]]) == [docs[0]], (p["id"], p["variant"], "poison passed")


def test_graded_levels():
    """Stricter provenance levels never keep more chunks, and the full level drops every stub."""
    from defense import FILTERS

    clean = load_corpus(f"{DATA}/corpus.jsonl")
    with open(f"{DATA}/poison.jsonl", encoding="utf-8") as f:
        stubs = [{"text": p["text"], "source": "x", "poison": True} for p in map(json.loads, f)]
    kept = [len(FILTERS[n](clean + stubs)) for n in ("prov25", "prov50", "prov75", "provenance")]
    assert kept == sorted(kept, reverse=True), kept
    assert not any(d["poison"] for d in FILTERS["provenance"](clean + stubs)), "a stub survived the 40-word cutoff"


def test_seq_stamps():
    """The index stamps ingest order itself; a later add is always later, even on a sliced index."""
    mk = lambda t: {"text": t, "source": "x", "poison": False}
    idx = Index.build([mk("a b"), mk("c d")])
    assert [d["seq"] for d in idx.docs] == [0, 1]
    idx.add([mk("e f"), {**mk("g h"), "seq": -5}])  # a writer-supplied seq is overwritten
    assert [d["seq"] for d in idx.docs] == [0, 1, 2, 3]
    sub = Index(idx.docs[1:2], idx.vecs[1:2])
    sub.add([mk("i j")])
    assert sub.docs[-1]["seq"] == 2


def test_tamper():
    """Tampered copies: all 50 x 10 name the target, never state the answer or an alias, and overlap a genuine chunk."""
    from attack import match, tamper
    from defense import _near_dup, _shingles

    docs = load_corpus(f"{DATA}/corpus.jsonl")
    with open(f"{DATA}/gold.jsonl", encoding="utf-8") as f:
        gold = [json.loads(line) for line in f]
    for g in gold:
        own = [_shingles(d["text"]) for d in docs if d["source"] == g["source_title"]]
        answers = (g["correct_answer"], *g.get("aliases", []))
        for v in range(10):
            t = tamper(g, docs, v)
            assert g["attacker_target"].lower() in t.lower(), (g["id"], v, "lost target")
            assert not any(match(t, a) for a in answers), (g["id"], v, "states the answer")
            assert any(_near_dup(_shingles(t), o) for o in own), (g["id"], v, "not a near-duplicate of a genuine chunk")


def test_neardup_calibration():
    """Genuine chunks of one article never look like near-duplicates (so the threshold costs nothing on clean data),
    a tampered copy does, and the two defenses act on it."""
    from collections import defaultdict

    from attack import ABSTAIN_TEXT, tamper
    from defense import DEFENSES, FILTERS, NEARDUP, _shingles, neardup_pairs

    docs = load_corpus(f"{DATA}/corpus.jsonl")
    by = defaultdict(list)
    for d in docs:
        by[d["source"]].append(_shingles(d["text"]))
    worst = max(len(a & b) / min(len(a), len(b)) for sh in by.values() for i, a in enumerate(sh) for b in sh[i + 1:])
    assert worst < NEARDUP, (worst, NEARDUP)
    print(f"clean same-article max containment {worst:.3f} < NEARDUP {NEARDUP}")

    with open(f"{DATA}/gold.jsonl", encoding="utf-8") as f:
        g = json.loads(f.readlines()[2])  # q003
    genuine = [{**d, "seq": i} for i, d in enumerate(docs)]
    original = next(d for d in genuine if d["source"] == g["source_title"] and g["correct_answer"] in d["text"])
    copy = {"text": tamper(g, docs, 0), "source": g["source_title"], "poison": True, "seq": 10_000}
    other = next(d for d in genuine if d["source"] != g["source_title"])
    cands = [copy, original, other]  # best-first, copy ranked on top
    assert neardup_pairs(cands) == [(0, 1)]
    assert FILTERS["earliest"](cands) == [original, other], "earliest keeps the original, drops the later copy"
    assert FILTERS["earliest"]([original, other]) == [original, other]
    assert neardup_pairs([original, other]) == []
    assert DEFENSES["neardup"]({}, g["question"], cands) == ABSTAIN_TEXT


def test_manifest_preingest():
    """A manifest regenerated after the poison was ingested admits it: the pre-ingest setting is the undefended row."""
    from defense import FILTERS
    from rag import text_hash

    clean = load_corpus(f"{DATA}/corpus.jsonl")
    with open(f"{DATA}/poison.jsonl", encoding="utf-8") as f:
        stubs = [{"text": json.loads(l)["text"], "source": "x", "poison": True, "seq": 0} for l in f]
    manifest = {text_hash(d["text"]) for d in clean + stubs}
    assert all(text_hash(s["text"]) in manifest for s in stubs)
    assert FILTERS["manifest_preingest"](stubs[:3]) == stubs[:3]


def test_answer_all():
    """answer_all runs every defense, in order, on one retrieval; with a tampered copy retrieved next to its
    original, neardup abstains and earliest answers. A stub reader keeps this offline."""
    import attack
    from defense import ALL_DEFENSES, answer_all

    with open(f"{DATA}/gold.jsonl", encoding="utf-8") as f:
        row = json.loads(f.readlines()[2])  # q003
    idx = Index.build(load_corpus(f"{DATA}/corpus.jsonl"))
    real, attack.answer = attack.answer, (lambda q, docs: "FAKE")
    try:
        out = list(answer_all(idx, row, {}, {}, [0, 1, 2], "tampered"))
    finally:
        attack.answer = real
    assert [n for n, _ in out] == ALL_DEFENSES
    ans = dict(out)
    assert all(isinstance(a, str) for a in ans.values())
    assert ans["neardup"] == attack.ABSTAIN_TEXT and ans["earliest"] == "FAKE" and ans["none"] == "FAKE"


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:  # python test_r2.py <poison file> -> check only that file
        test_poison_set(sys.argv[1])
        print("poison file ok")
        sys.exit()
    test_chunk_merges_short_paragraphs()
    test_retrieve_finds_the_relevant_chunk()
    test_add_injects_into_live_index()
    test_gold_set_is_grounded_and_retrievable()
    test_poison_set()
    test_pad_poison()
    test_manifest()
    test_graded_levels()
    test_seq_stamps()
    test_tamper()
    test_neardup_calibration()
    test_manifest_preingest()
    test_answer_all()
    print("all tests passed")
