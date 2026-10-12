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
    print("all tests passed")
