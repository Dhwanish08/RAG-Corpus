"""Defenses: sit between retrieve and answer. Evaluated separately.

manifest: origin check. sha256 of every genuine chunk is recorded at ingest
  (data/manifest.txt); a retrieved chunk whose hash is not in it was injected
  afterwards. Beats padding because it ignores how the text looks.

provenance: ingest-format check. Genuine chunks come out of rag.chunk(), which
  merges paragraphs up to ~200 words, so a retrieved chunk far shorter than that
  did not come from the normal ingest path. Drop those, then answer from the rest.
voting: score-outlier filter, then answer from each chunk on its own and take the
  majority answer (isolate-then-aggregate).
neardup: copy detection. Two retrieved chunks of the SAME source that share most of their
  5-word phrases are two versions of one passage (e.g. a genuine chunk and a tampered copy
  of it); the text cannot say which is true, so abstain.
earliest: same detection, but trust the index's own ingest order (seq, stamped by Index.add,
  never by the writer): keep the earliest version of each near-duplicate group.
manifest_preingest: the manifest regenerated AFTER the poison was ingested; it already holds
  the poison's hash, so nothing is filtered (the manifest's stated blind spot, by construction).

None reads the `poison` flag -- only text, source, retrieval score and ingest order.

ponytail: MIN_WORDS=40 sits near the 1st percentile of clean corpus chunk lengths
(29) but was set after seeing that all 500 poison docs are <= 37 words, so its 100%
catch rate reflects how the poison was written. An attacker who pads poison past the
threshold defeats it; upgrade path is a content-hash manifest of trusted chunks.
Voting fails once poison is the majority of the top-5 and cuts clean accuracy because
single chunks often cannot answer alone.
"""
import os
import re
import statistics
from functools import partial

from attack import ABSTAIN_TEXT, DATA, OVERFETCH, _norm, ask_one, cached_answer
from rag import load_corpus, text_hash, write_manifest

MIN_WORDS = 40  # provenance: shorter chunks are not ingest-format
Z_DROP = 1.5  # voting: drop chunks scoring more than this many std above the top-5 mean
MIN_KEEP = 2  # never filter below this many chunks
NEARDUP = 0.5  # same-source chunks whose 5-gram containment is at least this are near-duplicates
# Chosen from CLEAN data only: over all 70,987 same-article genuine pairs the highest containment is 0.243
# (Jaccard 0.124), so 0.5 is about twice the clean maximum. Cross-article genuine chunks can overlap far
# more (Darmstadtium/Dubnium 0.978), which is why the test is restricted to one source. test_r2 re-checks it.
SHINGLE = 5


def ingest_filter(docs: list[dict], min_words: int = MIN_WORDS) -> list[dict]:
    keep = [d for d in docs if len(d["text"].split()) >= min_words]
    return keep or docs  # nothing passes (not seen with 4x over-fetch): fall back rather than answer from nothing


_manifest = None


def manifest_filter(docs: list[dict]) -> list[dict]:
    """Keep only chunks whose sha256 is in the ingest-time manifest (data/manifest.txt).
    Judges origin, not appearance, so padding or paraphrasing poison cannot get it past.
    Never reads the poison flag.

    ponytail: the manifest is built from the corpus on first use if the file is missing;
    in a real deployment it is written once at ingest and kept where the attacker can't
    write. Poison present before ingest is hashed in and passes.
    """
    global _manifest
    if _manifest is None:
        path = f"{DATA}/manifest.txt"
        if not os.path.exists(path):
            write_manifest(load_corpus(f"{DATA}/corpus.jsonl"), path)
        with open(path, encoding="utf-8") as f:
            _manifest = set(f.read().split())
    keep = [d for d in docs if text_hash(d["text"]) in _manifest]
    return keep or docs  # same fallback as ingest_filter


def _shingles(text: str) -> set:
    w = re.findall(r"\w+", text.lower())
    return {tuple(w[i:i + SHINGLE]) for i in range(max(len(w) - SHINGLE + 1, 1))}


def _near_dup(a: set, b: set) -> bool:
    """Containment: share of the smaller chunk's phrases found in the other (catches a copy with one fact swapped,
    and a short doc padded with sentences copied from a genuine chunk)."""
    return len(a & b) / min(len(a), len(b)) >= NEARDUP


def neardup_pairs(docs: list[dict]) -> list[tuple[int, int]]:
    sh = [_shingles(d["text"]) for d in docs]
    return [(i, j) for i in range(len(docs)) for j in range(i + 1, len(docs))
            if docs[i]["source"] == docs[j]["source"] and _near_dup(sh[i], sh[j])]


def earliest_filter(docs: list[dict]) -> list[dict]:
    """Of each group of same-source near-duplicates keep only the earliest-ingested chunk (lowest seq).
    Survivors come back in the original best-first order.

    ponytail: trusts seq. Index.add stamps it, so a writer that only supplies text and source cannot backdate;
    an attacker who can forge the stamp (or tamper before ingest, when no earlier original exists) defeats it.
    Greedy O(n^2) over the 20 candidates.
    """
    kept = []
    for d in sorted(docs, key=lambda d: d["seq"]):
        s = _shingles(d["text"])
        if not any(k["source"] == d["source"] and _near_dup(s, ks) for k, ks in kept):
            kept.append((d, s))
    keep = {id(k) for k, _ in kept}
    return [d for d in docs if id(d) in keep]


def manifest_preingest_filter(docs: list[dict]) -> list[dict]:
    """A manifest built after the poison was ingested contains the poison's hash, so every chunk passes."""
    return docs


# graded strength: provNN applies NN% of the full 40-word cutoff; "provenance" is the 100% level
FILTERS = {
    **{f"prov{pct}": partial(ingest_filter, min_words=MIN_WORDS * pct // 100) for pct in (25, 50, 75)},
    "provenance": ingest_filter,
    "manifest": manifest_filter,
    "manifest_preingest": manifest_preingest_filter,
    "earliest": earliest_filter,
}


def _filtered(f):
    """Answer from the best 5 chunks that survive filter f (docs = 4x over-fetched candidates, best first)."""
    return lambda cache, question, docs: cached_answer(cache, question, f(docs)[:5])


def filter_outliers(docs: list[dict]) -> list[dict]:
    scores = [d["score"] for d in docs]
    if len(docs) <= MIN_KEEP:
        return docs
    mu, sd = statistics.mean(scores), statistics.pstdev(scores)
    if sd < 1e-6:
        return docs
    keep = [d for d in docs if (d["score"] - mu) / sd <= Z_DROP]
    return keep if len(keep) >= MIN_KEEP else docs


def _is_idk(a: str) -> bool:
    return "don t know" in _norm(a) or "do not know" in _norm(a) or not a.strip()


def _agree(a: str, b: str) -> bool:
    na, nb = f" {_norm(a)} ", f" {_norm(b)} "
    return na in nb or nb in na


def defended_answer(cache, question: str, docs: list[dict]) -> str:
    docs = filter_outliers(docs)
    per_chunk = [(d, cached_answer(cache, question, [d])) for d in docs]  # docs arrive best-score first
    voters = [(d, a) for d, a in per_chunk if not _is_idk(a)]
    if not voters:
        return cached_answer(cache, question, docs)  # no chunk answered alone; fall back to joint context
    groups: list[list[tuple]] = []
    for d, a in voters:
        for g in groups:
            if _agree(a, g[0][1]):
                g.append((d, a))
                break
        else:
            groups.append([(d, a)])
    # most votes wins; ties go to the group holding the best-scoring chunk (docs are sorted, so the earliest group)
    best = max(groups, key=lambda g: (len(g), -groups.index(g)))
    return min((a for _, a in best), key=len)  # shortest phrasing in the winning group


def neardup_answer(cache, question: str, docs: list[dict]) -> str:
    """Abstain if the 4x over-fetched candidates hold two versions of one passage, else answer from the best 5.
    Cost: one injected copy is enough to make the question unanswerable (the attack becomes denial of service)."""
    if neardup_pairs(docs):
        return ABSTAIN_TEXT
    return cached_answer(cache, question, docs[:5])


DEFENSES = {**{n: _filtered(f) for n, f in FILTERS.items()}, "neardup": neardup_answer, "voting": defended_answer}


ALL_DEFENSES = ["none", "prov25", "prov50", "prov75", "provenance", "manifest", "manifest_preingest", "neardup", "earliest", "voting"]  # display order; "none" = undefended reader


def answer_all(idx, row, poison, cache, variants=(), style=""):
    """Run every defense on the SAME retrieval and yield (name, answer) one at a time (the demo's single call).
    The 20 candidates are retrieved once; each defense gets what eval_config gives it (all 20 for the over-fetching
    ones, the best 5 for the undefended reader and voting), so the answers share cache keys with the sweeps."""
    top, _ = ask_one(idx, row, poison, cache, variants, lambda c, q, d: "", 4, style)
    for name in ALL_DEFENSES:
        fn = cached_answer if name == "none" else DEFENSES[name]
        yield name, fn(cache, row["question"], top if name in OVERFETCH else top[:5])
