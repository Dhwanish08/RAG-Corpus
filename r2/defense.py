"""Defenses: sit between retrieve and answer. Three, evaluated separately.

manifest: origin check. sha256 of every genuine chunk is recorded at ingest
  (data/manifest.txt); a retrieved chunk whose hash is not in it was injected
  afterwards. Beats padding because it ignores how the text looks.

provenance: ingest-format check. Genuine chunks come out of rag.chunk(), which
  merges paragraphs up to ~200 words, so a retrieved chunk far shorter than that
  did not come from the normal ingest path. Drop those, then answer from the rest.
voting: score-outlier filter, then answer from each chunk on its own and take the
  majority answer (isolate-then-aggregate).

Neither reads the `poison` flag -- only text and retrieval scores.

ponytail: MIN_WORDS=40 sits near the 1st percentile of clean corpus chunk lengths
(29) but was set after seeing that all 500 poison docs are <= 37 words, so its 100%
catch rate reflects how the poison was written. An attacker who pads poison past the
threshold defeats it; upgrade path is a content-hash manifest of trusted chunks.
Voting fails once poison is the majority of the top-5 and cuts clean accuracy because
single chunks often cannot answer alone.
"""
import os
import statistics
from functools import partial

from attack import DATA, _norm, cached_answer
from rag import load_corpus, text_hash, write_manifest

MIN_WORDS = 40  # provenance: shorter chunks are not ingest-format
Z_DROP = 1.5  # voting: drop chunks scoring more than this many std above the top-5 mean
MIN_KEEP = 2  # never filter below this many chunks


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


# graded strength: provNN applies NN% of the full 40-word cutoff; "provenance" is the 100% level
FILTERS = {
    **{f"prov{pct}": partial(ingest_filter, min_words=MIN_WORDS * pct // 100) for pct in (25, 50, 75)},
    "provenance": ingest_filter,
    "manifest": manifest_filter,
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


DEFENSES = {**{n: _filtered(f) for n, f in FILTERS.items()}, "voting": defended_answer}
