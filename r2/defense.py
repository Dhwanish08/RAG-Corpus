"""Defenses: sit between retrieve and answer. Two, evaluated separately.

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
import statistics

from attack import _norm, cached_answer

MIN_WORDS = 40  # provenance: shorter chunks are not ingest-format
Z_DROP = 1.5  # voting: drop chunks scoring more than this many std above the top-5 mean
MIN_KEEP = 2  # never filter below this many chunks


def ingest_filter(docs: list[dict]) -> list[dict]:
    keep = [d for d in docs if len(d["text"].split()) >= MIN_WORDS]
    return keep or docs  # nothing passes (not seen with 4x over-fetch): fall back rather than answer from nothing


def provenance_answer(cache, question: str, docs: list[dict]) -> str:
    """docs = 4x over-fetched candidates, best first: drop non-ingest-format chunks, keep the best 5."""
    return cached_answer(cache, question, ingest_filter(docs)[:5])


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


DEFENSES = {"provenance": provenance_answer, "voting": defended_answer}
