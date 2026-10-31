"""Core RAG module: chunk, embed, retrieve, answer.

Everything else (attack.py, defense.py, app.py) imports this. The index is a
plain numpy array so it supports runtime append (np.vstack) -- the live demo
injects poison into a loaded index without rebuilding anything.
"""
import hashlib
import json

import numpy as np
from sentence_transformers import SentenceTransformer

CHUNK_WORDS = 200
EMBED_MODEL = "all-MiniLM-L6-v2"

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def chunk(text: str, title: str = "") -> list[str]:
    """Split into ~CHUNK_WORDS-word chunks, merging short paragraphs."""
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, cur = [], []
    cur_words = 0
    for p in paras:
        words = p.split()
        if cur_words + len(words) > CHUNK_WORDS and cur:
            chunks.append(" ".join(cur))
            cur, cur_words = [], 0
        cur.append(p)
        cur_words += len(words)
    if cur:
        chunks.append(" ".join(cur))
    prefix = f"{title}: " if title else ""
    return [prefix + c for c in chunks]


def embed(texts: list[str]) -> np.ndarray:
    """L2-normalized embeddings, shape (n, 384)."""
    vecs = get_embedder().encode(texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


class Index:
    """Brute-force cosine-similarity index over doc chunks. No FAISS needed
    at this scale (a few thousand chunks) -- a dot product is exact and
    sub-millisecond, and a plain array is trivial to append to live.
    """

    def __init__(self, docs: list[dict], vecs: np.ndarray):
        # list of {"text": ..., "source": ..., "poison": bool, "seq": int}; seq is the ingest
        # order, stamped by the index itself (never supplied by whoever writes the doc)
        self.docs = docs
        self.vecs = vecs  # (n, 384)

    @classmethod
    def build(cls, docs: list[dict]) -> "Index":
        docs = [{**d, "seq": i} for i, d in enumerate(docs)]
        vecs = embed([d["text"] for d in docs])
        return cls(docs, vecs)

    def add(self, docs: list[dict]):
        """Append new docs (e.g. poison) to the live index, in place, stamped as ingested after everything already in it."""
        nxt = max((d["seq"] for d in self.docs), default=-1) + 1
        docs = [{**d, "seq": nxt + k} for k, d in enumerate(docs)]
        new_vecs = embed([d["text"] for d in docs])
        self.docs = self.docs + docs
        self.vecs = np.vstack([self.vecs, new_vecs])

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        qv = embed([query])[0]
        scores = self.vecs @ qv
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        top = top[np.argsort(-scores[top])]
        return [{**self.docs[i], "score": float(scores[i])} for i in top]


def load_corpus(path: str) -> list[dict]:
    """Load corpus.jsonl, chunk every article, return doc dicts."""
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            for c in chunk(rec["text"], rec["title"]):
                docs.append({"text": c, "source": rec["title"], "poison": False})
    return docs


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_manifest(docs: list[dict], path: str):
    """Trusted ingest record: one sha256 per chunk, written when the corpus is ingested."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted({text_hash(d["text"]) for d in docs})) + "\n")


ANSWER_PROMPT = """Answer the question using ONLY the context below. \
If the context does not contain the answer, say "I don't know." \
Be concise -- state the answer directly, don't hedge or repeat the question.

Context:
{context}

Question: {question}
Answer:"""

_reader_tokenizer = None
_reader_model = None


def get_reader(model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
    """Lazy-load the 4-bit local reader. Not needed for retrieval-only tests."""
    global _reader_tokenizer, _reader_model
    if _reader_model is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        _reader_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _reader_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16),
            device_map="auto",
        )
    return _reader_tokenizer, _reader_model


def answer(question: str, docs: list[dict], max_new_tokens: int = 100) -> str:
    """Generate an answer from retrieved docs. Context-only by prompt design --
    that constraint is what makes poisoning (and the defense) meaningful.
    """
    tok, model = get_reader()
    context = "\n\n".join(d["text"] for d in docs)
    prompt = ANSWER_PROMPT.format(context=context, question=question)
    messages = [{"role": "user", "content": prompt}]
    inputs = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
