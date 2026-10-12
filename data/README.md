# Data files

- `corpus.jsonl` — one line per Wikipedia article: `{"title": ..., "text": ...}`. From `r2/scrape.py`.
- `gold.jsonl` — one line per eval question:
  ```json
  {"id": "q001", "question": "...", "correct_answer": "...", "source_title": "...", "attacker_target": "..."}
  ```
  `source_title` is the article the correct answer comes from (must match a `title` in corpus.jsonl).
  `attacker_target` is the specific false answer the matching poison doc pushes.
- `poison.jsonl` — 500 poison documents, 10 per gold question:
  ```json
  {"id": "q001", "style": "keyword" | "natural" | "authoritative", "variant": 0-9, "text": "..."}
  ```
  `id` links back to the `gold.jsonl` question it targets. Variants 0-2 are one doc per style (keyword, natural, authoritative); variants 3-9 are paraphrases cycling the same styles (keyword, natural, authoritative, keyword, natural, authoritative, keyword). A poison count of *c* uses variants `0..c-1`. When injected, a doc's `source` is the gold `source_title` (the attacker spoofs the article). Every doc states `attacker_target` and never contains the correct answer or its aliases; `r2/test_r2.py` enforces this and that each doc retrieves into the top-5 alone.
- `gold.jsonl` also has an optional `"aliases": [...]` list: alternative accepted wordings of `correct_answer` for scoring.
- `results.csv` — attack sweep output, written by `r2/attack.py`.
- `gen_cache.*` — `shelve` disk cache of reader generations, keyed on (question, retrieved doc texts). Lets `attack.py` and the Gradio demo re-run instantly without the GPU. Safe to delete; it just gets rebuilt.
