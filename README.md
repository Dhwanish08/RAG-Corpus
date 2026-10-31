# R2: RAG corpus poisoning, attack and defense

A laptop-scale testbed that measures how easily someone who can add documents to a RAG system's corpus can change its answers, and how well three defenses hold up. Nothing is trained: a pretrained reader answers from retrieved text, and the benchmark is the attack success rate (ASR).

- **System:** 125 Wikipedia articles on chemical elements (3,799 chunks), MiniLM retriever (top 5), 4-bit Qwen2.5-3B-Instruct reader (fits a 6 GB GPU). 50 factual questions, clean accuracy **94%**.
- **Attack:** 500 hand-written poison documents (10 per question) injected under the true article's title. Two attackers: short stubs (10-37 words) and an adaptive one that pads the stub to 60+ words with real article sentences.
- **Defenses** (none reads the `poison` flag; it exists for metrics only):
  - `provenance`: drop retrieved chunks shorter than 40 words (graded versions `prov25/50/75` use 10/20/30 words).
  - `manifest`: keep only chunks whose SHA-256 was recorded at ingest (`data/manifest.txt`).
  - `voting`: score-outlier filter, then majority vote over per-chunk answers (ablation; costs clean accuracy and collapses when poison is the majority).

## Results (125 articles, 50 questions)

ASR % at 1 / 3 / 5 / 10 injected poison documents. Clean accuracy is 94% for every defense.

| defense | ASR, short stubs | ASR, padded poison |
|---|---|---|
| none | 58 / 84 / 94 / 96 | 54 / 82 / 88 / 94 |
| prov25 (10 words) | 62 / 84 / 94 / 96 | 56 / 82 / 88 / 94 |
| prov50 (20 words) | 46 / 66 / 74 / 82 | 56 / 82 / 88 / 94 |
| prov75 (30 words) | 0 / 0 / 0 / 0 | 56 / 82 / 88 / 94 |
| provenance (40 words) | 0 / 0 / 0 / 0 | 56 / 82 / 88 / 94 |
| manifest | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |

The poison reaches the top-5 for every question at every count, so the effect is a reader effect, not a retrieval failure. Plots: `figures/asr_vs_poison_count.png`, `figures/graded_provenance.png`; the plotted numbers are in `figures/figure_data.md`. Full tables: `python attack.py --summary`.

## Caveats (read before quoting the numbers)

- The length filter works only because the poison stubs are short; the 40-word cut-off was set after seeing their lengths. Padding defeats it completely (ASR back to the undefended level).
- The manifest's 0% holds **by construction** under the threat model "poison is injected after ingest". It does not help if poison is already in the corpus when the manifest is built, the manifest must be regenerated whenever the corpus legitimately changes, and the file must be out of the attacker's reach.
- 50 questions, one domain, one reader, one embedder; each question is worth 2 points. Scoring is whole-word string matching with aliases (no LLM judge).

## Setup

Python 3.12. The pinned versions in `requirements.txt` are the ones the results came from (CUDA 12.1 build of torch; use plain `torch==2.5.1` on a CPU-only machine, though the reader will then be slow). The first run downloads the Qwen2.5-3B-Instruct and MiniLM weights.

```
pip install -r requirements.txt
```

All generated answers are cached in `data/gen_cache.*`, so the demo and every sweep replay offline; the reader loads only for a question/context that was never run.

## Run

Scripts use relative paths, so run them from `r2/`. On Windows use Python's UTF-8 mode (`-X utf8`); the corpus contains characters cp1252 cannot print.

```
cd r2
python -X utf8 test_r2.py                 # self-checks, no framework
python -X utf8 attack.py --summary        # print the results table from data/results.csv
python -X utf8 attack.py --clean          # clean baseline
python -X utf8 attack.py --poison         # attack sweep (about 15 min uncached)
python -X utf8 attack.py --defense        # every defense (or name them: manifest prov25 prov50 prov75 provenance voting)
python -X utf8 attack.py --adaptive       # padded-poison attacker vs none, provenance levels, manifest
python -X utf8 plots.py                   # figures/ and figure_data.md
cd ..
python -X utf8 app.py                     # Gradio demo at http://127.0.0.1:7860
```

Only one process should write the answer cache at a time (do not run a sweep while the demo is open).

The demo: pick a question, the number of poison documents, the attacker (stubs or padded) and the defense, then press "Run all three" to see clean, poisoned and defended answers side by side with the retrieved chunks. A good showcase is q003: clean 1944, poisoned 1954; with the padded attacker the provenance filter still says 1954 while the manifest restores 1944.

## Layout

| path | what |
|---|---|
| `r2/rag.py` | chunking, embeddings, index, reader, manifest writer |
| `r2/attack.py` | poison injection, padded attacker, scoring, sweeps, `--summary` |
| `r2/defense.py` | provenance (graded), manifest, voting |
| `r2/plots.py` | figures and the data table |
| `r2/test_r2.py` | self-checks (corpus, gold set, poison set, padding, manifest, graded levels) |
| `r2/scrape.py` | builds `data/corpus.jsonl` from Wikipedia |
| `app.py` | Gradio demo |
| `data/` | corpus, 50 gold questions, 500 poison docs (see `data/README.md`), `manifest.txt`, `results.csv`, answer cache |
| `figures/` | generated plots and table |
| `PROGRESS.md` | run log, change log, what is done and next |
| `R2-Project-Report.docx` | the written project report |

The attack is run only against a local copy of public text with made-up false facts; no real system or third party is targeted.
