"""Gradio demo: ask -> inject poison -> ask again -> defense on.

Run:  python app.py        (answers replay from data/gen_cache; the local reader
loads only if a question/poison combination was never run before)
"""
import os
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(HERE, "r2"))  # attack.py/rag.py use ../data relative paths
sys.path.insert(0, os.getcwd())

import gradio as gr  # noqa: E402
import pandas as pd  # noqa: E402
import shelve  # noqa: E402

from attack import CACHE_PATH, DATA, OVERFETCH, ask_one, cached_answer, load_jsonl, load_poison, load_results, match, summarize  # noqa: E402
from defense import DEFENSES, FILTERS  # noqa: E402
from rag import Index, load_corpus  # noqa: E402

GOLD = {g["id"]: g for g in load_jsonl(f"{DATA}/gold.jsonl")}
POISON = load_poison()
print("loading corpus index...", file=sys.stderr)
IDX = Index.build(load_corpus(f"{DATA}/corpus.jsonl"))
CACHE = shelve.open(CACHE_PATH)
LOCK = threading.Lock()  # dbm cache and the reader are not thread-safe


DEFENSE_CHOICES = [
    ("Provenance, 40-word cutoff (100%)", "provenance"),
    ("Hash manifest", "manifest"),
    ("Provenance 75% (30 words)", "prov75"),
    ("Provenance 50% (20 words)", "prov50"),
    ("Provenance 25% (10 words)", "prov25"),
]


def ask(qid: str, count: int, defense: str, attack: str = "stub"):
    row = GOLD[qid]
    fn = cached_answer if defense == "none" else DEFENSES[defense]
    with LOCK:
        top, gen = ask_one(IDX, row, POISON, CACHE, list(range(int(count))), fn, OVERFETCH.get(defense, 1), pad=attack == "padded")
        CACHE.sync()
    if match(gen, row["attacker_target"]):
        verdict = "❌ **Attack succeeded** - the answer states the attacker's value."
    elif match(gen, row["correct_answer"], tuple(row.get("aliases", ()))):
        verdict = "✅ **Correct**"
    else:
        verdict = "⚠️ Neither the correct answer nor the attacker's value."
    kept = {id(d) for d in (FILTERS[defense](top)[:5] if defense in FILTERS else top[:5])}
    chunks = pd.DataFrame([
        {
            "rank": i,
            "score": round(d["score"], 3),
            "source": d["source"],
            "words": len(d["text"].split()),
            "used": "yes" if id(d) in kept else "dropped by defense",
            "poison (hidden from defense)": "POISON" if d["poison"] else "",
            "text": d["text"][:160],
        }
        for i, d in enumerate(top[:10], 1)
    ])
    md = f"**Answer:** {gen}\n\n{verdict}\n\n_Correct: {row['correct_answer']} · Attacker target: {row['attacker_target']}_"
    return md, chunks


def results_table():
    rows = summarize(load_results())
    return pd.DataFrame([
        {
            "defense": r["defense"], "corpus articles": r["corpus"], "poison docs": r["count"], "style": r["style"],
            "accuracy": f"{r['accuracy']:.0%}", "attack success": f"{r['asr']:.0%}",
            "poison in top-5": f"{r['poison_retrieval']:.0%}",
            "recovery": "-" if r["recovery"] is None else f"{r['recovery']:.0%}",
        }
        for r in rows
    ])


def run_all(qid, count, attack, defense):
    return (*ask(qid, 0, "none"), *ask(qid, count, "none", attack), *ask(qid, count, defense, attack))


with gr.Blocks(title="RAG poisoning demo") as demo:
    gr.Markdown("# RAG corpus poisoning: attack and defense\nA question is answered from a small Wikipedia chemistry corpus. "
                "Inject fake documents and watch the answer flip, then turn the defense on.")
    with gr.Row():
        qid = gr.Dropdown([(f"{k}: {v['question']}", k) for k, v in GOLD.items()], value="q003", label="Question", scale=3)
        count = gr.Radio([1, 3, 5, 10], value=3, label="Poison docs to inject", scale=1)
    with gr.Row():
        attack = gr.Radio([("Short stubs (naive)", "stub"), ("Padded to chunk length (adaptive)", "padded")], value="stub", label="Attacker")
        defense = gr.Dropdown(DEFENSE_CHOICES, value="provenance", label="Defense")
    with gr.Row():
        b1 = gr.Button("1 · Ask (no poison)")
        b2 = gr.Button("2 · Inject poison, ask again", variant="stop")
        b3 = gr.Button("3 · Defense on, ask again", variant="primary")
        ball = gr.Button("Run all three")
    outs = []
    with gr.Row():
        for title in ("Clean", "Poisoned", "Defended"):
            with gr.Column():
                gr.Markdown(f"### {title}")
                a = gr.Markdown()
                c = gr.Dataframe(label="Top-5 retrieved chunks", wrap=True)
                outs += [a, c]
    b1.click(lambda q: ask(q, 0, "none"), qid, outs[0:2])
    b2.click(lambda q, n, a: ask(q, n, "none", a), [qid, count, attack], outs[2:4])
    b3.click(lambda q, n, a, d: ask(q, n, d, a), [qid, count, attack, defense], outs[4:6])
    ball.click(run_all, [qid, count, attack, defense], outs)
    gr.Markdown("## Attack and defense results (50 questions)")
    table = gr.Dataframe(value=results_table, wrap=True)

if __name__ == "__main__":
    demo.launch()
