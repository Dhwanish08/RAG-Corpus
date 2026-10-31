"""Gradio demo: pick a question and an attacker, press one button. Every defense answers the same
retrieved chunks, and the report says which of them returned the correct answer.

Run:  python app.py        (answers replay from data/gen_cache; the local reader loads only if a
question / poison / defense combination was never run before, which is when the progress bar matters)
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

from attack import ABSTAIN_TEXT, CACHE_PATH, DATA, ask_one, load_jsonl, load_poison, load_results, match, summarize  # noqa: E402
from defense import ALL_DEFENSES, answer_all  # noqa: E402
from rag import Index, load_corpus  # noqa: E402

GOLD = {g["id"]: g for g in load_jsonl(f"{DATA}/gold.jsonl")}
POISON = load_poison()
print("loading corpus index...", file=sys.stderr)
IDX = Index.build(load_corpus(f"{DATA}/corpus.jsonl"))
CACHE = shelve.open(CACHE_PATH)
LOCK = threading.Lock()  # dbm cache and the reader are not thread-safe

LABELS = {
    "none": "No defense (poisoned)",
    "prov25": "Provenance filter, 10 words (25%)",
    "prov50": "Provenance filter, 20 words (50%)",
    "prov75": "Provenance filter, 30 words (75%)",
    "provenance": "Provenance filter, 40 words (100%)",
    "manifest": "Hash manifest",
    "manifest_preingest": "Hash manifest built after the poison",
    "neardup": "Near-duplicate detection (abstains)",
    "earliest": "Earliest ingest wins",
    "voting": "Voting",
}
COLUMNS = ["Defense", "Answer", "Result"]


def judge(row: dict, gen: str) -> str:
    if gen.startswith(ABSTAIN_TEXT):
        return "🛑 abstained"
    if match(gen, row["attacker_target"]):
        return "❌ attack succeeded"
    if match(gen, row["correct_answer"], tuple(row.get("aliases", ()))):
        return "✅ correct"
    return "⚠️ neither"


def report(row: dict, done: dict) -> str:
    """Summary line: the correct answer and which defenses returned it. done = {name: (answer, result)}."""
    defs = ALL_DEFENSES[1:]  # "none" is the undefended baseline, not a defense
    prov = ["prov25", "prov50", "prov75", "provenance"]
    same = len({done[n][1] for n in prov}) == 1  # the four cut-offs behave alike: name them once
    names = [n for n in defs if not (same and n in prov[1:])]
    label = lambda n: "Provenance filter (all four cut-offs)" if same and n == "prov25" else LABELS[n]
    right = [label(n) for n in names if done[n][1] == "✅ correct"]
    wrong = [f"{label(n)} — {done[n][1].split(' ', 1)[1]}" for n in names if done[n][1] != "✅ correct"]
    n_right = sum(done[n][1] == "✅ correct" for n in defs)
    md = f"### Correct answer: {row['correct_answer']}  \n_Attacker's target: {row['attacker_target']}_\n\n"
    md += f"✅ **Correct with {n_right} of {len(defs)} defenses:** " + (", ".join(right) or "none") + "\n\n"
    if wrong:
        md += "❌ **Not correct:** " + ", ".join(wrong) + "\n\n"
    if done["none"][1] == "✅ correct":
        md += "ℹ️ Even with no defense the answer survived: this attack did not flip this question."
    else:
        md += f"ℹ️ With no defense the poisoned answer is _{done['none'][0]}_ ({done['none'][1].split(' ', 1)[1]})."
    return md


def ask_all(qid: str, count: int, attack: str, progress=gr.Progress()):
    """Generator: yields (summary, table) after every defense, so the table fills in while the slow ones run."""
    row = GOLD[qid]
    n = len(ALL_DEFENSES)
    table, done = [], {}
    with LOCK:
        yield "⏳ Starting ...", pd.DataFrame(columns=COLUMNS)
        _, clean = ask_one(IDX, row, POISON, CACHE)  # no poison: the reference answer
        table.append(("No poison (reference)", clean, judge(row, clean)))
        runner = answer_all(IDX, row, POISON, CACHE, list(range(int(count))), "" if attack == "stub" else attack)
        for k, name in enumerate(ALL_DEFENSES):
            progress(k / n, desc=f"{LABELS[name]} ({k + 1} of {n})")
            yield f"⏳ Running **{LABELS[name]}** ({k + 1} of {n}) ...", pd.DataFrame(table, columns=COLUMNS)
            got, gen = next(runner)
            assert got == name
            done[name] = (gen, judge(row, gen))
            table.append((LABELS[name], gen, done[name][1]))
        CACHE.sync()
    yield report(row, done), pd.DataFrame(table, columns=COLUMNS)


def results_table():
    rows = summarize(load_results())
    return pd.DataFrame([
        {
            "defense": r["defense"], "corpus articles": r["corpus"], "poison docs": r["count"], "style": r["style"],
            "accuracy": f"{r['accuracy']:.0%}", "attack success": f"{r['asr']:.0%}",
            "poison in top-5": f"{r['poison_retrieval']:.0%}", "abstained": f"{r['abstain']:.0%}",
            "recovery": "-" if r["recovery"] is None else f"{r['recovery']:.0%}",
        }
        for r in rows
    ])


with gr.Blocks(title="RAG poisoning demo") as demo:
    gr.Markdown("# RAG corpus poisoning: attack and defense\nA question is answered from a small Wikipedia chemistry corpus. "
                "Pick a question, how many fake documents to inject and the attacker, then press the button: every defense answers "
                "the same retrieved chunks and the report lists which ones returned the correct answer.")
    with gr.Row():
        qid = gr.Dropdown([(f"{k}: {v['question']}", k) for k, v in GOLD.items()], value="q015", label="Question", scale=3)
        count = gr.Radio([1, 3, 5, 10], value=3, label="Poison docs to inject", scale=1)
    attack = gr.Radio([("Short stubs (naive)", "stub"), ("Padded to chunk length (adaptive)", "padded"),
                       ("Tampered copy of a genuine chunk (one fact swapped)", "tampered")], value="tampered", label="Attacker")
    run = gr.Button("Run all defenses", variant="primary")
    summary = gr.Markdown()
    table = gr.Dataframe(label="Every defense, same retrieved chunks", headers=COLUMNS, wrap=True, interactive=False)
    run.click(ask_all, [qid, count, attack], [summary, table])
    gr.Markdown("## Attack and defense results (50 questions)")
    gr.Dataframe(value=results_table, wrap=True)

if __name__ == "__main__":
    demo.launch()
