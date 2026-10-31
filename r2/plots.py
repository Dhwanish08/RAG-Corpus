"""Result figures + data table -> ../figures/. Run from r2/ after attack.py has filled results.csv:
    python -X utf8 plots.py

Static light-mode PNGs for the report/README (dataviz skill: categorical slots 1-3, validated all-pairs,
thin 2px lines, markers with a surface ring, hairline grid, legend + marker shapes, leader-line labels, table view in figure_data.md).
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from attack import load_results, summarize  # noqa: E402

OUT = "../figures"
SURFACE, GRID, AXIS, INK2, MUTED = "#fcfcfb", "#e1e0d9", "#c3c2b7", "#52514e", "#898781"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"  # categorical slots 1-5 (slot 5 unused)
COUNTS = [1, 3, 5, 10]
DEFENSES = ["none", "prov25", "prov50", "prov75", "provenance", "manifest", "manifest_preingest", "neardup", "earliest"]
CUTOFF = {"none": 0, "prov25": 10, "prov50": 20, "prov75": 30, "provenance": 40}  # words dropped below
plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 8.5, "axes.facecolor": SURFACE})


def style(ax, xlabel, ylabel=None):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.set_ylim(-4, 104)
    ax.set_yticks(range(0, 101, 20))
    ax.tick_params(colors=MUTED, length=0)
    ax.set_xlabel(xlabel, color=INK2)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2)


def line(ax, xs, ys, color, marker, label, size=8):
    ax.plot(xs, ys, color=color, linewidth=2, solid_capstyle="round", solid_joinstyle="round", marker=marker,
            markersize=size, markeredgecolor=SURFACE, markeredgewidth=2, label=label)


def legend_below(fig, ax):
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, labelcolor=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.09, 1, 1))


def fig_asr(S):
    """Attack success rate vs poison count: stub vs padded poison, no defense vs provenance vs manifest."""
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 3.0), sharey=True, facecolor=SURFACE)
    series = [("No defense", "none", BLUE, "o", 11), ("Provenance (40 words)", "provenance", ORANGE, "s", 7), ("Hash manifest", "manifest", AQUA, "^", 8)]
    for ax, (st, title) in zip(axes, [("mixed", "Short stub poison"), ("padded", "Padded poison")]):
        for label, d, color, marker, size in series:  # widest marker first so overlapping lines stay visible
            line(ax, COUNTS, [100 * S[d, c, st]["asr"] for c in COUNTS], color, marker, label, size)
        style(ax, "Poison documents injected", "Attack success rate (%)" if ax is axes[0] else None)
        ax.set_xticks(COUNTS)
        ax.set_title(title, color=INK2, fontsize=9, loc="left")
    if all(S["provenance", c, "mixed"]["asr"] == S["manifest", c, "mixed"]["asr"] for c in COUNTS):  # drawn on top of each other
        axes[0].text(0.5, 0.16, f"Provenance and manifest\nboth at {100 * S['manifest', 1, 'mixed']['asr']:.0f}%", transform=axes[0].transAxes, color=INK2, ha="center", fontsize=8)
    if all(abs(S["provenance", c, "padded"]["asr"] - S["none", c, "padded"]["asr"]) <= 0.05 for c in COUNTS):
        axes[1].text(0.5, 0.3, "Provenance tracks\nno defense", transform=axes[1].transAxes, color=INK2, ha="center", fontsize=8)
    legend_below(fig, axes[0])
    return fig


def fig_tampered(S):
    """Tampered-copy attacker: attack success and accuracy vs poison count. Provenance is left out (the copies are
    full length, so it equals no defense; see figure_data.md); manifest and earliest lie on top of each other."""
    series = [("No defense", "None", "none", BLUE, "o", 13), ("Hash manifest", "Manifest", "manifest", AQUA, "^", 11),
              ("Near-duplicate, abstain", "Neardup", "neardup", YELLOW, "D", 9), ("Earliest ingest wins", "Earliest", "earliest", MAGENTA, "p", 7)]  # widest first, so coinciding series stay visible
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.1), facecolor=SURFACE)
    for ax, key, ylabel, title in ((axes[0], "asr", "Attack success rate (%)", "Attack success"), (axes[1], "accuracy", "Accuracy (%)", "Accuracy under attack")):
        for label, _, d, color, marker, size in series:
            line(ax, COUNTS, [100 * S[d, c, "tampered"][key] for c in COUNTS], color, marker, label, size)
        style(ax, "Poison documents injected", ylabel)
        ax.set_xticks(COUNTS)
        ax.set_title(title, color=INK2, fontsize=9, loc="left")
    # direct labels with leader lines (ASR panel): lines converge, so labels sit in the margin at spread heights
    ax = axes[0]
    ax.set_xlim(0.5, 14)
    ends = sorted(((100 * S[d, COUNTS[-1], "tampered"]["asr"], short, color) for _, short, d, color, *_ in series), reverse=True)
    for (y, short, _), yl in zip(ends, (92, 68, 44, 20)):
        ax.annotate(short, xy=(COUNTS[-1], y), xytext=(10.9, yl), color=INK2, fontsize=7.5, va="center", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.8, shrinkA=0, shrinkB=5))
    if all(S[d, c, "tampered"]["asr"] == 0 for d in ("manifest", "neardup", "earliest") for c in COUNTS):  # drawn on top of each other
        ax.text(0.12, 0.14, "Manifest, neardup and\nearliest all at 0%", transform=ax.transAxes, color=INK2, fontsize=8)
    if all(S["manifest", c, "tampered"]["accuracy"] == S["earliest", c, "tampered"]["accuracy"] for c in COUNTS):
        axes[1].text(0.6, 0.62, f"Manifest and earliest\nboth at {100 * S['manifest', COUNTS[0], 'tampered']['accuracy']:.0f}%",
                     transform=axes[1].transAxes, color=INK2, fontsize=8)
    legend_below(fig, axes[0])
    return fig


def fig_graded(S, count=3):
    """Graded provenance: attack success and clean accuracy as the length cutoff tightens."""
    fig, ax = plt.subplots(figsize=(5.8, 3.2), facecolor=SURFACE)
    names = [d for d in DEFENSES if d in CUTOFF]
    xs = [CUTOFF[d] for d in names]
    line(ax, xs, [100 * S[d, count, "mixed"]["asr"] for d in names], BLUE, "o", "Attack success, stubs")
    line(ax, xs, [100 * S[d, count, "padded"]["asr"] for d in names], ORANGE, "s", "Attack success, padded")
    line(ax, xs, [100 * S[d, 0, "-"]["accuracy"] for d in names], AQUA, "^", "Clean accuracy")
    style(ax, "Provenance cutoff: chunks shorter than this many words are dropped", f"Percent ({count} poison docs)")
    ax.set_xticks(xs)
    ax.set_xticklabels(["none"] + [f"{CUTOFF[d]}\n({round(CUTOFF[d] / 40 * 100)}%)" for d in names[1:]])
    legend_below(fig, ax)
    return fig


def data_table(S):
    """Table view of every plotted number (percent at 1/3/5/10 poison docs)."""
    cell = lambda d, st, k: " / ".join(f"{100 * S[d, c, st][k]:.0f}" for c in COUNTS)
    lines = ["| defense | clean accuracy % | ASR % stub | ASR % padded | ASR % tampered | accuracy % tampered | abstained % tampered |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {d} | {100 * S[d, 0, '-']['accuracy']:.0f} | {cell(d, 'mixed', 'asr')} | {cell(d, 'padded', 'asr')} | "
              f"{cell(d, 'tampered', 'asr')} | {cell(d, 'tampered', 'accuracy')} | {cell(d, 'tampered', 'abstain')} |" for d in DEFENSES]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    S = {(s["defense"], s["count"], s["style"]): s for s in summarize(load_results()) if s["corpus"] == 125}
    os.makedirs(OUT, exist_ok=True)
    for name, fig in (("asr_vs_poison_count", fig_asr(S)), ("graded_provenance", fig_graded(S)), ("tampered_attack", fig_tampered(S))):
        path = f"{OUT}/{name}.png"
        fig.savefig(path, dpi=200, facecolor=SURFACE)
        assert os.path.getsize(path) > 10_000, path
        print("wrote", path)
    with open(f"{OUT}/figure_data.md", "w", encoding="utf-8") as f:
        f.write(data_table(S))
    print(data_table(S))
