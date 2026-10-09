"""Rebuild the figures from results/.

    python scripts/plot.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path("results")
FIGURES = Path("figures")

SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e6e5e1"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

MODELS = ["base", "strict", "lenient_last", "any_match", "format_bonus"]
MODEL_LABELS = {"base": "Base model\n(no training)", "strict": "Trained on\nstrict",
                "lenient_last": "Trained on\nlast-number", "any_match": "Trained on\nany-match",
                "format_bonus": "Trained on\nformat bonus"}
# colour follows the training condition in every figure
CONDITION_COLOURS = {"strict": BLUE, "lenient_last": ORANGE, "any_match": AQUA, "format_bonus": YELLOW}
CONDITION_LABELS = {"strict": "strict", "lenient_last": "last-number", "any_match": "any-match",
                    "format_bonus": "format bonus"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": GRID,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def graders_figure(summary: pd.DataFrame):
    """How each model scores under the grader that matters and under the lenient ones."""
    summary = summary.set_index("model").loc[MODELS]
    series = [("strict_acc", "Strict grader (the real objective)", BLUE),
              ("lenient_last_acc", "Last-number grader", ORANGE),
              ("any_match_acc", "Any-match grader", AQUA)]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    width = 0.24
    for k, (column, label, colour) in enumerate(series):
        xs = [i + (k - 1) * (width + 0.02) for i in range(len(MODELS))]
        values = summary[column] * 100
        ax.bar(xs, values, width, color=colour, label=label, linewidth=0)
        for x, v in zip(xs, values):
            ax.text(x, v + 1.2, f"{v:.0f}", ha="center", va="bottom", fontsize=8.5, color=INK_2)
        if column == "strict_acc":
            lo = (summary["strict_acc"] - summary["strict_lo"]) * 100
            hi = (summary["strict_hi"] - summary["strict_acc"]) * 100
            ax.errorbar(xs, values, yerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=1, capsize=2)
    ax.set_xticks(range(len(MODELS)), [MODEL_LABELS[m] for m in MODELS])
    ax.set_ylim(0, 100)
    ax.set_ylabel("GSM8K test problems passed (%)")
    ax.set_title("The same outputs, scored by three graders")
    ax.tick_params(axis="x", length=0)
    ax.legend(frameon=False, loc="upper left", ncols=3, fontsize=9, handlelength=1.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "graders.png", dpi=180)
    plt.close(fig)


def curves_figure():
    """Validation metrics during training, one line per training condition."""
    panels = [("strict", "Strict accuracy (%)", 100), ("well_formed", "Outputs with one valid answer tag (%)", 100),
              ("n_numbers", "Numbers written per output", 1)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (column, title, scale) in zip(axes, panels):
        for condition, colour in CONDITION_COLOURS.items():
            path = RESULTS / "curves" / f"{condition}_s0_eval.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            ax.plot(df["step"], df[column] * scale, color=colour, linewidth=2,
                    label=CONDITION_LABELS[condition])
        ax.set_title(title)
        ax.set_xlabel("Training step")
        ax.set_ylim(bottom=0)
    axes[0].legend(frameon=False, title="Training grader", fontsize=9, title_fontsize=9, alignment="left")
    fig.tight_layout()
    fig.savefig(FIGURES / "curves.png", dpi=180)
    plt.close(fig)


def main():
    FIGURES.mkdir(exist_ok=True)
    graders_figure(pd.read_csv(RESULTS / "summary.csv"))
    curves_figure()
    print("wrote", ", ".join(str(p) for p in sorted(FIGURES.glob("*.png"))))


if __name__ == "__main__":
    main()
