"""Turn the per-problem test outputs into the summary table used in the README.

    python scripts/analyze.py

Reads results/eval_test_<name>.jsonl for the base model and each trained
condition, and copies the training curves out of runs/ so the figures can be
rebuilt from results/ alone.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path("results")
RUNS = Path("runs")
CONDITIONS = ["strict", "lenient_last", "any_match", "format_bonus"]
SEED = 0
N_BOOT = 10_000


def load(name: str) -> pd.DataFrame:
    path = RESULTS / f"eval_test_{name}.jsonl"
    return pd.DataFrame(json.loads(line) for line in path.read_text().splitlines()).set_index("idx")


def bootstrap_ci(values: np.ndarray, rng) -> tuple[float, float]:
    """95% interval for the mean, resampling test problems."""
    idx = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    return tuple(np.percentile(values[idx].mean(axis=1), [2.5, 97.5]))


def main():
    rng = np.random.default_rng(0)
    frames = {"base": load("base")}
    frames.update({c: load(f"{c}_s{SEED}") for c in CONDITIONS})
    reference = frames["strict"]["strict"].to_numpy()

    rows = []
    for name, df in frames.items():
        own = "strict" if name == "base" else name
        strict = df["strict"].to_numpy()
        lo, hi = bootstrap_ci(strict, rng)
        row = {
            "model": name,
            "trained_on": "-" if name == "base" else own,
            "strict_acc": strict.mean(),
            "strict_lo": lo,
            "strict_hi": hi,
            # format_bonus pays up to 1.2, so its own-grader score is not an accuracy
            "own_grader_score": df[own].mean(),
            "lenient_last_acc": df["lenient_last"].mean(),
            "any_match_acc": df["any_match"].mean(),
            "well_formed": df["well_formed"].mean(),
            "multi_tag": (df["n_tags"] > 1).mean(),
            "tokens": df["n_tokens"].mean(),
            "truncated": df["truncated"].mean(),
            "numbers": df["n_numbers"].mean(),
            "distinct_numbers": df["n_distinct_numbers"].mean(),
        }
        if name not in ("base", "strict"):
            # paired difference against the strict-trained model on the same problems
            diff = strict - reference
            d_lo, d_hi = bootstrap_ci(diff, rng)
            row.update({"vs_strict": diff.mean(), "vs_strict_lo": d_lo, "vs_strict_hi": d_hi})
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "summary.csv", index=False, float_format="%.4f")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    curves = RESULTS / "curves"
    curves.mkdir(exist_ok=True)
    for c in CONDITIONS:
        run = RUNS / f"{c}_s{SEED}"
        for kind in ("train", "eval"):
            if (run / f"{kind}.csv").exists():
                shutil.copy(run / f"{kind}.csv", curves / f"{c}_s{SEED}_{kind}.csv")
        if (run / "config.json").exists():
            shutil.copy(run / "config.json", curves / f"{c}_s{SEED}_config.json")


if __name__ == "__main__":
    main()
