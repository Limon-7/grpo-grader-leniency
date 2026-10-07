"""Greedy evaluation of a model on a GSM8K split, scored by every grader.

    python -m grpo_gl.evaluate --name base --split val
    python -m grpo_gl.evaluate --name strict_s0 --adapter runs/strict_s0/adapter --split test
"""

import argparse
import json
from pathlib import Path

import torch

from .data import Problem, build_prompt, load_gsm8k
from .graders import GRADERS, behaviour
from .model import generate, load_model, load_tokenizer


def evaluate(model, tok, problems: list[Problem], max_new_tokens: int,
             batch_size: int = 256) -> tuple[dict, list[dict]]:
    """Returns (mean metrics, one record per problem)."""
    prompts = [build_prompt(tok, p.question) for p in problems]
    order = sorted(range(len(problems)), key=lambda i: len(prompts[i]))  # similar lengths, less padding
    records = [None] * len(problems)
    for start in range(0, len(order), batch_size):
        chunk = order[start : start + batch_size]
        outputs = generate(model, tok, [prompts[i] for i in chunk], max_new_tokens, sample=False)
        for i, out in zip(chunk, outputs):
            p = problems[i]
            record = {"idx": p.idx, "reference": p.answer, "output": out.text,
                      "n_tokens": len(out.token_ids), "truncated": float(out.truncated)}
            record.update({name: grade(out.text, p.answer) for name, grade in GRADERS.items()})
            record.update(behaviour(out.text))
            records[i] = record
    keys = [k for k in records[0] if k not in ("idx", "reference", "output")]
    metrics = {k: sum(r[k] for r in records) / len(records) for k in keys}
    metrics["multi_tag"] = sum(r["n_tags"] > 1 for r in records) / len(records)
    metrics["n"] = len(records)
    return metrics, records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="label used in the output file names")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = load_tokenizer(args.model)
    model = load_model(args.model, device, adapter=args.adapter)
    problems = load_gsm8k(args.split)[: args.limit]
    metrics, records = evaluate(model, tok, problems, args.max_new_tokens)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"eval_{args.split}_{args.name}"
    (out / f"{stem}.json").write_text(json.dumps(metrics, indent=2) + "\n")
    with open(out / f"{stem}.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
