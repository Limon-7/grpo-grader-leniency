"""GSM8K loading, fixed splits and prompt construction."""

import random
from dataclasses import dataclass

from datasets import load_dataset

from .graders import parse_number

SYSTEM_PROMPT = (
    "Solve the math problem step by step. Then give the final answer as a single "
    "number inside <answer></answer> tags, for example <answer>42</answer>."
)

VAL_SIZE = 500
SPLIT_SEED = 1234  # fixes which train problems are held out for validation


@dataclass
class Problem:
    idx: int
    question: str
    answer: float


def _reference(solution: str) -> float:
    # GSM8K solutions end with "#### <number>"
    value = parse_number(solution.split("####")[-1])
    if value is None:
        raise ValueError(f"cannot parse reference answer from: {solution[-80:]!r}")
    return value


def load_gsm8k(split: str) -> list[Problem]:
    """split is one of train, val, test. val is carved out of the official train split."""
    if split not in ("train", "val", "test"):
        raise ValueError(split)
    raw = load_dataset("openai/gsm8k", "main", split="test" if split == "test" else "train")
    problems = [Problem(i, r["question"], _reference(r["answer"])) for i, r in enumerate(raw)]
    if split == "test":
        return problems
    order = list(range(len(problems)))
    random.Random(SPLIT_SEED).shuffle(order)
    held_out = set(order[:VAL_SIZE])
    if split == "val":
        return [p for p in problems if p.idx in held_out]
    return [p for p in problems if p.idx not in held_out]


def build_prompt(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
