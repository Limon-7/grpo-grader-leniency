"""Ask the base model and each trained model the same question and grade the answers.

    python scripts/demo.py "A shop sells pens at 3 dollars each. How much do 12 pens cost?" 36
    python scripts/demo.py            # uses a built-in example

The second argument is the correct answer, used to show what each grader says.
"""

import sys
from pathlib import Path

import torch

from grpo_gl.data import build_prompt
from grpo_gl.graders import any_match, lenient_last, strict
from grpo_gl.model import generate, load_model, load_tokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
CONDITIONS = ["strict", "lenient_last", "any_match", "format_bonus"]
EXAMPLE = ("Maria reads 14 pages every day. Her book has 210 pages and she has already "
           "read 56. How many more days does she need to finish it?", 11.0)


def main():
    question, reference = EXAMPLE
    if len(sys.argv) >= 3:
        question, reference = sys.argv[1], float(sys.argv[2])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = load_tokenizer(MODEL)
    prompt = build_prompt(tok, question)
    print(f"Question: {question}\nCorrect answer: {reference:g}\n")
    for name in ["base"] + CONDITIONS:
        adapter = Path("runs") / f"{name}_s0" / "adapter"
        if name != "base" and not adapter.exists():
            continue
        model = load_model(MODEL, device, adapter=None if name == "base" else str(adapter))
        text = generate(model, tok, [prompt], max_new_tokens=300, sample=False)[0].text
        verdicts = "  ".join(f"{label}: {'pass' if grade(text, reference) else 'fail'}"
                             for label, grade in [("strict", strict), ("last-number", lenient_last),
                                                  ("any-match", any_match)])
        print(f"=== {name} ===\n{text.strip()}\n--> {verdicts}\n")
        del model


if __name__ == "__main__":
    main()
