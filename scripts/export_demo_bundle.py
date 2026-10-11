import json
import gc
import torch
from pathlib import Path

from grpo_gl.data import build_prompt
from grpo_gl.graders import any_match, lenient_last, strict
from grpo_gl.model import generate, load_model, load_tokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
CONDITIONS = ["strict", "lenient_last", "any_match", "format_bonus"]
OUT_DIR = Path("demo/bundle")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXAMPLES = [
    {
        "id": "maria_book",
        "title": "Maria's Book (Remaining Days)",
        "question": "Maria reads 14 pages every day. Her book has 210 pages and she has already read 56. How many more days does she need to finish it?",
        "answer": 11.0
    },
    {
        "id": "pens_cost",
        "title": "Pens Purchase (Simple Multiplication)",
        "question": "A shop sells pens at 3 dollars each. How much do 12 pens cost?",
        "answer": 36.0
    },
    {
        "id": "farm_animals",
        "title": "Farm Animals (Multi-step Addition)",
        "question": "A farmer has 15 cows and 24 sheep. He buys 7 more cows. How many animals does he have now in total?",
        "answer": 46.0
    },
    {
        "id": "bakery_cupcakes",
        "title": "Bakery Cupcakes (Multi-step Deduction)",
        "question": "A bakery bakes 60 cupcakes in the morning. They sell 25 cupcakes at lunch and 18 cupcakes in the afternoon. How many cupcakes are left?",
        "answer": 17.0
    }
]

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = load_tokenizer(MODEL)
    
    results = {ex["id"]: {
        "title": ex["title"],
        "question": ex["question"],
        "answer": ex["answer"],
        "models": {}
    } for ex in EXAMPLES}
    
    for name in ["base"] + CONDITIONS:
        print(f"Loading {name} model...")
        adapter = None if name == "base" else str(Path("runs") / f"{name}_s0" / "adapter")
        model = load_model(MODEL, device, adapter=adapter)
        
        for ex in EXAMPLES:
            qid = ex["id"]
            q = ex["question"]
            ans = ex["answer"]
            prompt = build_prompt(tok, q)
            text = generate(model, tok, [prompt], max_new_tokens=300, sample=False)[0].text.strip()
            
            results[qid]["models"][name] = {
                "text": text,
                "strict": bool(strict(text, ans)),
                "lenient_last": bool(lenient_last(text, ans)),
                "any_match": bool(any_match(text, ans))
            }
        
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    out_file = OUT_DIR / "examples.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Exported bundle to {out_file}")

if __name__ == "__main__":
    main()
