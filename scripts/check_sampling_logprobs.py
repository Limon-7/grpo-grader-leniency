"""Check that the gradient pass scores tokens the same way the sampler drew them.

Samples completions for a few prompts of different lengths, takes the
log-probability of each sampled token from the logits seen during generation
(left-padded, cached), and compares it with the teacher-forced recomputation
used for the loss (right-padded, no cache). A mismatch means the policy
gradient is being computed for a different distribution than the one sampled.
"""

import torch

from grpo_gl.data import build_prompt, load_gsm8k
from grpo_gl.model import (completion_logprobs, generation_config, load_model, load_tokenizer,
                           stop_token_ids, trim_at_stop)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@torch.no_grad()
def main():
    torch.manual_seed(0)
    tok = load_tokenizer(MODEL)
    model = load_model(MODEL, "cuda", lora_r=32)
    prompts = [build_prompt(tok, p.question) for p in load_gsm8k("val")[:6]]
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    out = model.generate(**enc, generation_config=generation_config(model, tok, 120, sample=True),
                         output_logits=True, return_dict_in_generate=True)
    step_logp = torch.stack(out.logits, dim=1).float().log_softmax(-1)  # [batch, steps, vocab]
    new_tokens = out.sequences[:, enc.input_ids.shape[1]:]
    stop_ids = set(stop_token_ids(model))

    prompt_ids, completion_ids, sampled = [], [], []
    for i in range(len(prompts)):
        ids, _ = trim_at_stop(new_tokens[i].tolist(), stop_ids)
        prompt_ids.append(enc.input_ids[i][enc.attention_mask[i].bool()].tolist())
        completion_ids.append(ids)
        sampled.append(step_logp[i, torch.arange(len(ids)), torch.tensor(ids)])
    recomputed, _ = completion_logprobs(model, prompt_ids, completion_ids, tok.pad_token_id)
    diff = (torch.cat(sampled) - recomputed).abs()
    print(f"tokens compared: {diff.numel()}")
    print(f"max abs difference: {diff.max().item():.2e}   mean: {diff.mean().item():.2e}")
    assert diff.max().item() < 1e-2, "sampling and scoring disagree"
    print("OK")


if __name__ == "__main__":
    main()
