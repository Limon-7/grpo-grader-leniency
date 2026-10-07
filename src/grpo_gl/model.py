"""Model loading, generation and teacher-forced log-probabilities."""

from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class Completion:
    prompt_ids: list[int]
    token_ids: list[int]  # ends with the stop token unless truncated
    text: str
    truncated: bool


def load_tokenizer(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"  # decoder-only generation needs the prompt flush against the new tokens
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(name: str, device: str, lora_r: int | None = None, lora_alpha: int | None = None,
               adapter: str | None = None):
    """Base model, optionally with fresh LoRA adapters (lora_r) or trained ones (adapter)."""
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
    if adapter is not None:
        model = PeftModel.from_pretrained(model, adapter)
    elif lora_r is not None:
        # Dropout stays at 0: the sampling pass and the gradient pass must see the same policy.
        config = LoraConfig(r=lora_r, lora_alpha=lora_alpha or 2 * lora_r, lora_dropout=0.0,
                            target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
        model = get_peft_model(model, config)
    return model.to(device).eval()


def stop_token_ids(model) -> list[int]:
    eos = model.generation_config.eos_token_id
    return list(eos) if isinstance(eos, (list, tuple)) else [eos]


def generation_config(model, tok, max_new_tokens: int, sample: bool, temperature: float = 1.0):
    """Explicit settings that replace the checkpoint's defaults.

    Qwen ships with top-k, top-p and a repetition penalty switched on. Left in
    place, the tokens would be drawn from a different distribution than the one
    the policy gradient assumes.
    """
    return GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=sample,
        temperature=temperature if sample else None,
        top_k=0 if sample else None,
        top_p=1.0 if sample else None,
        repetition_penalty=1.0,
        eos_token_id=stop_token_ids(model),
        pad_token_id=tok.pad_token_id,
    )


def trim_at_stop(token_ids: list[int], stop_ids: set[int]) -> tuple[list[int], bool]:
    """Keep tokens up to and including the first stop token. Returns (tokens, truncated)."""
    for i, t in enumerate(token_ids):
        if t in stop_ids:
            return token_ids[: i + 1], False
    return token_ids, True


@contextmanager
def merged_adapter(model):
    """Fold the LoRA weights into the base weights for the duration of the block.

    The function computed is the same. Decoding is one small forward pass per
    token, so the extra LoRA modules cost a large share of sampling time.
    """
    if not isinstance(model, PeftModel):
        yield
        return
    model.merge_adapter()
    try:
        yield
    finally:
        model.unmerge_adapter()


@torch.no_grad()
def generate(model, tok, prompts: list[str], max_new_tokens: int, sample: bool,
             temperature: float = 1.0, num_return: int = 1) -> list[Completion]:
    """Returns num_return completions per prompt, grouped in prompt order."""
    device = next(model.parameters()).device
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    input_ids = enc.input_ids.repeat_interleave(num_return, dim=0)
    attention = enc.attention_mask.repeat_interleave(num_return, dim=0)
    config = generation_config(model, tok, max_new_tokens, sample, temperature)
    with merged_adapter(model):
        out = model.generate(input_ids=input_ids, attention_mask=attention, generation_config=config)
    stop_ids = set(stop_token_ids(model))
    completions = []
    for row_in, row_mask, row_out in zip(input_ids, attention, out):
        prompt_ids = row_in[row_mask.bool()].tolist()
        token_ids, truncated = trim_at_stop(row_out[input_ids.shape[1]:].tolist(), stop_ids)
        text = tok.decode(token_ids, skip_special_tokens=True)
        completions.append(Completion(prompt_ids, token_ids, text, truncated))
    return completions


def completion_logprobs(model, prompt_ids: list[list[int]], completion_ids: list[list[int]],
                        pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-probability of every completion token given its prompt and the tokens before it.

    Returns a flat tensor over all completion tokens in the batch and, for each
    of those tokens, the index of the sequence it belongs to. The vocabulary
    projection is applied only at completion positions, which keeps the logits
    tensor small enough to backpropagate through.
    """
    hf = model.get_base_model() if isinstance(model, PeftModel) else model
    device = next(hf.parameters()).device
    lengths = [len(p) + len(c) for p, c in zip(prompt_ids, completion_ids)]
    input_ids = torch.full((len(lengths), max(lengths)), pad_id, dtype=torch.long)
    attention = torch.zeros_like(input_ids)
    rows, cols, targets = [], [], []
    for i, (p, c) in enumerate(zip(prompt_ids, completion_ids)):
        input_ids[i, : lengths[i]] = torch.tensor(p + c)
        attention[i, : lengths[i]] = 1
        for j, token in enumerate(c):
            rows.append(i)
            cols.append(len(p) + j - 1)  # the hidden state at t predicts token t + 1
            targets.append(token)
    rows = torch.tensor(rows, device=device)
    cols = torch.tensor(cols, device=device)
    targets = torch.tensor(targets, device=device)
    hidden = hf.base_model(input_ids=input_ids.to(device), attention_mask=attention.to(device)).last_hidden_state
    logits = hf.get_output_embeddings()(hidden[rows, cols])
    logp = -F.cross_entropy(logits.float(), targets, reduction="none")
    return logp, rows
