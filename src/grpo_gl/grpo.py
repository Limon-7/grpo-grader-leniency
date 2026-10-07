"""GRPO (Group Relative Policy Optimisation) for a causal language model with LoRA.

    python -m grpo_gl.grpo configs/strict.yaml
    python -m grpo_gl.grpo configs/strict.yaml steps=20 name=debug

A run that is stopped and started again with the same command continues from
its last checkpoint (written at every evaluation). Pass resume=false to start over.

Each step: sample a group of completions per prompt, grade them, standardise
the rewards within each group to get advantages, and take a clipped
policy-gradient step on the log-probabilities of the sampled tokens. There is
no value network; the group mean is the baseline.
"""

import csv
import json
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
import yaml

from .data import build_prompt, load_gsm8k
from .evaluate import evaluate
from .graders import GRADERS, behaviour, strict
from .model import Completion, completion_logprobs, generate, load_model, load_tokenizer

# Rewards that are not graders. Brevity is learnable from the first step, so it
# shows quickly whether the update moves the policy in the rewarded direction.
SANITY_REWARDS = {"brevity": lambda text, reference: -float(len(text))}


@dataclass
class Config:
    name: str = "debug"
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    reward: str = "strict"
    seed: int = 0
    steps: int = 200
    prompts_per_step: int = 16
    group_size: int = 8
    max_new_tokens: int = 300
    temperature: float = 1.0
    lr: float = 1e-5
    warmup_steps: int = 10
    max_grad_norm: float = 1.0
    clip_eps: float = 0.2
    beta: float = 0.0  # weight of the KL penalty towards the starting model
    updates_per_batch: int = 1
    loss_norm: str = "sequence"  # "sequence": mean over tokens per completion; "token": over all tokens
    micro_batch: int = 8  # completions per backward pass; the vocabulary-sized logits dominate memory
    lora_r: int = 32
    lora_alpha: int = 64
    eval_every: int = 20
    eval_size: int = 500
    out_dir: str = "runs"
    resume: bool = True


def load_config(argv: list[str]) -> Config:
    """A YAML file followed by any number of key=value overrides."""
    values = {}
    for arg in argv:
        if "=" in arg:
            key, value = arg.split("=", 1)
            values[key] = yaml.safe_load(value)
        else:
            values.update(yaml.safe_load(Path(arg).read_text()) or {})
    types = {f.name: f.type for f in fields(Config)}
    unknown = set(values) - set(types)
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    # YAML reads "1e-4" as a string, so cast every value to its field's type.
    return Config(**{k: types[k](v) for k, v in values.items()})


def group_advantages(rewards: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Standardise rewards within each group. rewards is [prompts, group_size]."""
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True)
    return (rewards - mean) / (std + eps)


def clipped_surrogate(logp: torch.Tensor, old_logp: torch.Tensor, advantage: torch.Tensor,
                      clip_eps: float) -> torch.Tensor:
    """Per-token PPO objective, to be maximised."""
    ratio = torch.exp(logp - old_logp)
    return torch.min(ratio * advantage, ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantage)


def kl_to_reference(logp: torch.Tensor, ref_logp: torch.Tensor) -> torch.Tensor:
    """Per-token estimate of KL(policy || reference) from tokens sampled from the policy.

    This is the non-negative estimator used in the GRPO paper.
    """
    log_ratio = ref_logp - logp
    return torch.exp(log_ratio) - log_ratio - 1


def _chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _logprobs(model, tok, batch: list[Completion]):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
        return completion_logprobs(model, [c.prompt_ids for c in batch],
                                   [c.token_ids for c in batch], tok.pad_token_id)


def update(model, tok, optimizer, completions: list[Completion], advantages: torch.Tensor,
           cfg: Config) -> dict:
    """One or more gradient steps on a sampled batch. advantages is flat, one per completion."""
    device = next(model.parameters()).device
    params = [p for p in model.parameters() if p.requires_grad]
    n_sequences = len(completions)
    n_tokens = sum(len(c.token_ids) for c in completions)

    # A completion with zero advantage has zero policy gradient, so it is only
    # worth a forward pass when the KL term is on.
    active = [i for i in range(n_sequences) if cfg.beta > 0 or advantages[i] != 0]
    active.sort(key=lambda i: len(completions[i].prompt_ids) + len(completions[i].token_ids))
    batches = list(_chunks(active, cfg.micro_batch))

    old, ref = {}, {}
    with torch.no_grad():
        if cfg.updates_per_batch > 1:
            for b, idx in enumerate(batches):
                old[b] = _logprobs(model, tok, [completions[i] for i in idx])[0]
        if cfg.beta > 0:
            with model.disable_adapter():
                for b, idx in enumerate(batches):
                    ref[b] = _logprobs(model, tok, [completions[i] for i in idx])[0]

    stats = {}
    for _ in range(cfg.updates_per_batch):
        optimizer.zero_grad(set_to_none=True)
        total_loss = total_kl = clipped = 0.0
        for b, idx in enumerate(batches):
            batch = [completions[i] for i in idx]
            logp, rows = _logprobs(model, tok, batch)
            # With a single update the sampling policy is the current policy, so the
            # ratio is exactly 1 and this is a plain policy gradient.
            old_logp = old[b] if old else logp.detach()
            adv = advantages[idx].to(device)[rows]
            per_token = -clipped_surrogate(logp, old_logp, adv, cfg.clip_eps)
            if cfg.beta > 0:
                kl = kl_to_reference(logp, ref[b])
                per_token = per_token + cfg.beta * kl
                total_kl += kl.sum().item()
            if cfg.loss_norm == "sequence":
                lengths = torch.tensor([len(c.token_ids) for c in batch], device=device)
                per_sequence = torch.zeros(len(batch), device=device).index_add(0, rows, per_token)
                loss = (per_sequence / lengths).sum() / n_sequences
            else:
                loss = per_token.sum() / n_tokens
            loss.backward()
            total_loss += loss.item()
            clipped += ((torch.exp(logp.detach() - old_logp) - 1).abs() > cfg.clip_eps).sum().item()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm).item() if batches else 0.0
        optimizer.step()
        active_tokens = max(1, sum(len(completions[i].token_ids) for i in active))
        stats = {"loss": total_loss, "grad_norm": grad_norm, "kl": total_kl / active_tokens,
                 "clip_frac": clipped / active_tokens}
    return stats


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _append_csv(path: Path, row: dict):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def _lora_state(model) -> dict:
    return {k: v for k, v in model.state_dict().items() if "lora_" in k}


def save_checkpoint(path: Path, step: int, model, optimizer, schedule, queue, order_rng):
    state = {
        "step": step,
        "lora": _lora_state(model),
        "optimizer": optimizer.state_dict(),
        "schedule": schedule.state_dict(),
        "queue": [p.idx for p in queue],
        "order_rng": order_rng.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)  # a run killed mid-write leaves the previous checkpoint intact


def _drop_rows_after(path: Path, step: int):
    """Remove log rows written after the checkpoint, which the resumed run will produce again."""
    if not path.exists():
        return
    if path.suffix == ".csv":
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        kept = [r for r in rows if int(r["step"]) <= step]
        path.unlink()
        for r in kept:
            _append_csv(path, r)
    else:
        lines = [line for line in path.read_text().splitlines() if json.loads(line)["step"] <= step]
        path.write_text("".join(line + "\n" for line in lines))


def train(cfg: Config):
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = Path(cfg.out_dir) / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "checkpoint.pt"
    resuming = cfg.resume and checkpoint.exists()
    if not resuming:
        for stale in ("train.csv", "eval.csv", "samples.jsonl", "checkpoint.pt"):
            (run_dir / stale).unlink(missing_ok=True)
    meta = {"config": asdict(cfg), "git_commit": _git_commit(), "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"}
    (run_dir / "config.json").write_text(json.dumps(meta, indent=2) + "\n")

    reward_fn = GRADERS.get(cfg.reward) or SANITY_REWARDS[cfg.reward]
    tok = load_tokenizer(cfg.model)
    model = load_model(cfg.model, device, lora_r=cfg.lora_r, lora_alpha=cfg.lora_alpha)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=cfg.lr, weight_decay=0.0)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / max(1, cfg.warmup_steps)))

    train_set = load_gsm8k("train")
    val_set = load_gsm8k("val")[: cfg.eval_size]
    order_rng = random.Random(cfg.seed)
    queue: list = []

    def run_eval(step: int):
        metrics, records = evaluate(model, tok, val_set, cfg.max_new_tokens, batch_size=256)
        _append_csv(run_dir / "eval.csv", {"step": step, **metrics})
        with open(run_dir / "samples.jsonl", "a") as f:
            for r in records[:20]:
                f.write(json.dumps({"step": step, **r}) + "\n")
        print(f"[eval {step}] strict={metrics['strict']:.3f} own={metrics.get(cfg.reward, float('nan')):.3f} "
              f"well_formed={metrics['well_formed']:.3f} tokens={metrics['n_tokens']:.0f} "
              f"numbers={metrics['n_numbers']:.1f}", flush=True)

    first_step = 1
    if resuming:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["lora"], strict=False)
        optimizer.load_state_dict(state["optimizer"])
        schedule.load_state_dict(state["schedule"])
        by_idx = {p.idx: p for p in train_set}
        queue = [by_idx[i] for i in state["queue"]]
        order_rng.setstate(state["order_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        first_step = state["step"] + 1
        for log in ("train.csv", "eval.csv", "samples.jsonl"):
            _drop_rows_after(run_dir / log, state["step"])
        print(f"resuming {cfg.name} from step {state['step']}", flush=True)
    else:
        run_eval(0)
    for step in range(first_step, cfg.steps + 1):
        started = time.time()
        if len(queue) < cfg.prompts_per_step:
            queue.extend(order_rng.sample(train_set, len(train_set)))
        problems = [queue.pop(0) for _ in range(cfg.prompts_per_step)]
        prompts = [build_prompt(tok, p.question) for p in problems]
        completions = generate(model, tok, prompts, cfg.max_new_tokens, sample=True,
                               temperature=cfg.temperature, num_return=cfg.group_size)
        references = [p.answer for p in problems for _ in range(cfg.group_size)]
        rewards = torch.tensor([reward_fn(c.text, ref) for c, ref in zip(completions, references)])
        grouped = rewards.view(cfg.prompts_per_step, cfg.group_size)
        advantages = group_advantages(grouped).flatten()
        sample_seconds = time.time() - started

        stats = update(model, tok, optimizer, completions, advantages, cfg)
        schedule.step()

        row = {
            "step": step,
            "reward": rewards.mean().item(),
            "strict": sum(strict(c.text, ref) for c, ref in zip(completions, references)) / len(completions),
            "well_formed": sum(behaviour(c.text)["well_formed"] for c in completions) / len(completions),
            "tokens": sum(len(c.token_ids) for c in completions) / len(completions),
            "truncated": sum(c.truncated for c in completions) / len(completions),
            "flat_groups": (grouped.std(dim=1) == 0).float().mean().item(),
            **stats,
            "lr": optimizer.param_groups[0]["lr"],
            "sample_seconds": sample_seconds,
            "step_seconds": time.time() - started,
        }
        _append_csv(run_dir / "train.csv", row)
        print(f"[{step}/{cfg.steps}] reward={row['reward']:.3f} strict={row['strict']:.3f} "
              f"tokens={row['tokens']:.0f} trunc={row['truncated']:.2f} flat={row['flat_groups']:.2f} "
              f"loss={row['loss']:+.4f} {row['step_seconds']:.0f}s", flush=True)
        if step % cfg.eval_every == 0 or step == cfg.steps:
            run_eval(step)
            save_checkpoint(checkpoint, step, model, optimizer, schedule, queue, order_rng)

    model.save_pretrained(run_dir / "adapter")


if __name__ == "__main__":
    train(load_config(sys.argv[1:]))
