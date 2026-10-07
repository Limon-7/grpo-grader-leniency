import json

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from grpo_gl.grpo import (_append_csv, _drop_rows_after, clipped_surrogate, group_advantages,
                           kl_to_reference, load_config)
from grpo_gl.model import completion_logprobs, trim_at_stop


def test_advantages_are_standardised_within_each_group():
    rewards = torch.tensor([[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 5.0]])
    adv = group_advantages(rewards)
    assert torch.allclose(adv.mean(dim=1), torch.zeros(2), atol=1e-6)
    assert torch.allclose(adv.std(dim=1), torch.ones(2), atol=1e-3)
    assert adv[0, 0] > 0 > adv[0, 1]


def test_group_with_identical_rewards_gets_zero_advantage():
    adv = group_advantages(torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]))
    assert torch.equal(adv, torch.zeros(2, 3))


def test_surrogate_gradient_is_the_advantage_when_the_ratio_is_one():
    logp = torch.tensor([-1.0, -2.0, -3.0], requires_grad=True)
    advantage = torch.tensor([1.5, -0.5, 0.0])
    clipped_surrogate(logp, logp.detach(), advantage, 0.2).sum().backward()
    assert torch.allclose(logp.grad, advantage)


def test_surrogate_stops_rewarding_moves_past_the_clip_range():
    old = torch.zeros(4)
    # ratios 1.5, 1.5, 0.5, 0.5 with advantages +, -, +, -
    logp = torch.log(torch.tensor([1.5, 1.5, 0.5, 0.5])).requires_grad_()
    advantage = torch.tensor([1.0, -1.0, 1.0, -1.0])
    clipped_surrogate(logp, old, advantage, 0.2).sum().backward()
    # No gradient where the policy already moved far in the helpful direction;
    # full gradient where it moved the wrong way.
    assert logp.grad[0] == 0 and logp.grad[3] == 0
    assert logp.grad[1] != 0 and logp.grad[2] != 0


def test_kl_estimate_is_zero_at_the_reference_and_positive_elsewhere():
    logp = torch.tensor([-1.0, -2.0])
    assert torch.equal(kl_to_reference(logp, logp), torch.zeros(2))
    assert (kl_to_reference(logp, logp + torch.tensor([0.5, -0.5])) > 0).all()


def test_trim_at_stop():
    assert trim_at_stop([5, 6, 9, 0, 0], {9}) == ([5, 6, 9], False)
    assert trim_at_stop([5, 6, 7], {9}) == ([5, 6, 7], True)


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    return Qwen2ForCausalLM(config).eval()


def test_completion_logprobs_match_a_direct_computation(tiny_model):
    prompts = [[3, 4, 5, 6, 7], [8, 9], [10, 11, 12]]
    completions = [[20, 21], [30, 31, 32, 33], [40]]
    with torch.no_grad():
        logp, rows = completion_logprobs(tiny_model, prompts, completions, pad_id=0)
        assert rows.tolist() == [0, 0, 1, 1, 1, 1, 2]
        expected = []
        for p, c in zip(prompts, completions):  # one sequence at a time, so no padding involved
            logits = tiny_model(torch.tensor([p + c])).logits[0].log_softmax(-1)
            expected += [logits[len(p) + j - 1, t].item() for j, t in enumerate(c)]
    assert torch.allclose(logp, torch.tensor(expected), atol=1e-5)


def test_config_overrides(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("reward: any_match\nsteps: 7\n")
    cfg = load_config([str(path), "lr=1e-4", "name=x"])
    assert (cfg.reward, cfg.steps, cfg.lr, cfg.name) == ("any_match", 7, 1e-4, "x")
    with pytest.raises(ValueError):
        load_config(["nonsense=1"])


def test_resume_drops_log_rows_written_after_the_checkpoint(tmp_path):
    log = tmp_path / "train.csv"
    for step in range(1, 8):
        _append_csv(log, {"step": step, "reward": step / 10})
    _drop_rows_after(log, 5)
    lines = log.read_text().splitlines()
    assert lines[0] == "step,reward" and len(lines) == 6 and lines[-1].startswith("5,")

    samples = tmp_path / "samples.jsonl"
    samples.write_text("".join(json.dumps({"step": s, "output": "x"}) + "\n" for s in (0, 25, 50)))
    _drop_rows_after(samples, 25)
    assert len(samples.read_text().splitlines()) == 2
