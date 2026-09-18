from __future__ import annotations
from typing import Callable, Dict, Literal, Optional, Tuple
import torch


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], Dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float = 1e-6,
    normalize_by_std: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    if len(rollout_responses) != len(repeated_ground_truths):
        raise ValueError("responses and ground truths must have same length")
    if group_size <= 0 or len(rollout_responses) % group_size != 0:
        raise ValueError("number of responses must be divisible by group_size")

    reward_dicts = [reward_fn(r, g) for r, g in zip(rollout_responses, repeated_ground_truths)]
    raw_rewards = torch.tensor([float(x["reward"]) for x in reward_dicts], dtype=torch.float32)
    grouped = raw_rewards.view(-1, group_size)
    means = grouped.mean(dim=1, keepdim=True)
    centered = grouped - means
    if normalize_by_std:
        # Assignment snapshots use torch.std default correction=1/sample std.
        stds = grouped.std(dim=1, keepdim=True)
        advantages = centered / (stds + advantage_eps)
    else:
        advantages = centered
    advantages = advantages.reshape(-1)

    meta: dict[str, float] = {
        "mean_reward": raw_rewards.mean().item(),
        "std_reward": raw_rewards.std().item() if raw_rewards.numel() > 1 else 0.0,
        "max_reward": raw_rewards.max().item(),
        "min_reward": raw_rewards.min().item(),
        "mean_advantage": advantages.mean().item(),
    }
    for key in ("format_reward", "answer_reward"):
        vals = [float(x[key]) for x in reward_dicts if key in x]
        if vals:
            meta[f"mean_{key}"] = sum(vals) / len(vals)
    return advantages, raw_rewards, meta


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    return -(raw_rewards_or_advantages * policy_log_probs)


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if cliprange < 0:
        raise ValueError("cliprange must be nonnegative")
    # Importance ratios are especially sensitive in BF16: exp() can overflow long before
    # the PPO/GRPO clipped branch is selected. Compute this path in FP32. This is
    # algebraically identical for ordinary ratios and preserves the assignment tests.
    log_ratio = policy_log_probs.float() - old_log_probs.float()
    ratio = torch.exp(log_ratio)
    adv = advantages.float()
    unclipped = ratio * adv
    clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    clipped = clipped_ratio * adv
    loss = -torch.minimum(unclipped, clipped)
    with torch.no_grad():
        clipped_mask = clipped < unclipped
        metadata = {
            "clip_fraction": clipped_mask.float().mean(),
            "clipped": clipped_mask,
            "ratio": ratio.detach(),
            "ratio_mean": ratio.mean(),
            "log_ratio_min": log_ratio.min().detach(),
            "log_ratio_max": log_ratio.max().detach(),
        }
    return loss, metadata


def compute_grpo_no_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    log_ratio = policy_log_probs.float() - old_log_probs.float()
    ratio = torch.exp(log_ratio)
    return -(ratio * advantages.float()), {
        "ratio": ratio.detach(), "ratio_mean": ratio.mean().detach(),
        "log_ratio_min": log_ratio.min().detach(), "log_ratio_max": log_ratio.max().detach(),
    }


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_no_clip"],
    raw_rewards: Optional[torch.Tensor] = None,
    advantages: Optional[torch.Tensor] = None,
    old_log_probs: Optional[torch.Tensor] = None,
    cliprange: Optional[float] = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if loss_type == "no_baseline":
        assert raw_rewards is not None
        return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), {}
    if loss_type == "reinforce_with_baseline":
        assert advantages is not None
        return compute_naive_policy_gradient_loss(advantages, policy_log_probs), {}
    if loss_type == "grpo_clip":
        assert advantages is not None and old_log_probs is not None and cliprange is not None
        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    if loss_type == "grpo_no_clip":
        assert advantages is not None and old_log_probs is not None
        return compute_grpo_no_clip_loss(advantages, policy_log_probs, old_log_probs)
    raise ValueError(f"unknown loss_type={loss_type}")


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    m = mask.to(dtype=tensor.dtype)
    return (tensor * m).sum(dim=dim) / m.sum(dim=dim)


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_no_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    length_normalization: str = "mean",
    constant_normalizer: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    per_token_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )
    if length_normalization == "mean":
        per_example = masked_mean(per_token_loss, response_mask, dim=1)
    elif length_normalization == "constant":
        if constant_normalizer is None or constant_normalizer <= 0:
            raise ValueError("constant_normalizer must be positive for constant length normalization")
        m = response_mask.to(per_token_loss.dtype)
        per_example = (per_token_loss * m).sum(dim=1) / constant_normalizer
    else:
        raise ValueError(f"unknown length_normalization={length_normalization}")
    unscaled = per_example.mean()
    loss = unscaled / gradient_accumulation_steps
    loss.backward()
    out_meta = {"loss": unscaled.detach(), **metadata}
    return loss, out_meta
