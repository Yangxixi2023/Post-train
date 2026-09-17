from __future__ import annotations
from typing import Dict, List
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def tokenize_prompt_and_output(
    prompt_strs: List[str],
    output_strs: List[str],
    tokenizer: PreTrainedTokenizerBase,
) -> Dict[str, torch.Tensor]:
    assert len(prompt_strs) == len(output_strs)
    if len(prompt_strs) == 0:
        raise ValueError("prompt_strs/output_strs must be non-empty")

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("tokenizer must define pad_token_id or eos_token_id")

    sequences: list[list[int]] = []
    masks: list[list[int]] = []
    for prompt, output in zip(prompt_strs, output_strs):
        p_ids = tokenizer.encode(prompt, add_special_tokens=False)
        o_ids = tokenizer.encode(output, add_special_tokens=False)
        ids = p_ids + o_ids
        if len(ids) < 2:
            raise ValueError("prompt+output must contain at least two tokens")
        sequences.append(ids)
        masks.append([0] * len(p_ids) + [1] * len(o_ids))

    max_len = max(map(len, sequences))
    batch = len(sequences)
    padded = torch.full((batch, max_len), pad_id, dtype=torch.long)
    response_mask_full = torch.zeros((batch, max_len), dtype=torch.bool)
    for i, (ids, mask) in enumerate(zip(sequences, masks)):
        n = len(ids)
        padded[i, :n] = torch.tensor(ids, dtype=torch.long)
        response_mask_full[i, :n] = torch.tensor(mask, dtype=torch.bool)

    # next-token alignment: logits from input_ids[:, t] score labels[:, t]
    return {
        "input_ids": padded[:, :-1],
        "labels": padded[:, 1:].clone(),
        "response_mask": response_mask_full[:, 1:],
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    # H(softmax(z)) = logsumexp(z) - E_p[z], numerically stable.
    lse = torch.logsumexp(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    return lse - torch.sum(probs * logits, dim=-1)


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> Dict[str, torch.Tensor]:
    logits = model(input_ids).logits
    log_probs = F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    out: Dict[str, torch.Tensor] = {"log_probs": log_probs}
    if return_token_entropy:
        out["token_entropy"] = compute_entropy(logits)
    return out


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    if normalize_constant == 0:
        raise ValueError("normalize_constant must be non-zero")
    masked = tensor * mask.to(dtype=tensor.dtype)
    return masked.sum(dim=dim) / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    batch_size = policy_log_probs.shape[0]
    nll = -policy_log_probs
    # Assignment convention: sum masked token NLL / normalize_constant, then mean over examples.
    unscaled = masked_normalize(nll, response_mask, normalize_constant, dim=None) / batch_size
    loss = unscaled / gradient_accumulation_steps
    loss.backward()
    return loss, {"loss": unscaled.detach()}
