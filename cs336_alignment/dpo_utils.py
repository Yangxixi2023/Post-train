from __future__ import annotations
import torch
import torch.nn.functional as F

ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{prompt}\n\n### Response:\n{response}"
)

def _sequence_logprob(model, input_ids: torch.Tensor) -> torch.Tensor:
    logits = model(input_ids).logits
    labels = input_ids[:, 1:]
    token_lp = F.log_softmax(logits[:, :-1], dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return token_lp.sum(dim=-1)

def compute_dpo_loss(model, ref_model, tokenizer, beta, prompt, response_chosen, response_rejected):
    """Per-instance DPO objective using the assignment's Alpaca formatting + EOS convention."""
    device = next(model.parameters()).device
    def encode(response: str):
        text = ALPACA_TEMPLATE.format(prompt=prompt, response=response) + tokenizer.eos_token
        return tokenizer.encode(text, return_tensors='pt').to(device)
    chosen = encode(response_chosen); rejected = encode(response_rejected)
    pi_c = _sequence_logprob(model, chosen); pi_r = _sequence_logprob(model, rejected)
    with torch.no_grad():
        ref_device = next(ref_model.parameters()).device
        ref_c = _sequence_logprob(ref_model, chosen.to(ref_device)).to(device)
        ref_r = _sequence_logprob(ref_model, rejected.to(ref_device)).to(device)
    chosen_reward = beta * (pi_c - ref_c)
    rejected_reward = beta * (pi_r - ref_r)
    loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()
    return loss, {'chosen_reward': chosen_reward.detach(), 'rejected_reward': rejected_reward.detach()}
