from __future__ import annotations
import torch
import torch.nn.functional as F

def _response_logprob(model,tokenizer,prompt,response):
    p=tokenizer.encode(prompt,add_special_tokens=False)
    r=tokenizer.encode(response,add_special_tokens=False)
    ids=torch.tensor([p+r],dtype=torch.long,device=next(model.parameters()).device)
    logits=model(ids[:,:-1]).logits
    labels=ids[:,1:]
    lp=F.log_softmax(logits,dim=-1).gather(-1,labels.unsqueeze(-1)).squeeze(-1)
    # label index len(prompt)-1 is the first response token.
    start=max(len(p)-1,0)
    return lp[:,start:].sum(dim=1)

def compute_dpo_loss(model,ref_model,tokenizer,beta,prompt,response_chosen,response_rejected):
    pi_c=_response_logprob(model,tokenizer,prompt,response_chosen)
    pi_r=_response_logprob(model,tokenizer,prompt,response_rejected)
    with torch.no_grad():
        ref_c=_response_logprob(ref_model,tokenizer,prompt,response_chosen)
        ref_r=_response_logprob(ref_model,tokenizer,prompt,response_rejected)
    logits=beta*((pi_c-pi_r)-(ref_c-ref_r))
    loss=-F.logsigmoid(logits).mean()
    return loss,{'chosen_reward':beta*(pi_c-ref_c),'rejected_reward':beta*(pi_r-ref_r)}
