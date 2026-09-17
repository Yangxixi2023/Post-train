from __future__ import annotations
import argparse, json, random, subprocess
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import SamplingParams
from cs336_alignment.sft_utils import tokenize_prompt_and_output, get_response_log_probs, sft_microbatch_train_step
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.train_utils import read_jsonl, extract_gold, format_prompt, init_vllm, load_policy_into_vllm_instance, evaluate_vllm, seed_everything, save_run_metadata

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',required=True); ap.add_argument('--train-data',required=True); ap.add_argument('--val-data',required=True); ap.add_argument('--output-dir',required=True)
    ap.add_argument('--ei-steps',type=int,default=2); ap.add_argument('--questions-per-step',type=int,default=32); ap.add_argument('--group-size',type=int,default=4)
    ap.add_argument('--sft-epochs',type=int,default=1); ap.add_argument('--lr',type=float,default=1e-5); ap.add_argument('--grad-accum',type=int,default=8)
    ap.add_argument('--max-seq-len',type=int,default=768); ap.add_argument('--max-new-tokens',type=int,default=256); ap.add_argument('--eval-examples',type=int,default=64)
    ap.add_argument('--seed',type=int,default=42); ap.add_argument('--policy-device',default='cuda:0'); ap.add_argument('--vllm-device',default='cuda:1'); ap.add_argument('--vllm-gpu-util',type=float,default=.5)
    args=ap.parse_args(); seed_everything(args.seed)
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    train=read_jsonl(args.train_data); val=read_jsonl(args.val_data)[:args.eval_examples]; tpl=Path('cs336_alignment/prompts/r1_zero.prompt').read_text()
    tok=AutoTokenizer.from_pretrained(args.model,trust_remote_code=True); tok.pad_token=tok.pad_token or tok.eos_token
    policy=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16,attn_implementation='sdpa',trust_remote_code=True).to(args.policy_device)
    policy.gradient_checkpointing_enable(); policy.config.use_cache=False
    opt=torch.optim.AdamW(policy.parameters(),lr=args.lr,weight_decay=0.0,betas=(.9,.95))
    llm=init_vllm(args.model,args.vllm_device,args.seed,args.vllm_gpu_util,max_model_len=max(1024,args.max_seq_len+args.max_new_tokens))
    valp=[format_prompt(x,tpl) for x in val]; valg=[extract_gold(x) for x in val]
    logf=(out/'metrics.jsonl').open('w')
    def eval_now(step):
        policy.eval(); load_policy_into_vllm_instance(policy,llm)
        m,_=evaluate_vllm(llm,r1_zero_reward_fn,valp,valg,temperature=1.0,max_tokens=args.max_new_tokens,seed=args.seed+step,output_path=str(out/f'eval_ei{step}.jsonl'))
        logf.write(json.dumps({'ei_step':step,'eval':m})+'\n'); logf.flush(); print({'ei_step':step,'eval':m},flush=True); policy.train(); return m
    eval_now(0)
    rng=random.Random(args.seed)
    rollout_params=SamplingParams(n=args.group_size,temperature=1.0,top_p=1.0,max_tokens=args.max_new_tokens,min_tokens=4,stop=['</answer>'],include_stop_str_in_output=True,seed=args.seed)
    for ei in range(1,args.ei_steps+1):
        policy.eval(); load_policy_into_vllm_instance(policy,llm)
        batch=rng.sample(train,min(args.questions_per_step,len(train)))
        prompts=[format_prompt(x,tpl) for x in batch]; golds=[extract_gold(x) for x in batch]
        outs=llm.generate(prompts,rollout_params,use_tqdm=True)
        expert=[]; total=0; correct_q=0
        for p,g,o in zip(prompts,golds,outs):
            any_ok=False
            for cand in o.outputs:
                total+=1
                if r1_zero_reward_fn(cand.text,g)['reward']==1.0:
                    expert.append({'prompt':p,'response':cand.text}); any_ok=True
            correct_q += int(any_ok)
        with (out/f'expert_ei{ei}.jsonl').open('w') as f:
            for x in expert: f.write(json.dumps(x,ensure_ascii=False)+'\n')
        stat={'ei_step':ei,'expert_examples':len(expert),'rollouts':total,'rollout_success_rate':len(expert)/max(1,total),'question_solve_rate':correct_q/max(1,len(batch))}
        logf.write(json.dumps(stat)+'\n'); logf.flush(); print(stat,flush=True)
        if not expert:
            eval_now(ei); continue
        # Filter examples that fit memory, then SFT over correct traces.
        fitted=[]
        for ex in expert:
            t=tokenize_prompt_and_output([ex['prompt']],[ex['response']],tok)
            if t['input_ids'].shape[1] <= args.max_seq_len: fitted.append(ex)
        policy.train()
        order=list(range(len(fitted))); entropy_sum=0.0; entropy_count=0
        for ep in range(args.sft_epochs):
            rng.shuffle(order); opt.zero_grad(set_to_none=True); accum=0; loss_sum=0.0
            for j,idx in enumerate(order):
                ex=fitted[idx]; t=tokenize_prompt_and_output([ex['prompt']],[ex['response']],tok)
                ids=t['input_ids'].to(args.policy_device); lab=t['labels'].to(args.policy_device); mask=t['response_mask'].to(args.policy_device)
                lp_out=get_response_log_probs(policy,ids,lab,True); lp=lp_out['log_probs']
                entropy_sum += float((lp_out['token_entropy'] * mask).sum().detach()); entropy_count += int(mask.sum())
                loss,_=sft_microbatch_train_step(lp,mask,args.grad_accum,1.0); loss_sum += loss.item()*args.grad_accum; accum+=1
                if accum==args.grad_accum or j==len(order)-1:
                    gn=torch.nn.utils.clip_grad_norm_(policy.parameters(),1.0); opt.step(); opt.zero_grad(set_to_none=True); accum=0
            logf.write(json.dumps({'ei_step':ei,'epoch':ep+1,'sft_loss_sum':loss_sum,'mean_token_entropy':entropy_sum/max(1,entropy_count)})+'\n'); logf.flush()
        eval_now(ei)
        ck=out/f'checkpoint_ei{ei}'; policy.save_pretrained(ck,safe_serialization=True); tok.save_pretrained(ck)
    final=out/'checkpoint_final'; policy.save_pretrained(final,safe_serialization=True); tok.save_pretrained(final)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(); save_run_metadata(str(out/'run.json'),{'git_commit':commit,'args':vars(args),'checkpoint':str(final.resolve())})
    logf.close()
if __name__=='__main__': main()
