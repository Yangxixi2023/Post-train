from __future__ import annotations
import argparse, json, random, subprocess
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import SamplingParams
from cs336_alignment.sft_utils import tokenize_prompt_and_output, get_response_log_probs
from cs336_alignment.grpo_utils import compute_group_normalized_rewards, grpo_microbatch_train_step
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn
from cs336_alignment.train_utils import read_jsonl, extract_gold, format_prompt, init_vllm, load_policy_into_vllm_instance, evaluate_vllm, seed_everything, save_run_metadata

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',required=True); ap.add_argument('--train-data',required=True); ap.add_argument('--val-data',required=True); ap.add_argument('--output-dir',required=True)
    ap.add_argument('--grpo-steps',type=int,default=4); ap.add_argument('--prompts-per-rollout',type=int,default=8); ap.add_argument('--group-size',type=int,default=4)
    ap.add_argument('--epochs-per-rollout',type=int,default=1); ap.add_argument('--loss-type',choices=['no_baseline','reinforce_with_baseline','grpo_clip','grpo_no_clip'],default='reinforce_with_baseline')
    ap.add_argument('--cliprange',type=float,default=.2); ap.add_argument('--normalize-by-std',action=argparse.BooleanOptionalAction,default=True)
    ap.add_argument('--length-normalization',choices=['mean','constant'],default='mean'); ap.add_argument('--constant-normalizer',type=float,default=256.0)
    ap.add_argument('--prompt-template',default='cs336_alignment/prompts/r1_zero.prompt'); ap.add_argument('--reward-type',choices=['r1_zero','question_only'],default='r1_zero')
    ap.add_argument('--lr',type=float,default=1e-5); ap.add_argument('--grad-accum',type=int,default=8); ap.add_argument('--train-batch-size',type=int,default=0,help='If >0, accumulate this many rollout samples per optimizer update (microbatch=1).'); ap.add_argument('--max-seq-len',type=int,default=768)
    ap.add_argument('--max-new-tokens',type=int,default=256); ap.add_argument('--eval-examples',type=int,default=64); ap.add_argument('--eval-every',type=int,default=1)
    ap.add_argument('--save-intermediate',action=argparse.BooleanOptionalAction,default=False); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--policy-device',default='cuda:0'); ap.add_argument('--vllm-device',default='cuda:1'); ap.add_argument('--vllm-gpu-util',type=float,default=.5)
    args=ap.parse_args(); seed_everything(args.seed)
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); rng=random.Random(args.seed)
    train=read_jsonl(args.train_data); val=read_jsonl(args.val_data)[:args.eval_examples]; tpl=Path(args.prompt_template).read_text(); reward_fn = r1_zero_reward_fn if args.reward_type == 'r1_zero' else question_only_reward_fn
    tok=AutoTokenizer.from_pretrained(args.model,trust_remote_code=True); tok.pad_token=tok.pad_token or tok.eos_token
    policy=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16,attn_implementation='sdpa',trust_remote_code=True).to(args.policy_device)
    policy.gradient_checkpointing_enable(); policy.config.use_cache=False
    opt=torch.optim.AdamW(policy.parameters(),lr=args.lr,weight_decay=0.0,betas=(.9,.95))
    llm=init_vllm(args.model,args.vllm_device,args.seed,args.vllm_gpu_util,max_model_len=max(1024,args.max_seq_len+args.max_new_tokens))
    valp=[format_prompt(x,tpl) for x in val]; valg=[extract_gold(x) for x in val]
    logf=(out/'metrics.jsonl').open('w')
    def eval_now(step):
        policy.eval(); load_policy_into_vllm_instance(policy,llm)
        m,_=evaluate_vllm(llm,reward_fn,valp,valg,temperature=1.0,max_tokens=args.max_new_tokens,seed=args.seed+1000+step,output_path=str(out/f'eval_grpo{step}.jsonl'))
        logf.write(json.dumps({'grpo_step':step,'eval':m})+'\n'); logf.flush(); print({'grpo_step':step,'eval':m},flush=True); policy.train(); return m
    eval_now(0)
    rollout_params=SamplingParams(n=args.group_size,temperature=1.0,top_p=1.0,max_tokens=args.max_new_tokens,min_tokens=4,stop=['</answer>'],include_stop_str_in_output=True)
    for step in range(1,args.grpo_steps+1):
        policy.eval(); load_policy_into_vllm_instance(policy,llm)
        qs=rng.sample(train,min(args.prompts_per_rollout,len(train))); prompts=[format_prompt(x,tpl) for x in qs]; golds=[extract_gold(x) for x in qs]
        outs=llm.generate(prompts,rollout_params,use_tqdm=True)
        flatp=[]; flatr=[]; flatg=[]
        for p,g,o in zip(prompts,golds,outs):
            for cand in o.outputs: flatp.append(p); flatr.append(cand.text); flatg.append(g)
        adv,raw,rmeta=compute_group_normalized_rewards(reward_fn,flatr,flatg,args.group_size,1e-6,args.normalize_by_std)
        # tokenize individually and keep only samples fitting max length. Retain group order; training can skip overlong samples.
        items=[]
        for i,(p,r) in enumerate(zip(flatp,flatr)):
            t=tokenize_prompt_and_output([p],[r],tok)
            if t['input_ids'].shape[1] <= args.max_seq_len: items.append((i,t))
        if not items:
            logf.write(json.dumps({'grpo_step':step,'skip':'all overlong','reward_meta':rmeta})+'\n'); logf.flush(); continue
        # old policy scores fixed before updates
        policy.eval(); old=[]
        with torch.inference_mode():
            for i,t in items:
                ids=t['input_ids'].to(args.policy_device); lab=t['labels'].to(args.policy_device)
                old.append(get_response_log_probs(policy,ids,lab,False)['log_probs'].detach().cpu())
        policy.train(); updates=0; clip_sum=0.0; loss_sum=0.0; entropy_sum=0.0; entropy_count=0; grad_norm_sum=0.0; accum_target=args.train_batch_size if args.train_batch_size>0 else args.grad_accum
        for ep in range(args.epochs_per_rollout):
            order=list(range(len(items))); rng.shuffle(order); opt.zero_grad(set_to_none=True); acc=0
            for pos,j in enumerate(order):
                orig,t=items[j]; ids=t['input_ids'].to(args.policy_device); lab=t['labels'].to(args.policy_device); mask=t['response_mask'].to(args.policy_device)
                lp=get_response_log_probs(policy,ids,lab,True)
                rr=raw[orig].view(1,1).to(args.policy_device); aa=adv[orig].view(1,1).to(args.policy_device); olp=old[j].to(args.policy_device)
                lt=args.loss_type
                # clipping is meaningful after old-policy rollout; it is valid even on the first update where ratio starts at 1.
                loss,meta=grpo_microbatch_train_step(lp['log_probs'],mask,accum_target,lt,rr,aa,olp,args.cliprange,args.length_normalization,args.constant_normalizer)
                loss_sum += float(loss.detach())*accum_target; clip_sum += float(meta.get('clip_fraction',torch.tensor(0.0))); entropy_sum += float((lp['token_entropy'] * mask).sum().detach()); entropy_count += int(mask.sum()); acc+=1
                if acc==accum_target or pos==len(order)-1:
                    gn=torch.nn.utils.clip_grad_norm_(policy.parameters(),1.0); grad_norm_sum += float(gn); opt.step(); opt.zero_grad(set_to_none=True); acc=0; updates+=1
        rec={'grpo_step':step,'reward_meta':rmeta,'samples':len(items),'updates':updates,'loss_sum':loss_sum,'mean_clip_fraction':clip_sum/max(1,len(items)*args.epochs_per_rollout),'mean_token_entropy':entropy_sum/max(1,entropy_count),'mean_grad_norm':grad_norm_sum/max(1,updates),'mean_response_tokens':sum(len(t['labels'][0]) for _,t in items)/max(1,len(items)),'train_batch_size':accum_target,'epochs_per_rollout':args.epochs_per_rollout}
        logf.write(json.dumps(rec)+'\n'); logf.flush(); print(rec,flush=True)
        if step%args.eval_every==0: eval_now(step)
        
        if args.save_intermediate:
            ck=out/f'checkpoint_grpo{step}'; policy.save_pretrained(ck,safe_serialization=True); tok.save_pretrained(ck)
    final=out/'checkpoint_final'; policy.save_pretrained(final,safe_serialization=True); tok.save_pretrained(final)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(); save_run_metadata(str(out/'run.json'),{'git_commit':commit,'args':vars(args),'checkpoint':str(final.resolve())})
    logf.close()
if __name__=='__main__': main()
