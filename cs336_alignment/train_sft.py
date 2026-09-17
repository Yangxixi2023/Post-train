from __future__ import annotations
import argparse, json, os, random, subprocess
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from cs336_alignment.sft_utils import tokenize_prompt_and_output, get_response_log_probs, sft_microbatch_train_step
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.train_utils import read_jsonl, extract_gold, format_prompt, init_vllm, load_policy_into_vllm_instance, evaluate_vllm, seed_everything, save_run_metadata

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',required=True); ap.add_argument('--train-data',required=True); ap.add_argument('--val-data',required=True)
    ap.add_argument('--output-dir',required=True); ap.add_argument('--steps',type=int,default=20); ap.add_argument('--lr',type=float,default=2e-5)
    ap.add_argument('--micro-batch-size',type=int,default=1); ap.add_argument('--grad-accum',type=int,default=8)
    ap.add_argument('--max-seq-len',type=int,default=768); ap.add_argument('--train-examples',type=int,default=512)
    ap.add_argument('--eval-examples',type=int,default=64); ap.add_argument('--eval-every',type=int,default=10)
    ap.add_argument('--max-new-tokens',type=int,default=256); ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--policy-device',default='cuda:0'); ap.add_argument('--vllm-device',default='cuda:1'); ap.add_argument('--vllm-gpu-util',type=float,default=.5)
    args=ap.parse_args(); seed_everything(args.seed)
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    data=read_jsonl(args.train_data); random.Random(args.seed).shuffle(data); data=data[:args.train_examples]
    val=read_jsonl(args.val_data)[:args.eval_examples]
    tpl=Path('cs336_alignment/prompts/r1_zero.prompt').read_text()
    tok=AutoTokenizer.from_pretrained(args.model,trust_remote_code=True); tok.pad_token=tok.pad_token or tok.eos_token
    policy=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16,attn_implementation='sdpa',trust_remote_code=True).to(args.policy_device)
    policy.gradient_checkpointing_enable(); policy.config.use_cache=False
    opt=torch.optim.AdamW(policy.parameters(),lr=args.lr,weight_decay=0.0,betas=(.9,.95))
    llm=init_vllm(args.model,args.vllm_device,args.seed,args.vllm_gpu_util,max_model_len=max(1024,args.max_seq_len+args.max_new_tokens))
    val_prompts=[format_prompt(x,tpl) for x in val]; val_golds=[extract_gold(x) for x in val]
    logf=(out/'train_metrics.jsonl').open('w')
    def do_eval(step):
        policy.eval(); load_policy_into_vllm_instance(policy,llm)
        m,_=evaluate_vllm(llm,r1_zero_reward_fn,val_prompts,val_golds,temperature=1.0,max_tokens=args.max_new_tokens,seed=args.seed+step,output_path=str(out/f'eval_step{step}.jsonl'))
        rec={'step':step,'eval':m}; logf.write(json.dumps(rec)+'\n'); logf.flush(); print(rec,flush=True); policy.train(); return m
    do_eval(0)
    policy.train(); opt.zero_grad(set_to_none=True)
    cursor=0
    for step in range(1,args.steps+1):
        loss_sum=0.0
        for _ in range(args.grad_accum):
            batch=[]
            while len(batch)<args.micro_batch_size:
                ex=data[cursor%len(data)]; cursor+=1
                t=tokenize_prompt_and_output([ex['prompt']],[ex['response']],tok)
                if t['input_ids'].shape[1] <= args.max_seq_len: batch.append(ex)
            t=tokenize_prompt_and_output([x['prompt'] for x in batch],[x['response'] for x in batch],tok)
            ids=t['input_ids'].to(args.policy_device); labels=t['labels'].to(args.policy_device); mask=t['response_mask'].to(args.policy_device)
            res=get_response_log_probs(policy,ids,labels,return_token_entropy=False)
            loss,_=sft_microbatch_train_step(res['log_probs'],mask,args.grad_accum,normalize_constant=1.0)
            loss_sum += loss.detach().item()*args.grad_accum
        gn=torch.nn.utils.clip_grad_norm_(policy.parameters(),1.0); opt.step(); opt.zero_grad(set_to_none=True)
        rec={'step':step,'train_loss':loss_sum/args.grad_accum,'grad_norm':float(gn)}; logf.write(json.dumps(rec)+'\n'); logf.flush(); print(rec,flush=True)
        if step%args.eval_every==0 or step==args.steps: do_eval(step)
    ckpt=out/'checkpoint_final'; policy.save_pretrained(ckpt,safe_serialization=True); tok.save_pretrained(ckpt)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    save_run_metadata(str(out/'run.json'),{'git_commit':commit,'args':vars(args),'seed':args.seed,'policy_gpu':args.policy_device,'vllm_gpu':args.vllm_device,'checkpoint':str(ckpt.resolve())})
    logf.close()
if __name__=='__main__': main()
