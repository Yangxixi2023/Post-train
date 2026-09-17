from __future__ import annotations
import argparse, json
from pathlib import Path
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn
from cs336_alignment.train_utils import read_jsonl, extract_gold, format_prompt, init_vllm, evaluate_vllm

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',required=True); ap.add_argument('--data',required=True)
    ap.add_argument('--prompt-template',default='cs336_alignment/prompts/r1_zero.prompt'); ap.add_argument('--reward-type',choices=['r1_zero','question_only'],default='r1_zero')
    ap.add_argument('--gpu',default='cuda:1'); ap.add_argument('--n',type=int,default=128)
    ap.add_argument('--temperature',type=float,default=1.0); ap.add_argument('--max-tokens',type=int,default=256)
    ap.add_argument('--seed',type=int,default=42); ap.add_argument('--gpu-memory-utilization',type=float,default=.55)
    ap.add_argument('--output',required=True)
    args=ap.parse_args()
    template=Path(args.prompt_template).read_text()
    data=read_jsonl(args.data)[:args.n]
    prompts=[format_prompt(x,template) for x in data]; golds=[extract_gold(x) for x in data]
    llm=init_vllm(args.model,args.gpu,args.seed,args.gpu_memory_utilization,max_model_len=2048)
    reward_fn=r1_zero_reward_fn if args.reward_type=='r1_zero' else question_only_reward_fn
    metrics,_=evaluate_vllm(llm,reward_fn,prompts,golds,temperature=args.temperature,
                            max_tokens=args.max_tokens,seed=args.seed,output_path=args.output)
    print(json.dumps(metrics,indent=2))
    Path(args.output+'.metrics.json').write_text(json.dumps(metrics,indent=2))
if __name__=='__main__': main()
