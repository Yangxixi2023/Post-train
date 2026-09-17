from __future__ import annotations
import argparse, json
from pathlib import Path

def convert(src: Path, dst: Path, template: str) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    n=0
    with src.open(encoding='utf-8') as fi, dst.open('w',encoding='utf-8') as fo:
        for line in fi:
            if not line.strip(): continue
            ex=json.loads(line); ans=str(ex['answer'])
            if '####' in ans: rationale, final=ans.rsplit('####',1)
            else: rationale, final=ans, ans
            final=final.strip().replace(',','')
            prompt=template.replace('{question}', ex['question'])
            response=f"{rationale.strip()}\n</think>\n<answer>{final}</answer>"
            fo.write(json.dumps({'question':ex['question'],'prompt':prompt,'response':response,'answer':final},ensure_ascii=False)+'\n')
            n+=1
    return n

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--gsm8k-dir',default='data/gsm8k')
    ap.add_argument('--output-dir',default='experiments/data')
    ap.add_argument('--prompt-template',default='cs336_alignment/prompts/r1_zero.prompt')
    args=ap.parse_args()
    template=Path(args.prompt_template).read_text()
    for split in ('train','test'):
        src=Path(args.gsm8k_dir)/f'{split}.jsonl'; dst=Path(args.output_dir)/f'gsm8k_{split}_r1.jsonl'
        print(split, convert(src,dst,template), dst)
if __name__=='__main__': main()
