from __future__ import annotations
import json, glob
from pathlib import Path

def load_json(p): return json.loads(Path(p).read_text())
def eval_curve(p, step_key):
    out=[]
    if not Path(p).exists(): return out
    for line in open(p):
        x=json.loads(line)
        if 'eval' in x: out.append({'step':x[step_key],**x['eval']})
    return out
summary={}
base='experiments/math_assignment/baseline/base_5000.jsonl.metrics.json'
if Path(base).exists(): summary['math_base_5000']=load_json(base)
summary['sft_size_sweep']={}
for p in sorted(glob.glob('experiments/math_assignment/sft/n*/train_metrics.jsonl')):
    name=Path(p).parent.name; summary['sft_size_sweep'][name]=eval_curve(p,'step')
fp='experiments/math_assignment/sft_filtered/train_metrics.jsonl'
if Path(fp).exists(): summary['sft_filtered']=eval_curve(fp,'step')
summary['ei']={}
for p in sorted(glob.glob('experiments/math_assignment/ei/*/metrics.jsonl')):
    summary['ei'][Path(p).parent.name]=eval_curve(p,'ei_step')
summary['grpo_ablations']={}
for p in sorted(glob.glob('experiments/math_assignment/grpo/*/metrics.saved.jsonl')):
    summary['grpo_ablations'][Path(p).parent.name]=eval_curve(p,'grpo_step')
old='experiments/math_assignment/leaderboard/metrics.jsonl'
if Path(old).exists(): summary['leaderboard_unstable']=eval_curve(old,'grpo_step')
pre='experiments/math_assignment/leaderboard_fixed100/metrics.jsonl'
res='experiments/math_assignment/leaderboard_fixed100_resume60/metrics.jsonl'
summary['leaderboard_fixed_part1']=eval_curve(pre,'grpo_step')
summary['leaderboard_fixed_part2']=eval_curve(res,'grpo_step')
final='experiments/math_assignment/final_fixed100_eval/validation_5000.jsonl.metrics.json'
if Path(final).exists(): summary['grpo_fixed100_5000']=load_json(final)
for name in ['sft_full','sft_filtered','ei_final']:
    p=f'experiments/math_assignment/final_stage_eval/{name}_5000.jsonl.metrics.json'
    if Path(p).exists(): summary[name+'_5000']=load_json(p)
Path('results/assignment5_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
