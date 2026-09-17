from __future__ import annotations
import json, glob, os
from pathlib import Path
import matplotlib.pyplot as plt
ROOT=Path('/workspace/assignment5-alignment'); OUT=ROOT/'experiments/results'; OUT.mkdir(parents=True,exist_ok=True)

def readj(path):
    return [json.loads(x) for x in open(path) if x.strip()]
def evals(path,key='step'):
    if not Path(path).exists(): return []
    return [x for x in readj(path) if 'eval' in x]
def plot_runs(runs,title,path,xkey):
    plt.figure(figsize=(7,4.5))
    any_=False
    for name,p in runs:
        es=evals(p)
        if not es: continue
        xs=[e.get(xkey,e.get('grpo_step',e.get('ei_step',e.get('step',0)))) for e in es]
        ys=[e['eval'].get('answer_reward',e['eval'].get('reward',0)) for e in es]
        plt.plot(xs,ys,marker='o',label=name); any_=True
    if any_:
        plt.xlabel(xkey); plt.ylabel('validation answer reward'); plt.title(title); plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(path,dpi=160)
    plt.close()

# SFT dataset-size curves
sft=[]
for p in sorted(glob.glob(str(ROOT/'experiments/sft_sweep/n*/metrics.saved.jsonl'))): sft.append((Path(p).parent.name,p))
plot_runs(sft,'SFT validation curves by unique dataset size',OUT/'sft_dataset_size_curves.png','step')
# EI validation and entropy
runs=[('G=4, epochs=1',ROOT/'experiments/ei_sweep/g4_e1/metrics.jsonl'),('G=8, epochs=2',ROOT/'experiments/ei_sweep/g8_e2/metrics.jsonl')]
plot_runs(runs,'Expert Iteration validation reward',OUT/'ei_validation_curves.png','ei_step')
plt.figure(figsize=(7,4.5)); any_=False
for name,p in runs:
    if not Path(p).exists(): continue
    rows=readj(p); pts=[x for x in rows if 'mean_token_entropy' in x]
    if pts:
        plt.plot([x['ei_step']+(x.get('epoch',1)-1)*.05 for x in pts],[x['mean_token_entropy'] for x in pts],marker='o',label=name); any_=True
if any_:
    plt.xlabel('EI step'); plt.ylabel('mean response-token entropy'); plt.title('EI response entropy over training'); plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(OUT/'ei_entropy.png',dpi=160)
plt.close()
# GRPO experiment groups
groot=ROOT/'experiments/grpo_assignment'
groups={
 'grpo_lr_sweep.png':[('2e-5',groot/'lr_2e5/metrics.saved.jsonl'),('3e-5',groot/'lr_3e5/metrics.saved.jsonl'),('4e-5',groot/'lr_4e5/metrics.saved.jsonl')],
 'grpo_baseline_ablation.png':[('reinforce+baseline',groot/'lr_3e5/metrics.saved.jsonl'),('no baseline',groot/'baseline_no_baseline/metrics.saved.jsonl')],
 'grpo_length_norm.png':[('masked mean',groot/'lr_3e5/metrics.saved.jsonl'),('constant normalize',groot/'length_constant/metrics.saved.jsonl')],
 'grpo_std_norm.png':[('std=True',groot/'lr_3e5/metrics.saved.jsonl'),('std=False',groot/'std_false/metrics.saved.jsonl')],
 'grpo_offpolicy.png':[('on-policy ref',groot/'lr_3e5/metrics.saved.jsonl'),('epochs2,batch256',groot/'offpolicy_e2_b256/metrics.saved.jsonl'),('epochs4,batch128',groot/'offpolicy_e4_b128/metrics.saved.jsonl'),('focused',groot/'offpolicy_focused/metrics.saved.jsonl')],
 'grpo_clip_ablation.png':[('GRPO-Clip',groot/'offpolicy_focused/metrics.saved.jsonl'),('GRPO-No-Clip',groot/'offpolicy_no_clip/metrics.saved.jsonl')],
 'grpo_prompt_ablation.png':[('R1-Zero',groot/'lr_3e5/metrics.saved.jsonl'),('question-only',groot/'prompt_question_only/metrics.saved.jsonl')],
}
for fn,r in groups.items(): plot_runs([(n,str(p)) for n,p in r],fn.replace('_',' ').replace('.png',''),OUT/fn,'grpo_step')
# off-policy wall-clock plot
plt.figure(figsize=(7,4.5)); any_=False
for name,p in groups['grpo_offpolicy.png']:
    if not Path(p).exists(): continue
    es=evals(p)
    if es and all('elapsed_seconds' in x for x in es):
        plt.plot([x['elapsed_seconds']/60 for x in es],[x['eval']['answer_reward'] for x in es],marker='o',label=name); any_=True
if any_:
    plt.xlabel('wall-clock minutes'); plt.ylabel('validation answer reward'); plt.title('Off-policy comparison vs wall-clock'); plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(OUT/'grpo_offpolicy_wallclock.png',dpi=160)
plt.close()
# compact JSON summary
summary={}
base=ROOT/'experiments/math_baseline_gsm8k/base_full.jsonl.metrics.json'
if base.exists(): summary['zero_shot_full_gsm8k']=json.load(open(base))
for name,p in [('SFT',ROOT/'experiments/formal_sft/run2/train_metrics.jsonl'),('EI',ROOT/'experiments/formal_ei/run1/metrics.jsonl'),('GRPO_initial',ROOT/'experiments/formal_grpo/run0_prev/metrics.jsonl')]:
    es=evals(p)
    if es: summary[name]={'final_eval':es[-1]['eval'],'best_answer_reward':max(x['eval']['answer_reward'] for x in es)}
leader=groot/'leaderboard_final/full_validation_1319.jsonl.metrics.json'
if leader.exists(): summary['leaderboard_full_gsm8k']=json.load(open(leader))
json.dump(summary,open(OUT/'summary.json','w'),indent=2)
print(json.dumps(summary,indent=2))
