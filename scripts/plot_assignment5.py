from __future__ import annotations
import json, math
from pathlib import Path
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')
root=Path('/workspace/assignment5-alignment')
out=root/'plots'; out.mkdir(exist_ok=True)

def rows(p):
    q=root/p
    return [json.loads(x) for x in q.open()] if q.exists() else []
def evals(p,key): return [(x[key],x['eval']['answer_reward']) for x in rows(p) if 'eval' in x]
# SFT size: best validation by number of examples
xs=[]; ys=[]
for n in [128,256,512,1024,1767]:
    c=evals(f'experiments/math_assignment/sft/n{n}/train_metrics.jsonl','step')
    if c: xs.append(n); ys.append(max(v for _,v in c))
fig,ax=plt.subplots(figsize=(7,4.2)); ax.plot(xs,[100*y for y in ys],marker='o',lw=2); ax.set_xscale('log',base=2); ax.set_xticks(xs,labels=xs); ax.set_xlabel('SFT examples'); ax.set_ylabel('Best validation answer reward (%)'); ax.set_title('MATH SFT data-size sweep'); fig.tight_layout(); fig.savefig(out/'sft_size_sweep.png',dpi=180); plt.close(fig)
# GRPO ablations compact bar chart using best eval
names={'lr_1e-5':'LR 1e-5','lr_2e-5':'LR 2e-5','lr_3e-5':'LR 3e-5','no_baseline':'No baseline','reinforce_baseline':'Baseline','length_constant':'Const length','std_false':'No std norm','offpolicy_e2_b128':'Off-policy 2ep','offpolicy_e4_b128':'Off-policy 4ep','offpolicy_focused':'Focused','clip_no':'No clip','question_only':'Question-only'}
vals=[]
for d,label in names.items():
    c=evals(f'experiments/math_assignment/grpo/{d}/metrics.saved.jsonl','grpo_step')
    if c: vals.append((label,100*max(v for _,v in c)))
fig,ax=plt.subplots(figsize=(10,5)); ax.bar([x[0] for x in vals],[x[1] for x in vals],color='#4C78A8'); ax.set_ylabel('Best validation answer reward (%)'); ax.set_title('GRPO ablations on MATH'); ax.tick_params(axis='x',rotation=55); fig.tight_layout(); fig.savefig(out/'grpo_ablations.png',dpi=180); plt.close(fig)
# unstable vs fixed diagnostics
old=rows('experiments/math_assignment/leaderboard/metrics.jsonl'); fixed=rows('experiments/math_assignment/leaderboard_fixed100/metrics.jsonl')
fig,axs=plt.subplots(2,1,figsize=(8,7),sharex=False)
for rr,label,color in [(old,'old 3e-5','#E45756'),(fixed,'fixed 5e-6 part 1','#54A24B')]:
    pts=[(x['grpo_step'],x.get('mean_grad_norm')) for x in rr if 'mean_grad_norm' in x and isinstance(x.get('mean_grad_norm'),(int,float)) and math.isfinite(x['mean_grad_norm'])]
    if pts: axs[0].plot([x for x,_ in pts],[max(y,1e-8) for _,y in pts],label=label,color=color)
axs[0].set_yscale('log'); axs[0].set_ylabel('Pre-clip grad norm (log)'); axs[0].legend(); axs[0].set_title('Late-stage GRPO instability and repair')
for rr,label,color in [(old,'old 3e-5','#E45756'),(fixed,'fixed 5e-6 part 1','#54A24B')]:
    pts=[(x['grpo_step'],100*x['eval']['answer_reward']) for x in rr if 'eval' in x]
    if pts: axs[1].plot([x for x,_ in pts],[y for _,y in pts],marker='o',label=label,color=color)
axs[1].set_xlabel('GRPO step'); axs[1].set_ylabel('Validation answer reward (%)'); axs[1].legend(); fig.tight_layout(); fig.savefig(out/'grpo_stability.png',dpi=180); plt.close(fig)
print('wrote',*[p.name for p in out.glob('*.png')])
