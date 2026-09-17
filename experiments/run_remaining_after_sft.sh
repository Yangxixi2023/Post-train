#!/usr/bin/env bash
set -euo pipefail
cd /workspace/assignment5-alignment
# Wait for SFT dataset-size sweep.
while ps -p "$(cat experiments/sft_sweep/orchestrator.pid)" >/dev/null 2>&1; do sleep 20; done

echo "[$(date -Is)] SFT sweep finished; starting EI sweep"
mkdir -p experiments/ei_sweep
for SPEC in 'g4_e1 4 1 61' 'g8_e2 8 2 62'; do
  set -- $SPEC; NAME=$1; G=$2; E=$3; SEED=$4; OUT="experiments/ei_sweep/$NAME"; mkdir -p "$OUT"
  echo "[$(date -Is)] EI START $NAME"
  .venv_run/bin/python -m cs336_alignment.train_ei \
    --model /workspace/models/Qwen2.5-Math-1.5B --train-data experiments/data/gsm8k_train_r1.jsonl --val-data experiments/data/gsm8k_test_r1.jsonl \
    --output-dir "$OUT" --ei-steps 5 --questions-per-step 512 --group-size "$G" --sft-epochs "$E" --lr 5e-6 --grad-accum 4 \
    --max-seq-len 768 --max-new-tokens 256 --eval-examples 256 --seed "$SEED" --policy-device cuda:0 --vllm-device cuda:1 --vllm-gpu-util 0.45 \
    > "$OUT/run.log" 2>&1
  echo "[$(date -Is)] EI DONE $NAME"
done

echo "[$(date -Is)] EI sweep finished"

echo "[$(date -Is)] Starting GRPO assignment sweeps from BASE model"
mkdir -p experiments/grpo_assignment
BASE=/workspace/models/Qwen2.5-Math-1.5B
run_grpo () {
  NAME="$1"; shift
  OUT="experiments/grpo_assignment/$NAME"; mkdir -p "$OUT"
  echo "[$(date -Is)] GRPO START $NAME $*"
  START=$(date +%s)
  .venv_run/bin/python -m cs336_alignment.train_grpo \
    --model "$BASE" --train-data experiments/data/gsm8k_train_r1.jsonl --val-data experiments/data/gsm8k_test_r1.jsonl \
    --output-dir "$OUT" --prompts-per-rollout 32 --group-size 8 --max-seq-len 768 --max-new-tokens 256 \
    --eval-examples 1024 --eval-every 2 --policy-device cuda:0 --vllm-device cuda:1 --vllm-gpu-util 0.45 "$@" \
    > "$OUT/run.log" 2>&1
  END=$(date +%s); echo $((END-START)) > "$OUT/wallclock_seconds.txt"
  cp "$OUT/metrics.jsonl" "$OUT/metrics.saved.jsonl"
  echo "[$(date -Is)] GRPO DONE $NAME seconds=$((END-START))"
  rm -rf "$OUT/checkpoint_final"
}
# LR sweep: canonical on-policy REINFORCE-with-group-baseline.
run_grpo lr_2e5 --grpo-steps 12 --lr 2e-5 --loss-type reinforce_with_baseline --train-batch-size 256 --epochs-per-rollout 1 --normalize-by-std --length-normalization mean --seed 71
run_grpo lr_3e5 --grpo-steps 12 --lr 3e-5 --loss-type reinforce_with_baseline --train-batch-size 256 --epochs-per-rollout 1 --normalize-by-std --length-normalization mean --seed 71
run_grpo lr_4e5 --grpo-steps 12 --lr 4e-5 --loss-type reinforce_with_baseline --train-batch-size 256 --epochs-per-rollout 1 --normalize-by-std --length-normalization mean --seed 71
BEST_LR=$(python3 - <<'PY'
import json,glob,os
best=(-1,None)
for p in glob.glob('experiments/grpo_assignment/lr_*/metrics.saved.jsonl'):
 vals=[json.loads(x) for x in open(p) if '"eval"' in x]
 v=vals[-1]['eval']['answer_reward'] if vals else -1
 lr={'lr_2e5':'2e-5','lr_3e5':'3e-5','lr_4e5':'4e-5'}[os.path.basename(os.path.dirname(p))]
 if v>best[0]: best=(v,lr)
print(best[1])
PY
)
echo "[$(date -Is)] BEST_LR=$BEST_LR"
# Baseline ablation; reinforce reference is best LR run above.
run_grpo baseline_no_baseline --grpo-steps 12 --lr "$BEST_LR" --loss-type no_baseline --train-batch-size 256 --epochs-per-rollout 1 --normalize-by-std --length-normalization mean --seed 71
# Length normalization ablation.
run_grpo length_constant --grpo-steps 12 --lr "$BEST_LR" --loss-type reinforce_with_baseline --train-batch-size 256 --epochs-per-rollout 1 --normalize-by-std --length-normalization constant --constant-normalizer 256 --seed 71
# Group std ablation.
run_grpo std_false --grpo-steps 12 --lr "$BEST_LR" --loss-type reinforce_with_baseline --train-batch-size 256 --epochs-per-rollout 1 --no-normalize-by-std --length-normalization mean --seed 71
# Broad off-policy sweep (GRPO-Clip): epochs and train batch size.
run_grpo offpolicy_e2_b256 --grpo-steps 8 --lr "$BEST_LR" --loss-type grpo_clip --train-batch-size 256 --epochs-per-rollout 2 --normalize-by-std --length-normalization mean --seed 72
run_grpo offpolicy_e4_b128 --grpo-steps 8 --lr "$BEST_LR" --loss-type grpo_clip --train-batch-size 128 --epochs-per-rollout 4 --normalize-by-std --length-normalization mean --seed 72
BEST_OFF=$(python3 - <<'PY'
import json
pairs=[]
for n in ['offpolicy_e2_b256','offpolicy_e4_b128']:
 vals=[json.loads(x) for x in open(f'experiments/grpo_assignment/{n}/metrics.saved.jsonl') if '"eval"' in x]
 pairs.append((vals[-1]['eval']['answer_reward'],n))
print(max(pairs)[1])
PY
)
echo "[$(date -Is)] BEST_OFF=$BEST_OFF"
if [ "$BEST_OFF" = offpolicy_e2_b256 ]; then OFF_E=2; OFF_B=256; else OFF_E=4; OFF_B=128; fi
# Focused off-policy run, longer than broad sweep.
run_grpo offpolicy_focused --grpo-steps 40 --lr "$BEST_LR" --loss-type grpo_clip --train-batch-size "$OFF_B" --epochs-per-rollout "$OFF_E" --normalize-by-std --length-normalization mean --seed 73
# Clipping ablation with same off-policy setup.
run_grpo offpolicy_no_clip --grpo-steps 12 --lr "$BEST_LR" --loss-type grpo_no_clip --train-batch-size "$OFF_B" --epochs-per-rollout "$OFF_E" --normalize-by-std --length-normalization mean --seed 73
# Prompt ablation. R1 reference = best LR on-policy run; question-only is matched setup.
run_grpo prompt_question_only --grpo-steps 12 --lr "$BEST_LR" --loss-type reinforce_with_baseline --train-batch-size 256 --epochs-per-rollout 1 --normalize-by-std --length-normalization mean --prompt-template cs336_alignment/prompts/question_only.prompt --reward-type question_only --seed 71
# Leaderboard-style final: train a persistent checkpoint with best off-policy setting.
OUT=experiments/grpo_assignment/leaderboard_final; mkdir -p "$OUT"
echo "[$(date -Is)] LEADERBOARD START lr=$BEST_LR epochs=$OFF_E batch=$OFF_B"
START=$(date +%s)
.venv_run/bin/python -m cs336_alignment.train_grpo \
  --model "$BASE" --train-data experiments/data/gsm8k_train_r1.jsonl --val-data experiments/data/gsm8k_test_r1.jsonl \
  --output-dir "$OUT" --grpo-steps 100 --prompts-per-rollout 32 --group-size 8 --epochs-per-rollout "$OFF_E" --loss-type grpo_clip \
  --cliprange 0.2 --train-batch-size "$OFF_B" --normalize-by-std --length-normalization mean --lr "$BEST_LR" \
  --max-seq-len 768 --max-new-tokens 256 --eval-examples 1024 --eval-every 10 --seed 74 \
  --policy-device cuda:0 --vllm-device cuda:1 --vllm-gpu-util 0.45 > "$OUT/run.log" 2>&1
END=$(date +%s); echo $((END-START)) > "$OUT/wallclock_seconds.txt"
echo "[$(date -Is)] LEADERBOARD TRAIN DONE seconds=$((END-START))"
# Full open-source validation under exact leaderboard eval sampling constraints.
.venv_run/bin/python -m cs336_alignment.evaluate --model "$OUT/checkpoint_final" --data experiments/data/gsm8k_test_r1.jsonl \
  --gpu cuda:1 --n 1319 --temperature 1.0 --max-tokens 1024 --seed 74 --gpu-memory-utilization 0.55 \
  --output "$OUT/full_validation_1319.jsonl" > "$OUT/full_validation.log" 2>&1

echo "[$(date -Is)] ALL MAIN ASSIGNMENT EXPERIMENTS FINISHED"
