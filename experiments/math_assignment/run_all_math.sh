#!/usr/bin/env bash
set -euo pipefail
cd /workspace/assignment5-alignment
BASE=/workspace/models/Qwen2.5-Math-1.5B
TRAIN=experiments/data/MATH/train.jsonl
VAL=experiments/data/MATH/validation.jsonl
SFT=experiments/data/MATH/sft.jsonl
SFTF=experiments/data/MATH/sft_filtered.jsonl
ROOT=experiments/math_assignment
# Wait exact 5k zero-shot baseline.
while ps -p "$(cat $ROOT/baseline/base_5000.pid)" >/dev/null 2>&1; do sleep 20; done
echo "[$(date -Is)] baseline done" >> $ROOT/orchestrator.log

# SFT size sweep + full filtered dataset. One pass over unique examples; 1024-example validation curves.
mkdir -p $ROOT/sft
for N in 128 256 512 1024 1767; do
  STEPS=$(( (N+3)/4 )); EVAL=$((STEPS/4)); [ $EVAL -lt 8 ] && EVAL=8
  OUT=$ROOT/sft/n$N; mkdir -p $OUT
  echo "[$(date -Is)] SFT n=$N start" >> $ROOT/orchestrator.log
  .venv_run/bin/python -m cs336_alignment.train_sft --model "$BASE" --train-data "$SFT" --val-data "$VAL" --output-dir "$OUT" \
    --steps "$STEPS" --lr 2e-5 --micro-batch-size 1 --grad-accum 4 --max-seq-len 1536 --train-examples "$N" \
    --eval-examples 256 --eval-every "$EVAL" --max-new-tokens 1024 --seed 101 --policy-device cuda:0 --vllm-device cuda:1 --vllm-gpu-util .45 > $OUT/run.log 2>&1
  cp $OUT/train_metrics.jsonl $OUT/metrics.saved.jsonl
  [ "$N" = 1767 ] || rm -rf $OUT/checkpoint_final
  echo "[$(date -Is)] SFT n=$N done" >> $ROOT/orchestrator.log
done
N=1408; STEPS=$(( (N+3)/4 )); OUT=$ROOT/sft/filtered1408; mkdir -p $OUT
.venv_run/bin/python -m cs336_alignment.train_sft --model "$BASE" --train-data "$SFTF" --val-data "$VAL" --output-dir "$OUT" \
  --steps "$STEPS" --lr 2e-5 --micro-batch-size 1 --grad-accum 4 --max-seq-len 1536 --train-examples "$N" \
  --eval-examples 256 --eval-every 88 --max-new-tokens 1024 --seed 101 --policy-device cuda:0 --vllm-device cuda:1 --vllm-gpu-util .45 > $OUT/run.log 2>&1
cp $OUT/train_metrics.jsonl $OUT/metrics.saved.jsonl
echo "[$(date -Is)] filtered SFT done" >> $ROOT/orchestrator.log

# EI: n_ei_steps=5; cover Db={512,1024,2048}, rollout G={4,8}, epochs={1,2}.
mkdir -p $ROOT/ei
for SPEC in 'b512_g4_e1 512 4 1 111' 'b1024_g8_e1 1024 8 1 112' 'b2048_g4_e2 2048 4 2 113'; do
 set -- $SPEC; NAME=$1; B=$2; G=$3; E=$4; SEED=$5; OUT=$ROOT/ei/$NAME; mkdir -p $OUT
 echo "[$(date -Is)] EI $NAME start" >> $ROOT/orchestrator.log
 .venv_run/bin/python -m cs336_alignment.train_ei --model "$BASE" --train-data "$TRAIN" --val-data "$VAL" --output-dir "$OUT" \
   --ei-steps 5 --questions-per-step "$B" --group-size "$G" --sft-epochs "$E" --lr 1e-5 --grad-accum 8 --max-seq-len 1536 \
   --max-new-tokens 1024 --eval-examples 256 --seed "$SEED" --policy-device cuda:0 --vllm-device cuda:1 --vllm-gpu-util .45 > $OUT/run.log 2>&1
 echo "[$(date -Is)] EI $NAME done" >> $ROOT/orchestrator.log
done

# GRPO assignment ablations. Base model, rollout batch 256 = 32 prompts x group 8.
mkdir -p $ROOT/grpo
run_grpo(){ NAME=$1; shift; OUT=$ROOT/grpo/$NAME; mkdir -p $OUT; echo "[$(date -Is)] GRPO $NAME start" >> $ROOT/orchestrator.log; START=$(date +%s); \
 .venv_run/bin/python -m cs336_alignment.train_grpo --model "$BASE" --train-data "$TRAIN" --val-data "$VAL" --output-dir "$OUT" \
 --prompts-per-rollout 32 --group-size 8 --max-seq-len 1536 --max-new-tokens 1024 --eval-examples 1024 --policy-device cuda:0 --vllm-device cuda:1 --vllm-gpu-util .45 "$@" > $OUT/run.log 2>&1; \
 END=$(date +%s); echo $((END-START)) > $OUT/wallclock_seconds.txt; cp $OUT/metrics.jsonl $OUT/metrics.saved.jsonl; echo "[$(date -Is)] GRPO $NAME done" >> $ROOT/orchestrator.log; }
# Learning-rate sweep; early-stop length 12 is permitted by handout for clear differences.
for LR in 1e-5 2e-5 3e-5; do run_grpo lr_${LR} --grpo-steps 12 --lr $LR --loss-type reinforce_with_baseline --train-batch-size 256 --grad-accum 256 --epochs-per-rollout 1 --normalize-by-std --length-normalization mean --eval-every 4 --seed 121; rm -rf $ROOT/grpo/lr_${LR}/checkpoint_final; done
BEST_LR=$(python3 - <<'PY'
import glob,json,os
z=[]
for p in glob.glob('experiments/math_assignment/grpo/lr_*/metrics.saved.jsonl'):
 v=[json.loads(x) for x in open(p) if '"eval"' in x][-1]['eval']['answer_reward']; z.append((v,os.path.basename(os.path.dirname(p)).split('lr_',1)[1]))
print(max(z)[1])
PY
)
echo "BEST_LR=$BEST_LR" >> $ROOT/orchestrator.log
# Baseline effect.
run_grpo no_baseline --grpo-steps 12 --lr $BEST_LR --loss-type no_baseline --train-batch-size 256 --grad-accum 256 --epochs-per-rollout 1 --normalize-by-std --length-normalization mean --eval-every 4 --seed 122; rm -rf $ROOT/grpo/no_baseline/checkpoint_final
# Keep matched baseline run for later comparisons.
run_grpo reinforce_best --grpo-steps 12 --lr $BEST_LR --loss-type reinforce_with_baseline --train-batch-size 256 --grad-accum 256 --epochs-per-rollout 1 --normalize-by-std --length-normalization mean --eval-every 4 --seed 122; rm -rf $ROOT/grpo/reinforce_best/checkpoint_final
# Length normalization.
run_grpo length_constant --grpo-steps 12 --lr $BEST_LR --loss-type reinforce_with_baseline --train-batch-size 256 --grad-accum 256 --epochs-per-rollout 1 --normalize-by-std --length-normalization constant --constant-normalizer 1024 --eval-every 4 --seed 123; rm -rf $ROOT/grpo/length_constant/checkpoint_final
# Group std normalization.
run_grpo std_false --grpo-steps 12 --lr $BEST_LR --loss-type reinforce_with_baseline --train-batch-size 256 --grad-accum 256 --epochs-per-rollout 1 --no-normalize-by-std --length-normalization mean --eval-every 4 --seed 124; rm -rf $ROOT/grpo/std_false/checkpoint_final
# Off-policy broad sweep and focused run.
run_grpo off_e2_b256 --grpo-steps 8 --lr $BEST_LR --loss-type grpo_clip --train-batch-size 256 --grad-accum 256 --epochs-per-rollout 2 --normalize-by-std --length-normalization mean --eval-every 2 --seed 125; rm -rf $ROOT/grpo/off_e2_b256/checkpoint_final
run_grpo off_e4_b128 --grpo-steps 8 --lr $BEST_LR --loss-type grpo_clip --train-batch-size 128 --grad-accum 128 --epochs-per-rollout 4 --normalize-by-std --length-normalization mean --eval-every 2 --seed 125; rm -rf $ROOT/grpo/off_e4_b128/checkpoint_final
BEST_OFF=$(python3 - <<'PY'
import json
z=[]
for n in ['off_e2_b256','off_e4_b128']:
 v=[json.loads(x) for x in open(f'experiments/math_assignment/grpo/{n}/metrics.saved.jsonl') if '"eval"' in x][-1]['eval']['answer_reward']; z.append((v,n))
print(max(z)[1])
PY
)
if [ "$BEST_OFF" = off_e2_b256 ]; then OE=2; OB=256; OA=256; else OE=4; OB=128; OA=128; fi
echo "BEST_OFF=$BEST_OFF" >> $ROOT/orchestrator.log
run_grpo off_focused --grpo-steps 40 --lr $BEST_LR --loss-type grpo_clip --train-batch-size $OB --grad-accum $OA --epochs-per-rollout $OE --normalize-by-std --length-normalization mean --eval-every 5 --seed 126; rm -rf $ROOT/grpo/off_focused/checkpoint_final
# Clip ablation.
run_grpo off_no_clip --grpo-steps 12 --lr $BEST_LR --loss-type grpo_no_clip --train-batch-size $OB --grad-accum $OA --epochs-per-rollout $OE --normalize-by-std --length-normalization mean --eval-every 4 --seed 127; rm -rf $ROOT/grpo/off_no_clip/checkpoint_final
# Prompt ablation.
run_grpo prompt_question_only --grpo-steps 12 --lr $BEST_LR --loss-type reinforce_with_baseline --train-batch-size 256 --grad-accum 256 --epochs-per-rollout 1 --normalize-by-std --length-normalization mean --eval-every 4 --prompt-template cs336_alignment/prompts/question_only.prompt --reward-type question_only --seed 128; rm -rf $ROOT/grpo/prompt_question_only/checkpoint_final
# Leaderboard-style final: 100 steps, keep checkpoint; exact full 5k validation with temp=1/max_tokens=1024.
OUT=$ROOT/leaderboard; mkdir -p $OUT; START=$(date +%s)
.venv_run/bin/python -m cs336_alignment.train_grpo --model "$BASE" --train-data "$TRAIN" --val-data "$VAL" --output-dir "$OUT" \
 --grpo-steps 100 --prompts-per-rollout 32 --group-size 8 --epochs-per-rollout $OE --loss-type grpo_clip --cliprange .2 --train-batch-size $OB --grad-accum $OA \
 --normalize-by-std --length-normalization mean --lr $BEST_LR --max-seq-len 1536 --max-new-tokens 1024 --eval-examples 1024 --eval-every 10 --seed 129 \
 --policy-device cuda:0 --vllm-device cuda:1 --vllm-gpu-util .45 > $OUT/run.log 2>&1
END=$(date +%s); echo $((END-START)) > $OUT/wallclock_seconds.txt
.venv_run/bin/python -m cs336_alignment.evaluate --model $OUT/checkpoint_final --data "$VAL" --gpu cuda:1 --n 5000 --temperature 1.0 --max-tokens 1024 --seed 129 --gpu-memory-utilization .55 --output $OUT/full_validation_5000.jsonl > $OUT/full_validation.log 2>&1

echo "[$(date -Is)] ALL MATH ASSIGNMENT EXPERIMENTS DONE" >> $ROOT/orchestrator.log
