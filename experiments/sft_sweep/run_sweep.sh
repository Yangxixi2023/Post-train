#!/usr/bin/env bash
set -euo pipefail
cd /workspace/assignment5-alignment
MODEL=/workspace/models/Qwen2.5-Math-1.5B
for N in 128 256 512 1024 7473; do
  STEPS=$(( (N + 3) / 4 ))
  EVAL=$(( STEPS / 4 )); [ "$EVAL" -lt 8 ] && EVAL=8
  OUT="experiments/sft_sweep/n${N}"
  mkdir -p "$OUT"
  echo "[$(date -Is)] START N=$N steps=$STEPS eval_every=$EVAL"
  .venv_run/bin/python -m cs336_alignment.train_sft \
    --model "$MODEL" --train-data experiments/data/gsm8k_train_r1.jsonl --val-data experiments/data/gsm8k_test_r1.jsonl \
    --output-dir "$OUT" --steps "$STEPS" --lr 1e-5 --micro-batch-size 1 --grad-accum 4 --max-seq-len 768 \
    --train-examples "$N" --eval-examples 128 --eval-every "$EVAL" --max-new-tokens 256 --seed 51 \
    --policy-device cuda:0 --vllm-device cuda:1 --vllm-gpu-util 0.45 > "$OUT/run.log" 2>&1
  cp "$OUT/train_metrics.jsonl" "$OUT/metrics.saved.jsonl"
  echo "[$(date -Is)] DONE N=$N"
  if [ "$N" != "7473" ]; then rm -rf "$OUT/checkpoint_final"; fi
done
