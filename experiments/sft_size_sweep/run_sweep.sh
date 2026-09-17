#!/bin/bash
set -e
cd /workspace/assignment5-alignment
for size in 128 256 512 1024 7473; do
  steps=$(( (size + 3) / 4 ))
  out="experiments/sft_size_sweep/n${size}"
  mkdir -p "$out"
  echo "START size=$size steps=$steps $(date -Is)" | tee -a experiments/sft_size_sweep/orchestrator.log
  .venv_run/bin/python -m cs336_alignment.train_sft \
    --model /workspace/models/Qwen2.5-Math-1.5B --train-data experiments/data/gsm8k_train_r1.jsonl --val-data experiments/data/gsm8k_test_r1.jsonl \
    --output-dir "$out" --steps "$steps" --lr 1e-5 --micro-batch-size 1 --grad-accum 4 --max-seq-len 768 \
    --train-examples "$size" --eval-examples 256 --eval-every "$steps" --max-new-tokens 256 --seed 42 \
    --policy-device cuda:0 --vllm-device cuda:1 --vllm-gpu-util 0.45 > "$out.log" 2>&1
  rm -rf "$out/checkpoint_final"
  echo "DONE size=$size $(date -Is)" | tee -a experiments/sft_size_sweep/orchestrator.log
done
