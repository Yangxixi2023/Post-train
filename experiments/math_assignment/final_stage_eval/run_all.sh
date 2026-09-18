#!/bin/bash
set -e
cd /workspace/assignment5-alignment
for spec in \
 'sft_full experiments/math_assignment/sft/n1767/checkpoint_final 301' \
 'sft_filtered experiments/math_assignment/sft/filtered1408/checkpoint_final 302' \
 'ei_final experiments/math_assignment/ei/b2048_g4_e2/checkpoint_final 303'; do
 set -- $spec; name=$1; model=$2; seed=$3
 echo "START $name $(date -Is)"
 .venv_run/bin/python -m cs336_alignment.evaluate --model "$model" --data experiments/data/MATH/validation.jsonl \
   --gpu cuda:1 --n 5000 --temperature 1.0 --max-tokens 1024 --seed "$seed" --gpu-memory-utilization .55 \
   --output "experiments/math_assignment/final_stage_eval/${name}_5000.jsonl" \
   > "experiments/math_assignment/final_stage_eval/${name}_5000.log" 2>&1
 echo "DONE $name $(date -Is)"; cat "experiments/math_assignment/final_stage_eval/${name}_5000.jsonl.metrics.json"
done
