#!/bin/bash
# Train the four grader conditions and evaluate each on the test set.
# usage: scripts/run_experiment.sh [gpu_index]
#
# Two runs share the GPU at a time. Sampling is limited by per-token overhead
# rather than by the GPU, so pairing them costs little. micro_batch=4 keeps the
# pair inside 24 GB; it changes memory use, not the gradient.
#
# Safe to stop and run again: training resumes from each run's last checkpoint
# and finished evaluations are skipped.
set -e
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${1:-0}
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
mkdir -p logs

train() { $PY -m grpo_gl.grpo configs/$1.yaml micro_batch=4 > logs/$1_s0.log 2>&1; }

train strict & train lenient_last & wait
train any_match & train format_bonus & wait

for r in strict lenient_last any_match format_bonus; do
    [ -f results/eval_test_${r}_s0.json ] && continue
    $PY -m grpo_gl.evaluate --name ${r}_s0 --adapter runs/${r}_s0/adapter --split test \
        > logs/eval_test_${r}_s0.log 2>&1
done
echo "experiment finished"
