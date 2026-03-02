#!/bin/bash
set -e

MODEL="${1:-openrouter/google/gemini-2.5-flash}"
STRATEGY="tool-calling"
TRIALS=5
CONCURRENCY=16

echo "Running with model: $MODEL"

for ENV in mimic_iv_star eicu_star; do
    for TASK in incre adapt; do
        echo "=== ${ENV} / ${TASK} ==="
        python run.py \
            --env $ENV \
            --task_type $TASK \
            --model $MODEL \
            --agent_strategy $STRATEGY \
            --temperature 0.0 \
            --num_trials $TRIALS \
            --max_concurrency $CONCURRENCY
    done
done
