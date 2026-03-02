#!/bin/bash
set -e

STRATEGY="tool-calling"
TRIALS=5
CONCURRENCY=16

MODELS=(
    "gemini/gemini-2.5-flash"
    "o4-mini"
    "gpt-4o"
    "gpt-4o-mini"
)

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "============================================"
    echo "Model: $MODEL"
    echo "============================================"
    python run.py \
        --env all \
        --task_type all \
        --model $MODEL \
        --agent_strategy $STRATEGY \
        --temperature 0.0 \
        --num_trials $TRIALS \
        --max_concurrency $CONCURRENCY
done
