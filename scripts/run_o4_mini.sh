#!/bin/bash
set -e

MODEL="o4-mini"
STRATEGY="tool-calling"
TRIALS=5
CONCURRENCY=16

for ENV in mimic_iv_star eicu_star; do
    for TASK in incre adapt; do
        echo "=== ${ENV} / ${TASK} ==="
        python run.py \
            --env $ENV \
            --task_type $TASK \
            --model $MODEL \
            --agent_strategy $STRATEGY \
            --num_trials $TRIALS \
            --max_concurrency $CONCURRENCY
    done
done
