# EHR-ChatQA

A benchmark for evaluating database agents through multi-turn EHR (Electronic Health Record) database conversations with simulated users.

Paper: [From Conversation to Query Execution: Benchmarking User and Tool Interactions for EHR Database Agents](https://openreview.net/forum?id=hLweUPBz7k) (ICLR 2026)

## Setup

Python >= 3.10

```bash
pip install -r requirements.txt
```

Create `.env` in the project root:
```bash
OPENAI_API_KEY=...       # FAISS embeddings + OpenAI models
GOOGLE_API_KEY=...       # Gemini models
TAVILY_API_KEY=...       # web_search tool
OPENROUTER_API_KEY=...   # (optional) OpenRouter
```

## Data

Two environments are included:
- **MIMIC-IV Star** — 145 incre + 40 adapt tasks
- **eICU Star** — 141 incre + 40 adapt tasks

### Database Files

SQLite databases and schema files are under each environment directory:
```
src/envs/{env_name}/{env_name}.sqlite   # SQLite database
src/envs/{env_name}/{env_name}.sql      # SQL schema definition
```

### Evaluation Tasks

Each environment provides two evaluation files (JSONL format, one JSON object per line):
```
src/envs/{env_name}/eval_incre.jsonl    # Incremental query tasks
src/envs/{env_name}/eval_adapt.jsonl    # Adaptive query tasks
```

### FAISS Index

The `value_similarity_search` tool uses a FAISS vector index for semantic matching. On first run, the index is built using `text-embedding-3-large` and cached at:
```
src/envs/{env_name}/faiss_index_{env_name}-text-embedding-3-large/
```

## Usage

Run a single task:
```bash
python run.py \
    --env mimic_iv_star \
    --task_type incre \
    --model gemini/gemini-2.5-flash \
    --agent_strategy tool-calling \
    --num_trials 1 \
    --task_ids 0
```

Full benchmark run:
```bash
python run.py \
    --env all \
    --task_type all \
    --model gemini/gemini-2.5-flash \
    --agent_strategy tool-calling \
    --num_trials 5 \
    --max_concurrency 16
```

Models can be specified as `gemini/gemini-2.5-flash` (Google), `gpt-4o` / `o4-mini` (OpenAI), `openrouter/google/gemini-2.5-flash` (OpenRouter), or `Qwen/Qwen3-32B` with `--api_base` (self-hosted).

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--env` | `all` | `mimic_iv_star`, `eicu_star`, `all` |
| `--task_type` | `all` | `incre`, `adapt`, `all` |
| `--model` | (required) | Agent model |
| `--agent_strategy` | (required) | `tool-calling` |
| `--temperature` | `0.0` | Agent sampling temperature |
| `--user_model` | `gemini/gemini-2.0-flash` | User simulator model |
| `--user_temperature` | `1.0` | User sampling temperature |
| `--user_strategy` | `nested-reflection` | `human`, `nested-reflection` |
| `--validation_model` | `gemini/gemini-2.5-flash` | Post-hoc validator model |
| `--num_trials` | `5` | Number of trials (k) |
| `--max_concurrency` | `1` | Parallel workers |
| `--max_retry` | `10` | Max retries on user error |
| `--timeout` | `600` | Per-task timeout (seconds) |
| `--max_agent_turns` | `30` | Max agent turns per conversation |
| `--task_ids` | `None` | Specific task IDs (space-separated) |
| `--api_base` | `None` | API base URL for self-hosted models |
| `--verbose` | `false` | Print conversations during execution |

## Evaluation

**IncreQA**: Agent's SQL result set is compared against the gold answer (exact set match).
**AdaptQA**: Agent's natural-language answer (`<answer>` tags) is compared via fuzzy string matching.

| Metric | Description |
|--------|-------------|
| SR-k | Success rate across k trials |
| Pass@k | Probability of at least 1 success in k trials |
| Pass^k | Probability of all k trials succeeding |
| Gap-k | Pass@k − Pass^k (inconsistency) |

```bash
python metric.py <result_file>
python metric.py <result_file> --by_env
python metric.py <result_file> --by_task_type
```

## Results

Results are saved to `results/` in JSONL format and checkpointed during execution. Re-running with the same configuration resumes from where it left off.

## Citation

```bibtex
@inproceedings{
lee2026from,
title={From Conversation to Query Execution: Benchmarking User and Tool Interactions for {EHR} Database Agents},
author={Gyubok Lee and Woosog Chay and Heeyoung Kwak and Yeong Hwa Kim and Haanju Yoo and Oksoon Jeong and Meong Hi Son and Edward Choi},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=hLweUPBz7k}
}
```
