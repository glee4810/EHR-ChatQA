import json
import re
import os
import time
import threading
import uuid
import logging
from datetime import datetime
from ast import literal_eval
from typing import Dict, Any, List, Optional
from argparse import Namespace
from math import comb
import pandas as pd
from collections import Counter
from tqdm import tqdm

from src.types import Action, EnvRunResult, CostInfo, EnvInfo, RewardInfo, ValidationResult
from langchain_community.vectorstores import FAISS
from sqlalchemy import text
from sqlalchemy.engine import Engine
import litellm
from litellm.exceptions import ContextWindowExceededError, RateLimitError
from litellm import completion, embedding
from langchain_core.embeddings import Embeddings
litellm.drop_params = True

def load_json(s: str):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        try:
            return literal_eval(s)
        except Exception:
            pass
        s2 = re.sub(r',\s*([}\]])', r'\1', s)
        s2 = re.sub(r'[\x00-\x1f]+', ' ', s2)
        try:
            return json.loads(s2)
        except Exception:
            print(f"[JSONDecodeError] Problematic string (truncated): {s[:500]}")
            raise


def process_answer(answer):
    """Process answer to standardized format."""
    answer = str(answer).lower().strip()
    try:
        answer = float(answer)
        if answer.is_integer():
            return str(int(answer))
        return str(round(answer, 4))
    except ValueError:
        return answer


def process_item(item):
    """Process individual item to string format."""
    try:
        item = round(float(item), 4)
    except:
        pass
    return str(item)


def process_result(result):
    """Process query result to standardized format."""
    try:
        result = literal_eval(result)
    except:
        pass
    if isinstance(result, str):
        return result
    else:
        return sorted([[process_item(c) for c in row] for row in result])[:100]  # only compare first 100 results


def initialize_vector_store(engine: Engine, embedding_model: str, faiss_path: str, columns_to_retrieve: Dict[str, List[str]]) -> FAISS:
    vector_store = None
    existing_metadata = set()
    embeddings_instance = None

    def get_litellm_embeddings(texts: List[str], batch_size: int = 100) -> List[List[float]]:
        all_embeddings = []
        if len(texts) <= 1:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                response = embedding(model=embedding_model, input=batch_texts)
                for d in response.data:
                    if isinstance(d, dict):
                        all_embeddings.append(d['embedding'])
                    else:
                        all_embeddings.append(d.embedding)
        else:
            num_batches = (len(texts) + batch_size - 1) // batch_size
            for i in tqdm(range(0, len(texts), batch_size), total=num_batches, desc="Embedding batches"):
                batch_texts = texts[i:i + batch_size]
                response = embedding(model=embedding_model, input=batch_texts)
                for d in response.data:
                    if isinstance(d, dict):
                        all_embeddings.append(d['embedding'])
                    else:
                        all_embeddings.append(d.embedding)
        return all_embeddings

    class LiteLLMEmbeddings(Embeddings):
        def embed_documents(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
            return get_litellm_embeddings(texts, batch_size)

        def embed_query(self, text: str) -> List[float]:
            return get_litellm_embeddings([text])[0]

    embeddings_instance = LiteLLMEmbeddings()

    if os.path.exists(faiss_path):
        try:
            vector_store = FAISS.load_local(faiss_path, embeddings_instance, allow_dangerous_deserialization=True)
            # Ensure the embedding function is properly set
            vector_store.embedding_function = embeddings_instance
            # print(f"Successfully loaded existing FAISS index from {faiss_path}")
        except Exception as e:
            print(f"Warning: Failed to load existing FAISS index: {e}")
            print("Creating new vector store...")
            vector_store = None

        if vector_store is not None:
            for doc in vector_store.docstore._dict.values():
                meta = doc.metadata
                if ("table" in meta and "column" in meta and meta["table"] in columns_to_retrieve and meta["column"] in columns_to_retrieve[meta["table"]]):
                    existing_metadata.add((meta["table"], meta["column"]))

    texts_to_embed = []
    metadatas_for_embedding = []

    def query_as_list(engine: Engine, table: str, column: str) -> List[str]:
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT DISTINCT {column} FROM {table} WHERE {column} IS NOT NULL;"))
            res = result.fetchall()
        res = [el for sub in res for el in sub]
        return res
    
    total_columns = sum(len(columns) for columns in columns_to_retrieve.values())
    for table, columns in columns_to_retrieve.items():
        for column in columns:            
            if (table, column) not in existing_metadata:
                values = query_as_list(engine, table, column)
                texts_to_embed.extend(values)
                metadatas_for_embedding.extend([{"table": table, "column": column} for _ in values])

    if texts_to_embed:
        print(f"Starting vectorization for {len(texts_to_embed)} items...")
        new_embeddings = get_litellm_embeddings(texts_to_embed)

        if vector_store is None:
            vector_store = FAISS.from_embeddings(
                text_embeddings=list(zip(texts_to_embed, new_embeddings)),
                embedding=embeddings_instance,
                metadatas=metadatas_for_embedding
            )
        else:
            vector_store.add_embeddings(
                text_embeddings=list(zip(texts_to_embed, new_embeddings)),
                metadatas=metadatas_for_embedding
            )

        vector_store.save_local(faiss_path)

    return vector_store


def count_agent_turns(messages):
    """Count the number of assistant turns in conversation."""
    return sum(1 for item in messages if item["role"] == "assistant")


def parse_model_name(model):
    """Extract model name from path."""
    return os.path.basename(model)


def qwen_parse_tool_calls(content: str):
    """Parse tool calls from Qwen model output."""
    tool_calls = []
    offset = 0

    content = content.replace('function_input', 'arguments')
    for i, m in enumerate(re.finditer(r"<tool_call>\n(.+)?\n</tool_call>", content)):
        if i == 0:
            offset = m.start()
        try:
            func = json.loads(m.group(1))
            if isinstance(func["arguments"], str):
                func["arguments"] = json.dumps(json.loads(func["arguments"]))
            if isinstance(func["arguments"], dict):
                func["arguments"] = json.dumps(func["arguments"])
            tool_calls.append({"type": "function", "function": func, "id": str(uuid.uuid4())})
        except json.JSONDecodeError as e:
            print(f"[Qwen] Failed to parse tool calls: the content is {m.group(1)} and {e}")
            continue
    
    if tool_calls:
        content = content[:offset].strip() if offset > 0 else ""
        message = {
            "role": "assistant", 
            "content": content.split('</think>')[-1].strip(),
            "tool_calls": tool_calls, 
            "function_call": None
        }
    else:
        message = {
            "role": "assistant", 
            "content": content.split('</think>')[-1].strip(),
            "tool_calls": None, 
            "function_call": None
        }
    return message


def llama_parse_tool_calls(content: str):
    """Parse tool calls from Llama model output."""
    tool_calls = []
    offset = 0
    
    for i, m in enumerate(re.finditer(r"```json\n(.+)?\n```", content, re.DOTALL)):
        if i == 0:
            offset = m.start()
        try:
            func = json.loads(m.group(1))
            if isinstance(func["parameters"], str):
                func["parameters"] = json.dumps(json.loads(func["parameters"]))
            if isinstance(func["parameters"], dict):
                func["parameters"] = json.dumps(func["parameters"])
            func["arguments"] = func.pop("parameters")
            tool_calls.append({"type": "function", "function": func, "id": str(uuid.uuid4())})
        except json.JSONDecodeError as e:
            print(f"[Llama] Failed to parse tool calls: the content is {m.group(1)} and {e}")
            continue
    
    if tool_calls:
        content_before = content[:offset].strip() if offset > 0 else ""
        message = {
            "role": "assistant", 
            "content": content_before, 
            "tool_calls": tool_calls, 
            "function_call": None
        }
    else:    
        message = {
            "role": "assistant", 
            "content": content, 
            "tool_calls": None, 
            "function_call": None
        }
    return message
    # return {k: v for k, v in message.items() if v is not None}


def gemini_parse_tool_calls(input_text):
    """Parse tool calls from Gemini model output."""
    output = {"role": "assistant"}

    # Check for tool call (tool_call or tool_code)
    tool_call_match = re.search(r'```(?:tool_call|tool_code)\n(.*?)\n```', input_text, re.DOTALL)
    
    if not tool_call_match:
        # Case 1: No tool call, just content
        output["content"] = input_text.strip()
        return output

    # Tool call exists
    tool_call_code = tool_call_match.group(1).strip()
    
    # Extract content before tool call, if any
    content = input_text[:tool_call_match.start()].strip()
    output["content"] = content if content else None

    # Parse tool call
    # Match function call, handling print(default_api.<func>) or direct <func>
    match = re.search(r'(?:print\(default_api\.)?(\w+)\((.*?)\)\)?', tool_call_code)
    if not match:
        raise ValueError("Invalid tool call format")

    function_name = match.group(1)
    arguments = match.group(2)

    arg_dict = {}
    if arguments:
        for key, value in re.findall(r'(\w+)=[\'"]([^\'"]*)[\'"]', arguments):
            arg_dict[key] = value

    # Build tool_calls structure
    output["tool_calls"] = [
        {
            "index": 0,
            "function": {
                "arguments": json.dumps(arg_dict),
                "name": function_name
            },
            "id": f"call_{str(uuid.uuid4())}",
            "type": "function"
        }
    ]
    output["function_call"] = None

    return output


def get_completion(model: str, messages: List[Dict[str, Any]], temperature: float, 
                  tools: Optional[List[Dict[str, Any]]] = None, api_base: Optional[str] = None,
                  response_format=None, n=1, parallel_tool_calls=False):

    if is_supported_closed_llm(model):
        res = completion(
            messages=messages,
            model=model,
            temperature=temperature,
            tools=tools,
            parallel_tool_calls=parallel_tool_calls if tools else None,
            response_format=response_format,
            n=n
        )
    elif is_supported_gemini_llm(model):
        res = completion(
            messages=messages,
            model=model,
            temperature=temperature,
            tools=tools,
            parallel_tool_calls=parallel_tool_calls if tools else None,
            response_format=response_format,
            n=n
        )
    elif is_supported_reasoning_llm(model):
        res = completion(
            messages=messages,
            model=model,
            tools=tools,
            parallel_tool_calls=parallel_tool_calls if tools else None,
            response_format=response_format,
            n=n
        )
    elif is_supported_open_source_llm(model):
        res = completion(
            messages=messages,
            model=model,
            custom_llm_provider="openai",
            temperature=temperature,
            api_base=api_base,
            tools=tools,
            parallel_tool_calls=parallel_tool_calls if tools else None,
            response_format=response_format,
            n=n
        )
    elif model.startswith("openrouter/"):
        res = completion(
            messages=messages,
            model=model,
            temperature=temperature,
            tools=tools,
            parallel_tool_calls=parallel_tool_calls if tools else None,
            response_format=response_format,
            n=n
        )
    else:
        raise ValueError(f"Model {model} is not supported")

    return res


def get_action(model: str, messages: List[Dict[str, Any]], temperature: float, 
               tools: Optional[List[Dict[str, Any]]] = None, api_base: Optional[str] = None):
    cost = 0.0
    done = False
    next_message = {'role': 'assistant', 'content': ''}
    action = Action(name='respond', kwargs={"content": ""})
    
    max_retries = 1000
    for attempt in range(max_retries):
        try:
            res = get_completion(model, messages, temperature, tools, api_base)
            if hasattr(res, '_hidden_params') and 'response_cost' in res._hidden_params and res._hidden_params["response_cost"]:
                cost += res._hidden_params["response_cost"]

            if not res.choices:
                print(f"[LLM] Empty choices returned from API (treating as empty response)")
                next_message = {'role': 'assistant', 'content': ''}
                action = Action(name='respond', kwargs={"content": ""})
                break

            next_message = res.choices[0].message.model_dump()
            content = next_message.get("content") or ""

            if 'llama' in model.lower() and content and '```json' in content:
                next_message = llama_parse_tool_calls(content)
                content = next_message.get("content") or ""

            if 'gemini' in model.lower() and content and '```tool_call' in content:
                next_message = gemini_parse_tool_calls(content)
                content = next_message.get("content") or ""

            if 'qwen' in model.lower():
                next_message = qwen_parse_tool_calls(content)

            if next_message.get("tool_calls") and len(next_message["tool_calls"]) > 0 and next_message["tool_calls"][0]["function"] is not None:
                tool_call = next_message["tool_calls"][0]
                action = Action(
                    name=tool_call["function"]["name"],
                    kwargs=json.loads(tool_call["function"]["arguments"]),
                )
            else:
                action = Action(name='respond', kwargs={"content": next_message.get("content") or ""})
            break

        except ContextWindowExceededError as e:
            print(f"[LLM] Context window exceeded: {e}")
            done = True
            break

        except RateLimitError as e:
            print(f"[LLM] Rate limit hit (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(10)

        except Exception as e:
            print(f"[LLM] Error (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(3)

    return next_message, action, done, cost


def is_supported_closed_llm(model: str) -> bool:
    supported_models = [
        'gpt-4o-mini', 
        'gpt-4o',
        'gpt-4.1-mini', 
        'gpt-4.1'
    ]
    return model in supported_models

def is_supported_gemini_llm(model: str) -> bool:
    supported_models = [
        'gemini/gemini-2.0-flash',
        'gemini/gemini-2.5-flash',
        'gemini/gemini-2.5-flash-lite',
        'gemini/gemini-2.5-flash-preview-09-2025',
        'gemini/gemini-2.5-flash-lite-preview-09-2025'
    ]
    return model in supported_models

def is_supported_reasoning_llm(model: str) -> bool:
    # These models do not support the temperature parameter
    supported_models = [
        'gpt-5',
        'gpt-5-mini',
        'gpt-5-nano',
        'o3-mini',
        'o4-mini'
    ]
    return model in supported_models

def is_supported_open_source_llm(model: str) -> bool:
    supported_models = [
        'meta-llama/Llama-3.3-70B-Instruct',
        'Qwen/Qwen3-32B',
        'nicoboss/Qwen-3-32B-Medical-Reasoning',
        'prithivMLmods/Qwen-UMLS-7B-Instruct',
        'Qwen/Qwen2.5-7B-Instruct'
    ]
    return model in supported_models


def display_metrics(results: List[EnvRunResult], num_trials: int) -> None:

    valid_ids = [f"{r.db_id}-{r.task_type}-{r.task_id}" for r in results if r.reward is not None]

    counts = Counter(valid_ids)
    invalid_tasks = [(key, count) for key, count in counts.items() if count != num_trials]
    
    if invalid_tasks:
        for key, count in invalid_tasks:
            print(f"Task {key} has {count} trials. {num_trials} trials required.")
        raise AssertionError(f"All tasks should have {num_trials} results, but some tasks have different counts.")

    filtered = [r for r in results if r.reward is not None]
    rewards = [r.reward for r in filtered]
    avg_reward = round(sum(rewards) / len(rewards) * 100, 1)

    success_counts: Dict[str, int] = {}
    for r in filtered:
        key = f'{r.db_id}-{r.task_type}-{r.task_id}'
        success_counts[key] = success_counts.get(key, 0) + (1 if r.reward == 1 else 0)

    denom_cache = {k: comb(num_trials, k) for k in range(1, num_trials + 1)}

    pass_at_k, pass_hat_k = {}, {}
    for k in range(1, num_trials + 1):
        denom = denom_cache[k]
        no_succ = sum(comb(num_trials - s, k) / denom for s in success_counts.values()) / len(success_counts)
        pass_at_k[k] = round((1 - no_succ) * 100, 1)
        all_succ = sum(comb(s, k) / denom for s in success_counts.values()) / len(success_counts)
        pass_hat_k[k] = round(all_succ * 100, 1)

    print('# Trajectory:', len(filtered))
    print(f"SR-{num_trials}: {avg_reward}")
    print(f"Pass@{num_trials}: {pass_at_k[num_trials]}")
    print(f"Pass^{num_trials}: {pass_hat_k[num_trials]}")
    print(f"Gap-{num_trials}: {round(pass_at_k[num_trials] - pass_hat_k[num_trials], 1)}")


def save_checkpoint(ckpt_path: str, results: List[EnvRunResult]) -> None:
    with open(ckpt_path, "w") as f:
        for r in results:
            f.write(json.dumps(r.model_dump()) + "\n")


def update_checkpoint(ckpt_path: str, result: EnvRunResult, lock: threading.Lock) -> None:
    with lock:
        with open(ckpt_path, "a") as f:
            f.write(json.dumps(result.model_dump()) + "\n")


def get_ckpt_name(config: Namespace, add_time: bool = True) -> str:
    
    model_name = parse_model_name(config.model)
    user_model_name = parse_model_name(config.user_model)
    
    if is_supported_reasoning_llm(model_name):
        agent_part = f"{config.env}-{config.task_type}-{config.agent_strategy}-{model_name}"
    else:
        agent_part = f"{config.env}-{config.task_type}-{config.agent_strategy}-{model_name}-{config.temperature}"
    
    if not config.task_ids:
        task_part = f"k={config.num_trials}_range_{config.start_index}-{config.end_index}"
    elif len(config.task_ids) < 20:
        task_part = f"k={config.num_trials}_range_{'-'.join(map(str, config.task_ids))}"
    else:
        task_part = f"k={config.num_trials}_range_{'-'.join(map(str, config.task_ids[:20]))}..."
    
    if is_supported_reasoning_llm(user_model_name):
        user_part = f"user-{config.user_strategy}-{user_model_name}"
    else:
        user_part = f"user-{config.user_strategy}-{user_model_name}-{config.user_temperature}"

    time_part = "_" + datetime.now().strftime("%Y%m%d%H%M%S") if add_time else ""
    
    return agent_part + "_" + task_part + "_" + user_part + time_part


def load_results(config: Namespace, idx: List[str]) -> List[EnvRunResult]:

    ckpt_name = get_ckpt_name(config, add_time=False)
    files = sorted([f for f in os.listdir(config.result_dir) if f.endswith('.jsonl')], key=lambda x: int(x.replace('.jsonl', '').split('_')[-1]))[::-1]
    load_prev_file = None
    for file in files:
        if ckpt_name in file:
            load_prev_file = file
            break

    if load_prev_file is None:
        return []

    prev_results = []
    filepath = os.path.join(config.result_dir, load_prev_file)
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                prev_results.append(json.loads(line))
    print(f"Loading previous results from {filepath}")

    if config.env == "all":
        filtered_results = [r for r in prev_results if r['task_type'] == config.task_type and r['task_id'] in idx]
    else:
        filtered_results = [r for r in prev_results if r['task_type'] == config.task_type and r['task_id'] in idx and r['db_id'] == config.env]
    return [EnvRunResult(**result) for result in filtered_results]


def dummy_error_result(isolated_env, error_decision: str, error_msg: str, reward: float = None) -> EnvRunResult:
    return EnvRunResult(
        db_id=isolated_env.db_id,
        task_type=isolated_env.task.task_type,
        task_id=isolated_env.task.task_id,
        sample_id=str(uuid.uuid4()),
        reward=reward,
        info=EnvInfo(task=isolated_env.task, reward_info=RewardInfo(reward=None, info=None)),
        messages=[],
        cost=CostInfo(agent_cost=None, user_cost=None, eval_cost=None, total_cost=None),
        validation=ValidationResult(decision=error_decision, reason=error_msg, eval_cost=0.0),
        retry=None,
        retry_reason=[]
    )
