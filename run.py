import sys
sys.stdout.reconfigure(line_buffering=True)

import os
import uuid
from tqdm import tqdm
from collections import Counter
from argparse import ArgumentParser, Namespace
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from src.agents.base import AgentTimeoutError
import litellm
import traceback
import io
litellm.suppress_debug_info = True
file_lock = threading.Lock()

import json

from src.envs import get_env
from src.agent_factory import get_agent
from src.types import EnvRunResult, CostInfo, ValidationResult
from src.utils import save_checkpoint, display_metrics, update_checkpoint, load_results, dummy_error_result, get_ckpt_name
from validator import user_validator
from dotenv import load_dotenv


load_dotenv()


def _parse_validation_explanation(reason: str) -> str | None:
    try:
        parsed = json.loads(reason)
        return parsed.get('explanation', '')
    except (json.JSONDecodeError, KeyError):
        return reason


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--env", type=str, default="all", choices=["mimic_iv_star", "eicu_star", "all"], help="Environment name for fetching user instructions")
    parser.add_argument("--task_type", type=str, default="all", choices=["incre", "adapt", "all"], help="Task type to use")
    parser.add_argument("--model", type=str, required=True, help="The agent model to use")
    parser.add_argument("--api_base", type=str, default=None, help="The API base to use")
    parser.add_argument("--agent_strategy", type=str, required=True, choices=["tool-calling"], help="The agent strategy to use")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for the action model")
    parser.add_argument("--user_model", type=str, default='gemini/gemini-2.0-flash', help="The user model to use")
    parser.add_argument("--user_temperature", type=float, default=1.0, help="Sampling temperature for the user model")
    parser.add_argument("--user_strategy", type=str, default='nested-reflection', choices=["human", "nested-reflection"], help="The user strategy to use")
    parser.add_argument("--result_dir", type=str, default="results", help="Directory to save the results")
    parser.add_argument("--num_trials", type=int, default=5, help="Number of trials (k) to run")
    parser.add_argument("--max_concurrency", type=int, default=1, help="Maximum concurrency level")
    parser.add_argument("--start_index", type=int, default=0, help="Start index for tasks")
    parser.add_argument("--end_index", type=int, default=-1, help="End index for tasks (-1 for all)")
    parser.add_argument("--task_ids", nargs='+', type=int, default=None, help="Specific task ids to run")
    parser.add_argument("--max_agent_turns", type=int, default=30, help="Maximum number of agent turns")
    parser.add_argument("--max_retry", type=int, default=10, help="Maximum number of simulation retries")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout for each task in seconds")
    parser.add_argument("--validation_model", type=str, default='gemini/gemini-2.5-flash', help="The validation model to use")
    parser.add_argument("--verbose", action="store_true", help="Print user-agent conversations during execution")
    parser.add_argument("--validation_trials", type=int, default=1, help="Number of validation trials")
    return parser.parse_args()


def run(config: Namespace):

    if any(x in config.model.lower() for x in ("llama", "qwen", "gpt-oss")):
        assert config.api_base is not None, f"api_base is required for {config.model}"

    if config.task_type == "all":
        task_types_to_run = ["incre", "adapt"]
        print("Running all task types: incre, adapt")
    else:
        task_types_to_run = [config.task_type]

    for task_type in task_types_to_run:
        config.task_type = task_type
        print(f"\n{'='*60}")
        print(f"Running task type: {task_type}")
        print(f"{'='*60}")
        _run_single(config)


def _run_single(config: Namespace):

    if config.env == "all":
        envs_to_run = ["mimic_iv_star", "eicu_star"]
        print("Running all environments: mimic_iv_star, eicu_star")
    else:
        envs_to_run = [config.env]

    ckpt_name = get_ckpt_name(config)
    ckpt_path = os.path.join(config.result_dir, ckpt_name + '.jsonl')
    os.makedirs(config.result_dir, exist_ok=True)
    
    print(f"Loading user with strategy: {config.user_strategy}")
    
    # Initialize environments and agents for each env
    envs = {}
    agents = {}
    env_task_indices = {}
    
    for env_name in envs_to_run:
        envs[env_name] = get_env(
            env_name=env_name,
            task_type=config.task_type,
            user_strategy=config.user_strategy,
            user_model=config.user_model,
            user_temperature=config.user_temperature,
            api_base=config.api_base
        )
        agents[env_name] = get_agent(
            tools_info=envs[env_name].tools_info,
            model=config.model,
            api_base=config.api_base,
            temperature=config.temperature,
            agent_strategy=config.agent_strategy,
            rule=envs[env_name].rule,
            verbose=config.verbose
        )
        
        # Determine task indices for this environment
        total_tasks = len(envs[env_name].tasks)
        end_index = total_tasks if config.end_index == -1 else min(config.end_index, total_tasks)
        idx = config.task_ids if config.task_ids else list(range(config.start_index, end_index))
        env_task_indices[env_name] = idx
        
        task_info = f"{config.task_ids}" if config.task_ids else f"{config.start_index} to {end_index}"
        print(f"Running tasks for {env_name}: {task_info} (checkpoint path: {ckpt_path})")
    
    # Collect all task indices for loading results
    all_task_ids = []
    for indices in env_task_indices.values():
        all_task_ids.extend(indices)
    all_task_ids = sorted(set(all_task_ids))
    
    results = load_results(config, idx=[str(i) for i in all_task_ids])
    
    # Build task list to run
    if config.env == "all":
        idx_to_run = []
        for env_name in envs_to_run:
            env_task_pairs = [(env_name, task_id) for task_id in env_task_indices[env_name]]
            idx_to_run.extend(env_task_pairs)
        
        idx_to_run = idx_to_run * config.num_trials
        
        existing_pairs = [(r.db_id, int(r.task_id)) for r in results if r.reward is not None]
        result_counter = Counter(idx_to_run) - Counter(existing_pairs)
        idx_to_run = list(result_counter.elements())
        
    else:
        # Single environment
        env_name = config.env
        idx = env_task_indices[env_name]
        idx_to_run = idx * config.num_trials
        existing_idx = [int(r.task_id) for r in results if r.reward is not None]
        result_counter = Counter(idx_to_run) - Counter(existing_idx)
        idx_to_run = list(result_counter.elements())

    if len(idx_to_run) == 0:
        print("No new tasks to run. All tasks have been loaded from checkpoint.")
        display_metrics(results, config.num_trials)
        return

    save_checkpoint(ckpt_path, results)

    def _run(task_item) -> EnvRunResult:
        try:
            if isinstance(task_item, tuple):
                current_env_name, task_idx = task_item
            else:
                current_env_name = config.env
                task_idx = task_item
                
            retry = 0
            result = None
            retry_reason = []

            while retry < config.max_retry:
                try:
                    
                    isolated_env = get_env(
                        env_name=current_env_name,
                        task_type=config.task_type,
                        user_strategy=config.user_strategy,
                        user_model=config.user_model,
                        user_temperature=config.user_temperature,
                        api_base=config.api_base,
                        task_index=str(task_idx),
                        retry_reason=retry_reason
                    )

                    response = agents[current_env_name].run(
                        isolated_env, str(task_idx), config.max_agent_turns, config.timeout
                    )

                    user_cost = isolated_env.user.get_total_cost()
                    result = EnvRunResult(
                        db_id=isolated_env.task.db_id,
                        task_type=isolated_env.task.task_type,
                        task_id=isolated_env.task.task_id,
                        sample_id=str(uuid.uuid4()),
                        reward=response.reward,
                        info=response.info,
                        messages=response.messages,
                        cost=CostInfo(
                            agent_cost=response.agent_cost,
                            user_cost=user_cost,
                            eval_cost=0.0,
                            total_cost=round(response.agent_cost + user_cost, 8),
                        ),
                        validation=None,
                        retry=retry,
                        retry_reason=retry_reason
                    )

                    validation_result = user_validator(
                        messages=response.messages,
                        env=isolated_env,
                        model=config.validation_model,
                        api_base=config.api_base,
                        n=config.validation_trials,
                        max_agent_turns=config.max_agent_turns
                    )
                    result.cost.eval_cost = round(result.cost.eval_cost + validation_result.eval_cost, 8)
                    result.cost.total_cost = round(result.cost.total_cost + validation_result.eval_cost, 8)
                    result.validation = validation_result

                    if validation_result.decision == 'user_error':
                        result.reward = None
                        update_checkpoint(ckpt_path, result, file_lock)
                        if config.user_strategy == "nested-reflection":
                            explanation = _parse_validation_explanation(validation_result.reason)
                            if explanation:
                                retry_reason.append(explanation)
                        retry += 1
                        print(
                            "⚠️ ",
                            f"Retry {retry}/{config.max_retry} |",
                            f"ckpt_path={ckpt_name}",
                            f"task_id={task_idx}",
                            f"User error during simulation: {validation_result.reason}"
                        )
                    else:
                        update_checkpoint(ckpt_path, result, file_lock)
                        break
                
                except AgentTimeoutError as e:
                    error_reason = f"Agent timeout: {e}"
                    result = dummy_error_result(isolated_env, 'no_error', error_reason, reward=0.0)
                    update_checkpoint(ckpt_path, result, file_lock)
                    print("❌", f"ckpt_path={ckpt_name}", f"task_id={task_idx}", error_reason)
                    break

            if result is not None and retry >= config.max_retry:
                result.retry_exhausted = True
                update_checkpoint(ckpt_path, result, file_lock)
                print("⚠️ Retry exhausted |", f"ckpt_path={ckpt_name}", f"task_id={task_idx}")

            if result and result.reward == 1:
                print("✅", f"ckpt_path={ckpt_name}", f"task_id={task_idx}", result.info)
            elif result and result.reward == 0:
                print("❌", f"ckpt_path={ckpt_name}", f"task_id={task_idx}", result.info)

            print("-----")
            return result
        
        except KeyboardInterrupt:
            print("Keyboard interrupt. Exiting...")
            exit(0)

        except Exception as e:
            tb_str = io.StringIO()
            traceback.print_exc(file=tb_str)
            error_details = tb_str.getvalue()
            error_reason = f"Unexpected error during simulation: {error_details}"
            if 'isolated_env' in dir():
                result = dummy_error_result(isolated_env, 'other', error_reason)
                update_checkpoint(ckpt_path, result, file_lock)
            else:
                result = None
            print("⚠️", f"ckpt_path={ckpt_name}", f"task_id={task_idx}", error_reason)
            return result

    max_workers = max(1, min(config.max_concurrency, len(idx_to_run)))
    new_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run, t): t for t in idx_to_run}
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Running"):
            pass
        for f in futures:
            try:
                new_results.append(f.result())
            except Exception:
                continue

    results.extend(new_results)
    display_metrics(results, config.num_trials)

if __name__ == "__main__":
    config = parse_arguments()
    run(config)
