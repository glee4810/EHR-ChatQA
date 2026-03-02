import time
from litellm import completion
from litellm.exceptions import RateLimitError
from typing import List, Optional, Dict, Any

from src.agents.base import Agent, AgentTimeoutError
from src.envs.base import Env
from src.types import AgentRunResult
from src.utils import get_action

TOOL_CALLING_INSTRUCTION = """Instruction:
- Your task is to generate an SQL query based on the user's question and the database schema provided below.
{database_schema}

- Keep the following format for your response:
{
    'reasoning': <Reasoning>, 
    'sql_query': <SQL Query>
}

- If you cannot answer the question, respond with "I cannot answer the question."
- If you cannot generate a valid SQL query, respond with "I cannot generate a valid SQL query."
"""

class ToolCallingAgent(Agent):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        rule: str,
        model: str,
        api_base: Optional[str] = None,
        temperature: float = 0.0,
        verbose: bool = False
    ):
        self.tools_info = [tool for tool in tools_info if tool['function']["name"] in TOOL_SETS]
        self.rule = rule
        self.model = model
        self.api_base = api_base
        self.temperature = temperature
        self.verbose = verbose
        self.instruction = TOOL_CALLING_INSTRUCTION + '\n' + self.rule
    def run(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30, agent_timeout: int = 600
    ) -> AgentRunResult:
        agent_cost = 0.0
        agent_elapsed = 0.0
        env_reset_res = env.reset(task_index=task_index)
        obs_user = env_reset_res.observation
        env_info = env_reset_res.info.model_dump()
        reward = 0.0
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.instruction},
            {"role": "user", "content": obs_user},
        ]
        
        if self.verbose:
            print(f"\n{'='*50}")
            print(f"[USER]: {obs_user}")
            print(f"{'='*50}")
        
        done = False
        for step in range(1, max_num_steps + 1):
            t0 = time.time()
            next_message, action, done, cost = get_action(model = self.model,
                                                          messages = messages,
                                                          temperature =self.temperature,
                                                          api_base =self.api_base,
                                                          tools = self.tools_info)
            agent_elapsed += time.time() - t0
            agent_cost += cost
            if agent_elapsed > agent_timeout:
                raise AgentTimeoutError(f"Agent LLM cumulative time exceeded {agent_timeout}s ({agent_elapsed:.1f}s)")
            env_response = env.step(action)
            reward = env_response.reward
            env_info = {**env_info, **env_response.info.model_dump()}
            if action.name != 'respond':
                next_message["tool_calls"] = next_message["tool_calls"][:1]
                if self.verbose:
                    tool_name = next_message["tool_calls"][0]["function"]["name"]
                    tool_args = next_message["tool_calls"][0]["function"]["arguments"]
                    print(f"[AGENT]: Using tool '{tool_name}' with args: {tool_args}")
                    print(f"[TOOL RESULT]: {env_response.observation}")
                    print(f"{'-'*30}")
                
                messages.extend(
                    [
                        next_message,
                        {
                            "role": "tool",
                            "tool_call_id": next_message["tool_calls"][0]["id"],
                            "name": next_message["tool_calls"][0]["function"]["name"],
                            "content": env_response.observation,
                        },
                    ]
                )
            else:
                if self.verbose:
                    print(f"[AGENT]: {next_message.get('content', '')}")
                    print(f"[USER]: {env_response.observation}")
                    print(f"{'-'*30}")
                
                messages.extend(
                    [
                        next_message,
                        {"role": "user", "content": env_response.observation},
                    ]
                )
            if done or env_response.done:
                break

        return AgentRunResult(
            reward=reward,
            messages=messages,
            agent_cost=round(agent_cost, 8),
            info=env_info
        )
