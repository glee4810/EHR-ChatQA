import abc
from typing import Optional
from src.envs.base import Env
from src.types import AgentRunResult


class AgentTimeoutError(Exception):
    pass


class Agent(abc.ABC):
    @abc.abstractmethod
    def run(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30, agent_timeout: int = 600
    ) -> AgentRunResult:
        raise NotImplementedError
