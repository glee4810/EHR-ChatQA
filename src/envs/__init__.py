import logging
from typing import Optional, List
from src.envs.base import Env

def get_env(
    env_name: str,
    user_strategy: str,
    task_type: str,
    user_model: str,
    user_temperature: float = 1.0,
    api_base: Optional[str] = None,
    task_index: Optional[str] = None,
    retry_reason: Optional[List[str]] = None
) -> Env:
    if env_name == "mimic_iv_star":
        from src.envs.mimic_iv_star import MimicIVStarEnv
        return MimicIVStarEnv(
            user_strategy=user_strategy,
            user_model=user_model,
            user_temperature=user_temperature,
            task_type=task_type,
            task_index=task_index,
            api_base=api_base,
            retry_reason=retry_reason
        )
    elif env_name == "eicu_star":
        from src.envs.eicu_star import eICUStarEnv
        return eICUStarEnv(
            user_strategy=user_strategy,
            user_model=user_model,
            user_temperature=user_temperature,
            task_type=task_type,
            task_index=task_index,
            api_base=api_base,
            retry_reason=retry_reason
        )
    else:
        raise ValueError(f"Unknown environment: {env_name}")
