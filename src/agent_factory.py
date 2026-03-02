from typing import List, Dict, Any, Optional
from src.agents.base import Agent

def get_agent(
    tools_info: List[Dict[str, Any]],
    model: str,
    api_base: Optional[str] = None,
    agent_strategy: Optional[str] = None,
    temperature: float = 0.0,
    rule: str = "",
    verbose: bool = False
) -> Agent:
    if agent_strategy == "tool-calling":
        from src.agents.tool_calling_agent import ToolCallingAgent
        return ToolCallingAgent(
            tools_info=tools_info,
            model=model,
            api_base=api_base,
            temperature=temperature,
            rule=rule,
            verbose=verbose
        )
    else:
        raise ValueError(f"Agent strategy {agent_strategy} not implemented")