from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class Task(BaseModel):
    db_id: str
    task_id: str
    task_type: str
    instruction: str
    gold_sql: str
    gold_answer: List[Any]
    
class Action(BaseModel):
    name: str
    kwargs: Dict[str, Any]

class Tool(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]
    required: List[str]

class RewardInfo(BaseModel):
    reward: Optional[float] = None
    info: Optional[Dict[str, Any]] = None

class EnvInfo(BaseModel):
    task: Task
    reward_info: RewardInfo

class EnvResponse(BaseModel):
    observation: str
    reward: Optional[float] = None
    done: bool
    info: EnvInfo

class AgentRunResult(BaseModel):
    reward: Optional[float] = None
    messages: List[Dict[str, Any]]
    agent_cost: Optional[float] = None
    info: EnvInfo

class CostInfo(BaseModel):
    agent_cost: Optional[float] = None
    user_cost: Optional[float] = None
    eval_cost: Optional[float] = None
    total_cost: Optional[float] = None

class ValidationResult(BaseModel):
    decision: str
    reason: str
    eval_cost: float

class EnvRunResult(BaseModel):
    db_id: str
    task_type: str
    task_id: str
    sample_id: str
    reward: Optional[float] = None
    info: EnvInfo
    messages: List[Dict[str, Any]]
    cost: CostInfo
    validation: Optional[ValidationResult] = None
    retry: Optional[int] = None
    retry_reason: Optional[List[str]] = None
    retry_exhausted: bool = False

class ValidationOutputFormat(BaseModel):
    explanation: str
    evidence: str
    broken_rule: str
    result: str

class ReActOutputFormat(BaseModel):
    thought: str
    response: str

class ReflectionOutputFormat(BaseModel):
    reflection: str
    new_response: str
