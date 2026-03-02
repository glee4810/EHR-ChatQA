import abc
import json
import time
from litellm.exceptions import ContextWindowExceededError
from typing import Optional, List, Dict, Any
from src.utils import get_completion
from src.types import ReflectionOutputFormat
from src.types import Task


class BaseUser(abc.ABC):
    @abc.abstractmethod
    def reset(self, task: Optional[Task] = None) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, content: str) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def get_total_cost(self) -> float:
        raise NotImplementedError


class HumanUser(BaseUser):
    def __init__(self) -> None:
        super().__init__()
        self.messages: List[Dict[str, Any]] = []
        self.total_cost = 0.0

    def reset(self, task: Optional[Task] = None) -> str:
        pass

    def step(self, content: str) -> str:
        pass

    def get_total_cost(self) -> float:
        return round(self.total_cost, 8)


def check_user_error(content: str):
    if 'SELECT' in content or 'tool_call' in content or 'tool_code' in content or 'default_api' in content or 'print(' in content or '```' in content or 'Instruction:' in content:
        return True, "You generated AI-assistant-like content (e.g., SQL, tool calls, or code). Revise your previous response to follow the instruction and rules in the system prompt correctly."
    if '###END###' in content and content.strip() != '###END###':
        return True, "You should generate only '###END###' if you want to end the conversation. If you don't want to end the conversation, do not add '###END###' to your response."
    return False, None


class LLMUser(BaseUser):
    def __init__(self, model: str, temperature: float = 1.0, api_base: Optional[str] = None) -> None:
        super().__init__()
        self.messages: List[Dict[str, Any]] = []
        self.model = model
        self.total_cost = 0.0
        self.temperature = temperature
        self.api_base = api_base
        self.max_attempts = 3
        self.db_id = None
        self.task_id = None
        self.task_type = None

    def generate_next_message(self, messages: List[Dict[str, Any]]) -> str:
        for _ in range(self.max_attempts):
            try:
                res = get_completion(model=self.model, messages=messages, temperature=self.temperature, api_base=self.api_base)
                if hasattr(res, '_hidden_params') and 'response_cost' in res._hidden_params and res._hidden_params["response_cost"]:
                    self.total_cost += res._hidden_params["response_cost"]
                next_message = res.choices[0].message.model_dump()
                flag_error, error_msg  = check_user_error(next_message["content"])
                if flag_error:
                    messages.append({"role": "assistant", "content": str(next_message["content"])})
                    messages.append({"role": "user", "content": error_msg})
                    raise ValueError(error_msg)
                self.messages.append(next_message)
                return next_message["content"].strip()
            except ContextWindowExceededError as e:
                print("⚠️ Context window exceeded:", e)
                return '###END###'
            except Exception as e:
                time.sleep(3)
        return '###END###'

    def build_system_prompt(self, instruction: Optional[str]) -> str:
        instruction_display = "\n\nInstruction: " + instruction if instruction is not None else ""
        prompt = f"""Your task is to simulate a user with no knowledge of SQL or database management systems, who needs specific information from an EHR database and relies on the DB agent for help.{instruction_display}

Rules:
- The current time is 2100-12-31 23:59:00.
- Start with a short, broad question that reflects the overall goals from the instruction.
- Use your own words to describe your goals for the DB agent.
- Do not reveal all your goals at once. Instead, share them gradually, one or two sentences at a time.
- Speak casually and directly, without functionally unnecessary phrases (like "please" or "thank you") that make the tone sound like an AI assistant.
- Do not generate SQL, code snippets, empty messages, or AI-assistant-like outputs. Stay in the role of a user, not a DB agent.
- If the DB agent requests specific tables or column names, instruct it to locate them independently (unless the instruction says otherwise).
- If the DB agent requests writing or reviewing SQL queries, or summarizing the overall goal, instruct it to complete the task independently.
- If the DB agent gives an intermediate answer, don't complete it yourself. Instead, instruct it to finalize it (e.g., performing calculations like time differences or rephrasing answers).
- If the DB agent's answer seems satisfactory (even though you do not know whether it is correct or whether the requested data actually exists), ask the DB agent to double check that their final answer covers all goals raised. If not, request any missing parts.
- If the DB agent's answer covers all goals raised, generate only "###END###" to end the conversation. Do not add it after every reply. Use it only once in the final message.
- Do not deviate from what is specified in the instruction, such as failing to mention the top N ranked tied results to retrieve, requesting medication order records or prescription records instead of administered records, requesting 6 months of data instead of 180 days, asking follow-up questions when they are not specified in the instruction, or revealing disallowed information before the DB agent mentions it."""

        return prompt

    def reset(self, task: Optional[Task] = None) -> str:
        instruction = ""
        if task is not None:
            instruction = task.instruction
            self.db_id = task.db_id
            self.task_id = task.task_id
            self.task_type = task.task_type
        self.messages = [
            {"role": "system", "content": self.build_system_prompt(instruction=instruction)},
            {"role": "user", "content": "Hi! How can I help you today?"},
        ]
        new_message = self.generate_next_message(self.messages)
        return new_message

    def step(self, content: str) -> str:
        self.messages.append({"role": "user", "content": content})
        new_message = self.generate_next_message(self.messages)
        return new_message

    def get_total_cost(self) -> float:
        return round(self.total_cost, 8)


class VerifierUser(LLMUser):
    def __init__(self, model: str, temperature: float = 1.0, api_base: Optional[str] = None, retry_reason: Optional[List[str]] = None) -> None:
        super().__init__(model=model, temperature=temperature, api_base=api_base)
        self.retry_reason = retry_reason if retry_reason is not None else []

    def generate_next_message(self, messages: List[Dict[str, Any]]) -> str:
        last_message = None
        for _ in range(self.max_attempts):
            try:
                res = get_completion(model=self.model, messages=messages, temperature=self.temperature, api_base=self.api_base)
                if hasattr(res, '_hidden_params') and 'response_cost' in res._hidden_params and res._hidden_params["response_cost"]:
                    self.total_cost += res._hidden_params["response_cost"]
                next_message = res.choices[0].message.model_dump()
                flag_error, error_msg = check_user_error(next_message["content"])
                if flag_error:
                    messages.append({"role": "assistant", "content": str(next_message["content"])})
                    messages.append({"role": "user", "content": error_msg})
                    raise ValueError(error_msg)
                last_message = next_message
                if len(messages) > 2:
                    if self.verifier(messages, next_message["content"]):
                        self.messages.append(next_message)
                        return next_message["content"].strip()
                else:
                    self.messages.append(next_message)
                    return next_message["content"].strip()
            except ContextWindowExceededError as e:
                print("⚠️ Context window exceeded:", e)
                return '###END###'
            except Exception as e:
                time.sleep(3)
        if last_message is not None:
            self.messages.append(last_message)
            return last_message["content"].strip()
        return '###END###'

    def verifier(self, messages: List[Dict[str, Any]], response: str) -> bool:
        instruction_rules = messages[0]['content']
        error_cases = self.retry_reason
        if len(error_cases) > 0:
            previous_error_cases = ''
            for i, error_case in enumerate(error_cases):
                previous_error_cases += f'{i+1}. {error_case}\n'
            instruction_rules += '\n\nPrevious user mistakes:\n' + previous_error_cases
        prompt = """{instruction_rules}

Conversation:
{conversation}

User Response:
{response}"""

        prompt = prompt.format(
            instruction_rules=instruction_rules,
            conversation=self.display_conversation_user(messages[2:]),
            response=response
        )

        verifier_messages = [
            {"role": "system", "content": "You are a supervisor of the User in the conversation. You are given a conversation history between the User and the DB Agent. The User has generated a response, and your goal is to verify whether the User's response correctly aligns with the instruction and rules. Answer 'yes' if the User's response aligns with the conversation and correctly follows the criteria; otherwise, answer 'no'."},
            {"role": "user", "content": prompt},
        ]

        res = get_completion(model=self.model, messages=verifier_messages, temperature=0.0, api_base=self.api_base)
        if hasattr(res, '_hidden_params') and 'response_cost' in res._hidden_params and res._hidden_params["response_cost"]:
            self.total_cost += res._hidden_params["response_cost"]
        next_message = res.choices[0].message.model_dump()
        return next_message["content"] and 'yes' == next_message["content"].strip().lower()

    def display_conversation_user(self, messages: List[Dict[str, Any]]) -> str:
        log = []
        for item in messages:
            if item["role"] == "assistant":
                log.append(f"[User]: {item['content'].strip()}")
            elif item["role"] == "user" and item['content']:
                log.append(f"[DB Agent]: {item['content'].strip()}")
        if len(log) > 0:
            return "\n".join(log)
        else:
            return "N/A"

class ReflectionUser(VerifierUser):
    def __init__(self, model: str, temperature: float = 1.0, api_base: Optional[str] = None, retry_reason: Optional[List[str]] = None) -> None:
        super().__init__(model=model, temperature=temperature, api_base=api_base, retry_reason=retry_reason)
        self.reflection_max_attempts = 3

    def generate_next_message(self, messages: List[Dict[str, Any]]) -> str:
        for _ in range(self.max_attempts):
            try:
                res = get_completion(model=self.model, messages=messages, temperature=self.temperature, api_base=self.api_base)
                if hasattr(res, '_hidden_params') and 'response_cost' in res._hidden_params and res._hidden_params["response_cost"]:
                    self.total_cost += res._hidden_params["response_cost"]
                next_message = res.choices[0].message.model_dump()
                flag_error, error_msg = check_user_error(next_message["content"])
                if flag_error:
                    messages.append({"role": "assistant", "content": str(next_message["content"])})
                    messages.append({"role": "user", "content": error_msg})
                    raise ValueError(error_msg)
                if len(messages) > 2:
                    if self.verifier(messages, next_message["content"]):
                        self.messages.append(next_message)
                        return next_message["content"].strip()
                    last_response = next_message["content"]
                    for _ in range(self.reflection_max_attempts):
                        new_response = self.reflection(messages, last_response)
                        if self.verifier(messages, new_response):
                            next_message["content"] = new_response
                            self.messages.append(next_message)
                            return new_response.strip()
                        last_response = new_response
                    next_message["content"] = last_response
                    self.messages.append(next_message)
                    return last_response.strip()
                else:
                    self.messages.append(next_message)
                    return next_message["content"].strip()
            except ContextWindowExceededError as e:
                print("⚠️ Context window exceeded:", e)
                return '###END###'
            except Exception as e:
                time.sleep(3)
        return '###END###'

    def reflection(self, messages: List[Dict[str, Any]], response: str):
        instruction_rules = messages[0]['content']
        error_cases = self.retry_reason
        if len(error_cases) > 0:
            previous_error_cases = ''
            for i, error_case in enumerate(error_cases):
                previous_error_cases += f'{i+1}. {error_case}\n'
            instruction_rules += '\n\nPrevious user mistakes:\n' + previous_error_cases
        prompt = """{instruction_rules}

Conversation:
{conversation}

User Response:
{response}"""

        prompt = prompt.format(
            instruction_rules=instruction_rules,
            conversation=self.display_conversation_user(messages[2:]),
            response=response
        )

        reflection_messages = [
            {"role": "system", "content": "You are a supervisor of the User in the conversation. You are given the conversation history between the User and the DB Agent. The User's response has been flagged as not aligned with the instruction and rules. You need to generate a Reflection on what went wrong in the conversation and propose a revised User response that fixes the issue."},
            {"role": "user", "content": prompt},
        ]

        res = get_completion(model=self.model, messages=reflection_messages, temperature=0.0, api_base=self.api_base, response_format=ReflectionOutputFormat)
        if hasattr(res, '_hidden_params') and 'response_cost' in res._hidden_params and res._hidden_params["response_cost"]:
            self.total_cost += res._hidden_params["response_cost"]
        next_message = res.choices[0].message.model_dump()
        return json.loads(next_message["content"])['new_response'].strip()


def load_user(
    user_strategy: str,
    model: str,
    temperature: float = 1.0,
    api_base: Optional[str] = None,
    retry_reason: Optional[List[str]] = None
) -> BaseUser:
    if user_strategy == "human":
        return HumanUser()
    return ReflectionUser(model=model, temperature=temperature, api_base=api_base, retry_reason=retry_reason)
