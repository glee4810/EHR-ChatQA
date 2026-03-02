from typing import List, Dict, Any
from dotenv import load_dotenv
from typing import Optional
from src.utils import load_json, get_completion
from src.types import ValidationOutputFormat, ValidationResult
from src.utils import count_agent_turns
import traceback
import io
import json
import time

load_dotenv()

user_validator_prompt = '''Your task is to simulate a user with no knowledge of SQL or database management systems, who needs specific information from an EHR database and relies on the DB agent for help.

Instruction: {instruction}

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
- Do not deviate from what is specified in the instruction, such as failing to mention the top N ranked tied results to retrieve, requesting medication order records or prescription records instead of administered records, requesting 6 months of data instead of 180 days, asking follow-up questions when they are not specified in the instruction, or revealing disallowed information before the DB agent mentions it.

Conversation:
{conversation}

Gold SQL:
{gold_sql}

Types of common user errors:
- The user gives away their goals all at once in the same turn.
- The user acts like a DB agent or AI assistant instead of the user (e.g., writing, reviewing, or executing SQL queries, calling external APIs, or responding to the DB agent in a machine assistant way).
- The user asks for information that is slightly different from what is specified in the instruction (e.g., requesting medication order records or prescription records instead of administered records, or requesting 6 months of data instead of 180 days).
- The user confirms values that differ from those in the gold SQL, unless specified otherwise in the instruction (e.g., requesting data for just "diabetes" when the gold SQL uses LIKE "%diabetes%").
- The user mentions information beyond the instruction, including related or unrelated details not specified (e.g., asking follow-up questions not in the instruction).
- The user does not provide all the detailed conditions specified in the instruction before ending the conversation. These conditions may include, for example, retrieving all tied ranked results, specifying the top "N" results to retrieve, handling duplicate patient records, or indicating keywords to include or exclude when searching for data. However, if the DB agent retrieves no relevant data, these conditions are not required.
- The user does not provide all the detailed conditions specified in the predicates of the gold SQL, either explicitly or implicitly, before ending the conversation.
- The user does not double-check with the DB agent to see if the agent's final answer satisfies all the information the user provided before ending the conversation.
- The user violates any other rules specified in the rules or the user instruction.

You must respond in JSON format with the following fields:
- explanation: Provide a clear and concise explanation of why you made the decision.
- broken_rule: If a user error is found, provide the exact rule or instruction that the user violated. If no error is found, provide an empty string.
- evidence: If a user error is found, provide the exact user response that caused the error. If no error is found, provide an empty string.
- result: Answer "user_error" if a user error is found. Answer "no_error" if no user error is found.'''


agent_validator_prompt = '''Below are the general rules for the DB agent:
- The DB agent must assume the user has no knowledge of SQL, databases, or stored values, and cannot execute queries.
- The DB agent must interact with the user only in natural language and must not show raw SQL queries.
- The DB agent must not modify the database schema or contents. The following commands are forbidden: INSERT, UPDATE, DELETE, DROP, ALTER.
- The DB agent must always explain answers in natural language, including the reasoning or conditions used to arrive at those answers. If SQL references are necessary, the DB agent must explain them in terms understandable to someone with no SQL knowledge.
- The DB agent must clearly explain when a question cannot be answered (e.g., due to limitations of SQL or empty results) and ask the user to rephrase or modify the request.
- The DB agent must generate a non-empty response. It must include either a message or a tool call.

{grading_rules}

Conversation:
{conversation}

You must respond in JSON format with the following fields:
- explanation: Provide a clear and concise explanation of why you made the decision.
- broken_rule: If an agent error is found, provide the exact rule that the agent violated. If no error is found, provide an empty string.
- evidence: If an agent error is found, provide the exact agent response that caused the error. If no error is found, provide an empty string.
- result: Answer "agent_error" if an agent error is found. Answer "no_error" if no agent error is found.'''


def display_conversation(messages):
    if len(messages) == 0:
        raise ValueError("Trajectory is empty")
    log = []
    for item in messages:
        if item["role"] == "system":
            continue
        if item["role"] == "user":
            log.append("-----")
            log.append(f"[USER]:")
            log.append(f"{item['content'].strip()}")
            log.append("-----")
        elif item["role"] == "assistant" and item['content']:
            log.append(f"[DB AGENT]:")
            log.append(f"{item['content'].strip()}")
    return "\n".join(log)


def display_conversation_agent(messages):
    if len(messages) == 0:
        raise ValueError("Trajectory is empty")
    log = ["-----"]
    for item in messages:
        if item["role"] == "system":
            continue
        if item["role"] == "user":
            log.append(f"[USER]:")
            log.append(f"(Visible to user) {item['content'].strip()}")
            log.append("-----")
        elif item["role"] == "assistant":
            log.append(f"[DB AGENT]:")
            if item['content']:
                log.append(f"(Visible to user) {item['content'].strip()}")
            if 'tool_calls' in item and item['tool_calls']:
                for tool_call in item['tool_calls']:
                    # log.append(f"{tool_call['function']['name']}({tool_call['function']['arguments']})")
                    if tool_call['function']['name'] == 'sql_execute':
                        log.append(f"(Hidden from user) {tool_call['function']['name']}({tool_call['function']['arguments']})")
                    else:
                        log.append(f"(Hidden from user) (omitted)")
            log.append("-----")
        elif item["role"] == "tool":
            log.append(f"[TOOL]:")
            # log.append(f"{item['content'].strip()}")
            if item['name'] == 'sql_execute':
                try:
                    from ast import literal_eval
                    content = literal_eval(item['content'])
                    if len(content) > 5:
                        display_lines = str(content[:5]) + f' ... (+{len(content) - 5} more)'
                    else:
                        display_lines = str(content)
                except:
                    display_lines = item['content']
                log.append(f"(Hidden from user) {display_lines}")
            else:
                log.append(f"(Hidden from user) (omitted)")
            log.append("-----")
        
    return "\n".join(log)


def user_validator(messages: List[Dict[str, Any]], env: Dict[str, Any], model: str, api_base: Optional[str] = None, n=1, temperature=0.0, max_agent_turns=30) -> ValidationResult:

    if count_agent_turns(messages) == max_agent_turns:
        reason = {'broken_rule': '', 'evidence': '', 'explanation': 'The agent reached the maximum number of agent turns.'}
        return ValidationResult(decision='no_error', reason=json.dumps(reason), eval_cost=0.0)
    flag = False
    for message in messages:
        if message['role'] == 'assistant' and message['content'] is None and message['tool_calls'] is None:
            flag = True
            break
    if flag:
        reason = {'broken_rule': '', 'evidence': '', 'explanation': 'The agent generated an empty response.'}
        return ValidationResult(decision='no_error', reason=json.dumps(reason), eval_cost=0.0)

    validator_messages = [
        {"role": "system", "content": "Your task is to determine whether [USER] accurately followed the provided rules and user instruction during their conversation with [DB AGENT]. Errors are defined as any deviations from the rules or user instruction. You must carefully review the rules, user instruction, conversation between [USER] and [DB AGENT], and the gold SQL query to identify any errors made by [USER]."},
        {"role": "user", "content": user_validator_prompt.format(instruction=env.task.instruction, 
                                                                 conversation=display_conversation(messages),
                                                                 gold_sql=env.task.gold_sql)},
    ]

    max_retries = 10
    for attempt in range(max_retries):
        try:
            response = get_completion(model=model, messages=validator_messages, temperature=temperature, response_format=ValidationOutputFormat, api_base=api_base, n=n)
            break
        except Exception as e:
            tb_str = io.StringIO()
            traceback.print_exc(file=tb_str)
            error_details = tb_str.getvalue()
            error_reason = f"Unexpected error during simulation: {error_details}"
            if 'generate_requests_per_model_per_day' in error_reason:
                print('Gemini-2.5-Flash exceeded the daily limit')
                exit(0)
            print(f"[Validator] Error (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(3)
    else:
        reason = {'broken_rule': '', 'evidence': '', 'explanation': 'Validator failed after max retries.'}
        return ValidationResult(decision='no_error', reason=json.dumps(reason), eval_cost=0.0)
    results = [load_json(m.message.content) for m in response.choices]

    eval_cost = 0.0
    if hasattr(response, '_hidden_params') and 'response_cost' in response._hidden_params and response._hidden_params["response_cost"]:
        eval_cost = response._hidden_params["response_cost"]
            
    if sum([r['result'] == 'user_error' for r in results]) > 0:
        decision = 'user_error'
        reason = [r for r in results if r['result'] == 'user_error'][0]
    else:
        decision = 'no_error'
        reason = [r for r in results if r['result'] == 'no_error'][0]

    reason = {'broken_rule': reason['broken_rule'], 
              'evidence': reason['evidence'],
              'explanation': reason['explanation']}

    return ValidationResult(decision=decision, reason=json.dumps(reason), eval_cost=eval_cost)


def agent_validator(messages: List[Dict[str, Any]], env: Dict[str, Any], model: str, api_base: Optional[str] = None, n=1, temperature=0.0, max_agent_turns=30) -> ValidationResult:

    if count_agent_turns(messages) == max_agent_turns:
        reason = {'broken_rule': 'The DB agent must limit each conversation to 30 interactions (including user exchanges and tool calls) and 600 seconds total.', 'evidence': '', 'explanation': 'The agent reached the maximum number of agent turns.'}
        return ValidationResult(decision='agent_error', reason=json.dumps(reason), eval_cost=0.0)
    flag = False
    for message in messages:
        if message['role'] == 'assistant' and message['content'] is None and message['tool_calls'] is None:
            flag = True
            break
    if flag:
        reason = {'broken_rule': ' The DB agent must generate a non-empty response, which must include either a message or a tool call.', 'evidence': '', 'explanation': 'The agent generated an empty response.'}
        return ValidationResult(decision='agent_error', reason=json.dumps(reason), eval_cost=0.0)

    if env.task_type == "incre":
        from src.envs.rules import task_type_incremental
        grading_rules = task_type_incremental
    elif env.task_type == "adapt":
        from src.envs.rules import task_type_adaptive
        grading_rules = task_type_adaptive

    validator_messages = [
        {"role": "system", "content": "Your task is to determine whether [DB Agent] accurately followed the provided rules during their conversation with [User]. Errors are defined as any deviations from the rules. You must carefully review the rules and conversation between [USER] and [DB AGENT] to identify any errors made by [DB Agent]. Note that you are not evaluating the correctness of the SQL queries based on the user's request. Instead, you are checking whether the agent followed the rules and clearly and accurately explained the SQL executed via sql_execute, while assuming that the user has no knowledge of SQL or database management systems."},
        {"role": "user", "content": agent_validator_prompt.format(grading_rules=grading_rules,
                                                                  conversation=display_conversation_agent(messages))},
    ]

    if ('llama' in model.lower() or 'qwen' in model.lower()) and temperature == 0.0:
        temperature = 0.1

    response = get_completion(model=model, messages=validator_messages, temperature=temperature, response_format=ValidationOutputFormat, api_base=api_base, n=n)
    results = [load_json(m.message.content) for m in response.choices]

    eval_cost = 0.0
    if hasattr(response, '_hidden_params') and 'response_cost' in response._hidden_params and response._hidden_params["response_cost"]:
        eval_cost = response._hidden_params["response_cost"]
            
    if sum([r['result'] == 'agent_error' for r in results]) > 0:
        decision = 'agent_error'
        reason = [r for r in results if r['result'] == 'agent_error'][0]
    else:
        decision = 'no_error'
        reason = [r for r in results if r['result'] == 'no_error'][0]

    reason = {'broken_rule': reason['broken_rule'], 
              'evidence': reason['evidence'],
              'explanation': reason['explanation']}

    return ValidationResult(decision=decision, reason=json.dumps(reason), eval_cost=eval_cost)

