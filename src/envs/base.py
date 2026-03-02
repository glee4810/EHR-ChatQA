import re
import random
import logging
from src.types import Tool
from src.utils import process_result, process_answer
from typing import Dict, List, Type, Optional

import sqlite3
from src.envs.user import load_user
from src.types import (
    Action,
    Task,
    EnvInfo,
    EnvResponse,
    RewardInfo,
)

numeric_to_words = {'two hundred and forty-seven': '247', 'two hundred forty-seven': '247',
                    'six hundred and fifty-nine': '659', 'six hundred fifty-nine': '659',
                    'thirty five': '35', 'thirty-five': '35',
                    'forty-eight': '48', 'forty eight': '48',
                    'thirty-seven': '37', 'thirty seven': '37',
                    'twenty-eight': '28', 'twenty eight': '28',
                    'seventy-two': '72', 'seventy two': '72',
                    'eleven': '11', 'thirty': '30',
                    'three': '3', 'four': '4', 'five': '5', 'six': '6',
                    'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
                    'fourteen': '14', 'sixteen': '16', 'nineteen': '19',
                    'zero': '0', 'two': '2', 'one': '1'}

class Env(object):
    def __init__(
        self,
        tools: List[Type[Tool]],
        tasks: List[Task],
        user_strategy: str,
        user_model: str,
        user_temperature: float,
        db_path: str,
        task_type: str,
        api_base: Optional[str] = None,
        task_index: Optional[str] = None,
        rule: Optional[str] = "",
        retry_reason: Optional[List[str]] = None
    ) -> None:
        super().__init__()
        if retry_reason is None:
            retry_reason = []
        self.tools_map: Dict[str, Type[Tool]] = {
            tool.get_info()["function"]["name"]: tool for tool in tools
        }
        self.tools_info = [tool.get_info() for tool in tools]
        self.rule = rule
        self.user = load_user(
            user_strategy=user_strategy, model=user_model, temperature=user_temperature, api_base=api_base, retry_reason=retry_reason
        )
        self.actions: List[Action] = []
        self.db_path = db_path
        self.task_type = task_type
        self.tasks = tasks
        if tasks is not None:
            if task_index is not None:
                self.task_index = int(task_index)
            else:
                self.task_index = random.randint(0, len(tasks)-1)
            self.task = self.tasks[self.task_index]

    def reset(self, task_index: Optional[str] = None) -> EnvResponse:
        if task_index is not None:
            self.task_index = int(task_index)
        else:
            self.task_index = random.randint(0, len(self.tasks)-1)
        self.task = self.tasks[self.task_index]
        self.actions = []
        initial_observation = self.user.reset(self.task)
        return EnvResponse(
            observation=initial_observation,
            reward=0.0,
            done=False,
            info=EnvInfo(task=self.task, reward_info=RewardInfo())
        )

    def step(self, action: Action) -> EnvResponse:
        self.actions.append(action)

        info = EnvInfo(task=self.task, reward_info=RewardInfo())
        reward = 0.0
        done = False
        if action.name == 'respond':
            observation = self.user.step(action.kwargs["content"])
            if observation:
                done = "###END###" in observation
            else:
                observation = ""
        elif action.name in self.tools_map:
            try:
                observation = self.tools_map[action.name].invoke(**action.kwargs)
            except Exception as e:
                observation = f"Error: {e}"
        else:
            observation = f"Unknown action {action.name}"
        if done:
            if self.task_type == 'incre':
                reward_info = self.calculate_reward_sql()
            elif self.task_type == 'adapt':
                reward_info = self.calculate_reward_ans()
            else:
                raise NotImplementedError("Task type must be either 'incre' or 'adapt'")
            reward = reward_info.reward
            info.reward_info = reward_info
        return EnvResponse(
            observation=observation, 
            reward=reward,
            done=done,
            info=info)

    def calculate_reward_sql(self) -> RewardInfo:

        reward = 0.0
        gold_answer = process_result(self.task.gold_answer)
        pred_sql = []
        pred_answer = []
        sql_actions = [action.kwargs["query"].strip() for action in self.actions if action.name == 'sql_execute' and 'query' in action.kwargs and action.kwargs['query'] is not None]
        if len(sql_actions) > 0:
            for sql in sql_actions:
                conn = sqlite3.connect(f"file:{self.db_path}?immutable=1", uri=True)
                try:
                    cursor = conn.cursor()
                    cursor.execute(sql)
                    pred_sql_answer = cursor.fetchall()
                    pred_sql_answer = process_result(pred_sql_answer)
                    if pred_sql_answer and len(pred_sql_answer) > 0:
                        converted_pred_sql_answer = list(zip(*pred_sql_answer))
                        for i in range(len(converted_pred_sql_answer)):
                            pred_sql.append(sql)
                            pred_answer.append(converted_pred_sql_answer[i])
                            if sorted(set([r for r in converted_pred_sql_answer[i] if r != 'None'])) == sorted(set([el[0] for el in gold_answer])):
                                reward = 1.0
                                break
                except sqlite3.Error as e:
                    pred_sql.append(sql)
                    pred_answer.append(str(e))
                finally:
                    conn.close()
                if reward == 1.0:
                    break

        reward_info = RewardInfo(reward=reward, info={'pred_sql': pred_sql, 'pred_answer': pred_answer})
        return reward_info

    def calculate_reward_ans(self) -> RewardInfo:

        reward = 0.0
        gold_answer = [process_answer(ans[0]) for ans in process_result(self.task.gold_answer)]
        gold_answer = [numeric_to_words[str(gold_ans)] if str(gold_ans) in numeric_to_words else gold_ans for gold_ans in gold_answer]
        pred_response = []
        pred_answer = []
        response_actions = [action.kwargs["content"] for action in self.actions if action.name == 'respond' and 'content' in action.kwargs and action.kwargs['content'] is not None]
        if len(response_actions) > 0:
            for response in response_actions:
                pred_response.append(response)
                pattern = r'(?:<answer>|```answer|```\s*\n)(.*?)(?:</answer>|```)'
                match = re.search(pattern, response, re.DOTALL)
                if match:
                    reward = 1.0
                    pred_ans = match.group(1).strip()                                        
                    pred_answer.append(pred_ans)
                    pred_ans = pred_ans.lower()
                    for gold_ans in gold_answer:                        
                        if str(pred_ans) in numeric_to_words:
                            pred_ans = numeric_to_words[str(pred_ans)]
                        if str(gold_ans).lower() not in str(pred_ans).lower():
                            reward = 0.0
                            break
                    
                    if reward == 1.0:
                        # to avoid false positive
                        pred_ans_to_check = pred_ans
                        for gold_ans in sorted(gold_answer, key=len)[::-1]:
                            pred_ans_to_check = pred_ans_to_check.replace(gold_ans, '', 1)
                        patterns = ['and', ',', ';', '.0000', '.000', '.00', '.0', '%', 'days', 'day'] # order matters
                        for pattern in patterns:
                            pred_ans_to_check = pred_ans_to_check.replace(pattern, '')
                        pred_ans_to_check = pred_ans_to_check.rstrip('0').rstrip('.').strip()
                        if len(pred_ans_to_check) == 0:
                            reward = 1.0
                        else:
                            reward = 0.0

                else:
                    pred_answer.append('N/A')
                if reward == 1.0:
                    break
                
        reward_info = RewardInfo(reward=reward, info={'pred_response': pred_response, 'pred_answer': pred_answer})
        return reward_info
