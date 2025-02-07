# agent.py

import os
import ast
import re
import json
import time
import io
import sys
import contextlib
import threading
import queue
from typing import List, Optional
import streamlit as st

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain.tools import StructuredTool
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLDatabaseTool,
)
from langchain_core.messages import SystemMessage, ToolMessage, AIMessage, HumanMessage, convert_to_openai_messages
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.vectorstores import FAISS
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt, Command
from langchain_core.tools.render import ToolsRenderer, render_text_description

####################################################################
embedding_model = 'text-embedding-3-small'
# embedding_model = 'text-embedding-3-large'
####################################################################

# A helper function for formatting the conversation history (used for logging)
def get_numbered_history(messages: List) -> str:
    history_lines = []
    turn = 1
    for message in messages:
        message_type = message.__class__.__name__
        header = f"Turn {turn} [{message_type}]:"
        turn += 1
        
        content_lines = []
        if message.content:
            if message_type == "ToolMessage":
                content_lines.append(f"Observation: {message.content}")
            else:
                content_lines.append(message.content.strip())
        
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                tool_name = tc.get("name", "UnknownTool")
                tool_args = tc.get("args", {})
                tool_lines = [
                    f"Action: {tool_name}",
                    f"Action Input: {json.dumps(tool_args, indent=2)}"
                ]
                content_lines.append('\n'.join(tool_lines))
        indented_content = "\n".join("    " + line for line in "\n".join(content_lines).splitlines())
        history_lines.append(f"{header}\n{indented_content}")
    return "\n\n".join(history_lines)

def parse_sql(response: str) -> str:
    pattern = r'```sql([\s\S]*?)```'
    matches = re.findall(pattern, response)
    if matches:
        return matches[-1].strip()
    stripped = response.strip()
    if stripped.lower().startswith("select") or stripped.lower().startswith("with"):
        return stripped
    raise ValueError("No SQL found in the response")

def parse_json(response: str) -> str:
    pattern = r'```json([\s\S]*?)```'
    matches = re.findall(pattern, response)
    if matches:
        return json.loads(matches[-1].strip())
    raise ValueError("No JSON found in the response")

def query_as_list(db: SQLDatabase, table: str, column: str) -> List[str]:
    query = f"SELECT DISTINCT {column} FROM {table}"
    res = db.run(query)
    res = [el for sub in ast.literal_eval(res) for el in sub if el]
    res = [re.sub(r"\\b\\d+\\b", "", string).strip() for string in res]
    return list(set(res))

def _initialize_vector_store(db: SQLDatabase, embeddings: OpenAIEmbeddings, faiss_path: str) -> FAISS:
    if os.path.exists(faiss_path):
        return FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
    columns_to_retrieve = {
        "d_icd_diagnoses": ["long_title"],
        "d_icd_procedures": ["long_title"],
        "prescriptions": ["drug"],
        "d_items": ["label"],
        "d_labitems": ["label"]
    }
    texts = []
    metadatas = []
    for table, columns in columns_to_retrieve.items():
        for column in columns:
            values = query_as_list(db, table, column)
            texts.extend(values)
            metadatas.extend([{"table": table, "column": column} for _ in values])
    vector_store = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
    vector_store.save_local(faiss_path)
    return vector_store

class ValueRetrieverInput(BaseModel):
    table: str = Field(..., description="The table name from which to retrieve values")
    column: str = Field(..., description="The column name from which to retrieve values")
    value: Optional[str] = Field(
        None,
        description="Optional value for similarity search. If not provided, returns k sample values."
    )
    k: Optional[int] = Field(
        10,
        description="The number of sample or similar values to return."
    )

def value_retriever_tool_func(
    table: str,
    column: str,
    vector_store: FAISS,
    db: SQLDatabase,
    value: Optional[str] = None,
    k: int = 10
) -> str:
    try:
        if value:
            results = vector_store.similarity_search(
                query=value,
                k=k,
                filter={"table": table, "column": column}
            )
            docs_text = [doc.page_content for doc in results]
            unique_values = list(set(docs_text))
            if len(unique_values) == 0:
                return f"No matches found in {table}.{column} for '{value}'."
            elif len(unique_values) == 1:
                return (
                    f"I found one close match in {table}.{column}: '{unique_values[0]}'.\n"
                    "Is this correct?"
                )
            else:
                return (
                    f"I found multiple close matches in {table}.{column}:\n"
                    f"{json.dumps(unique_values, ensure_ascii=False, indent=2)}\n"
                    "Ask the user to clarify which value they are interested in (provide all the values in a readable format. these values are case-sensitive, so do not change the case)."
                )
        else:
            query = f"SELECT DISTINCT {column} FROM {table} LIMIT {k}"
            res = db.run(query)
            sample_vals = [row[0] for row in ast.literal_eval(res) if row[0]]
            sample_vals = list(set(sample_vals))
            return (
                f"Here are up to {k} sample values in {table}.{column}:\n"
                f"{json.dumps(sample_vals, ensure_ascii=False, indent=2)}\n"
                "Ask the user to clarify which value they are interested in (provide all the values in a readable format. these values are case-sensitive, so do not change the case). If the user confirms the value, proceed to the next step."
            )
    except Exception as e:
        return f"Error retrieving values: {str(e)}"

# Define the type for the agent’s state.
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_call_cache: dict

######################################################
# Cached heavy initialization
@st.cache_resource(show_spinner=False)
def get_static_agent(model_name: str, database_path: str, faiss_path: str, config: dict):
    # Initialize heavy resources.
    llm = ChatOpenAI(model_name=model_name, temperature=0.7)
    db = SQLDatabase.from_uri(f"sqlite:///{database_path}")
    sql_db_list_tables = ListSQLDatabaseTool(db=db)
    sql_db_schema = InfoSQLDatabaseTool(db=db)
    sql_db_query = QuerySQLDatabaseTool(db=db)
    faiss_path = 'faiss_index-'+embedding_model
    embeddings = OpenAIEmbeddings(model=embedding_model)
    vector_store = _initialize_vector_store(db, embeddings, faiss_path)

    def _value_retriever_wrapper(table, column, value=None, k=10) -> str:
        return value_retriever_tool_func(
            table=table,
            column=column,
            value=value,
            k=k,
            vector_store=vector_store,
            db=db
        )

    value_retriever_tool = StructuredTool(
        name="value_retriever",
        func=_value_retriever_wrapper,
        description=(
            "Search for column values. If 'value' is provided, perform a similarity search; "
            "otherwise, return sample values."
        ),
        args_schema=ValueRetrieverInput
    )

    @tool
    def human_feedback(question: str):
        """Ask for human clarification or confirmation."""
        pass

    tools = [sql_db_list_tables, sql_db_schema, sql_db_query, value_retriever_tool, human_feedback]
    llm_with_tools = llm.bind_tools(tools)
    tools_by_name = {tool.name: tool for tool in tools}

    def sql_node(state: AgentState) -> AgentState:
        system_prompt_content = f"""
Generate a SQL query that fully addresses the user request. You have access to the following tools:

{render_text_description(tools)}

Use the following format:

Question: The input question for which you must generate a SQL query.
Action: Choose one of [{", ".join([t.name for t in tools])}] or "Finish" if you are ready.
Action Input: Provide the input for the action. If you choose "Finish", this should be the final SQL query.
Observation: The result of the action.
... (You may iterate through Action/Action Input/Observation as needed.)

When you are ready to provide your final answer, output it as follows:
Final Answer: [Your final SQL query, not plain text]

After the conversation ends, your final SQL query will be executed using the 'sql_db_query' tool, and the query result will be returned.

Begin!
"""
        system_prompt = SystemMessage(content=system_prompt_content)
        conversation = [system_prompt] + state["messages"]
        response = llm_with_tools.invoke(conversation)
        
        print("============ Input ============", file=sys.stderr)
        print(get_numbered_history([system_prompt] + state["messages"]), file=sys.stderr)
        print("============ Raw Input ============", file=sys.stderr)
        print('\n'.join([str(turn) for turn in convert_to_openai_messages(state["messages"])]), file=sys.stderr)
        print("============ Output ============", file=sys.stderr)
        print("response", response, file=sys.stderr)

        if response.content != "":
            print(f"{response.content}", file=sys.stderr)

        new_state = AgentState(
            messages=[response],
            tool_call_cache=state["tool_call_cache"]
        )        
        return new_state

    def tool_node(state: AgentState) -> AgentState:
        outputs = []
        for tool_call in state["messages"][-1].tool_calls:
            if tool_call["name"] == "human_feedback":
                continue
            call_key = f"{tool_call['name']}:{json.dumps(tool_call['args'], sort_keys=True)}"
            if call_key in state["tool_call_cache"]:
                tool_result = state["tool_call_cache"][call_key]
            else:
                tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
                state["tool_call_cache"][call_key] = tool_result

            outputs.append(
                ToolMessage(
                    content=tool_result,
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
            
            print("============ Tool Call ============", file=sys.stderr)
            print('\n'.join([str(turn) for turn in convert_to_openai_messages(outputs)]), file=sys.stderr)
            print(f"Action: {tool_call['name']}", file=sys.stderr)
            print(f"Action Input: {tool_call['args']}", file=sys.stderr)
            print(f"Observation: {tool_result}", file=sys.stderr)

            print(f"Action: {tool_call['name']}")
            print(f"Action Input: {tool_call['args']}")
            print(f"Observation: {tool_result}")
            print()

        new_state = AgentState(
            messages=outputs,
            tool_call_cache=state["tool_call_cache"]
        )
        return new_state

    def ask_human(state: AgentState) -> AgentState:
        feedback_tool_call = state["messages"][-1].tool_calls[0]
        prompt = feedback_tool_call["args"].get("question", "Please provide clarification:")
        human_response = 'raw input'
        tool_message = ToolMessage(
            content=human_response,
            name="human_feedback",
            tool_call_id=feedback_tool_call["id"],
        )
        interrupt(prompt)
        return AgentState(
            messages=[tool_message],
            tool_call_cache=state["tool_call_cache"]
        )

    def should_continue(state: AgentState) -> str:
        last_message = state["messages"][-1]
        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return END
        tool_names = [tool_call["name"] for tool_call in last_message.tool_calls]
        if "human_feedback" in tool_names:
            return "ask_human"
        else:
            return "tool_node"

    graph = StateGraph(AgentState)
    graph.add_node("sql_node", sql_node)
    graph.add_node("tool_node", tool_node)
    graph.add_node("ask_human", ask_human)

    graph.add_edge(START, "sql_node")
    graph.add_conditional_edges(
        "sql_node",
        should_continue,
        path_map={"ask_human": "ask_human",
                  "tool_node": "tool_node",
                  END: END}
    )
    graph.add_edge("ask_human", "sql_node")
    graph.add_edge("tool_node", "sql_node")
    
    memory = MemorySaver()
    graph = graph.compile(checkpointer=memory)

    return {
        "graph": graph,
        "sql_db_query": sql_db_query
    }

# Build the agent using the cached heavy resources and a fresh mutable state.
def create_db_agent(
    model_name: str = "gpt-4o-mini",
    database_path: str = "mimic_iv.sqlite",
    faiss_path: str = "faiss_index",
    config: dict = {}
):
    static_agent = get_static_agent(model_name, database_path, faiss_path, config)
    return {
        "graph": static_agent["graph"],
        "sql_db_query": static_agent["sql_db_query"],
        "config": config,
        "dialog_state": AgentState(messages=[], tool_call_cache={})
    }

def run_agent_stream(agent: dict, user_input: str, config: dict = {}):
    # Append the user's question.
    prefix = "Question: "
    agent["dialog_state"]["messages"].append(HumanMessage(content=prefix + user_input))

    output_queue = queue.Queue()

    class QueueWriter:
        def write(self, s):
            if s.strip():
                output_queue.put(s)
        def flush(self):
            pass

    writer = QueueWriter()

    def worker():
        with contextlib.redirect_stdout(writer):
            final_state = agent["graph"].invoke(agent["dialog_state"], config)
        agent["dialog_state"] = final_state
        output_queue.put(None)

    thread = threading.Thread(target=worker)
    thread.start()

    while True:
        try:
            chunk = output_queue.get(timeout=0.1)
        except queue.Empty:
            if not thread.is_alive():
                break
            continue
        if chunk is None:
            break
        for line in chunk.splitlines():
            if line.strip():
                yield {"type": "intermediate", "text": line}

    thread.join()

    last_message = agent["dialog_state"]["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            if tool_call["name"] == "human_feedback":
                yield {
                    "type": "interrupt",
                    "text": tool_call["args"].get("question", "").strip(),
                    "tool_call_id": tool_call["id"]
                }
                break
    if "Final Answer:" in last_message.content:
        response = last_message.content
        try:
            query_text = response.split("Final Answer:")[1].strip()
            query_result = agent["sql_db_query"].invoke({"query": query_text})
            response += f"\n\nQuery Result:\n{query_result}"
        except Exception as e:
            response += f"\n\nFailed to execute query: {e}"
        yield {"type": "final", "text": response}
    else:
        match = re.search(r'"question":\s*"([^"]+)"', last_message.content)
        if match:
            extracted_question = match.group(1)
            yield {"type": "other", "text": extracted_question}
        else:
            yield {"type": "other", "text": last_message.content}
