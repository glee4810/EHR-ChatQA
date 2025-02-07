# streamlit run app.py --server.port 8502

import os
import time
from dotenv import load_dotenv
import streamlit as st
from agent import create_db_agent, run_agent_stream
from langchain_core.tools import tool
from langgraph.types import Command, interrupt
from langchain_core.messages import SystemMessage, ToolMessage, AIMessage, HumanMessage, convert_to_openai_messages, convert_to_messages

####################################################################################################################################
if not os.path.exists("faiss_index-text-embedding-3-small"):
    import gdown
    import zipfile

    @st.cache_data(show_spinner=False)
    def download_and_extract_model(url, output_zip="faiss_index-text-embedding-3-small.zip", extract_to="."):
        if not os.path.exists(output_zip):
            gdown.download(url, output_zip, quiet=False)
        with zipfile.ZipFile(output_zip, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

    file_id = '1_2mfOKJgy5sBOi-enmuz0uBfNxena3s-'
    file_url = f'https://drive.google.com/uc?id={file_id}&export=download'

    # Call the function (this will be cached)
    download_and_extract_model(file_url)

if not os.path.exists(".env"):
    with open(".env", "w") as env_file:
        api_key = st.secrets['OPENAI_API_KEY']
        env_file.write(f"OPENAI_API_KEY={api_key}\n")
####################################################################################################################################

load_dotenv()

about_text = """
### About

With this application, you can retrieve patient data by asking questions in natural language from electronic medical records.
Powered by a database agent, the application takes user queries, clarifies them if needed, and generates SQL to retrieve the answers.

This version provides access to the processed MIMIC-IV Demo database, 
which includes the following tables (17 tables):
- patients (basic patient details)  
- admissions (hospital stays)  
- diagnoses_icd (diagnoses)  
- procedures_icd (treatment)  
- prescriptions (medications)  
- labevents (lab test results)  
- costs (treatment and medication expenses, etc.)  
- chartevents (vital signs)  
- inputevents (e.g., intravenous fluids)  
- outputevents (e.g., urine output)  
- microbiologyevents (infection test results)  
- icustays (ICU admission records)
- transfers (patient movements in the hospital)  
- d_icd_diagnoses (ICD-9/10 diagnosis codes)  
- d_icd_procedures (ICD-9/10 procedure codes)  
- d_items (codes for vitals, input, output, etc.)  
- d_labitems (codes for lab tests)  

v1.0.0"""

def format_text(text: str, expanded: bool = False) -> str:
    lines = text.splitlines()
    processed_lines = []
    for line in lines:
        if line.startswith("Action:"):
            line = line.replace("Action:", '<span style="color: blue; font-weight:bold;">Action:</span>', 1)
        elif line.startswith("Action Input:"):
            line = line.replace("Action Input:", '<span style="color: green; font-weight:bold;">Action Input:</span>', 1)
        elif line.startswith("Observation:"):
            line = line.replace("Observation:", '<span style="color: orange; font-weight:bold;">Observation:</span>', 1)
        processed_lines.append(line)
    formatted_text = "<br>".join(processed_lines)
    if expanded:
        details_html = (
            "<details open>"
            "<summary>Intermediate Steps (click to collapse)</summary>"
            + formatted_text +
            "</div>"
            "</details>"
        )
    else:
        details_html = (
            "<details>"
            "<summary>Intermediate Steps (click to expand)</summary>"
            + formatted_text +
            "</div>"
            "</details>"
        )
    return details_html

def update_intermediate_message(details_html: str):
    if "intermediate_message_index" in st.session_state:
        idx = st.session_state.intermediate_message_index
        st.session_state.messages[idx]["content"] = details_html
    else:
        st.session_state.messages.append({
            "role": "assistant",
            "content": details_html,
            "is_intermediate": True  # Optional flag for clarity.
        })
        st.session_state.intermediate_message_index = len(st.session_state.messages) - 1

def append_message(role: str, content: str):
    st.chat_message(role).markdown(content, unsafe_allow_html=True)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    st.session_state.messages.append({"role": role, "content": content})

def run_app():
    model_name = 'gpt-4o-mini'
    database_path = "mimic_iv.sqlite"
    st.set_page_config(page_title="Interact with the Database", page_icon="👋")
    st.title("Chat with Database Agent")
    st.text(f"Powered by {model_name}")

    with st.sidebar:
        st.markdown(about_text)
        st.markdown("[MIMIC-IV Demo](https://physionet.org/content/mimic-iv-demo/2.2/)", unsafe_allow_html=True)

    # Initialize session state variables if not already set.
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "waiting_for_clarification" not in st.session_state:
        st.session_state.waiting_for_clarification = False
    if "intermediate_steps" not in st.session_state:
        st.session_state.intermediate_steps = []
    if "config" not in st.session_state:
        st.session_state.config = {
            "configurable": {"thread_id": "1"},
            "recursion_limit": 20
        }
    if "agent" not in st.session_state:
        st.session_state.agent = create_db_agent(model_name, database_path, config=st.session_state.config)
    if "tool_call_id" not in st.session_state:
        st.session_state.tool_call_id = None

    if st.button("🔄 Refresh Chat"):
        st.session_state.messages = []
        st.session_state.waiting_for_clarification = False
        st.session_state.intermediate_steps = []
        st.session_state.config = {
            "configurable": {"thread_id": "user_session_1"},
            "recursion_limit": 20
        }
        st.session_state.agent = create_db_agent(model_name, database_path, config=st.session_state.config)
        st.session_state.tool_call_id = None
        st.rerun()

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"], unsafe_allow_html=True)

    if st.session_state.waiting_for_clarification:
        prompt_text = "Please respond to the clarifying question."
    else:
        prompt_text = "Type your question about the database."

    user_input = st.chat_input(prompt_text)
    if user_input:
        append_message("user", user_input)

        if st.session_state.waiting_for_clarification == False:

            intermediate_placeholder = None
            st.session_state.intermediate_steps = []
            if "intermediate_message_index" in st.session_state:
                del st.session_state.intermediate_message_index
                            
            for event in run_agent_stream(st.session_state.agent, user_input, config=st.session_state.config):
                event_type = event.get("type")
                text = event.get("text", "")

                if event_type == "interrupt":
                    st.session_state.waiting_for_clarification = True
                    st.session_state.tool_call_id = event.get("tool_call_id", "")
                    append_message("assistant", text)
                    break
                elif event_type == "final":
                    append_message("assistant", text)
                elif event_type == "intermediate":
                    if intermediate_placeholder is None:
                        intermediate_placeholder = st.chat_message("assistant").empty()
                    st.session_state.intermediate_steps.append(text)
                    combined_text = "\n".join(st.session_state.intermediate_steps)
                    formatted_text = format_text(combined_text, expanded=True)
                    intermediate_placeholder.markdown(formatted_text, unsafe_allow_html=True)
                    update_intermediate_message(formatted_text)
                    time.sleep(0.1)
                else:
                    append_message("assistant", text)                    

        if st.session_state.waiting_for_clarification == True:

            new_state = {
                    "messages": [ToolMessage(content=user_input, tool_call_id=st.session_state.tool_call_id)],
                    "tool_call_cache": st.session_state.agent["dialog_state"]["tool_call_cache"]
                }
            st.session_state.agent["graph"].update_state(st.session_state.config, new_state)
            st.session_state.waiting_for_clarification = False
            st.session_state.tool_call_id = None

if __name__ == "__main__":
    run_app()
