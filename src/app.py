import app_ui
import streamlit as st
import config
import logging
import prompts
from streamlit_chat import message
from inference import ChatOpenAIRetrieval
from typing import Literal
from openai import Model

@st.cache_resource()
def initialize_retriever(
    system_template: str,
    user_template: str,
    llm_type: str = "gpt-3.5-turbo",
    emb_type: str = "text-embedding-ada-002",
    embedding_store: Literal["faiss"] = "faiss",
):
    
    retriever = ChatOpenAIRetrieval(
        system_template=system_template,
        user_template=user_template,
        emb_path=config.EMBSTORE_DICT[embedding_store],
        openai_api_key=st.session_state.oai_api_key,
        llm_type=llm_type,
        embedding_type=emb_type,
        embedding_store=embedding_store
    )
    
    return retriever

def get_text():
    st.text_input("Enter patient case scenario below: ", key="query", on_change=clear_text)
    return st.session_state["temp"]

def clear_text():
    st.session_state["temp"] = st.session_state["query"]
    st.session_state["query"] = ""
    
def handler_verify_key():
    """Handle OpenAI key verification"""
    oai_api_key = st.session_state.open_ai_key_input
    try: 
        model_list = [model_info["id"] for model_info in Model.list(api_key=oai_api_key)["data"]]
        st.session_state.model_list = model_list
        st.session_state.oai_api_key = oai_api_key

    except Exception as e: 
        with openai_key_container: 
            st.error(f"{e}")
        logging.error(f"{e}")
    
embedding_models = ["text-embedding-ada-002"]

st.set_page_config("Physician Medical Assistant", layout="wide")
st.title("AI physician assistant for colonoscopy interval recommendation")

openai_key_container = st.container()

if "oai_api_key" not in st.session_state:
    st.write(app_ui.need_api_key_msg)
    col1, col2 = st.columns([6,4])
    col1.text_input(
        label="Enter OpenAI API Key",
        key="open_ai_key_input",
        type="password",
        autocomplete="current-password",
        on_change=handler_verify_key,
        placeholder=app_ui.helper_api_key_placeholder,
        help=app_ui.helper_api_key_prompt)
    with openai_key_container:
        st.empty()
        st.write("---")
else:
    if "gpt-4" in st.session_state.model_list:
        llm_models = ["gpt-4", "gpt-3.5-turbo"]
    else:
        llm_models = ["gpt-3.5-turbo"]
        
    with st.sidebar:
        st.header("OpenAI Settings")
        llm_type = st.radio("LLM", llm_models)
        emb_type = st.radio("Embedding Model", embedding_models)
        
    prompt = prompts.colonoscopy1

    retriever = initialize_retriever(
        system_template=prompt["system_template"],
        user_template=prompt["user_template"],
        llm_type=llm_type,
        emb_type=emb_type
    )

    # if 'generated' not in st.session_state:
    #     st.session_state['generated'] = []
    # if 'past' not in st.session_state:
    #     st.session_state['past'] = []

    message(app_ui.welcome_msg)
    
    convo = st.empty()
    query = st.empty()
    spinner = st.empty()

    ### with conversation history
    # with convo.container():
    #     with query:
    #         user_query = get_text()
    #     if user_query:
    #         response = retriever(user_query)
    #         st.session_state["past"].append(user_query)
    #         st.session_state["generated"].append(response)
    #     if st.session_state["generated"]:
    #         for i in range(len(st.session_state["generated"])):
    #             message(st.session_state["past"][i], is_user=True)
    #             message(st.session_state["generated"][i])
    
    if "temp" not in st.session_state:
        st.session_state["temp"] = ""
    
    with query.container():
        user_query = get_text()
    
    with convo.container():
        if user_query:
            message(user_query, is_user=True)
            with spinner.container():
                with st.spinner(text = "Generating guidelines for this patient. Please wait."):
                    response = retriever(user_query)
            message(response)
            