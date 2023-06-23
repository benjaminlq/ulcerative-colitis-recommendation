"""Streamlit App File
"""
import logging
from typing import Literal, Optional

import streamlit as st
from langchain.callbacks import get_openai_callback
from langchain.prompts.chat import ChatPromptTemplate
from openai import Model
from streamlit_chat import message

import app_ui
from config import EMBSTORE_DICT
from inference import ChatOpenAIRetrieval
from prompts import polyp


@st.cache_resource()
def initialize_retriever(
    _prompt_template: ChatPromptTemplate,
    llm_type: str = "gpt-3.5-turbo",
    emb_type: str = "text-embedding-ada-002",
    embedding_store: Literal["faiss"] = "faiss",
    verbose: Optional[bool] = None,
):
    """Initialize QA Retriever Chain for inference

    Args:
        system_template (str): System Template sets the context and expected behaviour of the LLM.
        user_template (str): Specific User Input which requires user to enter a query for RetrievalQA
            chain to extract relevant information related to sources.
        llm_type (str, optional): Type of LLM model to be used for extraction. Defaults to "gpt-3.5-turbo".
        emb_type (str, optional): Type of Embedding model to be used. Convert the query into embedding
            and find the most relevant text chunk from document store based
        on similarity (e.g. cosine distance). Defaults to "text-embedding-ada-002".
        embedding_store (Literal[faiss], optional): Type of document store to use. Defaults to "faiss".
        verbose (Optional[bool]): Whether chain query returns prompt and COT from LLM. Default to None.
    Returns:
        ChatOpenAIRetrieval: Retrieval Instance for extracting answer from source documents to queries.
    """

    retriever = ChatOpenAIRetrieval(
        prompt_template=_prompt_template,
        emb_path=EMBSTORE_DICT["polyp"][embedding_store],
        openai_api_key=st.session_state.oai_api_key,
        llm_type=llm_type,
        embedding_type=emb_type,
        embedding_store=embedding_store,
        verbose=verbose,
    )

    return retriever


def get_text():
    """Prompt users to input query.

    Returns:
        str: User's input query
    """
    st.text_input(
        "Enter patient case scenario below: ", key="query", on_change=clear_text
    )
    return st.session_state["temp"]


def clear_text():
    """This function helps to clear the previous text input from the input field.
    Temporary assign the input value to "temp" and clear "query" from session_state."""
    st.session_state["temp"] = st.session_state["query"]
    st.session_state["query"] = ""


def update_cost(callback):
    st.session_state["token_counter"]["total"] += callback.total_tokens
    st.session_state["token_counter"]["prompt"] += callback.prompt_tokens
    st.session_state["token_counter"]["completion"] += callback.completion_tokens
    st.session_state["total_cost"] += callback.total_cost


def print_cost(cost_container):
    with cost_container:
        st.write(
            "Total Cost: {:.2f} USD\n\nTotal Tokens: {}\n\nCompletion Tokens: {}\n\nPrompt Tokens: {}".format(
                st.session_state["total_cost"],
                st.session_state["token_counter"]["total"],
                st.session_state["token_counter"]["completion"],
                st.session_state["token_counter"]["prompt"],
            )
        )


def handler_verify_key():
    """Function to verify whether input OpenAI API Key is working."""
    oai_api_key = st.session_state.open_ai_key_input
    try:
        model_list = [
            model_info["id"] for model_info in Model.list(api_key=oai_api_key)["data"]
        ]
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
if "token_counter" not in st.session_state:
    st.session_state["token_counter"] = {
        "total": 0,
        "prompt": 0,
        "completion": 0,
    }

if "total_cost" not in st.session_state:
    st.session_state["total_cost"] = 0.0

if "oai_api_key" not in st.session_state:
    st.write(app_ui.need_api_key_msg)
    col1, col2 = st.columns([6, 4])
    col1.text_input(
        label="Enter OpenAI API Key",
        key="open_ai_key_input",
        type="password",
        autocomplete="current-password",
        on_change=handler_verify_key,
        placeholder=app_ui.helper_api_key_placeholder,
        help=app_ui.helper_api_key_prompt,
    )
    with openai_key_container:
        st.empty()
        st.write("---")
else:
    if "gpt-4" in st.session_state.model_list:
        llm_models = ["gpt-4", "gpt-3.5-turbo"]
    else:
        llm_models = ["gpt-3.5-turbo"]

    with st.sidebar:
        st.title("OpenAI Settings")
        llm_type = st.radio("LLM", llm_models)
        emb_type = st.radio("Embedding Model", embedding_models)
        # st.title("API Cost")
        # cost_container = st.empty()
        # print_cost(cost_container)

    prompt = polyp.CHAT_PROMPT_TEMPLATE

    retriever = initialize_retriever(
        _prompt_template=prompt,
        llm_type=llm_type,
        emb_type=emb_type,
    )

    # if 'generated' not in st.session_state:
    #     st.session_state['generated'] = []
    # if 'past' not in st.session_state:
    #     st.session_state['past'] = []

    message(app_ui.welcome_msg)

    convo = st.empty()
    query = st.empty()
    spinner = st.empty()

    # with conversation history
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
                with st.spinner(
                    text="Generating guidelines for this patient. Please wait."
                ):
                    with get_openai_callback() as cb:
                        response = retriever(user_query)
                        update_cost(cb)
                        # cost_container.empty()
                        # print_cost(cost_container)
            message(response)

# streamlit run src/app.py
