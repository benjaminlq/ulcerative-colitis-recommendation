from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import RetrievalQAWithSourcesChain
import os
import streamlit as st
import config
import re
from streamlit_chat import message

st.set_page_config("Physician Medical Chatbot")
# while ("OPENAI_API_KEY" not in os.environ.keys()) or (not os.environ["OPENAI_API_KEY"]):
#     message("PLease enter your OpenAI API Key for me to proceed.")
#     api_key = st.text_input(
#     "Please input your OpenAI API Key here:",
#     type = "password"
#     ) 
#     os.environ["OPENAI_API_KEY"] = api_key
#     emb_model = OpenAIEmbeddings(openai_api_key=api_key)
#     try:
#         emb_model.embed_query("test")
#         message("Successfully registered API Key")
#     except:
#         os.environ["OPENAI_API_KEY"] = None
#         continue
    
@st.cache_resource()
def initialize_model(
    llm_type: str = "gpt-3.5-turbo",
    emb_type: str = "text-embedding-ada-002",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512
):
    
    embedder = OpenAIEmbeddings(model = emb_type)
    llm = ChatOpenAI(model_name = llm_type, temperature=temperature, top_p=top_p, max_tokens=max_tokens)

    if datastore_type == "FAISS":
        docsearch = FAISS.load_local(
            os.path.join(config.EMBSTORE_DIR, "faiss"),
            embedder,
        )
        print("FAISS Datastore successfully loaded")
    
    return llm, embedder, docsearch

def get_text():
    input_text = st.text_input("You: ")
    return input_text
    
datastore_type = "FAISS"
llm_models = ["gpt-4", "gpt-3.5-turbo"]

embedding_models = ["text-embedding-ada-002"]
temperature = 0.0
top_p = 1.0
max_tokens = 800

with st.sidebar:
    while ("OPENAI_API_KEY" not in os.environ.keys()) or (not os.environ["OPENAI_API_KEY"]):
        api_key = st.text_input(
        "Please input your OpenAI API Key here:",
        type = "password"
        ) 
        os.environ["OPENAI_API_KEY"] = api_key
        emb_model = OpenAIEmbeddings(openai_api_key=api_key)
        try:
            emb_model.embed_query("test")
            message("Successfully registered API Key")
        except:
            os.environ["OPENAI_API_KEY"] = None
            continue
        
    st.header("OpenAI Settings")
    llm_type = st.radio("LLM", llm_models)
    emb_type = st.radio("Embedding Model", embedding_models)
    
    # with st.expander("Advanced Settings"):
    #     temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, step=0.1, value=0.0)
    #     top_p = st.slider("Top p", min_value=0.0, max_value=1.0, step=0.01, value=1.00)
    #     max_tokens = st.slider("Maximum Tokens", min_value=64, max_value=960, step = 8, value=512)
    
llm, embedder, docsearch = initialize_model(llm_type, emb_type, temperature, top_p, max_tokens)
    
system_template="""
Make reference to the context given to assess the scenario. If you don't know the answer, just say that "I don't know", don't try to make up an answer.
You are a physician asssitant advising a patient on their next colonoscopy to detect colorectal cancer (CRC). 
Analyse the colonoscopy results if any and list all high risk features. 
Analyse the patient profile and list all risk factors. 
Finally, provide the number of years to the next colonoscopy. If there is more than one reason to do a colonoscopy, pick the shortest time span. 
----------------
{summaries}
"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
]
prompt = ChatPromptTemplate.from_messages(messages)
 
chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
    reduce_k_below_max_tokens=True
)

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

welcome_msg = """
This is an AI chatbot that advises on time to next colonoscopy based on guidelines from the USA. Please enter your patient case scenario below, for example:

A 45 year old Chinese woman with no family history of colon cancer and otherwise well. There is no previous colonoscopy.
"""
message(welcome_msg)
convo = st.empty()
query = st.empty()

        
with convo.container():
    with query:
        user_query = get_text()
    if user_query:
        result = chain(user_query)
        sources = [str(re.findall( r'[ \w-]+?(?=\.)', name)[0]) for name in (list(set([doc.metadata['source'] for doc in result['source_documents']])))]
        response = f"""{result['answer']}
        \nRelevant sources: {', '.join(sources)}
        """
        st.session_state["past"].append(user_query)
        st.session_state["generated"].append(response)
    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"])):
            message(st.session_state["past"][i], is_user=True)
            message(st.session_state["generated"][i])
        