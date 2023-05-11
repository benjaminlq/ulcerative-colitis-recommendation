from langchain.prompts.base import BasePromptTemplate
from typing import Callable, Literal
import config
import re
import random
import openai
import time
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

class Retrieval_Interface:
    def __init__(
        self,
        llm_type: str,
        emb_path: str,
        openai_api_key: str,
        embedding_type: str = "text-embedding-ada-002",
        embedding_store: Literal["faiss"] = "faiss",
    ):
        self.prompt_template = None
        self.llm_type = llm_type
        self.chain = None
        if embedding_store.lower() == "faiss":
            self.embedder = OpenAIEmbeddings(model = embedding_type, openai_api_key=openai_api_key)
            self.docsearch = FAISS.load_local(emb_path, self.embedder)
            print("FAISS Datastore successfully loaded")
        else:
            raise Exception("Please specify correct embedding store type.")
        self.token_counter = 0
        self.dollar_counter = 0.0
    
    def __call__(
        self,
        query: str
    ):
        result = self._response(query)
        response = Retrieval_Interface._format_result(query, result)
        return response        

    def _retry_with_exponential_backoff(
        func,
        initial_delay: float = 1,
        exponential_base: float = 2,
        jitter: bool = True,
        max_retries: int = 5,
        errors: tuple = (openai.error.RateLimitError,),
    ):
        """Retry a function with exponential backoff."""
        def wrapper(*args, **kwargs):
            # Initialize variables
            num_retries = 0
            delay = initial_delay

            # Loop until a successful response or max_retries is hit or an exception is raised
            while True:
                try:
                    return func(*args, **kwargs)

                # Retry on specified errors
                except errors as e:
                    # Increment retries
                    num_retries += 1

                    # Check if max retries has been reached
                    if num_retries > max_retries:
                        raise Exception(
                            f"Maximum number of retries ({max_retries}) exceeded."
                        )

                    # Increment the delay
                    delay *= exponential_base * (1 + jitter * random.random())

                    # Sleep for the delay
                    time.sleep(delay)

                # Raise exceptions for any errors not specified
                except Exception as e:
                    raise e

        return wrapper
    
    @_retry_with_exponential_backoff
    def _response(
        self, query  
    ):
        return self.chain(query)
    
    @classmethod
    def _get_chain(
        cls, llm: Callable, docsearch: Callable, prompt_template: BasePromptTemplate
    ):
        return RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            retriever=docsearch.as_retriever(),
            chain_type="stuff",
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template},
            reduce_k_below_max_tokens=True
            )
    
    @classmethod
    def _format_result(
        cls, query: str, result: str
    ):
        sources = [str(re.findall( r'[ \w-]+?(?=\.)', name)[0]) for name in (list(set([doc.metadata['source'] for doc in result['source_documents']])))]
        response = f"""### Scenario: 
        {query}
        \n### Response: 
        {result['answer']}
        \n### Relevant sources:
        {', '.join(sources)}
        """
        return response
        
class ChatOpenAIRetrieval(Retrieval_Interface):
    def __init__(
        self,
        system_template: str,
        user_template: str,
        emb_path: Callable,
        openai_api_key: str,
        llm_type: str = "gpt-3.5-turbo",
        embedding_type: str = "text-embedding-ada-002",
        embedding_store: Literal["faiss"] = "faiss",
        temperature: float = config.TEMPERATURE,
        top_p: float = config.TOP_P,
        max_tokens: int = config.MAX_TOKENS
    ):
        super(ChatOpenAIRetrieval, self).__init__(llm_type, emb_path, openai_api_key, embedding_type, embedding_store)
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.llm_type = llm_type
        self.llm = ChatOpenAI(
            model_name = self.llm_type,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            openai_api_key=openai_api_key)
        messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(user_template)
        ]
        self.prompt_template = ChatPromptTemplate.from_messages(messages)
        self.chain = ChatOpenAIRetrieval._get_chain(
            llm = self.llm,
            docsearch = self.docsearch,
            prompt_template = self.prompt_template
        )
        
    def _check_model(self):
        