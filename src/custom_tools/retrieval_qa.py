from pydantic import root_validator
from langchain.tools import BaseTool
from langchain.base_language import BaseLanguageModel
from langchain.vectorstores.base import VectorStore
from langchain.chains import RetrievalQAWithSourcesChain
from typing import Literal, Optional
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun

class QASearchTool(BaseTool):
    name: str = "Docsearch QA Tool"
    description: str = "Use this tool to search for document and answer questions related to treatment of ulcerative colitis"
    llm: BaseLanguageModel
    docsearch: VectorStore
    chain_type: Literal["stuff", "map_reduce"] = "stuff"
    k: int = 4
    max_tokens_limit: int = 4500
    verbose: bool = True

    @root_validator()
    def generate_qa_chain(cls, values):
        values["chain"] = RetrievalQAWithSourcesChain.from_chain_type(
            llm=values["llm"],
            chain_type=values["chain_type"],
            retriever=values["docsearch"].as_retriever(search_kwargs={"k":values["k"]}),
            return_source_documents=True,
            reduce_k_below_max_tokens=True,
            max_tokens_limit=values["max_tokens_limit"],
            verbose=values["verbose"]
            )
        return values

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
        ) -> str:
        return self.chain(query)

    def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
        ) -> str:
        return NotImplementedError