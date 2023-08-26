from langchain.chains import LLMChain
from langchain.tools.base import BaseTool
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain.base_language import BaseLanguageModel
from typing import Optional, Dict
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from pydantic import root_validator

class GeneralKnowledgeTool(BaseTool):
    name: str = "General Knowledge"
    description: str = "Useful for general knowledge question"
    llm: BaseLanguageModel
    
    @root_validator()
    def initiate_llm_chain(cls, values: Dict) -> Dict:
        values["llm_chain"] = LLMChain(
            prompt=ChatPromptTemplate.from_messages([HumanMessagePromptTemplate.from_template("Question: {question}")]),
            llm=values["llm"]
        )
        return values
    
    def _run(
        self, 
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
        ) -> str:
        return self.llm_chain.run(query)
    
    def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
        ) -> str:
        return NotImplementedError