from langchain.base_language import BaseLanguageModel
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.vectorstores.base import VectorStore
from langchain.tools import BaseTool
from typing import List
from custom_tools import HumanTool, GoogleSerperTool, QASearchTool, GeneralKnowledgeTool

class RetrievalToolkit(BaseToolkit):
    llm: BaseLanguageModel
    docsearch: VectorStore
    
    class Config:
        arbitrary_types_allowed = True
    
    def get_tools(self) -> List[BaseTool]:
        tools = [
            QASearchTool(llm = self.llm, docsearch = self.docsearch),
            GeneralKnowledgeTool(llm=self.llm)
        ]
        return tools

class HumanRetrievalToolkit(BaseToolkit):
    llm: BaseLanguageModel
    docsearch: VectorStore

    class Config:
        arbitrary_types_allowed = True
    
    def get_tools(self) -> List[BaseTool]:
        tools = [
            HumanTool(),
            QASearchTool(llm = self.llm, docsearch = self.docsearch),
            GeneralKnowledgeTool(llm=self.llm)
        ]
        return tools

class HumanSearchRetrievalToolkit(BaseToolkit):
    serper_api_key: str
    llm: BaseLanguageModel
    docsearch: VectorStore
    
    class Config:
        arbitrary_types_allowed = True
    
    def get_tools(self) -> List[BaseTool]:
        tools = [
            HumanTool(),
            GoogleSerperTool(serper_api_key=self.serper_api_key),
            QASearchTool(llm = self.llm, docsearch = self.docsearch),
            GeneralKnowledgeTool(llm=self.llm)
        ]
        return tools
    
if __name__ == "__main__":
    import os, json
    from langchain.vectorstores import FAISS
    from langchain.embeddings import OpenAIEmbeddings
    from config import MAIN_DIR
    
    with open(os.path.join(MAIN_DIR, "auth", "api_keys.json"), "r") as f:
        api_keys = json.load(f)
    
    os.environ["OPENAI_API_KEY"] = api_keys["OPENAI_API_KEY"]
    docsearch = FAISS.load_local(
        os.path.join(MAIN_DIR, "data", "emb_store", "uc", "faiss",
                     "text-embedding-ada-002", "v8-add-tables_2500_500"),
        embeddings=OpenAIEmbeddings()
        )
    
    toolkit = HumanSearchRetrievalToolkit(
        serper_api_key=api_keys["SERP_API"],
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
        docsearch=docsearch
    )
    
    tools = toolkit.get_tools()
    for idx, tool in enumerate(tools):
        print(f"Tool number {idx+1}: {tool.name}")