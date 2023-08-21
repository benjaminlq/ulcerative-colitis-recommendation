import os
from abc import abstractmethod
from typing import List, Optional

from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.tools import BaseTool

from config import LOGGER, MAIN_DIR
from custom_parsers import DrugOutput, DrugParser

from .base import BaseExperiment


class BaseQuestionAnsweringAgent(BaseExperiment):
    def __init__(
        self,
        keys_json: str = os.path.join(MAIN_DIR, "auth", "api_keys.json"),
        temperature: float = 0,
        gt: Optional[str] = None,
        verbose: bool = True,
    ):
        super(BaseQuestionAnsweringAgent, self).__init__(
            keys_json=keys_json, temperature=temperature, gt=gt, verbose=verbose
        )

        self.questions = []
        self.answers = []
        self.intermediate_steps = []
        self.prompt_map = {}

        self.drug_parser = DrugParser.from_llm(
            parser=PydanticOutputParser(pydantic_object=DrugOutput),
            llm=ChatOpenAI(
                model_name="gpt-4", temperature=0, openai_api_key=self.openai_key
            ),
        )

    @abstractmethod
    def _create_agent_executor(self, agent, tools: List[BaseTool], **kwargs):
        return NotImplementedError
