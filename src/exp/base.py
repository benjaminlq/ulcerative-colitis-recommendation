"""BaseExperiment Module
"""
import json
import os.path as osp
from abc import abstractmethod
from typing import List, Optional, Union

import pandas as pd
from langchain.prompts import ChatPromptTemplate, PromptTemplate

from config import MAIN_DIR


class BaseExperiment:
    """Abstract Module for Experiment"""

    def __init__(
        self,
        llm_type: str = "gpt-3.5-turbo",
        keys_json: str = osp.join(MAIN_DIR, "auth", "api_keys.json"),
        temperature: float = 0,
        max_tokens: int = 512,
        gt: Optional[str] = None,
        verbose: bool = False,
    ):
        """Base Experiment Module

        Args:
            llm_type (str, optional): Type of LLM Model. Defaults to "gpt-3.5-turbo".
            keys_json (str, optional): Path to API Keys. Defaults to osp.join(MAIN_DIR, "auth", "api_keys.json").
            temperature (float, optional): Temperature Settings for LLM model. Defaults to 0.
            max_tokens (int, optional): Max_Tokens Settings for LLM model. Defaults to 512.
            gt (Optional[str], optional): Path to Ground Truth file. Defaults to None.
            verbose (bool, optional): Verbose Setting. Defaults to False.
        """
        self.llm_type = llm_type.lower()
        self.temperature = temperature
        self.max_tokens = max_tokens

        with open(keys_json, "r") as f:
            keys = json.load(f)

        self.openai_key = keys["OPENAI_API_KEY"]

        self.ground_truth = self.load_groundtruth(gt) if gt else None
        self.chain = None
        self.verbose = verbose

    @abstractmethod
    def run_test_cases(self, test_cases: Union[List[str], str], **kwargs):
        """Abstract method to run test cases

        Args:
            test_cases (Union[List[str], str]): List of test queries.
        """
        return NotImplementedError

    @staticmethod
    def convert_prompt_to_string(
        prompt: Union[PromptTemplate, ChatPromptTemplate]
    ) -> str:
        """Convert Prompt Object to string format

        Args:
            prompt (Union[PromptTemplate, ChatPromptTemplate]): Prompt Template

        Returns:
            str: Prompt String Template
        """
        return prompt.format(**{v: v for v in prompt.input_variables})

    def load_groundtruth(self, gt_path: str) -> pd.DataFrame:
        """Load Ground Truth information from .csv file

        Args:
            gt_path (str): Path to Ground Truth file

        Returns:
            pd.DataFrame: DataFrame containing Ground Truth data.
        """
        return pd.read_csv(gt_path, encoding="ISO-8859-1")

    @abstractmethod
    def reset(self, **kwargs):
        """Abstract Method for reset questions & answers"""
        return NotImplementedError

    @abstractmethod
    def save_json(self, output_path: str, **kwargs):
        """Save test cases result as json file

        Args:
            output_path (str): Path to output json file
        """
        return NotImplementedError

    @abstractmethod
    def load_json(self, json_path: str, reset: bool = False, **kwargs):
        """Abstract Method for Load Queries and Answers from Json file

        Args:
            json_path (str): Path to json output file to load into instance
            reset (bool, optional): If reset, clear queries and answers from memory before loading. Defaults to False.
        """
        return NotImplementedError

    @abstractmethod
    def write_csv(self, output_csv: str, **kwargs):
        """Abstract method to save outputs as .csv file

        Args:
            output_csv (str): Path to output csv file
        """
        return NotImplementedError
