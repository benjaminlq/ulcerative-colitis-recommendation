"""Prompts used for LLM
"""

from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

pregnancy_prompt = """You are a physician assistant giving advice on treatment for moderate to severe ulcerative colitis (UC). Make reference to the REFERENCE TEXT given to assess the scenario.

Based on the patient profile or medical history, make recommendations of biological drugs for UC treatment.If there are no suitable drugs, just say "I don't know", don't try to make up an answer.
For each recommended drug, explain the PROS and CONS in context of the patient profile.
Also return drugs which this patient should avoid given his profile or medical history and reasons why the drug should be avoided.

=========
REFERENCE TEXT:
{summaries}
"""

human_prompt = """
=========
PATIENT_PROFILE: {question}
=========
"""

FEATURE_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            pregnancy_prompt, input_variables=["summaries"]
        ),
        HumanMessagePromptTemplate.from_template(human_prompt),
    ]
)

combine_prompt = """You are a physician assistant giving advice on treatment for moderate to severe ulcerative colitis (UC). If you do not know the answer. just say that "I don't know", don't try to make up an answer.
Make reference to the REFERENCE ANSWERS given to assess the scenario.

Combine the answers in REFERENCE ANSWERS to recommend up to 2 TOP choices of biological drugs given patient profile. Explain the PROS and CONS of the choices recommended.
Prioritize drugs recommendations based on patients medical conditions and history.
Output your answer as a list of JSON objects with keys: drug_name, advantages, disadvantages.
=========
REFERENCE ANSWERS:
{summaries}

=========
COMBINED ANSWER:
"""

COMBINE_PROMPT_TEMPLATE = PromptTemplate.from_template(
    template=combine_prompt, input_variables=["summaries"]
)
