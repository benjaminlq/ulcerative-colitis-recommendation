"""Prompts used for LLM
"""

from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

pregnancy_prompt = """Make reference to the context given to assess the scenario. If you do not know the answer. just say that "I don't know", don't try to make up an answer.
=========
You are a physician assistant giving advice on treatment for moderate to severe ulcerative colitis (UC). Perform the following steps
1. Identify if patient is pregnant.
2. Search from REFERENCE TEXT the best biological drugs based on whether patient is pregnant.
3. Return up to 2 TOP choices of biological drugs with the PROS and CONS of the 2 choices.

=========
REFERENCE TEXT:
{summaries}
"""

human_prompt = """
=========
QUESTION: {question}
=========
"""

PREGNANCY_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            pregnancy_prompt, input_variables=["summaries"]
        ),
        HumanMessagePromptTemplate.from_template(human_prompt),
    ]
)
