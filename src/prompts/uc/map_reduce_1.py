from langchain.prompts import PromptTemplate

pregnancy_prompt = """Make reference to the context given to assess the scenario. If you do not know the answer. just say that "I don't know", don't try to make up an answer.
=========
You are a physician assistant giving advice on treatment for moderate to severe ulcerative colitis (UC). Perform the following steps
1. Identify if patient is pregnant.
2. Search from reference text best biological drugs based on whether patient is pregnant.
3. Return up to 2 TOP choices of biological drugs. Explain the PROS and CONS of the 2 choices.
=========
REFERENCE TEXT:
{summaries}
=========
QUESTION: {question}
=========
ANSWER:
"""

PREGNANCY_PROMPT_TEMPLATE = PromptTemplate(
    template=pregnancy_prompt,
    input_variables=["summaries", "question"],
)

pregnancy_prompt = """Make reference to the context given to assess the scenario. If you do not know the answer. just say that "I don't know", don't try to make up an answer.
=========
You are a physician assistant giving advice on treatment for moderate to severe ulcerative colitis (UC). Perform the following steps
1. Identify if patient is pregnant.
2. Search from reference text best biological drugs based on whether patient is pregnant.
3. Return up to 2 TOP choices of biological drugs. Explain the PROS and CONS of the 2 choices.
=========
REFERENCE TEXT:
{summaries}
=========
QUESTION: {question}
=========
ANSWER:
"""

PREGNANCY_PROMPT_TEMPLATE = PromptTemplate(
    template=pregnancy_prompt,
    input_variables=["summaries", "question"],
)
