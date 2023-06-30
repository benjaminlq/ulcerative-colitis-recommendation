"""Prompts used for LLM
"""

from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

# CHAT PROMTP TEMPLATE
system_prompt = """
Make reference to the context given to assess the scenario. If you do not know the answer. just say that "I don't know", don't try to make up an answer.
You are a physician assistant giving advice on treatment for moderate to severe ulcerative colitis (UC). Perform the following step

ANALYSE the given patient profile based on given query based on the following criteria:
- Newly Inducted patient (naive) or patient under maintenance
- Prior response to Infliximab
- Prior failure to Anti-TNF agents
- Prior failure to Vedolizumab
- Age
- Pregnancy
- Extraintestinale manifestations
- Pouchitis

RETURN up to 2 TOP choices of biological drugs given patient profile. Explain the PROS and CONS of the 2 choices.

{summaries}

"""

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            system_prompt, input_variables=["summaries"]
        ),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)
