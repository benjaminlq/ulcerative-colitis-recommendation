import os

from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate, AIMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.output_parsers import PydanticOutputParser
from parsers import DrugOutput

from config import PROMPT_DIR

# Prompt Here

### CHAT PROMTP TEMPLATE
system_prompt = """
Make reference to the context given to assess the scenario. If you do not know the answer. just say that "I don't know", don't try to make up an answer.
You are a physician assistant giving advice on treatment for moderate to severe ulcerative colitis (UC).

ANALYSE the given patient profile based on given query based on one of the following criteria:
- Whether treated patient is new patient or patient under maintenance
- Prior response to Infliximab
- Prior failure to Anti-TNF agents
- Prior failure to Vedolizumab
- Age
- Pregnancy
- Extraintestinale manifestations
- Pouchitis

FINALLY RETURN up to 2 TOP choices of biological drugs given patient profile. Explain the PROS and CONS of the 2 choices.
Output your answer as a list of JSON objects with keys: drug_name, advantages, disadvantages.

{summaries}

"""

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(system_prompt, input_variables = ["summaries"]),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)

print(PROMPT_TEMPLATE.format(summaries = "Summaries", question = "User Query"))

# Output
if __name__ == "__main__":
    output_path = os.path.join(PROMPT_DIR, "uc", "uc_chat.yaml")
    PROMPT_TEMPLATE.save(output_path)
    print("Conversion completed!!!")