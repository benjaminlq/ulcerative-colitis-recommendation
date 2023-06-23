from pydantic import BaseModel, Field


class DrugOutput(BaseModel):
    drug_name: str = Field(description="Name of the drug")
    advantages: str = Field(description="Advantages of the drug ")
    disadvantages: str = Field(description="Disadvantages of the drug")
