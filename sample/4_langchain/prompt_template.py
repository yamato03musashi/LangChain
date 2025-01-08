from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    "What is the capital of {country}?"
)

print(prompt.invoke({"country": "France"}))
