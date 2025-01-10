from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

output_parser = StrOutputParser()

prompt = ChatPromptTemplate.from_messages([
    ("system", "you are a helpful assistant"),
    ("human", "{input}"),
])

def upper_case(text: str) -> str:
    return text.upper()

chain = prompt | model | output_parser | RunnableLambda(upper_case)

print(chain.invoke({"input": "hello"}))


# @chain
# def upper_case(text: str) -> str:
#     return text.upper()

# chain = prompt | model | output_parser | upper_case

# print(chain.invoke({"input": "hello"}))