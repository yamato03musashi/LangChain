from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{input}"),
])

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

chain = prompt | llm

print(chain.invoke({"input": "What is the capital of France?"}))