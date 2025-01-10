from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

output_parser = StrOutputParser()

prompt = ChatPromptTemplate.from_messages([
    ("system", "ユーザーの質問にステップバイステップで回答してください。"),
    ("human", "{input}"),
])

steps_chain = prompt | model | output_parser

summarize_prompt = ChatPromptTemplate.from_messages([
    ("system", "ステップバイステップで考えた結論だけを出力してください。"),
    ("human", "{input}"),
])

summarize_chain = summarize_prompt | model | output_parser


chain = steps_chain | summarize_chain

print(chain.invoke({"input": "12 + 3 * 8"}))