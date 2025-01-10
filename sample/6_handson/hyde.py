# RAGの精度を上げる手法Hyde
# 質問に類似したドキュメントを取得するのではなく。回答に類似するドキュメントを取得する

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import GitLoader
from langchain_openai import OpenAIEmbeddings

# load env
from dotenv import load_dotenv
load_dotenv()

import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")


def file_filter(file_path: str) -> bool:
    return file_path.endswith(".mdx")

loader = GitLoader(
    repo_path="./langchain",
    branch="master",
    file_filter=file_filter,
)

docs = loader.load()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma.from_documents(docs, embeddings)

hypothtical_prompt = ChatPromptTemplate.from_template('''\
 以下の質問に対して回答してください。

 質問: {question}
''')

retriever = db.as_retriever()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 仮説を生成
hypothetical_chain = hypothtical_prompt | model | StrOutputParser()

hyde_rag_chain = {
  "question": RunnablePassthrough(),
  "context": hypothetical_chain | retriever # 生成した仮説をもとにドキュメントを取得
} | hypothtical_prompt | model | StrOutputParser()

response = hyde_rag_chain.invoke("LangChainとは？")

print(response)
