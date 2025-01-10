from langchain_community.document_loaders import GitLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

def file_filter(file_path: str) -> bool:
    return file_path.endswith(".mdx")

# Gitからデータを取得
loader = GitLoader(
    clone_url="https://github.com/langchain-ai/langchain",
    repo_path="./langchain",
    branch="master",
    file_filter=file_filter,
)

docs = loader.load()

print(len(docs))

# データをベクトル化
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma.from_documents(docs, embeddings)

prompt = ChatPromptTemplate.from_template('''\
 以下の文脈だけを踏まえて質問に回答してください。

 文脈: """
 {context}
 """

 質問: {question}
''')

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

retriever = db.as_retriever()

chain = {
    "question": RunnablePassthrough(),
    "context": retriever
} | prompt | model | StrOutputParser()

response = chain.invoke("LangChainとは？")

print(response)
