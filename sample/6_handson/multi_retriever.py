# 必要なライブラリのインポート
from langchain_community.retrievers import TavilySearchAPIRetriever
from enum import Enum
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import GitLoader
import os

# 環境変数の読み込み
from dotenv import load_dotenv
load_dotenv()

# LangChainのトレーシング設定
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# GitLoaderのファイルフィルター
def file_filter(file_path: str) -> bool:
    return file_path.endswith(".mdx")

# LangChainのドキュメントをGitHubから読み込み
loader = GitLoader(
    repo_path="./langchain",
    branch="master",
    file_filter=file_filter,
)
docs = loader.load()

# ベクトルストアの設定
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma.from_documents(docs, embeddings)

# RAGのプロンプトテンプレート
prompt = ChatPromptTemplate.from_template('''\
 以下の文脈だけを踏まえて質問に回答してください。

 文脈: """
 {context}
 """

 質問: {question}
''')

# LLMの設定
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Retrieverの設定
# LangChainドキュメント用のRetriever
retriever = TavilySearchAPIRetriever()
langchain_document_retriever = retriever.with_config(
    {"run_name": "langchain-document-retriever"}
)

# Web検索用のRetriever
web_retriever = TavilySearchAPIRetriever(k=3).with_config(
    {"run_name": "web-retriever"}
)

# Retrieverの種類を定義
class Routes(str, Enum):
    LANGCHAIN_DOCUMENT = "langchain-document"
    WEB = "web"

# Retriever選択の出力モデル
class RouteOutput(BaseModel):
    route: Routes

# Retriever選択のプロンプト
route_output = ChatPromptTemplate.from_template('''
質問に回答するために適切なRetrieverを選択してください。

質問: {question}
''')

# Retriever選択のチェーン
route_chain = (
    route_output
    | model.with_structured_output(RouteOutput)
    | (lambda x: x.route)
)

# 選択されたRetrieverを実行する関数
def routed_retriever(inp: dict[str, any]) -> list[Document]:
    question = inp["question"]
    route = inp["route"]

    if route == Routes.LANGCHAIN_DOCUMENT:
        return langchain_document_retriever.invoke(question)
    elif route == Routes.WEB:
        return web_retriever.invoke(question)
    else:
        raise ValueError(f"Invalid route: {route}")

# マルチRAGチェーンの構築    
route_rag_chain = (
    {
        "question": RunnablePassthrough(),
        "route": route_chain,
    }
    | RunnablePassthrough.assign(context=routed_retriever)
    | prompt
    | model
    | StrOutputParser()
)

# LangChainに関する質問のテスト
result = route_rag_chain.invoke("LangChainとは？")
print(result)

# 一般的な質問のテスト
result2 = route_rag_chain.invoke("東京の今日の気温は？")
print(result2)
