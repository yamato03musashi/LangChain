from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

class Person(BaseModel):
    name: str = Field(description="The name of the person")
    age: int = Field(description="The age of the person")

# パーサーを作成
output_parser = PydanticOutputParser(pydantic_object=Person)

# プロンプトを作成
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant \n\n{format_instructions}"),
    ("user", "{input}"),
])

# フォーマットを追加（指定したformat_instructionsを使用）
prompt_with_format_instructions = prompt.partial(
  format_instructions=output_parser.get_format_instructions()
)

model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0).bind(
    response_format={
        "type": "json_object"
    }
)

chain = prompt_with_format_instructions | model | output_parser

# サポートしているモデルではwith_structured_outputを使用できる
# chain = prompt | model.with_structured_output(Person)

# チェーンを実行
result = chain.invoke({"input": "John is 30 years old."})

# 結果を表示
print(type(result))
print(result)
