# 会話履歴を管理したい場合に使用

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("chat_history", optional=True),
    ("user", "{input}"),
])

prompt_value = prompt.invoke(
  {
    "chat_history": [
      HumanMessage(content="Hello, I'm John."),
      AIMessage(content="Nice to meet you, John. How can I help you today?")
    ],
    "input": "Do you know who I am?"
  }
)

print(prompt_value)