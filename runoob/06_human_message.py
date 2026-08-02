from langchain.messages import HumanMessage
from langchain.chat_models import init_chat_model

from dotenv import load_dotenv
load_dotenv()

# 创建一条用户消息
msg = HumanMessage(content="菜鸟教程 RUNOOB 是什么？")

print(f"类型: {msg.type}")        # human
print(f"内容: {msg.content}")      # 菜鸟教程 RUNOOB 是什么？

# 创建消息列表（代表多轮对话历史）
messages = [
    HumanMessage(content="你好"),
    HumanMessage(content="菜鸟教程有哪些课程？"),
    HumanMessage(content="Python 课程适合零基础吗？"),
]

model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)
response = model.invoke(messages)
print(f"\n模型回复: {response.content}")