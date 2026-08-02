from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage

model = init_chat_model("deepseek:deepseek-v4-flash")
agent = create_agent(
    model=model,
    system_prompt="你是菜鸟教程 RUNOOB 的助手。",
)

# stream_mode="messages" 逐个 Token 返回
print("实时流式输出：")
for msg_chunk, metadata in agent.stream(
    {"messages": [HumanMessage(content="用一句话介绍菜鸟教程 RUNOOB")]},
    stream_mode="messages",
):
    # msg_chunk 是 AIMessageChunk
    # 每个 chunk 只包含一小段内容
    if msg_chunk.content:
        print(msg_chunk.content, end="", flush=True)
print()  # 最后换行