from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()

@tool
def search_course(keyword: str) -> str:
    """在菜鸟教程 RUNOOB 搜索课程"""
    courses = {
        "python": "Python3 基础教程（30章，20小时）",
        "html": "HTML 基础教程（25章，15小时）",
    }
    return courses.get(keyword.lower(), "未找到相关课程")


agent = create_agent(
    model=init_chat_model("deepseek:deepseek-v4-flash", temperature=0),
    tools=[search_course],
    system_prompt="你是菜鸟教程 RUNOOB 的课程顾问。",
)

# 使用 updates 模式查看每一步
print("=== Agent 执行过程 ===\n")
for chunk in agent.stream(
    {"messages": [HumanMessage(content="帮我查一下 Python 课程")]},
    stream_mode="updates",
):
    for node_name, update in chunk.items():
        print(f"[{node_name}]", end=" ")
        if "messages" in update:
            for msg in update["messages"]:
                if msg.type == "ai":
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        calls = [tc['name'] for tc in msg.tool_calls]
                        print(f"请求调用: {calls}")
                    elif msg.content:
                        print(f"回复: {msg.content[:80]}")
                elif msg.type == "tool":
                    print(f"工具返回 [{msg.name}]: {msg.content}")