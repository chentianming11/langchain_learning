from langchain.agents import create_agent
from langchain.agents.middleware import before_model, after_model
from langchain.chat_models import init_chat_model
from langchain.tools import tool

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

@tool
def search_course(keyword: str) -> str:
    """在菜鸟教程 RUNOOB 搜索课程"""
    courses = {
        "python": "Python3 基础教程（30章，20小时）",
        "html": "HTML 基础教程（25章，15小时）",
    }
    return courses.get(keyword.lower(), "未找到相关课程")

@before_model
def notify_before(state, runtime):
    """在模型调用前发送自定义事件"""
    runtime.stream_writer({
        "type": "status",
        "message": "正在思考...",
    })
    return None


@after_model
def notify_after(state, runtime):
    """在模型调用后发送自定义事件"""
    last_msg = state["messages"][-1] if state.get("messages") else None
    has_tools = hasattr(last_msg, 'tool_calls') and last_msg.tool_calls

    if has_tools:
        tool_names = [tc['name'] for tc in last_msg.tool_calls]
        runtime.stream_writer({
            "type": "status",
            "message": f"正在调用工具: {', '.join(tool_names)}...",
        })
    else:
        runtime.stream_writer({
            "type": "status",
            "message": "回答已完成",
        })
    return None


agent = create_agent(
    model=init_chat_model("deepseek:deepseek-v4-flash", temperature=0),
    tools=[search_course],
    middleware=[notify_before, notify_after],
    system_prompt="你是菜鸟教程 RUNOOB 的课程顾问。",
)

# 使用 stream_mode=["updates", "custom"] 同时接收两种事件
print("=== 混合流式输出 ===\n")
for mode, chunk in agent.stream(
    {"messages": [HumanMessage(content="查一下 Python 课程")]},
    stream_mode=["updates", "custom"],
):
    if mode == "custom":
        print(f"[自定义事件] 状态: {chunk['message']}")
    elif mode == "updates":
        for node_name, update in chunk.items():
            # 部分节点（如 middleware）的 update 可能为 None，需跳过
            if not update:
                continue
            if "messages" in update:
                for msg in update["messages"]:
                    if msg.type == "ai" and msg.content:
                        print(f"[回复] {msg.content}")