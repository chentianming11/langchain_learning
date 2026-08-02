from typing import Annotated, Any
from dotenv import load_dotenv
load_dotenv()

from langchain.tools import tool, InjectedState
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage


@tool
def remember_preference(
    preference: str,
    state: Annotated[dict[str, Any], InjectedState],
) -> str:
    """记住用户的偏好设置。

    Args:
        preference: 用户的偏好内容
        state: 系统自动注入的当前 Agent 状态
    """
    # 从状态中获取之前的消息历史
    messages = state.get("messages", [])
    message_count = len(messages)

    # 可以读取状态中的任何字段
    previous_prefs = state.get("user_preferences", "无")

    return (
        f"已记住偏好: {preference}。"
        f"(当前对话共 {message_count} 条消息，"
        f"之前偏好: {previous_prefs})"
    )


# 注意：InjectedState 只在工具运行于 Agent 内部时由 ToolNode 自动注入，
# 不能脱离 Agent 直接 invoke（直接调用时没有 Agent 状态可注入，会因 state 缺失报错）。
# 下面通过 create_agent 运行，让 Agent 在调用工具时自动注入状态。
model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)
agent = create_agent(
    model=model,
    tools=[remember_preference],
    system_prompt="你是一个助手，用户提到偏好时调用 remember_preference 工具记住它。",
)

# 让 Agent 自行决定调用工具，state（含 messages）会被自动注入
result = agent.invoke({
    "messages": [HumanMessage(content="请记住我偏好暗色主题")]
})
print("=== Agent 自动调用工具时注入的状态结果 ===")
print(result["messages"][-1].content)