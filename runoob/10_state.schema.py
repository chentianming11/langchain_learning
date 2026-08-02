from typing import Annotated
from langchain.agents import create_agent, AgentState
from langchain.messages import HumanMessage
from langchain.tools import tool, InjectedState
from typing_extensions import TypedDict

from dotenv import load_dotenv
load_dotenv()
# 扩展 AgentState，添加自定义字段
class LearningAgentState(AgentState):
    """自定义状态，增加学习进度相关字段"""
    user_level: str                       # 用户等级
    completed_topics: list[str]           # 已完成的主题列表


@tool
def track_progress(
    topic: str,
    state: Annotated[dict, InjectedState],
) -> str:
    """记录用户的学习进度。

    Args:
        topic: 刚学完的主题名称
    """
    completed = state.get("completed_topics", [])
    completed.append(topic)
    return (
        f"已记录学习进度。当前已完成 {len(completed)} 个主题："
        f"{', '.join(completed)}"
    )


agent = create_agent(
    model="deepseek:deepseek-v4-flash",
    tools=[track_progress],
    state_schema=LearningAgentState,  # 使用自定义状态
    system_prompt="你是菜鸟教程 RUNOOB 的学习助手。",
)

# 运行时需要提供自定义状态的初始值
result = agent.invoke({
    "messages": [HumanMessage(content="我学完了 Python 基础，帮我记录一下")],
    "user_level": "入门",
    "completed_topics": ["HTML 基础"],
})

print(f"用户等级: {result.get('user_level')}")
print(f"已完成主题: {result.get('completed_topics')}")
print(f"回复: {result['messages'][-1].content[:100]}")