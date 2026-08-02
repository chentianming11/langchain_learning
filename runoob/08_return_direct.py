from dotenv import load_dotenv
load_dotenv()

from langchain.tools import tool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage


# 普通工具：结果返回给模型，模型再做总结
@tool
def search_normal(keyword: str) -> str:
    """搜索菜鸟教程 RUNOOB 的课程（普通模式）"""
    return f"搜索结果：Python3 基础教程、Python 数据分析、Python 爬虫入门"


# return_direct 工具：结果直接作为最终输出
@tool(return_direct=True)
def search_direct(keyword: str) -> str:
    """搜索菜鸟教程 RUNOOB 的课程（直接返回模式）。

    当用户只需要搜索结果，不需要额外分析时使用此工具。
    """
    return f"搜索结果：Python3 基础教程、Python 数据分析、Python 爬虫入门"


model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)

# 对比：普通模式 vs 直接返回模式
agent_normal = create_agent(
    model=model,
    tools=[search_normal],
    system_prompt="你是菜鸟教程的学习顾问。",
)

agent_direct = create_agent(
    model=model,
    tools=[search_direct],
    system_prompt="你是菜鸟教程的学习顾问。",
)

# 普通模式：模型会基于搜索结果再生成一段总结
result = agent_normal.invoke({
    "messages": [HumanMessage(content="搜索 Python 课程")]
})
print("=== 普通模式（模型会再加工）===")
print(result["messages"][-1].content[:150])

# 直接返回模式：工具结果就是最终答案
result = agent_direct.invoke({
    "messages": [HumanMessage(content="搜索 Python 课程")]
})
print("\n=== 直接返回模式（工具结果即最终答案）===")
print(result["messages"][-1].content[:150])