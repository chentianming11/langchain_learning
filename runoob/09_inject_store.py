from typing import Annotated
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langchain.tools import tool, InjectedStore
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()
# 创建 Store 并预置数据
store = InMemoryStore()
store.put(("runoob", "courses"), "catalog", {
    "data": {
        "Python3 基础教程": {"price": "免费", "duration": "20小时"},
        "Python 数据分析": {"price": "会员", "duration": "30小时"},
        "HTML 基础教程": {"price": "免费", "duration": "15小时"},
    }
})


@tool
def query_course_price(
    course_name: str,
    store: Annotated[BaseStore, InjectedStore()],
) -> str:
    """查询菜鸟教程 RUNOOB 中指定课程的价格信息。

    Args:
        course_name: 课程名称
    """
    item = store.get(("runoob", "courses"), "catalog")
    catalog = item.value["data"] if item else {}

    if course_name in catalog:
        info = catalog[course_name]
        return f"《{course_name}》- 价格：{info['price']}，学习时长：{info['duration']}"
    return f"未找到课程《{course_name}》"


model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)
agent = create_agent(
    model=model,
    tools=[query_course_price],
    store=store,  # 将 Store 传入 Agent
    system_prompt="你是菜鸟教程 RUNOOB 的课程顾问。",
)

result = agent.invoke({
    "messages": [HumanMessage(content="Python3 基础教程和 Python 数据分析分别多少钱？")]
})
print(result["messages"][-1].content)