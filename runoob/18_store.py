from dotenv import load_dotenv
load_dotenv()

from typing import Annotated
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langchain.tools import tool, InjectedStore
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage

# 创建 Store 并预置数据
store = InMemoryStore()
store.put(("runoob", "courses"), "catalog", {
    "Python3 基础教程": {"price": "免费", "hours": 20, "level": "入门"},
    "Python 数据分析": {"price": "会员", "hours": 30, "level": "进阶"},
    "Java 面向对象": {"price": "免费", "hours": 25, "level": "进阶"},
})

store.put(("runoob", "users"), "user_vip_001", {
    "name": "小明",
    "membership": "VIP",
    "joined": "2024-01-15",
})


@tool
def query_course_info(
    course_name: str,
    store: Annotated[BaseStore, InjectedStore()],
) -> str:
    """查询菜鸟教程 RUNOOB 中课程的详细信息。

    Args:
        course_name: 课程名称
    """
    item = store.get(("runoob", "courses"), "catalog")
    catalog = item.value if item else {}

    if course_name in catalog:
        info = catalog[course_name]
        return (
            f"《{course_name}》- 价格：{info['price']}，"
            f"时长：{info['hours']}小时，难度：{info['level']}"
        )
    return f"未找到课程《{course_name}》"


@tool
def get_user_membership(
    user_id: str,
    store: Annotated[BaseStore, InjectedStore()],
) -> str:
    """查询用户会员信息。

    Args:
        user_id: 用户 ID
    """
    item = store.get(("runoob", "users"), user_id)
    if item is None:
        return f"未找到用户 {user_id}"

    user = item.value
    return (
        f"用户 {user['name']}，{user['membership']} 会员，"
        f"注册日期 {user['joined']}"
    )


model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)
agent = create_agent(
    model=model,
    tools=[query_course_info, get_user_membership],
    store=store,
    system_prompt="你是菜鸟教程 RUNOOB 的课程顾问。",
)

# 查询课程信息（数据来自 Store）
result = agent.invoke({
    "messages": [HumanMessage(content="Python3 基础教程多少钱？")]
})
print(f"查询课程: {result['messages'][-1].content}")

# 查询用户信息（数据来自 Store）
result = agent.invoke({
    "messages": [HumanMessage(content="帮我查一下用户 user_vip_001 的信息")]
})
print(f"查询用户: {result['messages'][-1].content}")