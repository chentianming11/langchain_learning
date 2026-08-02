from typing import Annotated
from langchain.agents import create_agent, AgentState
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain.tools import tool, InjectedState
from typing_extensions import TypedDict
from dotenv import load_dotenv
load_dotenv()

# 扩展 AgentState，添加业务字段
class ShoppingAgentState(AgentState):
    """购物助手的状态"""
    cart: list[str]         # 购物车商品列表
    total_price: float      # 总价


@tool
def add_to_cart(
    item: str,
    price: float,
    state: Annotated[dict, InjectedState],
) -> str:
    """将商品添加到购物车。

    Args:
        item: 商品名称
        price: 商品价格
    """
    cart = state.get("cart", [])
    total = state.get("total_price", 0.0)

    return {
        "cart": cart + [item],
        "total_price": total + price,
        "messages": [],  # 不添加额外消息
    }


@tool
def view_cart(
    state: Annotated[dict, InjectedState],
) -> str:
    """查看购物车内容"""
    cart = state.get("cart", [])
    total = state.get("total_price", 0.0)
    if not cart:
        return "购物车为空"
    items = "、".join(cart)
    return f"购物车：{items}，总价：¥{total:.2f}"


model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)
agent = create_agent(
    model=model,
    tools=[add_to_cart, view_cart],
    state_schema=ShoppingAgentState,  # 使用自定义状态
    system_prompt="你是菜鸟教程 RUNOOB 商店的购物助手。",
)

# 初始状态包含空的购物车
result = agent.invoke({
    "messages": [HumanMessage(content="帮我加一本 Python 教程到购物车，价格 49.9")],
    "cart": ['Java教程'],
    "total_price": 10.0,
})

print(f"购物车: {result.get('cart', [])}")
print(f"总价: ¥{result.get('total_price', 0):.2f}")
print(f"回复: {result['messages'][-1].content}")