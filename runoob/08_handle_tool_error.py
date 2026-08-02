from langchain.tools import ToolException
from langchain_core.tools import StructuredTool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()


# handle_tool_error=True：出错时不再抛异常，
# 而是把 ToolException 的内容转成字符串，正常返回给模型
# 注意：langchain 1.x 的 @tool 装饰器已移除 handle_tool_error 关键字，
# 改用 StructuredTool.from_function 的 **kwargs 透传给 BaseTool 的同名字段
def get_weather(city: str) -> str:
    """查询指定城市的天气。

    Args:
        city: 城市名称，必须是中文全称，如 "杭州"、"北京"
    """
    weather_data = {
        "杭州": "晴，25°C",
        "北京": "多云，18°C",
        "上海": "小雨，22°C",
    }
    if city not in weather_data:
        # 城市不在数据中时抛出 ToolException
        raise ToolException(
            f"未收录城市 '{city}'。"
            f"可使用城市：{', '.join(weather_data.keys())}。"
            f"请使用中文城市全称。"
        )
    return f"{city}天气：{weather_data[city]}"


get_weather = StructuredTool.from_function(get_weather, handle_tool_error=True)


# 测试：用错误城市名调用
# handle_tool_error=True 时，错误信息会作为正常结果返回，而不是抛出异常
result = get_weather.invoke({"city": "北境"})
print(f"handle_tool_error=True: {result}")


# 如果想让所有工具的错误都由 Agent 统一处理（而不是逐个工具设置），
# 可以在 create_agent 内部使用的 ToolNode 层面配置。
# 在较新版本的 langchain 中，可以这样为整个 Agent 打开错误兜底：
model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)
agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="你是一个天气查询助手。",
)

result = agent.invoke({
    "messages": [HumanMessage(content="查询北境的天气")]
})
print("\n=== Agent 收到 handle_tool_error 转换后的错误信息，并据此向用户解释 ===")
print(result["messages"][-1].content[:200])