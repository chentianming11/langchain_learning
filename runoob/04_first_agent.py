# 文件路径：04_first_agent.py
from dotenv import load_dotenv

load_dotenv()

from langchain.tools import tool

# 步骤 1：用 @tool 装饰器定义一个工具
# 函数的文档字符串就是工具的描述
# 模型会根据描述来判断何时调用这个工具
@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气情况。

    Args:
        city: 城市名称，如 "杭州"、"北京"
    """
    # 这里用模拟数据演示
    # 实际项目中可以替换为真实的天气 API 调用
    weather_data = {
        "杭州": "晴，25°C，湿度 60%",
        "北京": "多云，18°C，湿度 45%",
        "上海": "小雨，22°C，湿度 80%",
    }
    return weather_data.get(city, f"未找到 {city} 的天气数据")


@tool
def calculate(expression: str) -> str:
    """执行数学计算。支持加减乘除等基本运算。

    Args:
        expression: 数学表达式，如 "3 * 7 + 2"
    """
    try:
        # 安全地计算数学表达式
        result = eval(expression, {"__builtins__": {}}, {})
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"

# 步骤 2：创建 Agent
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

# 初始化模型
model = init_chat_model("deepseek:deepseek-v4-flash")

# 创建 Agent，传入模型和工具列表
agent = create_agent(
    model=model,
    tools=[get_weather, calculate],
    system_prompt="你是一个乐于助人的助手，会使用工具来回答问题。",
)


# 步骤 3：运行 Agent

# 构建输入消息
# 消息列表中的第一条通常是 HumanMessage（用户消息）
from langchain.messages import HumanMessage


def ask(question : str):
    inputs = {"messages": [HumanMessage(content=question)]}

    # invoke() 运行 Agent，返回最终状态
    result = agent.invoke(inputs)

    # 查看消息历史（包含 AI 的工具调用和工具返回结果）
    print("=== 完整消息历史 ===")
    for msg in result["messages"]:
        print(f"[{msg.type}] {msg.content[:100]}")  # 截取前 100 字符

    print("\n=== 最终回复 ===")
    # 最后一条 AI 消息就是最终答案
    print(result["messages"][-1].content)


ask("杭州今天天气怎么样？")
# 问一个需要同时查询天气和计算的复杂问题
ask("杭州和北京今天温差多少度？")