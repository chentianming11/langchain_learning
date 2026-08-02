from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()

class WeatherReport(BaseModel):
    """天气报告"""
    city: str = Field(description="城市名称")
    temperature: float = Field(description="温度（摄氏度）")
    condition: str = Field(description="天气状况")
    humidity: int = Field(description="湿度百分比")


# ToolStrategy 会强制 tool_choice 让模型必须调用结构化输出工具，
# 而 deepseek-v4-flash 默认的 thinking 模式不支持强制 tool_choice，需先关闭
model = init_chat_model(
    "deepseek:deepseek-v4-flash",
    temperature=0,
    extra_body={"thinking": {"type": "disabled"}},
)

# 显式指定使用 ToolStrategy
agent = create_agent(
    model=model,
    response_format=ToolStrategy(schema=WeatherReport),
    system_prompt="你是天气助手，根据用户描述生成结构化天气报告。",
)

result = agent.invoke({
    "messages": [HumanMessage(content="杭州今天晴天，温度25度，湿度60%")]
})

report = result["structured_response"]
print(f"城市: {report.city}")
print(f"温度: {report.temperature}°C")
print(f"状况: {report.condition}")
print(f"湿度: {report.humidity}%")

# 查看执行过程——可以发现多了一条工具调用的消息
print(f"\n消息总数: {len(result['messages'])}")
for msg in result["messages"]:
    print(f"  [{msg.type}]", end="")
    if hasattr(msg, 'tool_calls') and msg.tool_calls:
        print(f" 调用: {[tc['name'] for tc in msg.tool_calls]}")
    elif msg.type == "tool":
        print(f" {msg.content[:60]}")
    else:
        print(f" {str(msg.content)[:60]}")