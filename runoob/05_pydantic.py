from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()

# 用 Pydantic 定义工具的参数结构
class WeatherInput(BaseModel):
    """查询指定城市的天气情况"""
    city: str = Field(description="城市名称，如 杭州、北京")
    unit: str = Field(
        default="celsius",
        description="温度单位，celsius（摄氏度）或 fahrenheit（华氏度）"
    )

class CalculatorInput(BaseModel):
    """执行数学计算"""
    expression: str = Field(
        description="要计算的数学表达式，如 '(3 + 5) * 2'"
    )

model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)

# 传入 Pydantic 模型，LangChain 自动转换为工具描述
model_with_tools = model.bind_tools([WeatherInput, CalculatorInput])

# 测试复杂场景
response = model_with_tools.invoke(
    "北京今天多少度？顺便帮我算一下 123 * 456"
)

print(f"模型请求了 {len(response.tool_calls)} 个工具调用：")
for tc in response.tool_calls:
    print(f"  {tc['name']}({tc['args']})")