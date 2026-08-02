from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model

model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)

# 用字典描述工具（OpenAI function calling 格式）
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如 杭州、北京"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

# bind_tools() 将工具绑定到模型
# 模型现在"知道"有 get_weather 这个工具可用
model_with_tools = model.bind_tools(tools)

# 问一个需要工具的问题
response = model_with_tools.invoke("杭州今天天气怎么样？")

# 检查模型是否请求调用工具
if response.tool_calls:
    print("模型请求调用以下工具：")
    for tc in response.tool_calls:
        print(f"  工具名: {tc['name']}")
        print(f"  参数: {tc['args']}")
        print(f"  调用ID: {tc['id']}")
else:
    print(f"模型直接回复: {response.content}")