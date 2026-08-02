from langchain.messages import HumanMessage, AIMessage, ToolMessage
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()

# 模拟一轮完整的工具调用对话
messages = [
    HumanMessage(content="杭州天气怎么样？"),

    # 模型请求调用工具
    AIMessage(
        content="",
        tool_calls=[
            {"name": "get_weather", "args": {"city": "杭州"},
             "id": "call_abc", "type": "tool_call"}
        ]
    ),

    # 工具返回结果（必须包含 tool_call_id 与上面的 id 对应）
    ToolMessage(
        content="晴，25°C，湿度 60%",
        tool_call_id="call_abc",   # 与 tool_call 的 id 对应
        name="get_weather",        # 工具名称
    ),
]

# deepseek-v4-flash 默认开启的 thinking 模式下，回传带 tool_calls 的 AIMessage
# 必须携带 reasoning_content；这里手动构造的 AIMessage 没有，故先关闭 thinking 模式
model = init_chat_model(
    "deepseek:deepseek-v4-flash",
    extra_body={"thinking": {"type": "disabled"}},
)
response = model.invoke(messages)
print(f"模型基于工具结果的回复: {response.content}")