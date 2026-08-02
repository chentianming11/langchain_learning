from langchain.messages import AIMessage

# 普通 AI 回复（无工具调用）
ai_msg = AIMessage(content="菜鸟教程是一个编程学习平台")

# 包含工具调用的 AI 回复
ai_with_tools = AIMessage(
    content="",  # 工具调用时 content 通常为空
    tool_calls=[
        {
            "name": "get_weather",
            "args": {"city": "杭州"},
            "id": "call_abc123",
            "type": "tool_call",
        }
    ]
)

print("=== 普通 AI 消息 ===")
print(f"content: {ai_msg.content}")
print(f"tool_calls: {ai_msg.tool_calls}")   # []

print("\n=== 含工具调用的 AI 消息 ===")
print(f"content: {ai_with_tools.content}")
print(f"tool_calls: {ai_with_tools.tool_calls}")
# [{'name': 'get_weather', 'args': {'city': '杭州'}, ...}]