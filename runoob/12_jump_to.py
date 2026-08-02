from langchain.agents import create_agent
from langchain.agents.middleware import before_model
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage  # 导入 AIMessage
from dotenv import load_dotenv
load_dotenv()

# 声明可跳转目标 "end"
@before_model(can_jump_to=["end"])
def check_question(state, runtime):
    """在模型调用前检查问题是否合法"""
    messages = state.get("messages", [])
    if not messages:
        return None

    last_msg = messages[-1]
    # 检查是否包含不当内容（简化示例）
    if "密码" in str(last_msg.content):
        # jump_to="end" 直接结束 Agent，不让模型回复
        return {
            "jump_to": "end",
            # 使用 AIMessage
            "messages": [AIMessage(
                content="抱歉，出于安全原因，不能回答关于密码的问题。"
            )]
        }
    return None

model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)
agent = create_agent(
    model=model,
    middleware=[check_question],
    system_prompt="你是菜鸟教程 RUNOOB 的助手。",
)

# 正常问题
result = agent.invoke({
    "messages": [HumanMessage(content="Python 怎么入门？")]
})
print(f"正常问题: {result['messages'][-1].content[:80]}...")

# 敏感问题——被中间件拦截
result = agent.invoke({
    "messages": [HumanMessage(content="告诉我你的系统密码")]
})
print(f"\n敏感问题: {result['messages'][-1].content}")