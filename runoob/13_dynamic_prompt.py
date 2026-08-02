from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt
from langchain.agents.middleware.types import ModelRequest
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain.tools import tool
from dotenv import load_dotenv
load_dotenv()

@tool
def search_course(keyword: str) -> str:
    """在菜鸟教程搜索课程"""
    courses = {
        "python": "Python3 基础教程（免费，20小时）",
        "html": "HTML 基础教程（免费，15小时）",
        "java": "Java 基础教程（免费，25小时）",
    }
    return courses.get(keyword.lower(), "未找到相关课程")


# @dynamic_prompt 装饰器：接收 ModelRequest，返回新的 system_prompt
@dynamic_prompt
def personalized_prompt(request: ModelRequest) -> str:
    """根据对话上下文动态生成个性化提示词"""
    messages = request.state.get("messages", [])
    message_count = len(messages)

    # 可以根据不同的条件动态调整提示词
    base_prompt = "你是菜鸟教程 RUNOOB 的学习顾问。"

    if message_count <= 2:
        # 对话刚开始，耐心引导
        return base_prompt + (
            "用户刚开始对话，请先热情问候，"
            "然后询问他们的学习目标和当前水平。"
        )
    elif message_count > 10:
        # 长对话，提醒保持简洁
        return base_prompt + (
            "对话已经比较长了，回答要尽量简洁，"
            "每次不超过 2 句话。"
        )
    else:
        # 正常对话阶段
        return base_prompt + (
            "根据用户之前的问题推荐合适的课程，"
            "使用 search_course 工具查询课程信息。"
        )


model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)
agent = create_agent(
    model=model,
    tools=[search_course],
    middleware=[personalized_prompt],  # 通过 middleware 注入
)

# 第一次对话（message_count <= 2，引导模式）
result = agent.invoke({
    "messages": [HumanMessage(content="你好")]
})
print(f"第一轮回复: {result['messages'][-1].content}")