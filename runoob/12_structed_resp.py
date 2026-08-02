from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()

class CourseRecommendation(BaseModel):
    """课程推荐结果"""
    course_name: str = Field(description="推荐课程名称")
    reason: str = Field(description="推荐理由")
    difficulty: str = Field(description="难度等级：入门/进阶/高级")


model = init_chat_model(
    "deepseek:deepseek-v4-flash",
    temperature=0,
    extra_body={"thinking": {"type": "disabled"}},
)
agent = create_agent(
    model=model,
    # 显式用 ToolStrategy 走「工具调用」方式获取结构化输出。
    # DeepSeek 不支持 response_format 的 json_schema 原生模式，
    # 默认 AutoStrategy 会被判定为支持并改用 ProviderStrategy 而报 400。
    response_format=ToolStrategy(schema=CourseRecommendation),
    system_prompt="你是菜鸟教程 RUNOOB 的学习顾问。",
)

result = agent.invoke({
    "messages": [HumanMessage(content="我想学编程，推荐一门适合零基础的课程")]
})

# 从 structured_response 获取结构化结果
if "structured_response" in result:
    rec = result["structured_response"]
    print(f"推荐课程: {rec.course_name}")
    print(f"推荐理由: {rec.reason}")
    print(f"难度等级: {rec.difficulty}")

# structured_response 不在 output schema 中
# 所以不会自动出现在返回给调用者的结果中（可配置）