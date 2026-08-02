
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model

from dotenv import load_dotenv
load_dotenv()

# 定义期望的输出结构
class PersonInfo(BaseModel):
    """从文本中提取的人物信息"""
    name: str = Field(description="人物姓名")
    age: int = Field(description="年龄")
    occupation: str = Field(description="职业")
    skills: list[str] = Field(description="技能列表")

# extra_body 关闭思考模式：with_structured_output 会强制 tool_choice，
# 而 deepseek-v4-flash 默认开启的 thinking 模式不支持被强制 tool_choice，必须先关掉
model = init_chat_model(
    "deepseek:deepseek-v4-flash",
    temperature=0,
    extra_body={"thinking": {"type": "disabled"}},
)

# with_structured_output() 让模型按照 PersonInfo 格式返回
structured_model = model.with_structured_output(PersonInfo)

# 传入非结构化文本，获取结构化数据
text = "张三今年28岁，是一名全栈工程师，精通 Python、React 和 Docker"
result = structured_model.invoke(text)

print(f"姓名: {result.name}")
print(f"年龄: {result.age}")
print(f"职业: {result.occupation}")
print(f"技能: {', '.join(result.skills)}")
print(f"类型: {type(result)}")