from dotenv import load_dotenv
load_dotenv()

from langchain.tools import tool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage


@tool
def search_courses(keyword: str) -> str:
    """在菜鸟教程 RUNOOB 中搜索课程。传入关键词返回相关课程列表。

    Args:
        keyword: 搜索关键词，如 python、html、java
    """
    courses = {
        "python": "Python3 基础教程、Python 数据分析、Python 爬虫入门",
        "html": "HTML 基础教程、HTML5 新特性、HTML 表单实战",
        "java": "Java 基础教程、Java 面向对象、Java Spring 框架",
    }
    return courses.get(keyword.lower(), f"未找到与 {keyword} 相关的课程")


@tool
def get_course_detail(course_name: str) -> str:
    """获取指定课程的详细信息，包括章节数和学习时长。

    Args:
        course_name: 课程名称，如 "Python3 基础教程"
    """
    details = {
        "python3 基础教程": "共 30 章，预计学习时长 20 小时，适合零基础入门",
        "html 基础教程": "共 25 章，预计学习时长 15 小时，适合零基础入门",
    }
    return details.get(
        course_name.lower(),
        f"{course_name} 详情：适合初学者，内容丰富，附带实战案例"
    )


# 创建 Agent
model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)
agent = create_agent(
    model=model,
    tools=[search_courses, get_course_detail],
    system_prompt="你是菜鸟教程 RUNOOB 的学习顾问，帮助用户找到合适的课程。",
)


# 运行 Agent
def ask(question: str):
    result = agent.invoke({"messages": [HumanMessage(content=question)]})
    print(f"用户: {question}")
    print(f"顾问: {result['messages'][-1].content}")
    print("-" * 60)


ask("我想学 Python，有什么课程推荐？")
ask("Python3 基础教程学完需要多久？")