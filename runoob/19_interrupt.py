from dotenv import load_dotenv
load_dotenv()

from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage


@tool
def delete_course(course_name: str) -> str:
    """删除课程（需要审批）。

    Args:
        course_name: 要删除的课程名称
    """
    # 暂停并等待审批
    approval = interrupt({
        "action": "delete_course",
        "course": course_name,
        "message": f"确认删除课程《{course_name}》？此操作不可撤销。"
    })

    if approval.get("confirmed"):
        return f"课程《{course_name}》已删除"
    else:
        return f"删除操作已取消"


checkpointer = InMemorySaver()
model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)
agent = create_agent(
    model=model,
    tools=[delete_course],
    checkpointer=checkpointer,
    system_prompt="你是菜鸟教程 RUNOOB 的管理员助手。",
)

config = {"configurable": {"thread_id": "admin-001"}}

# 第一步：发起删除请求（会触发中断）
print("=== 开始执行 ===")
result = agent.invoke(
    {"messages": [HumanMessage(content="请删除课程《过时的 Java 教程》")]},
    config=config,
)

# 检查 Agent 是否暂停了
state = agent.get_state(config)
print(f"状态: {state.next}")  # ('tools',) 表示在 tools 节点暂停
print(f"中断信息: {state.tasks[0].interrupts}")

# 第二步：人工审批（模拟用户点击"确认"）
print("\n=== 人工审批 ===")
resume_value = {"confirmed": True, "operator": "管理员张三"}
result = agent.invoke(
    Command(resume=resume_value),
    config=config,
)
print(f"最终回复: {result['messages'][-1].content}")