from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()

# 创建一个内存 Checkpointer
checkpointer = InMemorySaver()

model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)
agent = create_agent(
    model=model,
    checkpointer=checkpointer,  # 传入 Checkpointer
    system_prompt="你是菜鸟教程 RUNOOB 的助手。",
)

# 使用 thread_id 来标识对话线程
config = {"configurable": {"thread_id": "user-001"}}

# 第一轮
result1 = agent.invoke(
    {"messages": [HumanMessage(content="我叫小明，我在学 Python")]},
    config=config,
)
print(f"第一轮: {result1['messages'][-1].content}")

# 第二轮——使用相同的 thread_id，Agent 记住了！
result2 = agent.invoke(
    {"messages": [HumanMessage(content="我叫什么名字？我在学什么？")]},
    config=config,
)
print(f"第二轮: {result2['messages'][-1].content}")