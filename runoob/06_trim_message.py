from langchain.messages import (
    HumanMessage, AIMessage, SystemMessage, trim_messages
)
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import tiktoken
load_dotenv()


# 自定义 token 计数器：deepseek-v4-flash 没有内置 tokenizer 实现，
# model.get_num_tokens_from_messages() 会抛 NotImplementedError，
# 这里用 tiktoken 的 cl100k_base 自行统计每条消息的 token 数
_enc = tiktoken.get_encoding("cl100k_base")
# 每条消息的固定开销（role、分隔符等），与 OpenAI 的计数口径一致
_TOKENS_PER_MESSAGE = 4


def count_tokens(messages) -> int:
    """统计一组消息的 token 数"""
    total = 0
    for msg in messages:
        total += _TOKENS_PER_MESSAGE
        total += len(_enc.encode(msg.content if isinstance(msg.content, str) else str(msg.content)))
    total += 2  # 每轮对话的收尾开销
    return total

# 模拟一段很长的对话历史
messages = [
    SystemMessage(content="你是菜鸟教程 RUNOOB 的 AI 助手"),
    HumanMessage(content="Python 怎么入门？"),
    AIMessage(content="Python 入门可以从基础知识开始..."),
    HumanMessage(content="有推荐的 IDE 吗？"),
    AIMessage(content="推荐 VS Code 或 PyCharm..."),
    HumanMessage(content="如何安装第三方库？"),
    AIMessage(content="使用 pip install 命令..."),
    HumanMessage(content="NumPy 是什么？"),
    AIMessage(content="NumPy 是一个科学计算库..."),
    HumanMessage(content="pandas 和 NumPy 有什么区别？"),
]

# deepseek-v4-flash 默认开启的 thinking 模式下，回传带 tool_calls 的 AIMessage
# 必须携带 reasoning_content；这里手动构造的 AIMessage 没有，故先关闭 thinking 模式
model = init_chat_model(
    "deepseek:deepseek-v4-flash",
    extra_body={"thinking": {"type": "disabled"}},
)

# 裁剪消息以适应模型的上下文窗口（最多 1000 tokens）
# strategy="last" 保留最后的系统消息和最近的对话
trimmed = trim_messages(
    messages,
    max_tokens=1000,           # 最多保留 1000 tokens
    strategy="last",           # 保留最后的系统消息 + 最近的对话
    token_counter=count_tokens,    # 用自定义计数器，避免 deepseek 未实现 get_num_tokens_from_messages
    include_system=True,       # 始终保留 SystemMessage
    start_on="human",          # 裁剪后以 human 消息开头
)

print(f"裁剪前: {len(messages)} 条消息")
print(f"裁剪后: {len(trimmed)} 条消息")
for msg in trimmed:
    snippet = msg.content[:50] if isinstance(msg.content, str) else str(msg.content)[:50]
    print(f"  [{msg.type}] {snippet}...")