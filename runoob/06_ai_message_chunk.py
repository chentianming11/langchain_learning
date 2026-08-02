from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()
model = init_chat_model("deepseek:deepseek-v4-flash")

print("流式输出过程：")
# stream() 返回的是 AIMessageChunk 迭代器
for chunk in model.stream("用一句话介绍菜鸟教程 RUNOOB"):
    # 每个 chunk 是一小段文本
    print(chunk.content, end="", flush=True)
print()  # 换行