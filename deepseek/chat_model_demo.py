import os

from dotenv import load_dotenv

# 加载当前目录 .env 文件
load_dotenv()
from langchain.chat_models import init_chat_model

# 指定了 model，返回固定模型
model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0.7)
response = model.invoke("介绍菜鸟教程 RUNOOB")
print(response.content)