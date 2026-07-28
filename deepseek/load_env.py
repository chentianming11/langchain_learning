import os

from dotenv import load_dotenv

# 加载当前目录 .env 文件
load_dotenv()

# 获取 API Key
api_key = os.getenv("DEEPSEEK_API_KEY")

print(api_key)