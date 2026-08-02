from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()

model = init_chat_model("deepseek:deepseek-v4-flash")
response = model.invoke("介绍菜鸟教程 RUNOOB")

# AIMessage 包含丰富的元数据
print(f"内容: {response.content}")
print(f"消息ID: {response.id}")
print(f"模型名: {response.response_metadata.get('model_name')}")
print(f"完成原因: {response.response_metadata.get('finish_reason')}")

# usage_metadata 包含 Token 用量信息
if response.usage_metadata:
    print(f"输入 tokens: {response.usage_metadata.get('input_tokens')}")
    print(f"输出 tokens: {response.usage_metadata.get('output_tokens')}")
    print(f"总计 tokens: {response.usage_metadata.get('total_tokens')}")