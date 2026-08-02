

from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model

# extra_body 关闭思考模式：with_structured_output 会强制 tool_choice，
# 而 deepseek-v4-flash 默认开启的 thinking 模式不支持被强制 tool_choice，必须先关掉
model = init_chat_model(
    "deepseek:deepseek-v4-flash",
    temperature=0,
    extra_body={"thinking": {"type": "disabled"}},
)
# 直接传入 JSON Schema
json_schema = {
    "title": "SentimentAnalysis",
    "description": "情感分析结果",
    "type": "object",
    "properties": {
        "sentiment": {
            "type": "string",
            "enum": ["positive", "negative", "neutral"],
            "description": "情感倾向"
        },
        "confidence": {
            "type": "number",
            "description": "置信度，0~1"
        },
        "keywords": {
            "type": "array",
            "items": {"type": "string"},
            "description": "关键情感词"
        }
    },
    "required": ["sentiment", "confidence"]
}

structured_model = model.with_structured_output(json_schema)
result = structured_model.invoke("菜鸟教程 RUNOOB 真的太棒了，强烈推荐给所有编程新手！")

print(f"情感: {result['sentiment']}")
print(f"置信度: {result['confidence']}")
print(f"关键词: {result['keywords']}")