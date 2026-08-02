from pydantic import BaseModel, Field
from langchain.tools import tool


# 定义参数模型（提供更精细的参数控制）
class CourseSearchInput(BaseModel):
    """搜索课程参数"""
    keyword: str = Field(
        description="搜索关键词，支持模糊匹配",
        min_length=1,            # 最少 1 个字符
        max_length=50,           # 最多 50 个字符
    )
    category: str = Field(
        default="all",
        description="课程类别：all（全部）、frontend（前端）、backend（后端）、data（数据科学）",
        pattern=r"^(all|frontend|backend|data)$",  # 限定可选值
    )
    page: int = Field(
        default=1,
        description="页码，从 1 开始",
        ge=1,                    # 大于等于 1
        le=100,                  # 小于等于 100
    )


@tool(args_schema=CourseSearchInput)
def search_course(keyword: str, category: str = "all", page: int = 1) -> str:
    """在菜鸟教程 RUNOOB 中搜索课程"""
    return f"搜索 '{keyword}' (分类: {category}, 第 {page} 页)：共找到 15 条结果"


# 有效调用
print(search_course.invoke({"keyword": "Python", "category": "backend", "page": 1}))

# 无效调用（category 不在允许的值内）
try:
    search_course.invoke({"keyword": "Python", "category": "invalid"})
except Exception as e:
    print(f"参数校验失败: {e}")