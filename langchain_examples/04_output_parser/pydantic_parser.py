# =============================================================================
# 输出解析（三）— PydanticOutputParser：带类型验证的结构化输出
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 理解 Pydantic = "带检查的表单"（自动验证类型）
#   ✅ 使用 PydanticOutputParser 让 AI 输出严格类型化的数据
#   ✅ 定义嵌套数据模型（含列表、可选字段等）
# =============================================================================

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)


"""
JsonOutputParser vs PydanticOutputParser:

  JsonOutputParser:  返回 Python dict（字典）
    {"name": "小明", "age": "二十五"}  ← age 是字符串！程序可能崩溃

  PydanticOutputParser: 返回 Pydantic 对象（带类型验证）
    Person(name="小明", age=25)         ← age 必须是 int，否则直接报错

  生活化比喻: Pydantic = 带检查的表单
    填年龄时如果写了"二十五"，系统立刻提示"请填写数字"
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import PydanticOutputParser


# =============================================================================
# 示例 1: Pydantic + PydanticOutputParser 基础
# =============================================================================

def pydantic_output_parser_basic():
    """
    两步：定义数据模型 + 创建 Parser 绑定模型。

    PydanticOutputParser 会自动从模型定义生成格式说明告诉 AI，
    AI 输出 JSON 后自动验证并转换为 Pydantic 对象。
    """
    print(f"\n-- 示例 1: PydanticOutputParser 基础")

    # 第 1 步: 定义数据模型（带 Field 描述 — AI 会参考这些描述生成字段）
    class Person(BaseModel):
        """人员信息"""
        name: str = Field(description="姓名")
        age: int = Field(description="年龄（整数）")
        job: str = Field(description="职业")

    # 第 2 步: 创建 Parser（绑定模型）
    parser = PydanticOutputParser(pydantic_object=Person)

    prompt = PromptTemplate.from_template("""
从文本中提取人员信息。

文本: 小明今年25岁，是一名软件工程师。

{format_instructions}

JSON:""")

    prompt = prompt.partial(format_instructions=parser.get_format_instructions())

    model = ChatOllama(model="qwen3.5:2b")
    chain = prompt | model | parser

    result = chain.invoke({})
    print(f"  类型: {type(result).__name__}（Pydantic 对象，不是 dict）")
    print(f"  name: {result.name}（类型: {type(result.name).__name__}）")
    print(f"  age:  {result.age}（类型: {type(result.age).__name__}）")
    print(f"  job:  {result.job}")


# =============================================================================
# 示例 2: 复杂嵌套模型 — 电影信息提取
# =============================================================================

def complex_nested_model():
    """
    定义包含列表、浮点数、可选字段的复杂模型。

    这展示了 Pydantic 的真正威力——
    AI 输出的 genres 字段自动变成 Python list[str]，
    rating 自动变成 float，不需要手动类型转换。
    """
    print(f"\n-- 示例 2: 复杂嵌套模型 — 电影信息提取")

    class Movie(BaseModel):
        title: str = Field(description="电影名称")
        year: int = Field(description="上映年份")
        director: str = Field(description="导演")
        genres: List[str] = Field(description="类型列表")
        rating: float = Field(description="评分（0-10）")
        description: Optional[str] = Field(default=None, description="一句话简介")

    parser = PydanticOutputParser(pydantic_object=Movie)

    prompt = PromptTemplate.from_template("""
提取以下电影信息。

文本: 《肖申克的救赎》是1994年由弗兰克·德拉邦特执导，
类型包括剧情、犯罪，豆瓣评分9.7，是一部关于希望和自由的经典之作。

{format_instructions}

JSON:""")

    prompt = prompt.partial(format_instructions=parser.get_format_instructions())

    model = ChatOllama(model="qwen3.5:2b")
    chain = prompt | model | parser

    result = chain.invoke({})
    print(f"  标题: {result.title} ({result.year})")
    print(f"  导演: {result.director}")
    print(f"  类型: {', '.join(result.genres)}（自动变成 list）")
    print(f"  评分: {result.rating}（自动变成 float）")


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n>>> 04_output_parser/pydantic_parser — PydanticOutputParser\n")

    pydantic_output_parser_basic()
    complex_nested_model()

    # 接下来学习: 05_memory/conversation_memory.py（对话记忆）
