# =============================================================================
# Pydantic 强类型解析
# =============================================================================
#  
# 用途：教学演示 - 使用 PydanticOutputParser 进行强类型数据验证
#
# 核心概念：
#   - Pydantic = "带检查的数据模型"
#   - 类型安全的数据验证
# =============================================================================

# -----------------------------------------------------------------------------
# 运行前检查
# -----------------------------------------------------------------------------
# 1. 已安装 Ollama 服务
# 2. 已下载模型：ollama pull qwen3:4b
# -----------------------------------------------------------------------------

# 设置 UTF-8 编码（Windows 专用）
import sys
import io
sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer,
    encoding='utf-8',
    errors='replace',
    line_buffering=True
)


# =============================================================================
# 第一部分：理解 Pydantic
# =============================================================================
"""
什么是 Pydantic？

📝 定义
   Pydantic = Python 的数据验证库
   用类型注解定义数据结构，自动验证

🎯 为什么用 Pydantic？

   不用 Pydantic:
   data = {
       "name": "小明",
       "age": "二十五",  # 字符串！应该是数字
       "email": "not-an-email"  # 无效邮箱
   }
   → 程序运行时可能出错

   用 Pydantic:
   class Person(BaseModel):
       name: str
       age: int        # 必须是数字
       email: EmailStr  # 必须是有效邮箱

   → 自动验证，出错立刻知道

💡 生活化比喻
   Pydantic = "带检查的表单"
   填写信息时，系统会检查：
   - 年龄必须填数字
   - 邮箱必须带@
   - 必填项不能空
"""


# =============================================================================
# 示例 1: Pydantic 基础
# =============================================================================

def pydantic_basic():
    """
    Pydantic 基础用法

    先了解 Pydantic 是什么
    """
    print("=" * 60)
    print("示例 1: Pydantic 基础")
    print("=" * 60)

    from pydantic import BaseModel, Field

    # 1. 定义数据模型
    class Person(BaseModel):
        """人员信息"""
        name: str = Field(description="姓名")
        age: int = Field(description="年龄")
        email: str = Field(description="邮箱")

    # 2. 创建实例（会自动验证类型）
    try:
        # 正确用法
        person = Person(name="小明", age=25, email="xiaoming@example.com")
        print(f"创建成功：{person}")
        print(f"name: {person.name}")
        print(f"age: {person.age} (类型：{type(person.age)})")
        print()

        # 错误用法：类型不对会报错
        # Person(name="小红", age="二十五", email="...")
        # → ValidationError: age 必须是整数

    except Exception as e:
        print(f"验证错误：{e}")
        print()


# =============================================================================
# 示例 2: PydanticOutputParser 基础
# =============================================================================

def pydantic_output_parser():
    """
    使用 PydanticOutputParser 让 AI 输出结构化数据
    """
    print("=" * 60)
    print("示例 2: PydanticOutputParser 基础")
    print("=" * 60)

    from pydantic import BaseModel, Field
    from langchain_ollama import ChatOllama
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_core.prompts import PromptTemplate

    # 1. 定义数据模型
    class Person(BaseModel):
        """人员信息"""
        name: str = Field(description="姓名")
        age: int = Field(description="年龄")
        job: str = Field(description="职业")

    # 2. 创建 Parser
    parser = PydanticOutputParser(pydantic_object=Person)

    # 3. 创建 Prompt
    prompt = PromptTemplate.from_template("""
请从以下文本中提取人员信息。

文本：小明今年 25 岁，是一名软件工程师。

{format_instructions}

JSON 输出：""")

    # 4. 注入格式指令
    prompt_with_format = prompt.partial(
        format_instructions=parser.get_format_instructions()
    )

    # 5. 创建 Pipeline
    model = ChatOllama(model="qwen3:4b")
    chain = prompt_with_format | model | parser

    # 6. 调用
    result = chain.invoke({})

    print(f"解析结果：{result}")
    print(f"类型：{type(result)}")
    print(f"是 Person 对象：{isinstance(result, Person)}")
    print(f"name: {result.name}")
    print(f"age: {result.age}")
    print(f"job: {result.job}")
    print()


# =============================================================================
# 示例 3: 复杂数据模型
# =============================================================================

def complex_data_model():
    """
    定义复杂的数据模型

    包含列表、嵌套对象等
    """
    print("=" * 60)
    print("示例 3: 复杂数据模型")
    print("=" * 60)

    from typing import List, Optional
    from pydantic import BaseModel, Field
    from langchain_ollama import ChatOllama
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_core.prompts import PromptTemplate

    # 定义嵌套数据模型
    class Movie(BaseModel):
        """电影信息"""
        title: str = Field(description="电影名称")
        year: int = Field(description="上映年份")
        director: str = Field(description="导演")
        genres: List[str] = Field(description="类型列表")
        rating: float = Field(description="评分（0-10）")
        description: Optional[str] = Field(default=None, description="简介")

    # 创建 Parser
    parser = PydanticOutputParser(pydantic_object=Movie)

    # 创建 Prompt
    prompt = PromptTemplate.from_template("""
请从以下文本中提取电影信息。

文本：《肖申克的救赎》是 1994 年由弗兰克·德拉邦特执导的电影，
类型包括剧情、犯罪，豆瓣评分 9.7 分，是一部关于希望和自由经典之作。

{format_instructions}

JSON 输出：""")

    prompt_with_format = prompt.partial(
        format_instructions=parser.get_format_instructions()
    )

    model = ChatOllama(model="qwen3:4b")
    chain = prompt_with_format | model | parser

    result = chain.invoke({})

    print(f"电影信息:")
    print(f"  名称：{result.title}")
    print(f"  年份：{result.year}")
    print(f"  导演：{result.director}")
    print(f"  类型：{', '.join(result.genres)}")
    print(f"  评分：{result.rating}")
    print(f"  简介：{result.description}")
    print()


# =============================================================================
# 示例 4: 实用场景 - 新闻摘要
# =============================================================================

def news_summary():
    """
    使用 Pydantic 提取新闻摘要
    """
    print("=" * 60)
    print("示例 4: 新闻摘要（实用场景）")
    print("=" * 60)

    from typing import List
    from pydantic import BaseModel, Field
    from langchain_ollama import ChatOllama
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_core.prompts import PromptTemplate

    # 定义新闻摘要模型
    class NewsSummary(BaseModel):
        """新闻摘要"""
        headline: str = Field(description="标题")
        summary: str = Field(description="一句话摘要（50 字以内）")
        key_points: List[str] = Field(description="关键要点列表（3-5 条）")
        category: str = Field(description="分类（科技/财经/体育/娱乐等）")

    parser = PydanticOutputParser(pydantic_object=NewsSummary)

    prompt = PromptTemplate.from_template("""
请阅读以下新闻，提取摘要信息。

新闻：
OpenAI 发布了新一代 AI 模型 GPT-5，该模型在逻辑推理、代码生成和多语言理解方面有显著提升。
测试显示，GPT-5 在多项基准测试中超越了现有模型。该模型将于下月向付费用户开放，免费版将在明年推出。
专家分析，这次发布可能再次引发 AI 行业的竞争。

{format_instructions}

JSON 输出：""")

    prompt_with_format = prompt.partial(
        format_instructions=parser.get_format_instructions()
    )

    model = ChatOllama(model="qwen3:4b")
    chain = prompt_with_format | model | parser

    result = chain.invoke({})

    print(f"新闻摘要:")
    print(f"  标题：{result.headline}")
    print(f"  摘要：{result.summary}")
    print(f"  分类：{result.category}")
    print(f"  关键点:")
    for point in result.key_points:
        print(f"    - {point}")
    print()


# =============================================================================
# 示例 5: 错误处理与验证
# =============================================================================

def error_handling_and_validation():
    """
    Pydantic 的错误处理和验证
    """
    print("=" * 60)
    print("示例 5: 错误处理与验证")
    print("=" * 60)

    from pydantic import BaseModel, Field, ValidationError
    from langchain_ollama import ChatOllama
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_core.prompts import PromptTemplate

    class Product(BaseModel):
        """产品信息"""
        name: str = Field(description="产品名称")
        price: float = Field(description="价格（必须大于 0）")
        stock: int = Field(description="库存数量")

    parser = PydanticOutputParser(pydantic_object=Product)

    prompt = PromptTemplate.from_template("""
请提取以下文本中的产品信息。

文本：这款 iPhone 15 Pro 售价 7999 元，目前库存充足。

{format_instructions}

JSON 输出：""")

    prompt_with_format = prompt.partial(
        format_instructions=parser.get_format_instructions()
    )

    model = ChatOllama(model="qwen3:4b")
    chain = prompt_with_format | model | parser

    try:
        result = chain.invoke({})
        print(f"解析成功：{result}")
        print(f"name: {result.name}")
        print(f"price: {result.price}")
        print(f"stock: {result.stock}")

    except ValidationError as e:
        # Pydantic 验证错误
        print(f"数据验证失败：{e}")
        print()
        print("可能原因：")
        print("  1. AI 输出的 JSON 缺少必填字段")
        print("  2. 字段类型不匹配（如价格应该是数字）")
        print("  3. 数据格式不正确")

    except Exception as e:
        # 其他错误（如 JSON 解析失败）
        print(f"处理失败：{e}")
        print()
        print("提示：这可能是 JSON 格式问题，尝试：")
        print("  1. 在 Prompt 中强调'只输出 JSON'")
        print("  2. 使用 Few-Shot 示例")
        print("  3. 换用更强的模型")

    print()


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  Pydantic 强类型解析 - PydanticOutputParser")
    print("  说明：带类型验证的结构化输出")
    print("=" * 70 + "\n")

    print("【运行前检查】")
    print("  1. 已安装 Ollama 服务")
    print("  2. 已下载模型：ollama pull qwen3:4b")
    print()

    # 运行示例
    pydantic_basic()
    pydantic_output_parser()
    complex_data_model()
    news_summary()
    error_handling_and_validation()

    print("=" * 70)
    print("  接下来学习：05_memory/conversation_memory.py")
    print("=" * 70 + "\n")
