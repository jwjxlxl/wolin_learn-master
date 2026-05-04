# =============================================================================
# Prompt 模板基础
# =============================================================================
#  
# 用途：教学演示 - 使用 PromptTemplate 创建可复用的提示词模板
#
# 核心概念：
#   - Prompt Template = "填空题模板"
#   - 为什么需要模板？（复用、一致性、安全）
# =============================================================================

# -----------------------------------------------------------------------------
# 运行前检查
# -----------------------------------------------------------------------------
# 1. 已安装 Ollama 服务
# 2. 已下载模型：ollama pull qwen3.5:2b
# -----------------------------------------------------------------------------

# 设置 UTF-8 编码（Windows 专用）
import sys
import io
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer,
    encoding='utf-8',
    errors='replace',
    line_buffering=True
)


# =============================================================================
# 第一部分：理解 Prompt Template
# =============================================================================
"""
什么是 Prompt Template？

📝 普通 Prompt（每次手写）
   "请用英语解释机器学习"
   "请用中文解释机器学习"
   "请用日语解释机器学习"
   → 每次都重复写相似的内容

📋 Prompt Template（模板复用）
   template = "请用{language}解释{concept}"

   使用时填空：
   - language="英语", concept="机器学习"
   - language="中文", concept="深度学习"
   - language="日语", concept="神经网络"

为什么需要模板？
1. 复用：写一次，用多次
2. 一致性：保证格式统一
3. 安全：防止注入攻击
4. 易维护：修改一处，全局生效
"""


# =============================================================================
# 示例 1: 最简单的模板
# =============================================================================

def simplest_template():
    """
    最简单的 PromptTemplate 示例

    就像 Python 的 f-string，但可以复用
    """
    print("=" * 60)
    print("示例 1: 最简单的模板")
    print("=" * 60)

    from langchain_core.prompts import PromptTemplate

    # 创建模板
    # {variable} 是占位符，使用时填入实际值
    prompt = PromptTemplate.from_template(
        "请用{language}解释{concept}"
    )

    # 使用模板 - 方式 1：format() 方法
    formatted = prompt.format(language="英语", concept="机器学习")
    print(f"填好的 Prompt: {formatted}")
    print()

    # 调用模型
    from langchain_ollama import ChatOllama
    model = ChatOllama(model="qwen3.5:2b")
    response = model.invoke(formatted)
    print(f"AI 回复：{response.content}")
    print()


# =============================================================================
# 示例 2: 更复杂的模板
# =============================================================================

def complex_template():
    """
    包含多个变量的复杂模板

    可以设定角色、任务、格式要求等
    """
    print("=" * 60)
    print("示例 2: 复杂模板（多变量）")
    print("=" * 60)

    from langchain_core.prompts import PromptTemplate

    # 创建包含多个变量的模板
    prompt = PromptTemplate.from_template("""
你是一位{role}，请用{tone}的语气回答。

任务：{task}

要求：
1. 使用{language}语言
2. 长度不超过{max_length}字
3. {extra_requirement}

问题：{question}
""")

    # 填入具体值
    formatted = prompt.format(
        role="科学老师",
        tone="耐心",
        task="解释量子力学",
        language="中文",
        max_length=200,
        extra_requirement="用小学生的语言",
        question="什么是量子力学？"
    )

    print("填好的 Prompt:")
    print("-" * 40)
    print(formatted)
    print("-" * 40)
    print()

    # 调用模型
    from langchain_ollama import ChatOllama
    model = ChatOllama(model="qwen3.5:2b")
    response = model.invoke(formatted)
    print(f"AI 回复：{response.content}")
    print()


# =============================================================================
# 示例 3: 模板的多种创建方式
# =============================================================================

def different_ways_to_create_template():
    """
    PromptTemplate 的多种创建方式
    """
    print("=" * 60)
    print("示例 3: 模板的多种创建方式")
    print("=" * 60)

    from langchain_core.prompts import PromptTemplate

    # 方式 1: from_template (最常用)
    prompt1 = PromptTemplate.from_template("你好，{name}！")

    # 方式 2: 直接指定 template 和 input_variables
    prompt2 = PromptTemplate(
        template="你好，{name}！",
        input_variables=["name"]
    )

    # 两种方式等价
    print(f"方式 1: {prompt1.format(name='小明')}")
    print(f"方式 2: {prompt2.format(name='小明')}")
    print()


# =============================================================================
# 示例 4: 实用场景 - 翻译助手
# =============================================================================

def translation_assistant():
    """
    使用模板创建翻译助手
    """
    print("=" * 60)
    print("示例 4: 翻译助手（实用场景）")
    print("=" * 60)

    from langchain_core.prompts import PromptTemplate
    from langchain_ollama import ChatOllama

    # 创建翻译模板
    translate_prompt = PromptTemplate.from_template("""
你是一位专业翻译，请将以下文本从{source_lang}翻译成{target_lang}。

要求：
1. 保持原意
2. 翻译要自然流畅
3. 只输出翻译结果，不要解释

原文：{text}

译文：""")

    # 使用模板
    model = ChatOllama(model="qwen3.5:2b")

    # 翻译示例 1
    formatted = translate_prompt.format(
        source_lang="英语",
        target_lang="中文",
        text="Artificial intelligence will change the world."
    )
    print("【英译中】")
    response = model.invoke(formatted)
    print(f"译文：{response.content}")
    print()

    # 翻译示例 2
    formatted = translate_prompt.format(
        source_lang="中文",
        target_lang="英语",
        text="今天天气真好，适合出去散步。"
    )
    print("【中译英】")
    response = model.invoke(formatted)
    print(f"Translation: {response.content}")
    print()


# =============================================================================
# 示例 5: 实用场景 - 代码审查
# =============================================================================

def code_review_assistant():
    """
    使用模板创建代码审查助手
    """
    print("=" * 60)
    print("示例 5: 代码审查助手（实用场景）")
    print("=" * 60)



    # 创建代码审查模板
    review_prompt = PromptTemplate.from_template("""
你是一位经验丰富的{language}程序员，请审查以下代码。

审查要点：
1. 是否有语法错误
2. 是否有潜在 bug
3. 代码风格是否规范
4. 有什么改进建议

代码：
```{language}
{code}
```

审查结果：""")

    # 使用模板
    model = ChatOllama(model="qwen3.5:2b")

    code = """
def calculate_average(numbers):
    total = sum(numbers)
    return total / len(numbers)
"""

    formatted = review_prompt.format(
        language="Python",
        code=code
    )

    response = model.invoke(formatted)
    print(f"审查结果：{response.content}")
    print()


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  Prompt 模板基础")
    print("  说明：PromptTemplate 的使用方法")
    print("=" * 70 + "\n")

    print("【运行前检查】")
    print("  1. 已安装 Ollama 服务")
    print("  2. 已下载模型：ollama pull qwen3.5:2b")
    print()

    # 运行示例
    simplest_template()
    complex_template()
    different_ways_to_create_template()
    translation_assistant()
    code_review_assistant()

    print("=" * 70)
    print("  接下来学习：few_shot_prompt.py（Few-Shot 示例）")
    print("=" * 70 + "\n")
