# =============================================================================
# 顺序链
# =============================================================================
#  
# 用途：教学演示 - 使用 Sequential Chain 串联多个处理步骤
#
# 核心概念：
#   - SequentialChain = "多步骤流水线"
#   - 第一步输出 → 第二步输入
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
sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer,
    encoding='utf-8',
    errors='replace',
    line_buffering=True
)


# =============================================================================
# 第一部分：理解 Sequential Chain
# =============================================================================
"""
什么是 Sequential Chain？

🔗 定义
   SequentialChain = "顺序链" = 多个步骤按顺序执行
   上一步的输出 → 下一步的输入

🎯 使用场景

   写文章：
   主题 → 生成大纲 → 写正文 → 起标题 → 完成

   做翻译：
   原文 → 翻译 → 润色 → 校对 → 完成

   分析问题：
   问题 → 收集信息 → 分析原因 → 给出建议 → 完成

💡 生活化比喻
   SequentialChain = "工厂流水线"
   原料 → 加工 1 → 加工 2 → 组装 → 成品
"""


# =============================================================================
# 示例 1: 手动串联 Chain
# =============================================================================

def manual_chain_chain():
    """
    手动串联多个 Chain

    第一个 Chain 的输出作为第二个的输入
    """
    print("=" * 60)
    print("示例 1: 手动串联 Chain")
    print("=" * 60)

    from langchain_core.prompts import PromptTemplate
    from langchain_ollama import ChatOllama
    from langchain_core.output_parsers import StrOutputParser

    model = ChatOllama(model="qwen3.5:2b")
    parser = StrOutputParser()

    # Chain 1: 生成大纲
    outline_prompt = PromptTemplate.from_template("""
请为"{topic}"主题生成一个简单大纲，包含 3 个要点。
格式：1.xxx 2.xxx 3.xxx

大纲：""")
    outline_chain = outline_prompt | model | parser

    # Chain 2: 根据大纲写内容
    content_prompt = PromptTemplate.from_template("""
请根据以下大纲写一段内容：
{outline}

内容：""")
    content_chain = content_prompt | model | parser

    # 手动串联
    topic = "人工智能"
    print(f"主题：{topic}")
    print()

    # 第 1 步：生成大纲
    outline = outline_chain.invoke({"topic": topic})
    print(f"【大纲】{outline}")
    print()

    # 第 2 步：写内容（用第 1 步的输出）
    content = content_chain.invoke({"outline": outline})
    print(f"【内容】{content}")
    print()


# =============================================================================
# 示例 2: 文章生成器（多步骤）
# =============================================================================

def article_generator():
    """
    完整的多步骤文章生成器

    主题 → 大纲 → 正文 → 标题
    """
    print("=" * 60)
    print("示例 2: 文章生成器（多步骤）")
    print("=" * 60)

    from langchain_core.prompts import PromptTemplate
    from langchain_ollama import ChatOllama
    from langchain_core.output_parsers import StrOutputParser

    model = ChatOllama(model="qwen3.5:2b")
    parser = StrOutputParser()

    # 步骤 1: 生成大纲
    outline_prompt = PromptTemplate.from_template("""
请为"{topic}"主题生成 3 个要点大纲。
大纲：""")
    outline_chain = outline_prompt | model | parser

    # 步骤 2: 写正文
    content_prompt = PromptTemplate.from_template("""
请根据以下大纲写一篇 200 字的短文：
{outline}

短文：""")
    content_chain = content_prompt | model | parser

    # 步骤 3: 起标题
    title_prompt = PromptTemplate.from_template("""
请为以下内容起一个吸引人的标题（10 字以内）：
{content}

标题：""")
    title_chain = title_prompt | model | parser

    # 执行流程
    topic = "机器学习入门"
    print(f"主题：{topic}")
    print("=" * 40)

    # 第 1 步
    outline = outline_chain.invoke({"topic": topic})
    print(f"步骤 1 - 大纲:\n{outline}")
    print()

    # 第 2 步
    content = content_chain.invoke({"outline": outline})
    print(f"步骤 2 - 正文:\n{content}")
    print()

    # 第 3 步
    title = title_chain.invoke({"content": content})
    print(f"步骤 3 - 标题:\n{title}")
    print()

    print("=" * 40)
    print(f"完成！文章标题：《{title}》")


# =============================================================================
# 示例 3: 问题分析助手
# =============================================================================

def problem_analysis_chain():
    """
    问题分析助手

    问题 → 分析原因 → 给出建议
    """
    print("=" * 60)
    print("示例 3: 问题分析助手")
    print("=" * 60)

    from langchain_core.prompts import PromptTemplate
    from langchain_ollama import ChatOllama
    from langchain_core.output_parsers import StrOutputParser

    model = ChatOllama(model="qwen3.5:2b")
    parser = StrOutputParser()

    # 步骤 1: 分析原因
    cause_prompt = PromptTemplate.from_template("""
请分析以下问题的可能原因（列出 3 条）：
问题：{problem}

可能原因：""")
    cause_chain = cause_prompt | model | parser

    # 步骤 2: 给出建议
    advice_prompt = PromptTemplate.from_template("""
针对以下问题及其原因，给出 3 条实用建议：
问题：{problem}
原因分析：{causes}

建议：""")
    advice_chain = advice_prompt | model | parser

    # 执行流程
    problem = "我最近总是失眠，睡不好觉"
    print(f"问题：{problem}")
    print()

    # 第 1 步：分析原因
    causes = cause_chain.invoke({"problem": problem})
    print(f"【原因分析】{causes}")
    print()

    # 第 2 步：给出建议
    advice = advice_chain.invoke({"problem": problem, "causes": causes})
    print(f"【实用建议】{advice}")
    print()


# =============================================================================
# 示例 4: 代码审查助手
# =============================================================================

def code_review_chain():
    """
    代码审查助手

    代码 → 找问题 → 给建议 → 改代码
    """
    print("=" * 60)
    print("示例 4: 代码审查助手")
    print("=" * 60)

    from langchain_core.prompts import PromptTemplate
    from langchain_ollama import ChatOllama
    from langchain_core.output_parsers import StrOutputParser

    model = ChatOllama(model="qwen3.5:2b")
    parser = StrOutputParser()

    # 步骤 1: 找问题
    issue_prompt = PromptTemplate.from_template("""
请审查以下 Python 代码，指出存在的问题：
```python
{code}
```

问题：""")
    issue_chain = issue_prompt | model | parser

    # 步骤 2: 给改进建议
    suggestion_prompt = PromptTemplate.from_template("""
针对以下代码及其问题，给出改进建议：
代码：{code}
问题：{issues}

建议：""")
    suggestion_chain = suggestion_prompt | model | parser

    # 步骤 3: 改写代码
    refactor_prompt = PromptTemplate.from_template("""
请根据以下建议，改写代码：
原代码：
```python
{code}
```
改进建议：{suggestions}

改进后的代码：""")
    refactor_chain = refactor_prompt | model | parser

    # 测试代码
    code = """
def calc(a,b):
    result=a/b
    return result
"""

    print(f"原代码:\n{code}")
    print()

    # 第 1 步：找问题
    issues = issue_chain.invoke({"code": code})
    print(f"【问题】{issues}")
    print()

    # 第 2 步：给建议
    suggestions = suggestion_chain.invoke({"code": code, "issues": issues})
    print(f"【建议】{suggestions}")
    print()

    # 第 3 步：改写
    new_code = refactor_chain.invoke({"code": code, "suggestions": suggestions})
    print(f"【改进后代码】{new_code}")


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  顺序链 - Sequential Chain")
    print("  说明：多步骤流水线处理")
    print("=" * 70 + "\n")

    print("【运行前检查】")
    print("  1. 已安装 Ollama 服务")
    print("  2. 已下载模型：ollama pull qwen3.5:2b")
    print()

    # 运行示例
    manual_chain_chain()
    article_generator()
    problem_analysis_chain()
    # code_review_chain()  # 需要代码审查可取消注释

    print("=" * 70)
    print("  接下来学习：router_chain.py（路由链）")
    print("=" * 70 + "\n")
