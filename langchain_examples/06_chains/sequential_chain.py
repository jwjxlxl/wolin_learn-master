# =============================================================================
# 顺序链 — 多个步骤串联，上一步的输出作为下一步的输入
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 把复杂任务拆成多个步骤，依次执行
#   ✅ 理解 Sequential Chain = "工厂流水线"
# =============================================================================

import sys
import io

from utils import get_model

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)


"""
什么是 Sequential Chain？

  简单 Chain: prompt | model | parser（一个步骤）
  顺序 Chain: 步骤1 → 步骤2 → 步骤3（多步骤流水线）

  例如写文章: 生成大纲 → 写正文 → 起标题
  例如分析问题: 找原因 → 给建议 → 写方案

  生活化比喻: 工厂流水线
    原料 → 加工1 → 加工2 → 组装 → 成品
    上一步的输出自动传到下一步
"""

from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser


# =============================================================================
# 示例 1: 手动串联 — 上一步输出 → 下一步输入
# =============================================================================

def manual_sequential():
    """
    用 Python 变量手动串联两个 Chain。

    第 1 步的输出存到变量，然后作为第 2 步的输入。
    这是最灵活的串联方式——每一步的中间结果你都能看到和保存。
    """
    print(f"\n-- 示例 1: 手动串联两个 Chain")

    model = ChatOllama(model="qwen3.5:9b")
    parser = StrOutputParser()

    # Chain 1: 生成大纲
    outline_chain = (
        PromptTemplate.from_template('请为"{topic}"主题生成 3 个要点的简单大纲。\n大纲:')
        | model | parser
    )

    # Chain 2: 根据大纲写内容
    content_chain = (
        PromptTemplate.from_template("请根据以下大纲写一段内容:\n{outline}\n\n内容:")
        | model | parser
    )

    # 第 1 步
    outline = outline_chain.invoke({"topic": "人工智能"})
    print(f"[大纲]\n{outline}")

    # 第 2 步 — 用第 1 步的输出
    content = content_chain.invoke({"outline": outline})
    print(f"\n[内容]\n{content}")


# =============================================================================
# 示例 2: 完整流程 — 主题 → 大纲 → 正文 → 标题
# =============================================================================

def article_generator():
    """
    三步生成一篇短文：大纲 → 正文 → 标题。

    展示了一个完整的"内容创作流水线"——从主题到成品文章。
    """
    print(f"\n-- 示例 2: 文章生成器（三步流水线）")

    # model = ChatOllama(model="qwen3.5:9b")
    model = get_model("qwen")
    parser = StrOutputParser()

    # 定义三个 Chain
    outline_c = (
        PromptTemplate.from_template('为"{topic}"生成 3 个要点大纲。\n大纲:')
        | model | parser
    )
    content_c = (
        PromptTemplate.from_template("根据大纲写一篇 150 字短文:\n{outline}\n\n短文:")
        | model | parser
    )
    title_c = (
        PromptTemplate.from_template("为以下内容起一个 10 字以内的标题:\n{content}\n\n标题:")
        | model | parser
    )

    topic = "机器学习入门"
    outline = outline_c.invoke({"topic": topic})
    content = content_c.invoke({"outline": outline})
    title = title_c.invoke({"content": content})

    print(f"主题: {topic}")
    print(f"大纲: {outline}")
    print(f"正文: {content}")
    print(f"标题: 《{title}》")


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n>>> 06_chains/sequential_chain — 顺序链\n")

    # manual_sequential()
    article_generator()

    # 接下来学习: router_chain.py（路由链）
