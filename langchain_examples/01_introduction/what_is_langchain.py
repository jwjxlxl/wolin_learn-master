# =============================================================================
# 什么是 LangChain — 用通俗语言理解核心概念
# =============================================================================
#
# 本文件不需要安装任何依赖，直接运行即可。
# 目的：在动手写代码之前，先建立"LangChain 是什么、能做什么"的整体认知。
# =============================================================================

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)


def without_langchain():
    """
    演示不用 LangChain 时写 AI 应用有多繁琐。

    想象你要做一个"翻译助手"：调 API、拼 Prompt、解析结果、加记忆...
    每一步都要手写样板代码，换个服务商格式全变。
    """
    print(f"\n-- 没有 LangChain 时，写一个翻译助手需要...")

    print("""
    # 1. 手动拼接 Prompt（容易出错）
    prompt = f\"\"\"You are a translator. Translate from
    {source_language} to {target_language}.
    Text: {user_input}\"\"\"

    # 2. 不同服务商 API 格式不同，每种都要单独适配
    response = requests.post(url, headers={...}, json={...})

    # 3. 解析结果格式各不相同
    result = response.json()["choices"][0]["message"]["content"]

    # 4. 想加对话记忆？自己写历史管理
    # 5. 想加文件上传？自己解析 PDF / Word
    """)

    print("结论：每个 AI 应用都重复造这些轮子 → LangChain 帮你做掉这些")


def what_is_langchain():
    """
    用生活化比喻解释 LangChain = "AI 应用的乐高积木"。

    不用乐高（不用框架）：和泥、烧砖、砌墙 — 复杂、耗时
    用乐高（用 LangChain）：现成的积木块 — 快速拼装

    LangChain 提供的"积木块"：
      Model 块     — 调用 AI 模型
      Prompt 块    — 管理提示词模板
      Memory 块    — 保存对话历史
      Parser 块    — 解析模型输出
      Chain 块     — 串联多个功能
      Retriever 块 — 检索外部知识
      Tool 块      — 让 AI 调用外部工具
    """
    print(f"\n-- LangChain = AI 应用的乐高积木")

    components = {
        "Model（模型）":       {"比喻": "会聊天的大脑",        "代码": "ChatOllama(), ChatOpenAI()"},
        "Prompt（提示词模板）": {"比喻": "填空题的模板纸",      "代码": "PromptTemplate(template='用{语言}解释{概念}')"},
        "Output Parser（解析器）": {"比喻": "翻译官：把 AI 的话转成程序能用的数据", "代码": "StrOutputParser(), JsonOutputParser()"},
        "Memory（记忆）":      {"比喻": "记事本：AI 会失忆，需要外部记录", "代码": "InMemorySaver()"},
        "Chain（链）":        {"比喻": "流水线：点单→做菜→上菜", "代码": "prompt | model | parser"},
        "Retriever（检索器）":  {"比喻": "图书管理员：帮你找相关资料", "代码": "VectorStoreRetriever()"},
        "Tool（工具）":        {"比喻": "万能工具箱：AI 的手和脚", "代码": "get_weather(), calculator()"},
    }

    for name, info in components.items():
        print(f"  {name}")
        print(f"    比喻: {info['比喻']}")
        print(f"    示例: {info['代码']}")


def langchain_applications():
    """LangChain 能做什么应用？列举典型场景和用到的组件。"""
    print(f"\n-- LangChain 的典型应用场景")

    cases = [
        ("智能客服",   "自动回答用户问题，支持多轮对话",       "Chat Model + Memory + Prompt"),
        ("文档问答",   "上传 PDF，基于文档内容回答问题",        "RAG + Retriever + Chat Model"),
        ("数据分析",   "上传 Excel，用自然语言查询数据",        "Chat Model + Output Parser(JSON)"),
        ("研究助手",   "自动搜索资料，生成研究报告",            "Chain + Retriever + Chat Model"),
        ("代码助手",   "解释代码、生成代码、调试 Bug",          "Chat Model + Prompt Template"),
        ("Agent 智能体", "自主调用工具完成任务（查天气+计算+搜索）", "create_agent + Tools + Middleware"),
    ]

    for name, desc, comps in cases:
        print(f"  {name}: {desc}")
        print(f"    用到: {comps}")


if __name__ == '__main__':
    print("\n>>> 什么是 LangChain？—— 在写代码之前先建立整体认知\n")

    without_langchain()
    what_is_langchain()
    langchain_applications()

    print(f"\n-- 概念理解完成！接下来运行 first_chain.py 体验真正的代码")
