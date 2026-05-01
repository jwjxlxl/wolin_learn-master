# =============================================================================
# 什么是 LangChain
# =============================================================================
#  
# 用途：教学演示 - 用通俗语言解释 LangChain 是什么
#
# 核心概念：
#   - LangChain = "AI 应用的乐高积木"
#   - 为什么需要 LangChain：封装复杂性，提供标准化接口
#   - LangChain 能做什么：连接 LLM、管理记忆、处理文档、构建工作流
# =============================================================================

# -----------------------------------------------------------------------------
# 运行前检查
# -----------------------------------------------------------------------------
# 本文件不需要 API Key，直接运行即可
# 目的：理解概念，不需要实际调用模型
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
# 第一部分：没有 LangChain 时，我们怎么写 AI 应用？
# =============================================================================

def without_langchain_example():
    """
    模拟不使用 LangChain 的情况

    想象你要做一个"翻译助手"应用，需要：
    1. 调用 AI 模型
    2. 处理用户输入
    3. 格式化 Prompt
    4. 解析返回结果

    不用 LangChain，你可能要写很多"样板代码"...
    """
    print("=" * 60)
    print("没有 LangChain 时...")
    print("=" * 60)

    # 伪代码示例 - 展示复杂性
    print('''
    # 1. 手动拼接 Prompt（容易出错）
    prompt = f"""You are a translator. Translate the following text
    from {source_language} to {target_language}.
    Only output the translation, no explanations.

    Text: {user_input}

    Translation:"""

    # 2. 调用 API（每个服务商格式不同）
    import requests
    response = requests.post(
        api_url,
        headers={"Authorization": f"Bearer {api_key}"},
        json={"messages": [{"role": "user", "content": prompt}]}
    )

    # 3. 解析结果（不同 API 返回格式不同）
    if response.status_code == 200:
        result = response.json()
        translation = result["choices"][0]["message"]["content"]

        # 4. 清理输出（模型可能添加多余内容）
        translation = translation.strip()
        if translation.startswith('"'):
            translation = translation[1:-1]

    # 5. 想要添加"记忆"功能？自己写代码保存历史对话...
    # 6. 想要添加"文件上传"？自己解析 PDF/Word...

    print("看到了吗？需要写很多'基础设施'代码！")
    ''')


# =============================================================================
# 第二部分：LangChain 是什么？
# =============================================================================

def what_is_langchain():
    """
    用生活化比喻解释 LangChain

    核心比喻：
    - LangChain = "AI 应用的乐高积木"
    - 提供标准化的"积木块"，快速搭建 AI 应用
    """
    print("=" * 60)
    print("LangChain = AI 应用的乐高积木")
    print("=" * 60)

    print("""
    想象你要搭一个"房子"（AI 应用）：

    不用乐高（不用 LangChain）:
    ┌─────────────────────────┐
    │  要和泥、烧砖、砌墙...   │  ← 复杂、耗时
    └─────────────────────────┘

    用乐高（用 LangChain）:
    ┌─────────────────────────┐
    │  □□□  □□□  □□□  □□□    │  ← 现成的积木块
    │  □□□  □□□  □□□  □□□    │     快速拼装
    └─────────────────────────┘

    LangChain 提供的"积木块"：

    🟦 Model 块     - 调用 AI 模型（LLM、Chat Model）
    🟩 Prompt 块    - 管理提示词模板
    🟨 Memory 块    - 保存对话历史
    🟪 Parser 块    - 解析模型输出
    🟧 Chain 块     - 组合多个功能
    🟥 Retriever 块 - 检索外部知识
    """)


# =============================================================================
# 第三部分：LangChain 的核心组件
# =============================================================================

def langchain_components():
    """
    介绍 LangChain 的核心组件

    每个组件用一句话解释 + 生活化比喻
    """
    print("=" * 60)
    print("LangChain 的核心组件")
    print("=" * 60)

    components = {
        "Chat Model": {
            "作用": "调用 AI 模型，生成回复",
            "比喻": "会聊天的大脑",
            "代码示例": "ChatOpenAI(), ChatOllama()"
        },
        "Prompt Template": {
            "作用": "预定义 Prompt 模板",
            "比喻": "填空题模板",
            "代码示例": "PromptTemplate(template='用{语言}解释{概念}')"
        },
        "Output Parser": {
            "作用": "解析模型输出为结构化数据",
            "比喻": "翻译官（把 AI 的话转成程序能用的数据）",
            "代码示例": "PydanticOutputParser()"
        },
        "Memory": {
            "作用": "保存对话历史",
            "比喻": "记事本（AI 本身会失忆，需要外部记录）",
            "代码示例": "ConversationBufferMemory()"
        },
        "Chain": {
            "作用": "组合多个组件，形成完整功能",
            "比喻": "流水线（Prompt → Model → Parser）",
            "代码示例": "prompt | model | parser"
        },
        "Retriever": {
            "作用": "从外部数据源检索相关信息",
            "比喻": "图书管理员（帮你找相关资料）",
            "代码示例": "VectorStoreRetriever()"
        }
    }

    for name, info in components.items():
        print(f"\n🟦 {name}")
        print(f"   作用：{info['作用']}")
        print(f"   比喻：{info['比喻']}")
        print(f"   示例：{info['代码示例']}")


# =============================================================================
# 第四部分：LangChain 能做什么应用？
# =============================================================================

def langchain_use_cases():
    """
    展示 LangChain 的应用场景
    """
    print("\n" + "=" * 60)
    print("LangChain 能做什么应用？")
    print("=" * 60)

    use_cases = [
        {
            "名称": "智能客服机器人",
            "描述": "自动回答用户问题，支持多轮对话",
            "用到组件": "Chat Model + Memory + Prompt"
        },
        {
            "名称": "文档问答助手",
            "描述": "上传 PDF/Word 文档，基于文档内容回答问题",
            "用到组件": "RAG + Retriever + Chat Model"
        },
        {
            "名称": "数据分析助手",
            "描述": "上传 Excel，用自然语言查询数据",
            "用到组件": "Chat Model + Output Parser (JSON)"
        },
        {
            "名称": "研究助手",
            "描述": "自动搜索网络资料，生成研究报告",
            "用到组件": "Chain + Retriever + Chat Model"
        },
        {
            "名称": "代码助手",
            "描述": "解释代码、生成代码、调试 Bug",
            "用到组件": "Chat Model + Prompt Template"
        }
    ]

    for i, case in enumerate(use_cases, 1):
        print(f"\n{i}. {case['名称']}")
        print(f"   {case['描述']}")
        print(f"   用到：{case['用到组件']}")


# =============================================================================
# 第五部分：第一个 LangChain 代码预览
# =============================================================================

def preview_first_code():
    """
    预览下一个文件的代码，激发学习兴趣
    """
    print("\n" + "=" * 60)
    print("预告：下一个文件将写真正的代码！")
    print("=" * 60)

    print("""
    只需 3 行代码，就能调用 AI 模型：

    ┌─────────────────────────────────────────┐
    │ from langchain_ollama import ChatOllama │
    │                                         │
    │ model = ChatOllama(model="qwen3.5:2b")   │
    │                                         │
    │ response = model.invoke("你好！")        │
    └─────────────────────────────────────────┘

    这就是 LangChain 的魅力 - 简单、优雅、强大！

    接下来请运行：01_introduction/first_chain.py
    """)


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  什么是 LangChain？")
    print("  说明：用通俗语言解释 LangChain 的核心概念")
    print("=" * 70 + "\n")

    # 按顺序执行各个部分
    without_langchain_example()
    print()

    what_is_langchain()
    print()

    langchain_components()
    print()

    langchain_use_cases()
    print()

    preview_first_code()

    print("\n" + "=" * 70)
    print("  概念理解完成！接下来运行 first_chain.py 体验真正的代码")
    print("=" * 70 + "\n")
