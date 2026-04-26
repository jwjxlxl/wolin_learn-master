# =============================================================================
# 路由链
# =============================================================================
#  
# 用途：教学演示 - 使用 Router Chain 根据输入选择不同处理路径
#
# 核心概念：
#   - Router = "智能分线器"
#   - 根据问题类型选择不同处理路径
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
# 第一部分：理解 Router Chain
# =============================================================================
"""
什么是 Router Chain？

🔀 定义
   Router Chain = "路由链" = 根据输入自动选择处理路径

🎯 使用场景

   客服系统：
   用户问题 → 路由判断 → 技术问题→技术客服
                          售后问题→售后客服
                          投诉建议→人工客服

   助手应用：
   用户输入 → 路由判断 → 翻译→翻译链
                      总结→总结链
                      写代码→代码链

💡 生活化比喻
   Router Chain = "医院分诊台"
   病人 → 分诊护士判断 → 内科/外科/儿科...
"""


# =============================================================================
# 示例 1: 简单的 if-else 路由
# =============================================================================

def simple_if_else_router():
    """
    用简单的 if-else 实现路由

    最基础的路由方式
    """
    print("=" * 60)
    print("示例 1: 简单的 if-else 路由")
    print("=" * 60)

    from langchain_core.prompts import PromptTemplate
    from langchain_ollama import ChatOllama
    from langchain_core.output_parsers import StrOutputParser

    model = ChatOllama(model="qwen3:4b")
    parser = StrOutputParser()

    # 创建不同功能的 Chain
    translate_prompt = PromptTemplate.from_template(
        "请将以下文本翻译成英语：{text}"
    )
    translate_chain = translate_prompt | model | parser

    summarize_prompt = PromptTemplate.from_template(
        "请用一句话总结以下内容：{text}"
    )
    summarize_chain = summarize_prompt | model | parser

    explain_prompt = PromptTemplate.from_template(
        "请解释以下概念：{text}"
    )
    explain_chain = explain_prompt | model | parser

    # 路由函数
    def route(input_text, task_type):
        """根据任务类型选择 Chain"""
        if task_type == "translate":
            return translate_chain.invoke({"text": input_text})
        elif task_type == "summarize":
            return summarize_chain.invoke({"text": input_text})
        elif task_type == "explain":
            return explain_chain.invoke({"text": input_text})
        else:
            return "不支持的任务类型"

    # 测试
    texts = [
        ("你好，世界", "translate", "翻译"),
        ("人工智能是研究如何让计算机具有人类智能的学科...", "summarize", "总结"),
        ("机器学习", "explain", "解释"),
    ]

    for text, task, task_name in texts:
        print(f"【{task_name}】")
        result = route(text, task)
        print(f"结果：{result}")
        print()


# =============================================================================
# 示例 2: 智能路由（用 AI 判断类型）
# =============================================================================

def intelligent_router():
    """
    使用 AI 自动判断输入类型

    更智能的路由方式
    """
    print("=" * 60)
    print("示例 2: 智能路由（AI 判断类型）")
    print("=" * 60)

    from langchain_core.prompts import PromptTemplate
    from langchain_ollama import ChatOllama
    from langchain_core.output_parsers import StrOutputParser

    model = ChatOllama(model="qwen3:4b")

    # 第 1 步：用 AI 判断输入类型
    router_prompt = PromptTemplate.from_template("""
请判断以下用户输入属于哪种类型，只输出类型名称（translate/summarize/explain/other）：

用户输入：{input}

类型：""")

    router_chain = router_prompt | model | StrOutputParser()

    # 不同功能的 Chain
    translate_chain = (
        PromptTemplate.from_template("请将以下文本翻译成英语：{text}")
        | model | StrOutputParser()
    )

    summarize_chain = (
        PromptTemplate.from_template("请用一句话总结：{text}")
        | model | StrOutputParser()
    )

    explain_chain = (
        PromptTemplate.from_template("请解释：{text}")
        | model | StrOutputParser()
    )

    # 路由函数
    def smart_route(input_text):
        # 先判断类型
        task_type = router_chain.invoke({"input": input_text}).strip().lower()
        print(f"判断类型：{task_type}")

        # 根据类型选择 Chain
        if "translate" in task_type or "翻译" in task_type:
            return translate_chain.invoke({"text": input_text})
        elif "summarize" in task_type or "总结" in task_type:
            return summarize_chain.invoke({"text": input_text})
        elif "explain" in task_type or "解释" in task_type:
            return explain_chain.invoke({"text": input_text})
        else:
            return "我不知道该怎么处理这个输入..."

    # 测试
    inputs = [
        "把'你好'翻译成英语",
        "总结一下：人工智能是计算机科学的一个分支",
        "什么是深度学习？",
        "今天天气不错",  # other
    ]

    for input_text in inputs:
        print(f"输入：{input_text}")
        result = smart_route(input_text)
        print(f"结果：{result}")
        print()


# =============================================================================
# 示例 3: 客服路由系统
# =============================================================================

def customer_service_router():
    """
    模拟客服路由系统

    根据用户问题类型路由到不同处理流程
    """
    print("=" * 60)
    print("示例 3: 客服路由系统")
    print("=" * 60)

    from langchain_core.prompts import PromptTemplate
    from langchain_ollama import ChatOllama
    from langchain_core.output_parsers import StrOutputParser

    model = ChatOllama(model="qwen3:4b")

    # 路由判断
    router_prompt = PromptTemplate.from_template("""
请判断用户问题属于哪种类型：
- order: 订单相关问题
- shipping: 物流/发货问题
- refund: 退货/退款问题
- other: 其他问题

只输出类型名称（order/shipping/refund/other）

用户问题：{question}

类型：""")

    router_chain = router_prompt | model | StrOutputParser()

    # 不同问题的处理 Chain
    order_chain = (
        PromptTemplate.from_template("""
你是客服助手，请回答订单相关问题。

问题：{question}

回答：""")
        | model | StrOutputParser()
    )

    shipping_chain = (
        PromptTemplate.from_template("""
你是客服助手，请回答物流相关问题。

问题：{question}

回答：""")
        | model | StrOutputParser()
    )

    refund_chain = (
        PromptTemplate.from_template("""
你是客服助手，请回答退货退款相关问题。

问题：{question}

回答：""")
        | model | StrOutputParser()
    )

    other_chain = (
        PromptTemplate.from_template("""
你是客服助手，如果无法回答请引导用户联系人工客服。

问题：{question}

回答：""")
        | model | StrOutputParser()
    )

    # 路由函数
    def route_question(question):
        # 判断类型
        task_type = router_chain.invoke({"question": question}).strip().lower()
        print(f"问题类型：{task_type}")

        # 路由到对应 Chain
        chains = {
            "order": order_chain,
            "shipping": shipping_chain,
            "refund": refund_chain,
        }

        chain = chains.get(task_type, other_chain)
        return chain.invoke({"question": question})

    # 测试
    questions = [
        "我的订单什么时候发货？",
        "怎么申请退款？",
        "订单能修改地址吗？",
        "你们客服态度真差！",
    ]

    for question in questions:
        print(f"问题：{question}")
        answer = route_question(question)
        print(f"客服：{answer}")
        print()


# =============================================================================
# 示例 4: 多语言助手
# =============================================================================

def multilingual_assistant():
    """
    多语言助手

    自动识别输入语言并路由到对应处理
    """
    print("=" * 60)
    print("示例 4: 多语言助手")
    print("=" * 60)

    from langchain_core.prompts import PromptTemplate
    from langchain_ollama import ChatOllama
    from langchain_core.output_parsers import StrOutputParser

    model = ChatOllama(model="qwen3:4b")

    # 语言识别
    lang_prompt = PromptTemplate.from_template("""
请判断以下文本是什么语言，只输出语言名称（中文/英语/日语/其他）：

文本：{text}

语言：""")

    lang_chain = lang_prompt | model | StrOutputParser()

    # 不同语言的处理
    chinese_chain = (
        PromptTemplate.from_template("请用中文回复：{text}")
        | model | StrOutputParser()
    )

    english_chain = (
        PromptTemplate.from_template("Please respond in English: {text}")
        | model | StrOutputParser()
    )

    japanese_chain = (
        PromptTemplate.from_template("日本語で返信してください：{text}")
        | model | StrOutputParser()
    )

    # 路由函数
    def route_by_language(text):
        # 识别语言
        language = lang_chain.invoke({"text": text}).strip()
        print(f"识别语言：{language}")

        # 路由
        if "中文" in language:
            return chinese_chain.invoke({"text": text})
        elif "英语" in language or "English" in language:
            return english_chain.invoke({"text": text})
        elif "日语" in language or "日本語" in language:
            return japanese_chain.invoke({"text": text})
        else:
            return "抱歉，我还不会这种语言..."

    # 测试
    texts = [
        "你好，请介绍一下自己",
        "Hello, please introduce yourself",
        "こんにちは、自己紹介をお願いします",
    ]

    for text in texts:
        print(f"输入：{text}")
        response = route_by_language(text)
        print(f"回复：{response}")
        print()


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  路由链 - Router Chain")
    print("  说明：根据输入选择不同处理路径")
    print("=" * 70 + "\n")

    print("【运行前检查】")
    print("  1. 已安装 Ollama 服务")
    print("  2. 已下载模型：ollama pull qwen3:4b")
    print()

    # 运行示例
    simple_if_else_router()
    intelligent_router()
    customer_service_router()
    multilingual_assistant()

    print("=" * 70)
    print("  接下来学习：07_retrieval/document_loader.py")
    print("=" * 70 + "\n")
