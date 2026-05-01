# =============================================================================
# 问答机器人
# =============================================================================
#  
# 用途：实战项目 - 基于文档的智能问答机器人
#
# 核心概念：
#   - 综合应用 RAG 技术
#   - 构建实用的问答系统
#   - 处理真实场景问题
# =============================================================================

# -----------------------------------------------------------------------------
# 运行前检查
# -----------------------------------------------------------------------------
# 1. 已安装 Ollama 服务
# 2. 已下载模型：ollama pull qwen3.5:2b
# 3. 已完成前面章节的学习
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
# 第一部分：项目概述
# =============================================================================
"""
项目目标

🎯 构建一个基于文档的问答机器人
   - 用户可以上传自己的文档（产品手册、知识库等）
   - 机器人基于文档内容回答问题
   - 答案准确、有依据

📋 技术栈
   - LangChain：框架核心
   - Ollama：本地 LLM
   - FAISS：向量数据库
   - HuggingFace：Embedding 模型

📊 系统架构

   文档 → 加载 → 分块 → 向量化 → 存储
                              ↓
   用户问题 → 向量化 → 检索 → 拼接 → LLM → 答案
"""


# =============================================================================
# 示例 1: 简单的文档问答机器人
# =============================================================================

class SimpleDocQnABot:
    """
    简单的文档问答机器人

    适合小型知识库，快速上手
    """

    def __init__(self):
        """初始化机器人"""
        from langchain_core.prompts import PromptTemplate
        from langchain_ollama import ChatOllama
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.documents import Document

        self.documents = []
        self.model = ChatOllama(model="qwen3.5:2b")

        # RAG 提示模板
        self.rag_prompt = PromptTemplate.from_template("""
请基于以下参考资料回答问题。

参考资料：
{context}

要求：
1. 只根据参考资料回答，不要编造信息
2. 如果资料中没有答案，请说"抱歉，根据提供的资料，无法回答这个问题"
3. 回答要简洁明了，直接给出答案
4. 如果资料中有具体数据或步骤，请完整引用

问题：{question}

回答：""")

        self.rag_chain = self.rag_prompt | self.model | StrOutputParser()

    def add_document(self, content, source="unknown"):
        """
        添加文档到知识库

        Args:
            content: 文档内容
            source: 来源标识
        """
        from langchain_core.documents import Document

        doc = Document(page_content=content, metadata={"source": source})
        self.documents.append(doc)
        print(f"已添加文档：{source}（{len(content)} 字符）")

    def _retrieve(self, query, k=3):
        """
        检索相关文档

        简单关键词匹配，实际项目应该用向量检索
        """
        if not self.documents:
            return []

        results = []
        for doc in self.documents:
            # 计算相似度分数
            score = 0
            query_words = query.lower().split()
            for word in query_words:
                if len(word) > 1 and word in doc.page_content.lower():
                    score += 1
            if score > 0:
                results.append((score, doc))

        # 按分数排序
        results.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in results[:k]]

    def ask(self, question, show_context=False):
        """
        提问并获取答案

        Args:
            question: 问题
            show_context: 是否显示检索到的上下文

        Returns:
            答案字符串
        """
        if not self.documents:
            return "知识库为空，请先添加文档。"

        # 检索
        context_docs = self._retrieve(question)

        if not context_docs:
            return "抱歉，没有找到与问题相关的文档内容。"

        # 构建上下文
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            context_parts.append(f"[资料{i}]: {doc.page_content}")

        context = "\n\n".join(context_parts)

        # 显示检索结果（调试用）
        if show_context:
            print("\n【检索到的资料】")
            for i, doc in enumerate(context_docs, 1):
                print(f"{i}. 来源：{doc.metadata.get('source', 'unknown')}")
                print(f"   内容：{doc.page_content[:100]}...")
            print()

        # 生成答案
        answer = self.rag_chain.invoke({
            "context": context,
            "question": question
        })

        return answer

    def interactive_mode(self):
        """进入交互模式"""
        print("\n" + "=" * 50)
        print("问答机器人已启动！")
        print("输入问题获取答案，输入 'quit' 退出")
        print("=" * 50 + "\n")

        while True:
            question = input("你：").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                print("再见！")
                break

            if not question:
                continue

            print("思考中...", end="", flush=True)
            answer = self.ask(question, show_context=True)
            print(f"\nAI: {answer}\n")


def demo_simple_bot():
    """演示简单问答机器人"""
    print("=" * 60)
    print("示例 1: 简单的文档问答机器人")
    print("=" * 60)

    # 创建机器人
    bot = SimpleDocQnABot()

    # 添加知识库
    print("添加知识库...\n")

    bot.add_document(
        content="""
        Qwen 是阿里云开发的大语言模型系列。
        主要版本包括：Qwen、Qwen-Plus、Qwen-Max。
        Qwen-Plus 是中等规模版本，性价比高。
        Qwen-Max 是最大规模版本，能力最强。
        阿里云通过 API 提供服务。
        """,
        source="qwen_intro"
    )

    bot.add_document(
        content="""
        Ollama 是本地运行大模型的工具。
        支持 Llama、Qwen、Mistral 等开源模型。
        安装简单，一条命令即可运行。
        适合本地开发和测试。
        完全免费，开源。
        """,
        source="ollama_intro"
    )

    bot.add_document(
        content="""
        LangChain 是构建 AI 应用的框架。
        提供标准化组件：模型调用、记忆、检索等。
        支持 Python 和 JavaScript。
        核心概念：Chain（链）、Agent（智能体）、RAG（检索增强生成）。
        官网：https://langchain.com
        """,
        source="langchain_intro"
    )

    bot.add_document(
        content="""
        RAG 是检索增强生成（Retrieval-Augmented Generation）的缩写。
        工作流程：用户提问 → 检索知识库 → 拼接问题 + 资料 → LLM 生成答案。
        优点：减少幻觉、提供依据、知识可更新。
        应用场景：客服问答、文档助手、知识库查询。
        """,
        source="rag_intro"
    )

    print(f"\n知识库包含 {len(bot.documents)} 个文档\n")
    print("-" * 40 + "\n")

    # 测试问题
    test_questions = [
        "Qwen 有哪些版本？",
        "Ollama 是什么工具？",
        "LangChain 的核心概念有哪些？",
        "RAG 的工作流程是什么？",
        "Qwen-Max 的特点是什么？",
    ]

    for question in test_questions:
        print(f"问：{question}")
        answer = bot.ask(question)
        print(f"答：{answer}")
        print("-" * 40)
        print()


# =============================================================================
# 示例 2: 带向量检索的问答机器人
# =============================================================================

class VectorDocQnABot:
    """
    带向量检索的问答机器人

    使用 FAISS 向量数据库，支持语义搜索
    """

    def __init__(self, embedding_model=None):
        """
        初始化机器人

        Args:
            embedding_model: Embedding 模型名称
        """
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            from langchain_community.vectorstores import FAISS
            from langchain_core.prompts import PromptTemplate
            from langchain_ollama import ChatOllama
            from langchain_core.output_parsers import StrOutputParser

            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model or "paraphrase-multilingual-MiniLM-L12-v2"
            )
            self.vectorstore = None
            self.documents = []

            self.model = ChatOllama(model="qwen3.5:2b")

            self.rag_prompt = PromptTemplate.from_template("""
请基于以下参考资料回答问题。

参考资料：
{context}

要求：
1. 只根据参考资料回答
2. 如果资料中没有答案，请如实告知
3. 回答简洁明了

问题：{question}

回答：""")

            self.rag_chain = self.rag_prompt | self.model | StrOutputParser()

            print("向量检索机器人已初始化")

        except ImportError as e:
            print(f"需要安装依赖：{e}")
            print("运行：pip install faiss-cpu langchain-huggingface langchain-community")
            self.vectorstore = None

    def add_documents(self, documents):
        """
        批量添加文档

        Args:
            documents: Document 对象列表
        """
        if self.vectorstore is None:
            print("向量库未初始化")
            return

        self.documents.extend(documents)

        if self.documents:
            self.vectorstore = FAISS.from_documents(self.documents, self.embeddings)
            print(f"已添加 {len(documents)} 个文档，总计 {len(self.documents)} 个文档")
        else:
            print("没有文档可添加")

    def ask(self, question, k=3):
        """
        提问并获取答案

        Args:
            question: 问题
            k: 检索文档数量

        Returns:
            答案字符串
        """
        if not self.vectorstore:
            return "向量库未初始化"

        # 向量检索
        docs = self.vectorstore.similarity_search(question, k=k)

        if not docs:
            return "未找到相关文档"

        # 构建上下文
        context = "\n\n".join([doc.page_content for doc in docs])

        # 生成答案
        answer = self.rag_chain.invoke({
            "context": context,
            "question": question
        })

        return answer


def demo_vector_bot():
    """演示向量检索机器人"""
    print("=" * 60)
    print("示例 2: 带向量检索的问答机器人")
    print("=" * 60)

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document

        # 创建机器人
        bot = VectorDocQnABot()

        if bot.vectorstore is None:
            print("跳过此示例（需要安装依赖）")
            return

        # 准备文档
        documents = [
            Document(page_content="Python 是一种高级编程语言，由 Guido van Rossum 于 1989 年发明。"),
            Document(page_content="机器学习是让计算机从数据中学习规律的技术，无需显式编程。"),
            Document(page_content="深度学习使用神经网络，在图像识别和自然语言处理中效果很好。"),
            Document(page_content="大语言模型（LLM）是基于大量文本训练的深度学习模型。"),
            Document(page_content="Transformer 是深度学习模型架构，是 GPT 系列的基础。"),
            Document(page_content="Embedding 是将文本转换为数字向量的技术，用于语义搜索。"),
            Document(page_content="FAISS 是 Facebook 开发的向量相似度搜索库。"),
            Document(page_content="RAG 是检索增强生成，结合检索和生成的优势。"),
        ]

        # 添加文档
        bot.add_documents(documents)

        print()

        # 测试问题
        questions = [
            "Python 是什么时候发明的？",
            "什么是机器学习？",
            "深度学习用什么技术？",
            "如何把文字转成向量？",
        ]

        for question in questions:
            print(f"问：{question}")
            answer = bot.ask(question)
            print(f"答：{answer}")
            print("-" * 40)

    except Exception as e:
        print(f"示例运行失败：{e}")
        print("请确保已安装所需依赖")

    print()


# =============================================================================
# 示例 3: 客服问答机器人（完整版）
# =============================================================================

class CustomerServiceBot:
    """
    客服问答机器人

    完整的客服场景，包含多轮对话支持
    """

    def __init__(self):
        """初始化客服机器人"""
        from langchain_core.prompts import PromptTemplate
        from langchain_ollama import ChatOllama
        from langchain_core.output_parsers import StrOutputParser
        from langchain_classic.memory import ConversationBufferMemory
        from langchain_core.documents import Document

        # 知识库
        self.knowledge_base = [
            Document(
                page_content="发货时间：工作日 24 小时内发货，周末和节假日顺延。",
                metadata={"category": "shipping"}
            ),
            Document(
                page_content="运费说明：满 99 元包邮，不满收取 10 元运费。",
                metadata={"category": "shipping"}
            ),
            Document(
                page_content="退货政策：7 天无理由退货，商品需保持完好。",
                metadata={"category": "refund"}
            ),
            Document(
                page_content="退款时间：退货入库后 3-5 个工作日退款。",
                metadata={"category": "refund"}
            ),
            Document(
                page_content="客服时间：工作日 9:00-18:00 在线服务。",
                metadata={"category": "service"}
            ),
            Document(
                page_content="联系方式：客服邮箱 support@example.com。",
                metadata={"category": "service"}
            ),
        ]

        # 记忆
        self.memory = ConversationBufferMemory(return_messages=True)

        # 模型
        self.model = ChatOllama(model="qwen3.5:2b")

        # 提示模板
        self.prompt = PromptTemplate.from_template("""
你是电商客服助手，请友好、专业地回答用户问题。

知识库：
{context}

对话历史：
{history}

用户问题：{question}

回答：""")

        self.chain = self.prompt | self.model | StrOutputParser()

    def _retrieve(self, query, k=3):
        """检索相关文档"""
        results = []
        for doc in self.knowledge_base:
            score = sum(1 for word in query.lower() if word in doc.page_content.lower())
            if score > 0:
                results.append((score, doc))
        results.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in results[:k]]

    def chat(self, question):
        """
        对话

        Args:
            question: 用户问题

        Returns:
            回复字符串
        """
        # 检索
        context_docs = self._retrieve(question)
        context = "\n".join([doc.page_content for doc in context_docs])

        # 获取历史
        history = self.memory.load_memory_variables({}).get("chat_history", "")

        # 生成回复
        response = self.chain.invoke({
            "context": context,
            "history": history,
            "question": question
        })

        # 保存对话
        self.memory.save_context(
            {"input": question},
            {"output": response}
        )

        return response


def demo_customer_service():
    """演示客服机器人"""
    print("=" * 60)
    print("示例 3: 客服问答机器人（完整版）")
    print("=" * 60)

    bot = CustomerServiceBot()

    # 模拟对话
    conversations = [
        "什么时候发货？",
        "运费怎么算？",
        "可以退货吗？",
        "退款多久到账？",
    ]

    print("模拟对话：\n")

    for question in conversations:
        print(f"用户：{question}")
        response = bot.chat(question)
        print(f"客服：{response}")
        print()

    # 带上下文的追问
    print("-" * 40)
    print("追问测试：\n")

    print("用户：我周六下单什么时候发货？")
    response = bot.chat("我周六下单什么时候发货？")
    print(f"客服：{response}")
    print()


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  问答机器人 - Document Q&A Bot")
    print("  说明：基于文档的智能问答系统")
    print("=" * 70 + "\n")

    print("【运行前检查】")
    print("  1. 已安装 Ollama 服务")
    print("  2. 已下载模型：ollama pull qwen3.5:2b")
    print("  3. 已完成前面章节的学习")
    print()

    # 运行示例
    demo_simple_bot()
    # demo_vector_bot()  # 需要额外依赖，按需运行
    demo_customer_service()

    # 交互模式（按需运行）
    # bot = SimpleDocQnABot()
    # ... 添加文档 ...
    # bot.interactive_mode()

    print("=" * 70)
    print("  接下来学习：research_assistant.py（研究助手）")
    print("=" * 70 + "\n")
