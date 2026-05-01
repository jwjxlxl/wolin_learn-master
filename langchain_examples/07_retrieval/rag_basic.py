# =============================================================================
# RAG 基础示例
# =============================================================================
#  
# 用途：教学演示 - 检索增强生成（RAG）完整流程
#
# 核心概念：
#   - RAG = 检索 + 生成
#   - 先搜索相关知识，再让 AI 回答
#   - 解决 AI 幻觉问题
# =============================================================================

# -----------------------------------------------------------------------------
# 运行前检查
# -----------------------------------------------------------------------------
# 1. 已安装 Ollama 服务
# 2. 已下载模型：ollama pull qwen3.5:2b
# 3. 理解向量存储概念（学习过 vector_store.py）
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
# 第一部分：理解 RAG
# =============================================================================
"""
什么是 RAG？

📖 定义
   RAG = Retrieval-Augmented Generation = 检索增强生成

🔍 为什么需要 RAG？

   问题 1：AI 会"幻觉"（胡说八道）
   - LLM 基于训练数据生成，不知道最新信息
   - 不知道你的私有数据（公司文档、产品手册等）

   问题 2：AI 无法引用具体来源
   - 回答没有依据，难以验证

   🎯 RAG 解决方案
   1. 先检索相关知识库
   2. 把知识 + 问题一起给 AI
   3. AI 基于检索内容回答

💡 生活化比喻
   RAG = "开卷考试"
   - 闭卷考试（普通 LLM）：凭记忆回答，可能记错
   - 开卷考试（RAG）：先查资料，再回答，更准确

📊 RAG 流程

   用户问题
       ↓
   检索相关文档（向量搜索）
       ↓
   拼接：问题 + 相关文档
       ↓
   LLM 生成答案
       ↓
   返回给用户
"""


# =============================================================================
# 示例 1: 简单的 RAG 示例（手动实现）
# =============================================================================

def simple_rag_example():
    """
    手动实现一个简单的 RAG 流程

    不用向量数据库，用关键词匹配演示概念
    """
    print("=" * 60)
    print("示例 1: 简单的 RAG 示例（手动实现）")
    print("=" * 60)

    from langchain_core.prompts import PromptTemplate
    from langchain_ollama import ChatOllama
    from langchain_core.output_parsers import StrOutputParser

    # 知识库（模拟向量检索结果）
    knowledge_base = [
        {"content": "Qwen 是阿里云开发的大语言模型系列，包括 Qwen-Plus、Qwen-Max 等版本。", "source": "qwen_intro"},
        {"content": "Ollama 是本地运行大模型的工具，支持多种开源模型。", "source": "ollama_intro"},
        {"content": "LangChain 是构建 AI 应用的框架，提供标准化工具和组件。", "source": "langchain_intro"},
        {"content": "DeepSeek 是深度求索开发的大语言模型，以高性价比著称。", "source": "deepseek_intro"},
        {"content": "RAG 是检索增强生成技术，先检索知识再让 AI 回答。", "source": "rag_intro"},
    ]

    # 简单的检索函数（关键词匹配）
    def retrieve(query, k=2):
        """检索与问题最相关的文档"""
        results = []
        for doc in knowledge_base:
            # 简单评分：查询词在内容中出现的次数
            score = sum(1 for word in query if word in doc["content"])
            if score > 0:
                results.append((score, doc))

        # 按相似度排序
        results.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in results[:k]]

    # 创建 RAG 链
    rag_prompt = PromptTemplate.from_template("""
请基于以下参考资料回答问题。如果资料中没有答案，就说不知道。

参考资料：
{context}

问题：{question}

回答：""")

    model = ChatOllama(model="qwen3.5:2b")
    rag_chain = rag_prompt | model | StrOutputParser()

    # 测试
    test_cases = [
        ("Qwen 是谁开发的？", ["阿里云", "Qwen"]),
        ("Ollama 有什么用？", ["Ollama", "本地"]),
        ("LangChain 是什么？", ["LangChain", "框架"]),
    ]

    for question, keywords in test_cases:
        print(f"问题：{question}")
        print()

        # 第 1 步：检索
        context_docs = retrieve(keywords)
        context = "\n".join([doc["content"] for doc in context_docs])

        print(f"【检索到的资料】")
        for i, doc in enumerate(context_docs, 1):
            print(f"  {i}. [{doc['source']}] {doc['content'][:50]}...")
        print()

        # 第 2 步：生成答案
        answer = rag_chain.invoke({"context": context, "question": question})
        print(f"【AI 回答】{answer}")
        print()
        print("-" * 40)
        print()


# =============================================================================
# 示例 2: 完整的 RAG 流程（使用向量库）
# =============================================================================

def complete_rag_pipeline():
    """
    完整的 RAG 流程

    使用 FAISS 向量数据库和真实的 Embedding
    """
    print("=" * 60)
    print("示例 2: 完整的 RAG 流程（使用向量库）")
    print("=" * 60)

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain_core.prompts import PromptTemplate
        from langchain_ollama import ChatOllama
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.documents import Document

        # 知识库文档
        documents = [
            Document(page_content="Qwen 是阿里云开发的大语言模型系列，包括 Qwen-Plus、Qwen-Max 等版本。", metadata={"source": "qwen"}),
            Document(page_content="Ollama 是本地运行大模型的工具，支持多种开源模型。", metadata={"source": "ollama"}),
            Document(page_content="LangChain 是构建 AI 应用的框架，提供标准化工具和组件。", metadata={"source": "langchain"}),
            Document(page_content="DeepSeek 是深度求索开发的大语言模型，以高性价比著称。", metadata={"source": "deepseek"}),
            Document(page_content="RAG 是检索增强生成技术，先检索知识再让 AI 回答。", metadata={"source": "rag"}),
            Document(page_content="Embedding 是将文本转换为数字向量的技术。", metadata={"source": "embedding"}),
            Document(page_content="FAISS 是 Facebook 开发的向量相似度搜索库。", metadata={"source": "faiss"}),
        ]

        print("加载嵌入模型...")
        embeddings = HuggingFaceEmbeddings(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )

        print("创建向量存储...")
        vectorstore = FAISS.from_documents(documents, embeddings)

        # 创建 RAG 链
        rag_prompt = PromptTemplate.from_template("""
请基于以下参考资料回答问题。如果资料中没有答案，就说不知道。

参考资料：
{context}

问题：{question}

回答：""")

        model = ChatOllama(model="qwen3.5:2b")

        # 完整的 RAG 链
        rag_chain = (
            # 第 1 步：检索
            {"context": lambda x: vectorstore.similarity_search(x["question"], k=3),
             "question": lambda x: x["question"]}
            # 第 2 步：生成
            | rag_prompt | model | StrOutputParser()
        )

        # 测试
        questions = [
            "Qwen 是谁开发的？",
            "Ollama 有什么作用？",
            "LangChain 是什么框架？",
            "什么是 RAG 技术？",
        ]

        for question in questions:
            print(f"\n问题：{question}")
            answer = rag_chain.invoke({"question": question})
            print(f"回答：{answer}")

    except ImportError as e:
        print("需要安装依赖：")
        print("  pip install faiss-cpu langchain-huggingface langchain-community")
        print(f"\n错误详情：{e}")
        print("\n跳过此示例")

    print()


# =============================================================================
# 示例 3: RAG 问答机器人（实用版）
# =============================================================================

def rag_qna_bot():
    """
    实用的 RAG 问答机器人

    封装成类，方便复用
    """
    print("=" * 60)
    print("示例 3: RAG 问答机器人（实用版）")
    print("=" * 60)

    from langchain_core.prompts import PromptTemplate
    from langchain_ollama import ChatOllama
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.documents import Document

    class SimpleRAGBot:
        """简单的 RAG 问答机器人"""

        def __init__(self, knowledge_base):
            """
            初始化机器人

            Args:
                knowledge_base: 知识库文档列表
            """
            self.knowledge_base = knowledge_base
            self.model = ChatOllama(model="qwen3.5:2b")

            # 创建 RAG 提示
            self.rag_prompt = PromptTemplate.from_template("""
请基于以下参考资料回答问题。

要求：
1. 只根据参考资料回答
2. 如果资料中没有答案，明确说"根据提供的资料，无法回答这个问题"
3. 回答简洁明了

参考资料：
{context}

问题：{question}

回答：""")

            self.rag_chain = self.rag_prompt | self.model | StrOutputParser()

        def retrieve(self, query, k=2):
            """检索相关文档（简单关键词匹配）"""
            results = []
            for doc in self.knowledge_base:
                score = sum(1 for word in query.lower() if word in doc.page_content.lower())
                if score > 0:
                    results.append((score, doc))

            results.sort(key=lambda x: x[0], reverse=True)
            return [doc for _, doc in results[:k]]

        def ask(self, question):
            """
            提问并获取答案

            Args:
                question: 用户问题

            Returns:
                答案字符串
            """
            # 检索
            context_docs = self.retrieve(question)

            if not context_docs:
                return "抱歉，没有找到相关的参考资料。"

            context = "\n".join([doc.page_content for doc in context_docs])

            # 生成答案
            answer = self.rag_chain.invoke({
                "context": context,
                "question": question
            })

            return answer

    # 创建知识库
    knowledge_base = [
        Document(page_content="Qwen 是阿里云开发的大语言模型系列。"),
        Document(page_content="Ollama 是本地运行大模型的工具。"),
        Document(page_content="LangChain 是构建 AI 应用的框架。"),
        Document(page_content="DeepSeek 是深度求索开发的大语言模型。"),
        Document(page_content="RAG 是检索增强生成技术。"),
    ]

    # 创建机器人
    bot = SimpleRAGBot(knowledge_base)

    # 测试
    questions = [
        "Qwen 是谁开发的？",
        "Ollama 有什么用？",
        "LangChain 是什么？",
    ]

    for question in questions:
        print(f"问：{question}")
        answer = bot.ask(question)
        print(f"答：{answer}")
        print()


# =============================================================================
# 示例 4: 带来源引用的 RAG
# =============================================================================

def rag_with_citations():
    """
    RAG 答案带来源引用

    方便验证答案的可信度
    """
    print("=" * 60)
    print("示例 4: 带来源引用的 RAG")
    print("=" * 60)

    from langchain_core.prompts import PromptTemplate
    from langchain_ollama import ChatOllama
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.documents import Document

    # 带来源的知识库
    knowledge_base = [
        Document(
            page_content="Qwen 是阿里云于 2023 年发布的大语言模型系列，包括基础版、Plus、Max 等多个版本。",
            metadata={"source": "阿里云文档", "url": "https://example.com/qwen"}
        ),
        Document(
            page_content="Ollama 是一个开源工具，允许用户在本地运行 Llama、Qwen 等开源大模型。",
            metadata={"source": "Ollama 官网", "url": "https://ollama.ai"}
        ),
        Document(
            page_content="LangChain 是 2022 年发布的 AI 应用开发框架，提供模块化组件。",
            metadata={"source": "LangChain 文档", "url": "https://langchain.com"}
        ),
        Document(
            page_content="DeepSeek 由深度求索公司开发，以高性价比和强大的推理能力著称。",
            metadata={"source": "DeepSeek 官网", "url": "https://deepseek.com"}
        ),
    ]

    # 检索函数
    def retrieve_with_citations(query, k=2):
        """检索并返回带来源的文档"""
        results = []
        for doc in knowledge_base:
            score = sum(1 for word in query if word in doc.page_content)
            if score > 0:
                results.append((score, doc))

        results.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in results[:k]]

    # 创建带引用的提示
    citation_prompt = PromptTemplate.from_template("""
请基于以下参考资料回答问题。

参考资料：
{context}

要求：
1. 只根据参考资料回答
2. 在答案末尾标注来源，格式：[来源：xxx]
3. 如果有多条来源，都标注出来

问题：{question}

回答：""")

    model = ChatOllama(model="qwen3.5:2b")
    rag_chain = citation_prompt | model | StrOutputParser()

    # 测试
    questions = [
        "Qwen 是什么时候发布的？",
        "Ollama 支持哪些模型？",
    ]

    for question in questions:
        print(f"问题：{question}")
        print()

        # 检索
        context_docs = retrieve_with_citations(question)

        # 构建带来源的上下文
        context_parts = []
        for doc in context_docs:
            context_parts.append(f"{doc.page_content} [来源：{doc.metadata['source']}]")

        context = "\n".join(context_parts)

        print("【参考资料】")
        for doc in context_docs:
            print(f"  - {doc.page_content[:50]}... [来源：{doc.metadata['source']}]")
        print()

        # 生成答案
        answer = rag_chain.invoke({"context": context, "question": question})
        print(f"【答案】{answer}")
        print()
        print("-" * 40)
        print()


# =============================================================================
# 示例 5: 交互式 RAG 问答
# =============================================================================

def interactive_rag_qna():
    """
    交互式 RAG 问答

    可以持续提问的 RAG 系统
    """
    print("=" * 60)
    print("示例 5: 交互式 RAG 问答")
    print("=" * 60)
    print("(输入 'quit' 退出)\n")

    from langchain_core.prompts import PromptTemplate
    from langchain_ollama import ChatOllama
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.documents import Document

    # 知识库
    knowledge_base = [
        Document(page_content="Qwen 是阿里云开发的大语言模型系列。"),
        Document(page_content="Ollama 是本地运行大模型的工具。"),
        Document(page_content="LangChain 是构建 AI 应用的框架。"),
        Document(page_content="DeepSeek 是深度求索开发的大语言模型。"),
        Document(page_content="RAG 是检索增强生成技术，先检索再回答。"),
        Document(page_content="Embedding 是将文本转换为数字向量的技术。"),
        Document(page_content="FAISS 是 Facebook 开发的向量搜索库。"),
        Document(page_content="Prompt 是给 AI 的指令或问题。"),
        Document(page_content="Chain 是将多个组件串联起来的功能组合。"),
        Document(page_content="Memory 用于保存对话历史，让 AI 有记忆。"),
    ]

    # 检索函数
    def retrieve(query, k=3):
        results = []
        for doc in knowledge_base:
            score = sum(1 for word in query.lower() if word in doc.page_content.lower())
            if score > 0:
                results.append((score, doc))
        results.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in results[:k]]

    # 创建 RAG 链
    rag_prompt = PromptTemplate.from_template("""
请基于以下参考资料回答问题。如果资料中没有相关信息，请如实告知。

参考资料：
{context}

问题：{question}

回答：""")

    model = ChatOllama(model="qwen3.5:2b")
    rag_chain = rag_prompt | model | StrOutputParser()

    print("RAG 问答机器人已启动！")
    print("可以问关于 AI、大模型、LangChain 等问题\n")

    while True:
        question = input("你：").strip()
        if question.lower() == 'quit':
            break

        # 检索
        context_docs = retrieve(question)

        if not context_docs:
            print("AI: 抱歉，没有找到相关的参考资料。")
            continue

        context = "\n".join([doc.page_content for doc in context_docs])

        # 生成答案
        answer = rag_chain.invoke({"context": context, "question": question})
        print(f"AI: {answer}")
        print()


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  RAG 基础示例 - Retrieval-Augmented Generation")
    print("  说明：检索增强生成完整流程")
    print("=" * 70 + "\n")

    print("【运行前检查】")
    print("  1. 已安装 Ollama 服务")
    print("  2. 已下载模型：ollama pull qwen3.5:2b")
    print("  3. 已学习向量存储概念（vector_store.py）")
    print()

    # 运行示例
    simple_rag_example()
    # complete_rag_pipeline()  # 需要额外依赖，按需运行
    rag_qna_bot()
    rag_with_citations()

    # interactive_rag_qna()  # 交互式，按需运行

    print("=" * 70)
    print("  接下来学习：08_project/qna_bot.py（实战项目）")
    print("=" * 70 + "\n")
