# =============================================================================
# 实战项目（一）— 文档问答机器人
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 综合运用 Prompt + Model + RAG 构建完整问答系统
#   ✅ 封装可交互的问答机器人类
# =============================================================================

import sys
import io

from utils import get_model

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)


class DocQnABot:
    """
    基于文档的问答机器人。

    整合了前面学到的所有技能:
    - PromptTemplate（定义问答格式）
    - Chat Model（调用 LLM）
    - RAG 检索（关键词匹配找相关资料）
    - 封装（类 + 方法，方便复用）

    用法:
      bot = DocQnABot()
      bot.add_document("Python 是...", source="wiki")
      answer = bot.ask("Python 是什么？")
    """

    def __init__(self):
        from langchain_core.prompts import PromptTemplate
        from langchain_ollama import ChatOllama
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.documents import Document

        self.docs: list[Document] = []
        # self.model = ChatOllama(model="qwen3.5:2b")
        self.model = get_model("qwen")
        self.prompt = PromptTemplate.from_template("""基于以下资料回答问题。只根据资料回答，不要编造。

资料:
{context}

问题: {question}
回答:""")
        self.chain = self.prompt | self.model | StrOutputParser()

    def add_document(self, content: str, source: str = "unknown"):
        """添加文档到知识库。"""
        from langchain_core.documents import Document
        self.docs.append(Document(page_content=content, metadata={"source": source}))

    def retrieve(self, query: str, k: int = 2) -> list:
        """关键词匹配检索。"""
        scored = []
        for doc in self.docs:
            score = sum(1 for w in query.lower() if w in doc.page_content.lower())
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:k]]

    def ask(self, question: str) -> str:
        """提问并获取答案。"""
        if not self.docs:
            return "知识库为空，请先添加文档。"
        docs = self.retrieve(question)
        if not docs:
            return "抱歉，没有找到相关的资料。"
        context = "\n".join(d.page_content for d in docs)
        return self.chain.invoke({"context": context, "question": question})


def demo():
    """演示完整流程：添加知识库 → 多轮问答。"""
    print(f"\n-- 文档问答机器人演示")

    bot = DocQnABot()

    # 添加知识库
    bot.add_document("Qwen 是阿里云开发的大语言模型系列。主要版本: Qwen-Plus、Qwen-Max。", "qwen")
    bot.add_document("Ollama 是本地运行大模型的工具，支持 Llama、Qwen 等开源模型，完全免费。", "ollama")
    bot.add_document("LangChain 是构建 AI 应用的框架。核心概念: Chain(链)、Agent(智能体)、RAG(检索增强生成)。", "langchain")
    bot.add_document("RAG 是检索增强生成技术。流程: 用户提问→检索知识库→拼接资料→LLM 生成答案。优点: 减少幻觉、提供依据。", "rag")

    questions = [
        "Qwen 有哪些版本？",
        "Ollama 是什么工具？",
        "LangChain 的核心概念有哪些？",
        "RAG 的工作流程是什么？",
    ]

    for q in questions:
        print(f"\n问: {q}")
        print(f"答: {bot.ask(q)}")


if __name__ == '__main__':
    print("\n>>> 08_project/qna_bot — 文档问答机器人\n")
    demo()
    # 接下来学习: research_assistant.py（研究助手）
