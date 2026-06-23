# =============================================================================
# RAG 基础 — 检索增强生成：先查资料，再回答问题
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 理解 RAG = 检索 + 生成的完整流程
#   ✅ 手写一个最简单的 RAG 问答系统
#   ✅ 封装 RAG 机器人类
# =============================================================================

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)


"""
什么是 RAG？

  普通 LLM: 凭"记忆"回答 → 可能编造（幻觉）、不知道私有数据
  RAG:      先查资料 → 把资料 + 问题一起给 LLM → 基于资料回答

  流程: 用户问题 → 检索相关文档 → 拼接(文档+问题) → LLM 生成答案

  生活化比喻: RAG = "开卷考试"
    闭卷（普通 LLM）: 凭记忆回答，可能记错
    开卷（RAG）:     先翻书查资料，再写答案，更准确、可溯源
"""

from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from utils import get_model


# =============================================================================
# 示例 1: 最简单的 RAG — 手动检索 + 生成
# =============================================================================

def simple_rag():
    """
    不用向量数据库，用纯 Python 关键词匹配模拟检索过程。

    这样你能看清 RAG 的每一个步骤：
    1. 准备知识库（Document 列表）
    2. 关键词匹配检索
    3. 拼接上下文 + 问题
    4. LLM 基于上下文生成答案
    """
    print(f"\n-- 示例 1: 最简单的 RAG（关键词检索）")

    # 知识库
    knowledge = [
        {"content": "Qwen 是阿里云开发的大语言模型系列，包括 Qwen-Plus、Qwen-Max。", "source": "qwen"},
        {"content": "Ollama 是本地运行大模型的工具，支持多种开源模型。", "source": "ollama"},
        {"content": "LangChain 是构建 AI 应用的框架，提供标准化工具。", "source": "langchain"},
        {"content": "RAG 是检索增强生成技术，先检索知识再让 AI 回答。", "source": "rag"},
    ]

    def retrieve(query: str, k: int = 2):
        """关键词匹配：query 中的词在文档中出现越多，评分越高。"""
        scored = []
        for doc in knowledge:
            score = sum(1 for w in query if w in doc["content"])
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:k]]

    # RAG Chain
    model = ChatOllama(model="qwen3.5:2b")
    rag_prompt = PromptTemplate.from_template("""基于以下资料回答问题。如果资料中没有答案，就说不知道。

资料:
{context}

问题: {question}
回答:""")
    rag_chain = rag_prompt | model | StrOutputParser()

    # 测试
    question = "Qwen 是谁开发的？"
    docs = retrieve(question)
    context = "\n".join(d["content"] for d in docs)

    print(f"问题: {question}")
    print(f"检索到 {len(docs)} 篇资料:")
    for d in docs:
        print(f"  [{d['source']}] {d['content'][:40]}...")
    print(f"\n回答: {rag_chain.invoke({'context': context, 'question': question})}")


# =============================================================================
# 示例 2: 封装 RAG 机器人类
# =============================================================================

class SimpleRAGBot:
    """
    一个可直接使用的 RAG 问答机器人。

    用法:
      bot = SimpleRAGBot(knowledge_base)
      bot.ask("你们公司地址在哪？")
    """

    def __init__(self, knowledge_base: list[Document]):
        self.knowledge = knowledge_base
        self.model = get_model("qwen")
        self.prompt = PromptTemplate.from_template("""基于以下资料回答问题。如果资料中没有答案，就说不知道。

资料:
{context}

问题: {question}
回答:""")
        self.chain = self.prompt | self.model | StrOutputParser()

    def retrieve(self, query: str, k: int = 2) -> list[Document]:
        scored = []
        for doc in self.knowledge:
            score = sum(1 for w in query.lower() if w in doc.page_content.lower())
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:k]]

    def ask(self, question: str) -> str:
        docs = self.retrieve(question)
        if not docs:
            return "抱歉，没有找到相关的资料。"
        context = "\n".join(doc.page_content for doc in docs)
        return self.chain.invoke({"context": context, "question": question})


def rag_bot_demo():
    """演示 SimpleRAGBot 的使用。"""
    print(f"\n-- 示例 2: RAG 机器人类")

    kb = [
        Document(page_content="Qwen 是阿里云开发的大语言模型系列。"),
        Document(page_content="Ollama 是本地运行大模型的工具。"),
        Document(page_content="LangChain 是构建 AI 应用的框架。"),
        Document(page_content="RAG 是检索增强生成技术。"),
    ]

    bot = SimpleRAGBot(kb)
    for q in ["Qwen 是谁开发的？", "Ollama 有什么用？", "LangChain 是什么？"]:
        print(f"  问: {q}")
        print(f"  答: {bot.ask(q)}\n")


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n>>> 07_retrieval/rag_basic — RAG 基础\n")

    # simple_rag()
    rag_bot_demo()

    # 接下来学习: 08_project/qna_bot.py（实战问答机器人）
