# =============================================================================
# rag_qna_api — RAG 问答 API 封装
# =============================================================================
#
# 教学内容：封装完整的 RAG 问答流程（检索 + 生成）
# 核心功能：检索相关文档、构建 Prompt 上下文、调用 LLM 生成答案、
#           返回答案 + 引用来源、多轮对话
# 前置知识：完成 rag_retrieval_api.py 的学习
# =============================================================================

import os
import sys
from dotenv import load_dotenv
load_dotenv(override=True)

from rag_examples.milvus_config import MILVUS_URI, DEFAULT_DIMENSION

# 确保能导入同目录下的 rag_retrieval_api（两个文件属于同一教学模块）
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from rag_retrieval_api import RAGRetriever


# =============================================================================
# RAGQnA 类实现
# =============================================================================

class RAGQnA:
    """
    RAG 问答系统

    封装检索 + 生成的完整流程。

    类结构：
        ├─ __init__()          # 初始化检索器和 LLM
        ├─ ask()               # 问主治接口
        ├─ chat()              # 多轮对话（带历史记忆）
        ├─ _build_prompt()     # 构建 Prompt
        ├─ _generate()         # 调用 LLM 生成
        └─ _extract_sources()  # 提取引用来源

    RAG 流程：
        1. 用户提问 → 检索相关文档
        2. 拼接 Prompt → 上下文 + 问题
        3. LLM 生成 → 基于上下文回答
        4. 返回结果 → 答案 + 引用来源
    """

    def __init__(self, milvus_uri=MILVUS_URI, collection_name="knowledge_base",
                 embedding_model=None, dim=DEFAULT_DIMENSION, llm_model=None):
        """
        初始化 RAG 问答系统

        参数:
            milvus_uri: Milvus 连接 URI
            collection_name: Collection 名称
            embedding_model: Embedding 模型
            dim: 向量维度（默认 1024，对应 text-embedding-v4）
            llm_model: LLM 模型名称（默认 qwen-plus）
        """
        self.milvus_uri = milvus_uri
        self.collection_name = collection_name
        self.llm_provider = "qwen"  # 固定使用阿里云百炼 Qwen
        self.llm_model = llm_model

        # 初始化检索器（复用 RAGRetriever）
        self.retriever = RAGRetriever(
            milvus_uri=milvus_uri,
            collection_name=collection_name,
            embedding_model=embedding_model,
            dim=dim
        )

        print(f"RAGQnA 初始化完成")
        print(f"  LLM 模型：{self.llm_model or 'qwen-plus（默认）'}")

    def ask(self, question, top_k=5, use_rerank=False, stream=False):
        """
        RAG 问主治接口

        参数:
            question: 用户问题
            top_k: 检索文档数量
            use_rerank: 是否使用 Rerank
            stream: 是否流式输出（暂未实现）
        返回:
            {
                'answer': str,          # 答案
                'sources': list,        # 引用来源
                'retrieved_docs': list  # 检索到的文档
            }
        """
        # 1. 检索相关文档
        print(f"\n正在检索相关知识...")
        retrieved_docs = self.retriever.search(
            query=question,
            top_k=top_k,
            use_rerank=use_rerank
        )

        if not retrieved_docs:
            return {
                'answer': "未找到相关信息，无法回答该问题。",
                'sources': [],
                'retrieved_docs': []
            }

        print(f"检索到 {len(retrieved_docs)} 篇相关文档")

        # 2. 构建 Prompt
        prompt = self._build_prompt(question, retrieved_docs)

        # 3. 调用 LLM 生成答案
        print(f"正在生成答案...")
        answer = self._generate(prompt, stream=stream)

        # 4. 提取来源
        sources = self._extract_sources(retrieved_docs)

        return {
            'answer': answer,
            'sources': sources,
            'retrieved_docs': retrieved_docs
        }

    def _build_prompt(self, question, retrieved_docs):
        """
        构建 RAG Prompt

        参数:
            question: 用户问题
            retrieved_docs: 检索到的文档列表
        返回:
            Prompt 字符串
        """
        # 构建上下文
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            content = doc['content']
            context_parts.append(f"[{i}] {content}")

        context = "\n".join(context_parts)

        # Prompt 模板
        prompt = f"""你是一个智能助手，请根据以下信息回答问题。

相关信息：
{context}

问题：{question}

要求：
1. 基于上述信息回答，不要编造未知内容
2. 如果信息不足以完整回答，请说明
3. 引用来源时用 [1]、[2] 等标注
4. 回答简洁明了，重点突出

回答："""

        return prompt

    def _generate(self, prompt, stream=False):
        """
        调用 LLM 生成答案（使用阿里云百炼 Qwen API）

        参数:
            prompt: 输入 Prompt
            stream: 是否流式输出（暂不支持）
        返回:
            生成的答案
        """
        return self._generate_qwen(prompt)

    def _generate_qwen(self, prompt):
        """调用阿里云 Qwen API（使用 OpenAI 兼容接口）"""
        try:
            from openai import OpenAI

            # 获取 API Key
            api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("ALIYUN_API_KEY")
            if not api_key:
                print("  未找到 API Key，请设置环境变量 DASHSCOPE_API_KEY 或 ALIYUN_API_KEY")
                return self._mock_generate(prompt)

            # 初始化客户端（使用阿里云百炼兼容模式）
            client = OpenAI(
                api_key=api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )

            model = self.llm_model or "qwen-plus"

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一个智能助手，请根据提供的信息回答问题。"},
                    {"role": "user", "content": prompt}
                ]
            )

            return response.choices[0].message.content

        except ImportError:
            print("  需要安装：pip install openai")
            return self._mock_generate(prompt)
        except Exception as e:
            print(f"  Qwen API 调用失败：{e}")
            return self._mock_generate(prompt)

    def _mock_generate(self, prompt):
        """模拟生成（用于演示，无需 API Key）"""
        print("  （演示模式：模拟 LLM 回答）")

        # 简单提取 Prompt 中的关键信息做模板匹配
        if "机器学习" in prompt:
            return "根据相关信息，机器学习是人工智能的核心技术，通过训练数据让计算机自动学习规律。机器学习需要数学基础，包括线性代数和概率统计。[1]"
        elif "深度学习" in prompt:
            return "根据相关信息，深度学习使用多层神经网络，在图像识别和自然语言处理领域取得成功。深度学习是机器学习的子集。[1][2]"
        elif "Milvus" in prompt or "向量数据库" in prompt:
            return "根据相关信息，Milvus 是一个开源的向量数据库，专门用于存储和搜索向量数据，支持亿级向量毫秒级检索。[1]"
        else:
            return "根据检索到的相关信息，我已经找到了与您的问题相关的内容。建议您参考上述来源获取详细信息。[1]"

    def _extract_sources(self, retrieved_docs):
        """
        提取来源信息

        参数:
            retrieved_docs: 检索到的文档列表
        返回:
            来源列表
        """
        sources = []
        for i, doc in enumerate(retrieved_docs, 1):
            content = doc['content']
            sources.append({
                'id': i,
                'content': content[:100] + "..." if len(content) > 100 else content,
                'score': doc.get('vector_score', 0)
            })
        return sources

    def chat(self, question, history=None, top_k=3):
        """
        多轮对话（带历史记忆）

        参数:
            question: 当前问题
            history: 对话历史列表，每项为 {'role': 'user'/'assistant', 'content': '...'}
            top_k: 检索文档数量
        返回:
            {'answer': str, 'sources': list}
        """
        # 构建带历史的上下文
        context = ""
        if history:
            context = "对话历史：\n"
            for msg in history[-5:]:  # 最近 5 轮
                role = "用户" if msg['role'] == 'user' else "助手"
                context += f"{role}: {msg['content']}\n"
            context += "\n"

        # 检索
        retrieved_docs = self.retriever.search(query=question, top_k=top_k)

        # 构建 Prompt
        prompt_parts = ["你是一个智能助手，请根据以下信息和对话历史回答问题。"]
        if context:
            prompt_parts.append(context)
        prompt_parts.append("相关信息：")
        for i, doc in enumerate(retrieved_docs, 1):
            prompt_parts.append(f"[{i}] {doc['content']}")
        prompt_parts.append(f"\n问题：{question}\n\n回答：")

        prompt = "\n".join(prompt_parts)

        # 生成
        answer = self._generate(prompt)

        return {
            'answer': answer,
            'sources': self._extract_sources(retrieved_docs)
        }

    def close(self):
        """关闭连接"""
        self.retriever.close()
        print("RAGQnA 已关闭")


# =============================================================================
# 示例 1: 基础问答演示
# =============================================================================

def demo_basic_qna():
    """演示基础 RAG 问答功能：初始化、添加文档、检索 + 生成"""
    print(f"\n-- 示例 1: 基础问答演示")

    # 初始化 RAG 问答系统
    qna = RAGQnA(
        milvus_uri=MILVUS_URI,
        collection_name="demo_qna",
        dim=DEFAULT_DIMENSION,
        llm_model="qwen-plus"
    )

    # 创建 Collection 并添加测试文档
    qna.retriever.create_collection()

    documents = [
        "机器学习是人工智能的核心技术，通过训练数据让计算机自动学习规律。机器学习需要数学基础，包括线性代数和概率统计。",
        "深度学习使用多层神经网络，在图像识别和自然语言处理领域取得成功。深度学习是机器学习的子集。",
        "自然语言处理让计算机理解和生成人类语言，应用包括机器翻译和智能客服。",
        "计算机视觉让计算机能够看懂图像，用于人脸识别和自动驾驶。",
        "Milvus 是一个开源的向量数据库，专门用于存储和搜索向量数据。",
    ]

    qna.retriever.add_documents(documents, chunk_size=100)

    # 问答测试
    questions = [
        "机器学习需要什么基础？",
        "深度学习有什么应用？"
    ]

    for question in questions:
        print(f"\n用户：{question}")

        result = qna.ask(question, top_k=3)

        print(f"助手：{result['answer']}")

        if result['sources']:
            print("\n引用来源:")
            for src in result['sources'][:2]:
                print(f"  [{src['id']}] {src['content']}")

    qna.close()


# =============================================================================
# 示例 2: Prompt 模板定制
# =============================================================================

def demo_custom_prompt():
    """演示不同 Prompt 模板的构建方式和效果对比"""
    print(f"\n-- 示例 2: Prompt 模板定制")

    question = "机器学习是什么？"
    docs = [
        {"content": "机器学习是人工智能的核心技术。"},
        {"content": "机器学习通过数据训练模型。"},
    ]

    # 模板 1: 简洁版
    print("\n模板 1: 简洁版")
    print("-" * 50)
    context = "\n".join([f"[{i+1}] {d['content']}" for i, d in enumerate(docs)])
    prompt1 = f"""信息：{context}
问题：{question}
回答："""
    print(prompt1)

    # 模板 2: 详细版
    print("\n模板 2: 详细版（带要求）")
    print("-" * 50)
    prompt2 = f"""你是一个专业的 AI 助手。请根据以下信息回答问题。

相关信息：
{context}

问题：{question}

要求：
1. 基于上述信息回答
2. 引用来源用 [1]、[2] 标注
3. 回答简洁明了

回答："""
    print(prompt2)

    # 模板 3: 结构化版
    print("\n模板 3: 结构化版（分点回答）")
    print("-" * 50)
    prompt3 = f"""请根据以下信息回答问题：{question}

参考信息：
{context}

请按以下格式回答：
|【核心定义】用一句话概括
|【关键特点】列出 2-3 个特点
|【应用场景】列出 1-2 个应用场景

回答："""
    print(prompt3)


# =============================================================================
# 示例 3: 完整 RAG 流程解析
# =============================================================================

def rag_pipeline_explained():
    """完整 RAG 流程解析（纯文档展示）"""
    print(f"\n-- 示例 3: 完整 RAG 流程解析")

    print("""
RAG 问答完整流程
--------------------------------------------------------------

  1. 用户提问
     → "机器学习需要什么基础？"

  2. 问题向量化
     → Embedding 模型编码 → [0.1, -0.5, 0.8, ...]

  3. 向量检索
     → 在 Milvus 中查找最相似的文档
     → 返回 Top-5 相关文档

  4. (可选) Rerank 重排序
     → 用 CrossEncoder 对结果重新打分
     → 提升相关性最高的文档排名

  5. 构建 Prompt
     → 拼接：系统指令 + 相关文档 + 用户问题

  6. LLM 生成答案
     → 基于 Prompt 生成回答
     → 支持流式输出

  7. 返回结果
     → 答案 + 引用来源 + 检索文档

关键设计点:

1. 检索质量决定上限
   - 如果检索不到相关文档，LLM 再强也没用
   - 建议：使用混合检索 + Rerank

2. Prompt 设计影响回答质量
   - 清晰的指令让 LLM 知道如何回答
   - 要求引用来源可以减少幻觉

3. 上下文长度限制
   - LLM 有 token 限制，不能传入太多文档
   - 建议：Top-K=5-10，或按 token 数控制

延迟分析:

  阶段       耗时        优化方法
  --------------------------------------------------
  检索       10-50ms     建立索引、批量查询
  Rerank     100-500ms   可跳过或用轻量模型
  LLM 生成   1-5s        流式输出、用小模型

关键概念总结:

  RAGQnA           封装检索 + 生成的完整流程
  ask()            问主治接口
  _build_prompt()  构建 RAG Prompt
  _generate()      调用 LLM 生成
  chat()           多轮对话
""")


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == "__main__":
    # 示例 1: 基础问答
    demo_basic_qna()

    # 示例 2: Prompt 模板定制
    # demo_custom_prompt()

    # 示例 3: 完整 RAG 流程解析
    # rag_pipeline_explained()
