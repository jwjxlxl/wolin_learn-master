# =============================================================================
# RAG 完整流程 - 端到端示例
# =============================================================================
#  
# 用途：教学演示 - 完整的 RAG 流程闭环
#
# 核心流程：
#   文档加载 → 文档切片 → Embedding 向量化 → 存入 Milvus → 检索 → RAG 问答
# =============================================================================

# -----------------------------------------------------------------------------
# 运行前检查
# -----------------------------------------------------------------------------
# 1. 已安装依赖：pip install -r ../requirements.txt
# 2. 已配置 Milvus（使用远程服务）
# 3. 配置 API Key：确保 .env 文件中配置了 ALIYUN_API_KEY 或 DASHSCOPE_API_KEY
# -----------------------------------------------------------------------------

# 设置 UTF-8 编码（Windows 专用）

# 加载 .env 文件中的环境变量
from dotenv import load_dotenv
load_dotenv()

# 导入 Milvus 配置（优先使用 Docker Milvus）
import sys
import os
# 添加父目录到路径，允许导入配置
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from milvus_config import MILVUS_URI



# =============================================================================
# 第一部分：完整流程概览
# =============================================================================
"""
RAG 完整流程（端到端）

┌─────────────────────────────────────────────────────────────────────┐
│                        RAG 完整流程                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  【准备阶段】                                                        │
│       │                                                             │
│       ▼                                                             │
│   ┌─────────┐                                                       │
│   │ 1. 文档加载 │  读取 TXT/PDF/Word 等格式文档                       │
│   └─────┬───┘                                                       │
│         │                                                           │
│         ▼                                                           │
│   ┌─────────┐                                                       │
│   │ 2. 文档切片 │  滑动窗口/Fixed/AI 切片                             │
│   └─────┬───┘                                                       │
│         │                                                           │
│         ▼                                                           │
│   ┌─────────┐                                                       │
│   │ 3. Embedding│  调用模型生成向量                                  │
│   └─────┬───┘                                                       │
│         │                                                           │
│         ▼                                                           │
│   ┌─────────┐                                                       │
│   │ 4. 存入 Milvus│  创建 Collection+Index，插入向量                 │
│   └─────┬───┘                                                       │
│         │                                                           │
│  【知识库构建完成】                                                   │
│         │                                                           │
│         ▼                                                           │
│  【检索问答阶段】                                                     │
│         │                                                           │
│         ▼                                                           │
│   ┌─────────┐                                                       │
│   │ 5. 用户提问 │  输入问题                                          │
│   └─────┬───┘                                                       │
│         │                                                           │
│         ▼                                                           │
│   ┌─────────┐                                                       │
│   │ 6. 问题 Embedding│  同样模型生成查询向量                         │
│   └─────┬───┘                                                       │
│         │                                                           │
│         ▼                                                           │
│   ┌─────────┐                                                       │
│   │ 7. 向量检索 │  Milvus 检索 Top-K 相似文档                         │
│   └─────┬───┘                                                       │
│         │                                                           │
│         ▼                                                           │
│   ┌─────────┐                                                       │
│   │ 8. Rerank │  (可选) 重排序提升精度                               │
│   └─────┬───┘                                                       │
│         │                                                           │
│         ▼                                                           │
│   ┌─────────┐                                                       │
│   │ 9. LLM 生成  │  基于检索结果生成答案                             │
│   └─────┬───┘                                                       │
│         │                                                           │
│         ▼                                                           │
│      输出答案 + 引用来源                                            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
"""


# =============================================================================
# 完整流程封装类
# =============================================================================

class RAGPipeline:
    """
    RAG 完整流程封装

    涵盖：文档加载 → 切片 → Embedding → 存储 → 检索 → 问答
    """

    def __init__(self,
                 milvus_uri=MILVUS_URI,
                 collection_name="rag_knowledge",
                 dim=1024):
        """
        初始化 RAG 流程

        参数:
            milvus_uri: Milvus 连接 URI
            collection_name: Collection 名称
            dim: 向量维度（默认 1024，对应 text-embedding-v4）
        """
        from pymilvus import MilvusClient

        self.milvus_uri = milvus_uri
        self.collection_name = collection_name
        self.dim = dim

        # 初始化 Milvus 客户端
        self.client = MilvusClient(uri=milvus_uri)

        # 初始化 Embedding 模型（使用阿里云 API）
        self._init_embedding_model()

        print(f"✓ RAGPipeline 初始化完成")
        print(f"  Milvus URI: {milvus_uri}")
        print(f"  Collection: {collection_name}")
        print(f"  Embedding: text-embedding-v4 ({dim}维)")
        print()

    def _init_embedding_model(self):
        """初始化 Embedding 模型"""
        from openai import OpenAI
        import os

        # 获取 API Key
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("ALIYUN_API_KEY")

        # 初始化 OpenAI 兼容客户端（使用阿里云百炼）
        self.embedding_model = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        print(f"✓ 阿里云 Embedding API 初始化完成 (text-embedding-v4)")

    def embed_text(self, text):
        """
        生成文本向量（使用阿里云百炼 Embedding API）

        参数:
            text: 输入文本
        返回:
            向量列表
        """
        response = self.embedding_model.embeddings.create(
            model="text-embedding-v4",
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding

    def embed_batch(self, texts, batch_size=32):
        """
        批量生成向量

        参数:
            texts: 文本列表
            batch_size: 批次大小
        返回:
            向量列表
        """
        # 阿里云 API 批量调用（最多 25 条/批）
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.embedding_model.embeddings.create(
                model="text-embedding-v4",
                input=batch,
                encoding_format="float"
            )
            for item in response.data:
                all_embeddings.append(item.embedding)
        return all_embeddings

    def load_documents(self, file_paths):
        """
        步骤 1: 加载文档

        参数:
            file_paths: 文件路径列表
        返回:
            文档内容列表
        """
        print("\n" + "=" * 60)
        print("步骤 1: 加载文档")
        print("=" * 60)

        documents = []
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                documents.append({
                    'content': content,
                    'source': file_path
                })
                print(f"✓ 已加载：{file_path} ({len(content)} 字符)")
            except Exception as e:
                print(f"✗ 加载失败 {file_path}: {e}")

        print(f"\n共加载 {len(documents)} 个文档")
        return documents

    def chunk_documents(self, documents, chunk_size=500, chunk_overlap=50, method="sliding_window"):
        """
        步骤 2: 文档切片

        参数:
            documents: 文档列表
            chunk_size: 切片大小
            chunk_overlap: 切片重叠
            method: 切片方法 ("sliding_window", "fixed", "ai")
        返回:
            切片列表
        """
        print("\n" + "=" * 60)
        print("步骤 2: 文档切片")
        print("=" * 60)
        print(f"切片方法：{method}")
        print(f"切片大小：{chunk_size} 字符")
        print(f"重叠大小：{chunk_overlap} 字符")

        all_chunks = []

        for doc in documents:
            content = doc['content']
            source = doc['source']

            if method == "sliding_window":
                # 滑动窗口切片
                step = chunk_size - chunk_overlap
                start = 0
                chunk_idx = 0
                while start < len(content):
                    end = min(start + chunk_size, len(content))
                    chunk_text = content[start:end].strip()
                    if chunk_text:
                        all_chunks.append({
                            'text': chunk_text,
                            'source': source,
                            'chunk_idx': chunk_idx,
                            'start': start,
                            'end': end
                        })
                        chunk_idx += 1
                    start += step
                    if end >= len(content):
                        break

            elif method == "fixed":
                # 固定切片
                for i in range(0, len(content), chunk_size):
                    chunk_text = content[i:i + chunk_size].strip()
                    if chunk_text:
                        all_chunks.append({
                            'text': chunk_text,
                            'source': source,
                            'chunk_idx': len(all_chunks)
                        })

            elif method == "ai":
                # AI 切片（简化版：按段落）
                paragraphs = content.split('\n\n')
                for para in paragraphs:
                    para = para.strip()
                    if para:
                        all_chunks.append({
                            'text': para,
                            'source': source,
                            'chunk_idx': len(all_chunks)
                        })

        print(f"\n✓ 切片完成，共 {len(all_chunks)} 个切片")
        return all_chunks

    def create_collection(self):
        """
        步骤 3: 创建 Collection（建表前先删表）
        """
        print("\n" + "=" * 60)
        print("步骤 3: 创建 Collection")
        print("=" * 60)

        # 建表前先删表
        if self.client.has_collection(self.collection_name):
            print(f"删除已存在的集合：{self.collection_name}")
            self.client.drop_collection(self.collection_name)

        # 创建 Collection
        self.client.create_collection(
            collection_name=self.collection_name,
            dimension=self.dim,
            auto_id=True,
            metric_type="COSINE"
        )

        print(f"✓ Collection 创建完成：{self.collection_name}")
        print(f"  向量维度：{self.dim}")
        print(f"  度量类型：COSINE")

    def create_index(self, index_type="FLAT"):
        """
        步骤 4: 创建索引（先删除已有索引）

        参数:
            index_type: 索引类型 (FLAT, IVF_FLAT, HNSW)
        """
        print("\n" + "=" * 60)
        print("步骤 4: 创建索引")
        print("=" * 60)
        print(f"索引类型：{index_type}")

        # 先删除已有索引（需要先释放 collection）
        try:
            index_list = self.client.list_indexes(collection_name=self.collection_name)
            if index_list:
                print(f"释放 Collection...")
                self.client.release_collection(collection_name=self.collection_name)
                print(f"删除已有索引...")
                for idx in index_list:
                    self.client.drop_index(
                        collection_name=self.collection_name,
                        index_name=idx
                    )
        except Exception as e:
            print(f"删除索引警告：{e}")

        index_params = self.client.prepare_index_params()

        if index_type == "IVF_FLAT":
            index_params.add_index(
                field_name="vector",
                index_type="IVF_FLAT",
                metric_type="COSINE",
                params={"nlist": 10}
            )
        elif index_type == "HNSW":
            index_params.add_index(
                field_name="vector",
                index_type="HNSW",
                metric_type="COSINE",
                params={"M": 16, "efConstruction": 200}
            )
        else:  # FLAT
            index_params.add_index(
                field_name="vector",
                index_type="FLAT",
                metric_type="COSINE"
            )

        self.client.create_index(
            collection_name=self.collection_name,
            index_params=index_params
        )

        print(f"✓ 索引创建完成")

    def store_embeddings(self, chunks, batch_size=32):
        """
        步骤 5: Embedding 向量化并存入 Milvus

        参数:
            chunks: 切片列表
            batch_size: 批次大小
        """
        print("\n" + "=" * 60)
        print("步骤 5: Embedding 向量化并存储")
        print("=" * 60)
        print(f"待处理切片数：{len(chunks)}")
        print(f"批次大小：{batch_size}")

        # 提取文本
        texts = [chunk['text'] for chunk in chunks]

        # 批量生成向量
        print("\n正在生成向量...")
        embeddings = self.embed_batch(texts, batch_size=batch_size)

        print(f"✓ 向量生成完成")

        # 构造插入数据
        data_to_insert = []
        for chunk, emb in zip(chunks, embeddings):
            data_to_insert.append({
                "content": chunk['text'],
                "source": chunk.get('source', 'unknown'),
                "chunk_idx": chunk.get('chunk_idx', 0),
                "vector": emb  # Milvus 默认向量字段名
            })

        # 批量插入 Milvus
        print(f"\n正在存入 Milvus...")
        result = self.client.insert(
            collection_name=self.collection_name,
            data=data_to_insert
        )

        print(f"✓ 存储完成，共 {result.get('insert_count', len(data_to_insert))} 条数据")

        return len(data_to_insert)

    def search(self, query, top_k=5, use_rerank=False):
        """
        步骤 6: 检索相关文档

        参数:
            query: 查询问题
            top_k: 返回结果数
            use_rerank: 是否 Rerank
        返回:
            检索结果列表
        """
        # 生成查询向量
        query_vector = self.embed_text(query)

        # 确保 Collection 已加载（存储数据后 collection 可能未加载）
        self.client.load_collection(collection_name=self.collection_name)

        # 向量检索
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=top_k * 2 if use_rerank else top_k,
            output_fields=["content", "source", "chunk_idx"],
            search_params={"metric_type": "COSINE"}
        )

        # 提取结果
        retrieved = []
        for hit in results[0]:
            retrieved.append({
                'content': hit['entity']['content'],
                'source': hit['entity'].get('source', ''),
                'chunk_idx': hit['entity'].get('chunk_idx', 0),
                'score': hit['distance']
            })

        # Rerank（可选）
        if use_rerank:
            retrieved = self._rerank(query, retrieved, top_k)

        return retrieved[:top_k]

    def _rerank(self, query, results, top_k):
        """Rerank 重排序"""
        try:
            from FlagEmbedding import FlagReranker
            reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=False)

            pairs = [[query, r['content']] for r in results]
            scores = reranker.compute_score(pairs)

            for i, score in enumerate(scores):
                results[i]['rerank_score'] = score

            results.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
            return results[:top_k]

        except ImportError:
            print("  ⚠ Rerank 需要安装：pip install FlagEmbedding")
            return results[:top_k]

    def ask(self, question, top_k=5, use_rerank=False):
        """
        步骤 7: RAG 问答

        参数:
            question: 问题
            top_k: 检索文档数
            use_rerank: 是否 Rerank
        返回:
            答案 + 来源
        """
        # 检索相关文档
        print(f"\n📖 正在检索相关知识...")
        results = self.search(question, top_k=top_k, use_rerank=use_rerank)

        if not results:
            return {
                'answer': "未找到相关信息，无法回答。",
                'sources': []
            }

        # 构建上下文
        context = "\n\n".join([f"[{i+1}] {r['content']}" for i, r in enumerate(results)])

        # 构建 Prompt
        prompt = f"""你是一个智能助手，请根据以下信息回答问题。

相关信息：
{context}

问题：{question}

要求：
1. 基于上述信息回答
2. 引用来源用 [1]、[2] 等标注
3. 如果信息不足，说明无法回答

回答："""

        # 模拟 LLM 回答（实际使用时调用真实 LLM）
        answer = self._mock_llm_answer(question, results)

        return {
            'answer': answer,
            'sources': results
        }

    def _mock_llm_answer(self, question, results):
        """模拟 LLM 回答（演示用）"""
        if len(results) == 0:
            return "抱歉，未找到相关信息。"

        # 简单拼接前 3 条结果作为"回答"
        contents = [r['content'][:100] + "..." for r in results[:3]]
        answer = "根据检索到的信息：\n\n" + "\n".join(contents) + "\n\n以上信息可能与您的问题相关。"
        return answer

    def run_full_pipeline(self, file_paths, chunk_size=500, chunk_overlap=50,
                          chunk_method="sliding_window", index_type="FLAT"):
        """
        运行完整 RAG 流程

        参数:
            file_paths: 文档文件路径列表
            chunk_size: 切片大小
            chunk_overlap: 切片重叠
            chunk_method: 切片方法
            index_type: 索引类型
        """
        print("\n" + "=" * 70)
        print("         RAG 完整流程演示")
        print("=" * 70)

        # 步骤 1: 加载文档
        documents = self.load_documents(file_paths)

        # 步骤 2: 文档切片
        chunks = self.chunk_documents(documents, chunk_size, chunk_overlap, chunk_method)

        # 步骤 3: 创建 Collection
        self.create_collection()

        # 步骤 4: 创建索引
        self.create_index(index_type)

        # 步骤 5: Embedding 并存储
        self.store_embeddings(chunks)

        print("\n" + "=" * 70)
        print("         ✓ 知识库构建完成！")
        print("=" * 70)

        # 测试检索
        print("\n现在可以进行问答测试了！")
        print("调用方法：pipeline.ask('你的问题')")

    def close(self):
        """关闭连接"""
        print("\n✓ RAGPipeline 已关闭")


# =============================================================================
# 使用示例
# =============================================================================

def demo_full_pipeline():
    """
    演示完整 RAG 流程
    """
    print("=" * 70)
    print("         RAG 完整流程演示")
    print("=" * 70)

    # 导入 Milvus 配置（使用远程 Milvus 服务）
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from milvus_config import MILVUS_URI

    # 初始化 Pipeline（使用阿里云 Embedding API）
    pipeline = RAGPipeline(
        milvus_uri=MILVUS_URI,  # 使用远程 Milvus 服务
        collection_name="demo_rag",
        dim=1024  # text-embedding-v4 维度
    )

    # 准备测试文件（使用绝对路径）
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, '..', 'data', 'txt')
    file_paths = [
        os.path.join(data_dir, "milvus_intro.txt"),
        os.path.join(data_dir, "ai_intro.txt"),
    ]

    # 运行完整流程
    pipeline.run_full_pipeline(
        file_paths=file_paths,
        chunk_size=300,
        chunk_overlap=50,
        chunk_method="sliding_window",
        index_type="FLAT"
    )

    # 测试问答
    print("\n" + "-" * 50)
    print("测试问答：")
    print("-" * 50)

    questions = [
        "Milvus 是什么？",
        "人工智能包括哪些技术？"
    ]

    for question in questions:
        print(f"\n🙋 用户：{question}")
        result = pipeline.ask(question, top_k=3)
        print(f"🤖 助手：{result['answer']}")

        if result['sources']:
            print("\n📚 引用来源:")
            for i, src in enumerate(result['sources'][:2], 1):
                print(f"  [{i}] {src['content'][:50]}...")

    pipeline.close()


# =============================================================================
# 流程图总结
# =============================================================================

def pipeline_summary():
    """
    完整流程总结
    """
    print()
    print("=" * 60)
    print("RAG 完整流程总结")
    print("=" * 60)

    print("""
┌─────────────────────────────────────────────────────────┐
│ 完整 RAG 流程（可复用模板）                               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  # 1. 初始化 Pipeline                                    │
│  from milvus_config import MILVUS_URI                   │
│  pipeline = RAGPipeline(                                │
│      milvus_uri=MILVUS_URI,  # 使用远程 Milvus 服务      │
│      embedding_model="local",                           │
│      dim=1024                                           │
│  )                                                      │
│                                                         │
│  # 2. 运行完整流程（构建知识库）                         │
│  pipeline.run_full_pipeline(                            │
│      file_paths=["doc1.txt", "doc2.txt"],               │
│      chunk_size=500,                                    │
│      chunk_overlap=50                                   │
│  )                                                      │
│                                                         │
│  # 3. 问答                                              │
│  result = pipeline.ask("你的问题？", top_k=5)           │
│  print(result['answer'])                                │
│                                                         │
└─────────────────────────────────────────────────────────┘

💡 关键点：
1. Embedding 模型要一致（文档和查询用同一模型）
2. 向量维度要与模型匹配
3. 切片大小影响检索效果（建议 300-500）
4. 建表前先删表，确保可重复运行
""")


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  RAG 完整流程 - 端到端示例")
    print("  说明：文档加载 → 切片 → Embedding → 存储 → 检索 → 问答")
    print("=" * 70 + "\n")

    print("【流程说明】")
    print("  本示例演示完整的 RAG 流程闭环")
    print("  每一步都有详细说明和日志输出")
    print()

    # 运行演示
    print("-" * 60)
    demo_full_pipeline()
    print()

    pipeline_summary()

    print()
    print("=" * 70)
    print("  RAG 完整流程学习完成！")
    print("  现在你可以：")
    print("  1. 修改 file_paths 加载自己的文档")
    print("  2. 调整切片参数优化效果")
    print("  3. 切换 Embedding 模型（local/alinyun）")
    print("  4. 接入真实 LLM 生成答案")
    print("=" * 70 + "\n")
