# =============================================================================
# rag_retrieval_api — RAG 检索 API 封装
# =============================================================================
#
# 教学内容：封装 RAG 系统的检索功能，提供易用的 API
# 核心功能：文档加载与切片、向量生成与插入、多种检索方式（向量/关键字/混合）、
#           可选 Rerank 重排序
# 前置知识：完成 03_retrieval_methods/ 的学习
# 后续学习：rag_qna_api.py（RAG 问答 API）
# =============================================================================

import os
import sys
from dotenv import load_dotenv
load_dotenv()

from rag_examples.milvus_config import MILVUS_URI, DEFAULT_DIMENSION


# =============================================================================
# RAGRetriever 类实现
# =============================================================================

class RAGRetriever:
    """
    RAG 检索器

    封装 Milvus 向量库操作，提供文档检索功能。

    类结构：
        ├─ __init__()          # 初始化连接和模型
        ├─ add_documents()     # 添加文档到知识库
        ├─ search()            # 检索相关文档
        ├─ create_collection() # 创建 Collection
        ├─ query()             # 标量查询
        ├─ _embed()            # 生成向量（内部方法）
        └─ _rerank_results()   # Rerank 重排序（内部方法）

    设计原则：
        1. 低维度封装：暴露底层 API，便于教学理解
        2. 原子函数：每个方法独立可运行
        3. 灵活配置：支持多种检索方式和参数
        4. 友好错误：捕获常见错误并给出提示
    """

    def __init__(self, milvus_uri=MILVUS_URI, collection_name="knowledge_base",
                 embedding_model=None, dim=DEFAULT_DIMENSION):
        """
        初始化 RAG 检索器

        参数:
            milvus_uri: Milvus 连接 URI
            collection_name: Collection 名称
            embedding_model: Embedding 模型名称或路径（本地 sentence-transformers 模型）
            dim: 向量维度（默认 1024，对应 text-embedding-v4）
        """
        from pymilvus import MilvusClient

        self.milvus_uri = milvus_uri
        self.collection_name = collection_name
        self.dim = dim
        self.embedding_model = embedding_model
        self._model = None

        # 连接 Milvus
        self.client = MilvusClient(uri=milvus_uri)

        print(f"RAGRetriever 初始化完成")
        print(f"  Milvus URI: {milvus_uri}")
        print(f"  Collection: {collection_name}")

    def _init_embedding_model(self):
        """延迟加载 Embedding 模型（本地 sentence-transformers 模式）"""
        if self._model is None and self.embedding_model:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.embedding_model)
                print(f"Embedding 模型加载完成：{self.embedding_model}")
            except ImportError:
                print("需要安装：pip install sentence-transformers")
            except Exception as e:
                print(f"模型加载失败：{e}")

    def _embed(self, texts):
        """
        生成文本向量（使用阿里云百炼 Embedding API）

        参数:
            texts: 文本或文本列表
        返回:
            向量列表
        """
        from openai import OpenAI

        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("ALIYUN_API_KEY")

        if isinstance(texts, str):
            texts = [texts]

        # 没有 API Key 时使用模拟向量（降级方案，演示模式）
        if not api_key:
            print("  未找到 API Key，使用模拟向量（演示模式）")
            import random
            random.seed(42)
            return [[random.random() for _ in range(self.dim)] for _ in texts]

        try:
            client = OpenAI(
                api_key=api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )

            # 批量处理（API 限制每次最多 25 条）
            all_embeddings = []
            batch_size = 25

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = client.embeddings.create(
                    model="text-embedding-v4",
                    input=batch,
                    encoding_format="float"
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

            print(f"已生成 {len(all_embeddings)} 个 Embedding 向量")
            return all_embeddings

        except Exception as e:
            print(f"  Embedding API 调用失败：{e}，使用模拟向量")
            import random
            random.seed(42)
            return [[random.random() for _ in range(self.dim)] for _ in texts]

    def create_collection(self, schema_fields=None):
        """
        创建 Collection

        参数:
            schema_fields: 字段定义列表（可选，不传则使用默认 content + vector 字段）
        """
        from pymilvus import FieldSchema, CollectionSchema, DataType

        if self.client.has_collection(self.collection_name):
            print(f"集合 {self.collection_name} 已存在，先删除...")
            self.client.drop_collection(self.collection_name)

        if schema_fields:
            # 自定义字段
            schema = CollectionSchema(fields=schema_fields)
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                metric_type="COSINE"
            )
        else:
            # 默认字段（content + vector）
            self.client.create_collection(
                collection_name=self.collection_name,
                dimension=self.dim,
                auto_id=True,
                metric_type="COSINE"
            )

        print(f"Collection 创建完成：{self.collection_name}")

    def add_documents(self, documents, chunk_size=500, chunk_overlap=50):
        """
        添加文档到知识库

        参数:
            documents: 文档列表或文档字符串
            chunk_size: 切片大小
            chunk_overlap: 切片重叠
        返回:
            插入的文档数量
        """
        if isinstance(documents, str):
            documents = [documents]

        # 切片
        all_chunks = []
        for doc in documents:
            chunks = self._chunk_text(doc, chunk_size, chunk_overlap)
            all_chunks.extend(chunks)

        print(f"文档切片完成，共 {len(all_chunks)} 个切片")

        # 生成向量
        texts = [chunk['text'] for chunk in all_chunks]
        embeddings = self._embed(texts)

        # 构造插入数据
        data_to_insert = []
        for chunk, emb in zip(all_chunks, embeddings):
            data_to_insert.append({
                "content": chunk['text'],
                "metadata": chunk.get('metadata', {}),
                "vector": emb
            })

        # 批量插入
        result = self.client.insert(
            collection_name=self.collection_name,
            data=data_to_insert
        )

        print(f"插入 {result.get('insert_count', len(data_to_insert))} 条数据")
        return len(data_to_insert)

    def _chunk_text(self, text, chunk_size, chunk_overlap):
        """
        文本切片（滑动窗口）

        参数:
            text: 输入文本
            chunk_size: 切片大小
            chunk_overlap: 重叠大小
        返回:
            切片列表，每项包含 text 和 metadata
        """
        chunks = []
        step = chunk_size - chunk_overlap

        if step <= 0:
            step = chunk_size // 2

        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]

            chunks.append({
                'text': chunk_text,
                'metadata': {
                    'start': start,
                    'end': end
                }
            })

            start += step
            if end >= len(text):
                break

        return chunks

    def search(self, query, top_k=5, use_vector=True, use_keyword=False,
               filter=None, use_rerank=False, score_threshold=0.0):
        """
        检索相关文档

        参数:
            query: 查询文本
            top_k: 返回结果数量
            use_vector: 是否使用向量检索
            use_keyword: 是否使用关键字检索
            filter: 标量过滤条件
            use_rerank: 是否使用 Rerank
            score_threshold: 最低相似度阈值（0.0~1.0），低于此值的结果将被过滤
                - COSINE 度量下，1.0 表示完全相同，0.0 表示完全无关
                - 建议值：0.5~0.7，过低会返回无关内容，过高可能漏掉相关内容
        返回:
            检索结果列表
        """
        results = []

        # 1. 向量检索
        if use_vector:
            query_vector = self._embed([query])[0]

            search_params = {"metric_type": "COSINE", "params": {
            "radius": 0.5,
            "range_filter": 0.9
            }}
            if filter:
                search_params["filter"] = filter

            vector_results = self.client.search(
                collection_name=self.collection_name,
                data=[query_vector],
                limit=top_k * 2 if use_rerank else top_k,
                output_fields=["content", "metadata"],
                search_params=search_params
            )

            for hit in vector_results[0]:
                score = hit['distance']
                if score < score_threshold:
                    continue
                results.append({
                    'content': hit['entity']['content'],
                    'metadata': hit['entity'].get('metadata', {}),
                    'vector_score': score,
                    'keyword_score': 0.0
                })

        # 2. 关键字检索（简化实现，完整版参见 03_retrieval_methods）
        if use_keyword:
            keyword_results = self._keyword_search(query, top_k)
            # 注：混合检索的完整融合逻辑参见 03_retrieval_methods/keyword_search.py
            results.extend(keyword_results)

        # 3. Rerank 重排序
        if use_rerank and results:
            results = self._rerank_results(query, results, top_k)

        # 4. 截取 Top-K
        results = results[:top_k]

        return results

    def _keyword_search(self, query, top_k):
        """关键字检索（简化版，完整实现参见 03_retrieval_methods）"""
        # 实际实现应调用 03_keyword_search.py 中的 BM25 方法
        return []

    def _rerank_results(self, query, results, top_k):
        """
        Rerank 重排序

        使用 BGE-Reranker (CrossEncoder) 对检索结果进行重排序

        参数:
            query: 查询文本
            results: 待重排的结果列表
            top_k: 返回数量
        返回:
            重排后的结果列表
        """
        try:
            from sentence_transformers import CrossEncoder

            reranker = CrossEncoder('BAAI/bge-reranker-large')

            pairs = [[query, r['content']] for r in results]
            scores = reranker.predict(pairs)

            for i, score in enumerate(scores):
                results[i]['rerank_score'] = float(score)

            results.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
            return results[:top_k]

        except ImportError:
            print("  Rerank 需要安装：pip install sentence-transformers")
            return results[:top_k]
        except Exception as e:
            print(f"  Rerank 失败：{e}")
            return results[:top_k]

    def query(self, filter_expr, output_fields=None, limit=10):
        """
        标量查询

        参数:
            filter_expr: 过滤表达式，例如 "category == 'AI' and views > 1000"
            output_fields: 输出字段列表
            limit: 结果数量
        返回:
            查询结果列表
        """
        results = self.client.query(
            collection_name=self.collection_name,
            filter=filter_expr,
            output_fields=output_fields or ["content", "metadata"],
            limit=limit
        )

        return results

    def close(self):
        """关闭连接"""
        # MilvusClient 不需要显式关闭
        print("RAGRetriever 已关闭")





# =============================================================================
# 示例 4: API 参数详解
# =============================================================================

def api_parameters_explained():
    """API 参数说明（纯文档展示）"""
    print(f"\n-- 示例 4: API 参数详解")

    print("""
RAGRetriever 参数说明
--------------------------------------------------------------

初始化参数：
  milvus_uri: Milvus 连接 URI
    - 远程服务："http://localhost:19530"
    - Milvus Lite（不支持 Windows）："milvus_demo.db"

  collection_name: Collection 名称
    - 默认："knowledge_base"

  embedding_model: Embedding 模型名称（本地 sentence-transformers）
    - 中文："bge-large-zh-v1.5"
    - 英文："all-MiniLM-L6-v2"
    - None: 使用阿里云 text-embedding-v4 API（默认）

  dim: 向量维度
    - 需与 Embedding 模型匹配
    - text-embedding-v4: 1024
    - bge-large-zh: 1024
    - MiniLM: 384

search() 参数：
  query: 查询文本
  top_k: 返回结果数量（建议 5-20）
  use_vector: 是否使用向量检索（默认 True）
  use_keyword: 是否使用关键字检索（默认 False）
  filter: 标量过滤条件
    - 示例："category == 'AI' and views > 1000"
  use_rerank: 是否使用 Rerank（默认 False）

方法总结：
  __init__          初始化 Milvus 连接
  create_collection 创建 Collection
  add_documents     添加文档（自动切片 + 向量化）
  search            检索（支持向量/关键字/Rerank）
  query             标量查询（按字段过滤）
  close             关闭连接
""")


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == "__main__":
    # 示例 4: API 参数说明
    api_parameters_explained()
