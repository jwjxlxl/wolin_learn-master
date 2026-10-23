# =============================================================================
# hybrid_retriever — Milvus 混合检索器（稠密向量 + BM25 稀疏向量）
# =============================================================================
# 用途：把"创建 Collection + 插入数据 + 混合检索"封装成可复用类，
#       供 03_retrieval_methods 的混合检索演示和 05_rag_pipeline 的 RAG 共用。
#
# 核心概念：
#   - 双向量字段：dense（语义）+ sparse（关键词/BM25）
#   - BM25 Function：Milvus 自动把 text 分词并生成稀疏向量
#   - 结果融合：RRFRanker / WeightedRanker
# =============================================================================

from pymilvus import (
    DataType,
    Function,
    FunctionType,
    AnnSearchRequest,
    RRFRanker,
    WeightedRanker,
)


class HybridRetriever:
    """Milvus 混合检索器封装

    职责：
      1. 创建含 dense + sparse 双向量字段的 Collection
      2. 插入文档（dense 由调用方提供，sparse 由 Milvus 自动生成）
      3. 提供 纯向量 / 纯 BM25 / RRF 混合 / 加权混合 四类检索

    使用示例：
        client = MilvusClient(uri=MILVUS_URI, db_name=MILVUS_DB_NAME)
        emb = AliyunEmbedding()
        retriever = HybridRetriever(client, "hybrid_demo", emb)

        dense_vectors = [emb.embed(d) for d in documents]
        retriever.create_collection(dim=len(dense_vectors[0]))
        retriever.insert(documents, dense_vectors)

        results = retriever.hybrid_search("机器学习", emb.embed("机器学习"), top_k=3)
    """

    def __init__(self, client, collection_name, emb=None):
        self.client = client
        self.collection_name = collection_name
        self.emb = emb

    # ------------------------------------------------------------------
    # 创建 Collection
    # ------------------------------------------------------------------
    def create_collection(self, dim=1024):
        """创建含 dense + sparse 双向量字段的 Collection（会先 drop 同名旧集合）。

        BM25 Function 原理（Milvus 2.4+）：
          传统做法要在应用层手动「jieba 分词 + rank-bm25 算稀疏向量」再插入；
          现在只需定义这个 BM25 Function，把 text 自动映射成 sparse 稀疏向量，
          相当于把 BM25 计算从「应用层」下沉到「数据库层」。
          前提：text 字段要开 enable_analyzer=True 分词（中文必须 analyzer_params
          {"type": "chinese"}），Milvus 才能据此算出 BM25 稀疏向量。

        参数:
            dim: 稠密向量维度（text-embedding-v4 为 1024）
        """
        schema = self.client.create_schema()
        schema.add_field(
            field_name="id",
            datatype=DataType.INT64,
            is_primary=True,
            auto_id=True,
        )
        schema.add_field(
            field_name="text",
            datatype=DataType.VARCHAR,
            max_length=2000,
            enable_analyzer=True,
            analyzer_params={"type": "chinese"},  # 中文分词（BM25 全文检索必需）
        )
        schema.add_field(field_name="dense", datatype=DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)

        # BM25 函数：text → sparse（Milvus 自动分词并生成稀疏向量）
        bm25_function = Function(
            name="text_bm25_emb",
            input_field_names=["text"],
            output_field_names=["sparse"],
            function_type=FunctionType.BM25,
        )
        schema.add_function(bm25_function)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="dense",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        index_params.add_index(
            field_name="sparse",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params={"inverted_index_algo": "DAAT_MAXSCORE"},
        )

        # 清理旧 Collection（教学演示用）
        if self.client.has_collection(collection_name=self.collection_name):
            self.client.drop_collection(collection_name=self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )

    # ------------------------------------------------------------------
    # 插入数据
    # ------------------------------------------------------------------
    def insert(self, documents, dense_vectors=None):
        """插入文档列表并 flush。

        参数:
            documents: 原始文本列表
            dense_vectors: 与 documents 一一对应的稠密向量（可选）
                           不传则用 self.emb 现场生成
        返回:
            int: 插入的文档数
        """
        if dense_vectors is None:
            if self.emb is None:
                raise ValueError("insert 需要 self.emb 或显式传入 dense_vectors")
            dense_vectors = [self.emb.embed(doc) for doc in documents]

        data = [{"text": doc, "dense": vec} for doc, vec in zip(documents, dense_vectors)]
        self.client.insert(collection_name=self.collection_name, data=data)
        self.client.flush(collection_name=self.collection_name)  # 关键：flush 后数据才可检索
        return len(data)

    # ------------------------------------------------------------------
    # 检索（四类）
    # ------------------------------------------------------------------
    def search_dense(self, query_vec, top_k=3):
        """纯向量检索（语义）"""
        res = self.client.search(
            collection_name=self.collection_name,
            data=[query_vec],
            anns_field="dense",
            limit=top_k,
            output_fields=["text"],
        )
        return res[0]

    def search_sparse(self, query, top_k=3):
        """纯 BM25 关键字检索（精确匹配）"""
        res = self.client.search(
            collection_name=self.collection_name,
            data=[query],
            anns_field="sparse",
            limit=top_k,
            output_fields=["text"],
        )
        return res[0]

    def hybrid_search(self, query, query_vec, top_k=3, ranker="rrf", vector_weight=0.6):
        """混合检索：RRF 或 加权融合。

        参数:
            query: 原始查询文本（供稀疏/全文这一路使用）
            query_vec: 查询的稠密向量
            top_k: 每路召回数量
            ranker: "rrf" 或 "weighted"
            vector_weight: 加权融合时向量的权重（0-1）
        返回:
            list: 命中结果，每项为 {"distance": 分数, "entity": {"text": 文本}}
        """
        req_dense = AnnSearchRequest(
            data=[query_vec],
            anns_field="dense",
            param={"metric_type": "COSINE"},
            limit=top_k,
        )
        req_sparse = AnnSearchRequest(
            data=[query],  # 稀疏/全文这一路直接传原始文本，由 BM25 函数处理
            anns_field="sparse",
            param={"metric_type": "BM25"},
            limit=top_k,
        )

        if ranker == "rrf":
            ranker_obj = RRFRanker(k=60)
        elif ranker == "weighted":
            ranker_obj = WeightedRanker(vector_weight, 1 - vector_weight)
        else:
            raise ValueError(f"不支持的融合方式: {ranker}")

        res = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[req_dense, req_sparse],
            ranker=ranker_obj,
            limit=top_k,
            output_fields=["text"],
        )
        return res[0]

    # ------------------------------------------------------------------
    # RAG 检索入口
    # ------------------------------------------------------------------
    def retrieve(self, query, top_k=3, ranker="rrf", vector_weight=0.6):
        """RAG 检索入口：基于查询文本做混合检索，返回带编号的上下文列表。

        返回:
            list: [{"index": 1, "text": ..., "score": ...}, ...]
        """
        if self.emb is None:
            raise ValueError("retrieve 需要 self.emb 才能生成查询向量")

        query_vec = self.emb.embed(query)
        hits = self.hybrid_search(
            query, query_vec, top_k=top_k, ranker=ranker, vector_weight=vector_weight
        )
        return [
            {
                "index": i,
                "text": hit["entity"]["text"],
                "score": round(hit["distance"], 4),
            }
            for i, hit in enumerate(hits, 1)
        ]
