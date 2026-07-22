from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Milvus
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from dotenv import load_dotenv
import os
load_dotenv()
os.environ["DASHSCOPE_API_KEY"] = os.getenv("ALIYUN_API_KEY", "")

# 1. 加载文档
loader = TextLoader("test_doc.txt", encoding="utf-8")
documents = loader.load()

# 2. 切片
"""
    使用 LangChain 的 RecursiveCharacterTextSplitter 做智能分块。

    它按优先级尝试切分: 段落(\n\n) → 行(\n) → 句子(。)→ 字级别
    这样能尽量保持语义完整性，不会在句子中间断开。
"""
splitter = RecursiveCharacterTextSplitter(
    chunk_size=80,      # 每片约 500 token
    chunk_overlap=20,    # 切片间重叠 50 token（防止信息在边界丢失）
    separators=["\n", "。", "，", " "]
)
chunks = splitter.split_documents(documents)

# 3. 向量化并存入向量库
embeddings = DashScopeEmbeddings(model="text-embedding-v3")
vectorstore = Milvus.from_documents(
    chunks,
    embeddings,
    connection_args={"host": "192.168.142.128", "port": "19530"},
    collection_name="edu_policy",
)

# 4. 创建检索器（Retriever）
retriever = vectorstore.as_retriever(
    search_type="similarity",   # 语义相似度检索
    search_kwargs={"k": 3},     # 返回 top 3 相关文档
)


# 5. 使用检索器
docs = retriever.invoke("年假有多少天？")
for doc in docs:
    print(f"来源: {doc.metadata}")
    print(f"内容: {doc.page_content[:100]}...")
    print("---")

def hhh():

    # 关键词检索（BM25）
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 3

    # 向量检索
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 混合检索（两者结合，效果通常优于单独使用）
    ensemble = EnsembleRetriever(
        retrievers=[bm25, vector_retriever],
        weights=[0.3, 0.7],  # BM25 权重 30%，向量检索权重 70%
    )

    results = ensemble.invoke("年假多少天")

if __name__ == '__main__':
    hhh()