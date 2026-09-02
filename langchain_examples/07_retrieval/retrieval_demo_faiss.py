from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
import os
load_dotenv()
os.environ["DASHSCOPE_API_KEY"] = os.getenv("ALIYUN_API_KEY", "")

'''
    并不是所有的RAG都必须使用Milvus，简单的文档，可以使用FAISS向量存储，或者Chroma
'''

# 1. 加载文档
loader = TextLoader("test_doc.txt", encoding="utf-8")
documents = loader.load()

# 2. 切片（自动按语义边界切分）
splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,      # 每片约 500 token
    chunk_overlap=10,    # 切片间重叠 50 token（防止信息在边界丢失）
    separators=["\n\n", "\n", "。", "，", " "]
)
chunks = splitter.split_documents(documents)

# 3. 向量化并存入向量库

embeddings = DashScopeEmbeddings(model="text-embedding-v3")
# 定义一个Faiss向量存储 ，替换了Milvus
store = FAISS.from_documents(documents, embeddings)

# 语义搜索: "人工智能相关的技术" 应该找到 ai 来源的文档
query = "机器学习"
results = store.similarity_search(query, k=3)

print(f"搜索: '{query}'\n")
for i, doc in enumerate(results):
    print(f"  [{i + 1}] 来源: {doc.metadata['source']}")
    print(f"      内容: {doc.page_content}\n")
