# =============================================================================
# 03_from_mock_to_real — 从模拟向量到真实 Embedding 的过渡
# =============================================================================
# 用途：理解教学代码中大量使用的 mock 向量与真实项目的差异
# 难度：⭐⭐⭐（3 星）
#
# 核心概念：
#   1. Mock 向量的局限性和使用场景
#   2. 如何将 mock 替换为真实 Embedding API
#   3. 从"教学代码"到"生产代码"的改造步骤
# =============================================================================

import os
import random
from dotenv import load_dotenv
load_dotenv()


# =============================================================================
# 第一部分：Mock vs Real — 对比分析
# =============================================================================

def explain_mock_vs_real():
    """讲解 mock 向量和真实 Embedding 的差异"""
    print("=" * 60)
    print("第一部分：Mock vs Real Embedding")
    print("=" * 60)

    print("""
┌─────────────────────────────────────────────────────────┐
│ Mock 向量 vs 真实 Embedding                              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ Mock 向量（教学用）：                                    │
│   - 使用 random.seed(hash(text)) 生成                    │
│   - 特点：同一次运行中相同的文本得到相同的向量           │
│   - 目的：让学生无需 API Key 就能跑通检索流程            │
│   - 局限：向量没有语义信息，检索结果无意义               │
│                                                         │
│   示例代码：                                             │
│   ```python                                             │
│   def mock_embedding(text, dim=1024):                   │
│       random.seed(hash(text) % 10000)                   │
│       return [random.uniform(-1, 1) for _ in range(dim)]│
│   ```                                                    │
│                                                         │
│ 真实 Embedding（生产用）：                                │
│   - 调用 Embedding API 或本地模型                        │
│   - 特点：语义相似的文本向量距离近                       │
│   - 目的：获得真正有效的语义检索能力                     │
│                                                         │
│   示例代码：                                             │
│   ```python                                             │
│   from openai import OpenAI                             │
│   client = OpenAI(api_key="...",                        │
│       base_url="https://dashscope.aliyuncs.com/...")    │
│   response = client.embeddings.create(                   │
│       model="text-embedding-v4", input=text)            │
│   vector = response.data[0].embedding                   │
│   ```                                                    │
│                                                         │
└─────────────────────────────────────────────────────────┘

📊 Mock vs Real 快速对比：

| 维度 | Mock | Real |
|------|------|------|
| 需要 API Key | ❌ 不需要 | ✅ 需要 |
| 语义正确性 | ❌ 无 | ✅ 有 |
| 检索效果 | ❌ 随机 | ✅ 有效 |
| 代码调试 | ✅ 方便 | ❌ 依赖网络 |
| 教学适用 | ✅ 概念演示 | ✅ 实战项目 |

💡 教学策略：
   课程早期用 mock 降低门槛（聚焦"流程"）
   → 课程后期用 real 获得效果（聚焦"实战"）
""")


# =============================================================================
# 第二部分：如何逐步替换 mock
# =============================================================================

def step_by_step_replacement():
    """展示从 mock 到 real 的逐步替换过程"""
    print("\n" + "=" * 60)
    print("第二部分：从 Mock 到 Real 的替换步骤")
    print("=" * 60)

    print("""
┌─────────────────────────────────────────────────────────┐
│ 替换步骤                                                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ 第 1 步：安装依赖                                        │
│   pip install openai python-dotenv                      │
│                                                         │
│ 第 2 步：配置 API Key                                    │
│   在 .env 文件中设置 ALIYUN_API_KEY=sk-xxx               │
│                                                         │
│ 第 3 步：找到代码中的 mock_embedding() 函数              │
│   在 rag_examples 中搜索：                              │
│   - generate_mock_embeddings()                          │
│   - _random_embedding()                                │
│   - random.seed(hash(text))                            │
│                                                         │
│ 第 4 步：替换为真实 Embedding 调用                       │
│                                                         │
│   替换前（mock）：                                       │
│   ```python                                             │
│   def generate_mock_embeddings(texts, dim=1024):        │
│       random.seed(42)                                   │
│       vectors = []                                      │
│       for text in texts:                                │
│           random.seed(hash(text) % 10000)               │
│           vectors.append([random.random() for _ in ...])│
│       return vectors                                    │
│                                                         │
│   vectors = generate_mock_embeddings(documents)         │
│   ```                                                    │
│                                                         │
│   替换后（real）：                                       │
│   ```python                                             │
│   from rag_demo.util.embedding import generate_embedding│
│                                                         │
│   vectors = [generate_embedding(doc) for doc in documents]│
│   ```                                                    │
│                                                         │
│ 第 5 步：运行测试，验证检索效果                           │
│   好的结果：相关文档排在前面                              │
│   差的结果：随机排列（说明 Embedding 有问题）            │
│                                                         │
└─────────────────────────────────────────────────────────┘
""")


# =============================================================================
# 第三部分：通用 Embedding 封装（可直接用于替换 mock）
# =============================================================================

class UniversalEmbedder:
    """通用 Embedding 封装，同时支持 mock 和 real 模式

    用法：
        # Mock 模式（无需 API Key）
        embedder = UniversalEmbedder(mode="mock")

        # Real 模式（需要 API Key）
        embedder = UniversalEmbedder(mode="real")

    设计意图：
        这是 rag_demo/util/embedding.py 的简化教学版，
        展示了从 mock 到 real 的统一接口设计。
    """

    def __init__(self, mode="auto", dim=1024):
        self.dim = dim
        if mode == "auto":
            api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("ALIYUN_API_KEY")
            mode = "real" if api_key else "mock"
        self.mode = mode

        if mode == "real":
            from openai import OpenAI
            self.client = OpenAI(
                api_key=os.getenv("DASHSCOPE_API_KEY") or os.getenv("ALIYUN_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )

    def embed(self, text: str) -> list:
        """对单个文本生成向量"""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list) -> list:
        """批量生成向量"""
        if self.mode == "mock":
            return self._mock_embed(texts)
        else:
            return self._real_embed(texts)

    def _mock_embed(self, texts: list) -> list:
        """Mock 实现（仅用于演示）"""
        vectors = []
        for text in texts:
            random.seed(hash(text) % 10000)
            vectors.append([random.uniform(-1, 1) for _ in range(self.dim)])
        return vectors

    def _real_embed(self, texts: list) -> list:
        """真实 API 实现"""
        all_vectors = []
        batch_size = 25
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                model="text-embedding-v4",
                input=batch,
                encoding_format="float",
            )
            all_vectors.extend([item.embedding for item in response.data])
        return all_vectors


# =============================================================================
# 第四部分：验证替换效果
# =============================================================================

def demo_universal_embedder():
    """演示 UniversalEmbedder 的 mock 和 real 两种模式"""
    print("=" * 60)
    print("第三部分：通用 Embedder 演示")
    print("=" * 60)

    texts = [
        "人工智能是计算机科学的前沿领域",
        "机器学习让计算机从数据中学习",
        "今天天气很好适合户外运动",
    ]

    # 测试 mock 模式
    mock_embedder = UniversalEmbedder(mode="mock", dim=1024)
    print("\nMock 模式：")
    vectors = mock_embedder.embed_batch(texts)
    for i, (text, vec) in enumerate(zip(texts, vectors)):
        print(f"  '{text[:20]}...' → [{vec[0]:.4f}, {vec[1]:.4f}, ...]")

    # 测试 auto 模式（自动检测 API Key）
    auto_embedder = UniversalEmbedder(mode="auto", dim=1024)
    print(f"\nAuto 模式：当前使用 [{auto_embedder.mode}] 模式")
    if auto_embedder.mode == "real":
        print("  （检测到 API Key，使用真实 Embedding）")
        vectors = auto_embedder.embed_batch(texts)
        for i, (text, vec) in enumerate(zip(texts, vectors)):
            print(f"  '{text[:20]}...' → [{vec[0]:.4f}, {vec[1]:.4f}, ...]")
    else:
        print("  （未检测到 API Key，使用 Mock Embedding）")
        print("  💡 设置 ALIYUN_API_KEY 环境变量即可切换到真实模式")


# =============================================================================
# 第五部分：迁移检查清单
# =============================================================================

def migration_checklist():
    """从教学代码到生产代码的迁移检查清单"""
    print("\n" + "=" * 60)
    print("第四部分：迁移检查清单")
    print("=" * 60)

    print("""
从教学代码到生产代码，逐项检查：

□ 1. Embedding 替换
     □ 找到所有 mock_embedding() / _random_embedding() 调用
     □ 替换为 from rag_demo.util.embedding import generate_embedding
     □ 运行一次真实检索，验证结果相关性

□ 2. 配置管理
     □ 确保所有 API Key 通过 .env 管理
     □ 确保 MILVUS_URI 通过环境变量配置
     □ 不要在代码中硬编码任何凭证

□ 3. 错误处理
     □ 添加 API 调用失败的降级策略
     □ 添加 Milvus 连接失败的提示
     □ 添加数据格式校验

□ 4. 性能优化
     □ 批量插入改为批次（batch_size=100~1000）
     □ 使用 AUTOINDEX 代替手动指定索引
     □ 大量数据时考虑分区

□ 5. 质量评估
     □ 构建测试问题集
     □ 人工评估前 10 条检索结果的相关性
     □ 考虑引入 RAGAS 评估框架（见 06_rag_evaluation/）
""")


# =============================================================================
# 主程序
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  从 Mock 到 Real — Embedding 过渡指南")
    print("=" * 70 + "\n")

    explain_mock_vs_real()
    step_by_step_replacement()
    demo_universal_embedder()
    migration_checklist()

    print("\n" + "=" * 70)
    print("  迁移完成！你现在可以：")
    print("  1. 用 UniversalEmbedder 统一管理 mock/real 切换")
    print("  2. 将任意教学代码中的 mock 替换为真实 Embedding")
    print("  3. 进入 rag_demo 实战项目")
    print("=" * 70)
