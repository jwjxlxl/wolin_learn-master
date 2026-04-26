# =============================================================================
# 阿里云百炼 Embedding API 调用示例
# =============================================================================
#  
# 用途：教学演示 - 如何使用阿里云百炼 Embedding API
#
# 核心概念：
#   - DashScope SDK 使用
#   - text-embedding-v3 模型调用
#   - 批量向量生成
# =============================================================================

# -----------------------------------------------------------------------------
# 运行前检查
# -----------------------------------------------------------------------------
# 1. 已安装依赖：pip install dashscope
# 2. 已获取 API Key：https://bailian.console.aliyun.com/
# 3. 已设置环境变量：DASHSCOPE_API_KEY=your_key
# -----------------------------------------------------------------------------

# 设置 UTF-8 编码（Windows 专用）



# =============================================================================
# 第一部分：阿里云百炼 Embedding 简介
# =============================================================================
"""
阿里云百炼 Embedding API

📊 支持的模型
   - text-embedding-v3：最新版本，支持中英文，维度 1024/1536
   - text-embedding-v2：上一代，维度 1536
   - text-embedding-v1：基础版本

💡 text-embedding-v3 特点
   - 支持多语言（中文、英文等）
   - 支持多种维度（1024/1536 可选）
   - 支持输入长度：最大 8192 tokens
   - 支持向量归一化（直接用于余弦相似度）

📦 API 调用方式
   1. DashScope SDK（推荐）
   2. HTTP REST API

⚠️ 计费说明（参考）
   - 免费额度：每月一定额度免费
   - 超出后：按调用次数计费
   - 详情请查阅官网
"""


# =============================================================================
# 示例 1: 基础调用（SDK 方式）
# =============================================================================

def basic_embedding_with_sdk():
    """
    使用 DashScope SDK 调用 Embedding API
    """
    print("=" * 60)
    print("示例 1: 基础调用（SDK 方式）")
    print("=" * 60)

    try:
        from dashscope import TextEmbedding

        # 文本
        text = "人工智能是计算机科学的一个分支"

        print(f"输入文本：{text}")
        print(f"文本长度：{len(text)} 字符\n")

        # 调用 Embedding API
        print("正在调用 text-embedding-v3 模型...")
        result = TextEmbedding.call(
            model='text-embedding-v3',
            input=text
        )

        # 检查结果
        if result.status_code == 200:
            print("✓ 调用成功！\n")

            # 获取向量
            embedding = result.output['embeddings'][0]['embedding']

            print(f"向量维度：{len(embedding)}")
            print(f"向量预览：{embedding[:10]}...")  # 显示前 10 个元素

            # 向量特性
            import math
            norm = math.sqrt(sum(x * x for x in embedding))
            print(f"\n向量范数：{norm:.4f} (归一化后应接近 1.0)")

        else:
            print(f"✗ 调用失败：{result.code} - {result.message}")

    except ImportError:
        print("✗ 需要安装：pip install dashscope")
    except Exception as e:
        print(f"✗ 错误：{e}")


# =============================================================================
# 示例 2: 批量生成向量
# =============================================================================

def batch_embedding():
    """
    批量生成文本向量
    """
    print()
    print("=" * 60)
    print("示例 2: 批量生成向量")
    print("=" * 60)

    try:
        from dashscope import TextEmbedding

        # 批量文本
        texts = [
            "机器学习是人工智能的核心技术",
            "深度学习使用神经网络进行表征学习",
            "自然语言处理让计算机理解人类语言",
            "计算机视觉用于图像识别和分析",
        ]

        print(f"批量处理 {len(texts)} 条文本：\n")
        for i, text in enumerate(texts, 1):
            print(f"  [{i}] {text}")
        print()

        # 批量调用
        print("正在批量生成向量...")
        result = TextEmbedding.call(
            model='text-embedding-v3',
            input=texts  # 支持列表输入
        )

        if result.status_code == 200:
            print("✓ 批量生成成功！\n")

            embeddings = result.output['embeddings']

            for i, emb_data in enumerate(embeddings):
                text_index = emb_data.get('text_index', i)
                vector = emb_data['embedding']

                print(f"文本 [{i+1}] 向量：")
                print(f"   维度：{len(vector)}")
                print(f"   前 5 元素：{vector[:5]}")
                print()
        else:
            print(f"✗ 调用失败：{result.code}")

    except ImportError:
        print("✗ 需要安装：pip install dashscope")
    except Exception as e:
        print(f"✗ 错误：{e}")


# =============================================================================
# 示例 3: 封装 Embedding 工具类
# =============================================================================

class AliyunEmbedding:
    """
    阿里云 Embedding 工具类封装
    """

    def __init__(self, model='text-embedding-v3', api_key=None):
        """
        初始化 Embedding 工具

        参数:
            model: 模型名称
            api_key: API Key（可选，也可用环境变量）
        """
        from dashscope import TextEmbedding

        self.model = model
        self.text_embedding = TextEmbedding

        # 如果提供了 API Key
        if api_key:
            import dashscope
            dashscope.api_key = api_key

        print(f"✓ AliyunEmbedding 初始化完成")
        print(f"  模型：{model}")

    def embed(self, text):
        """
        生成单个文本的向量

        参数:
            text: 输入文本
        返回:
            向量列表
        """
        result = self.text_embedding.call(
            model=self.model,
            input=text
        )

        if result.status_code == 200:
            return result.output['embeddings'][0]['embedding']
        else:
            raise Exception(f"Embedding 失败：{result.code}")

    def embed_batch(self, texts, batch_size=10):
        """
        批量生成向量

        参数:
            texts: 文本列表
            batch_size: 批次大小
        返回:
            向量列表
        """
        all_embeddings = []

        # 分批处理
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            result = self.text_embedding.call(
                model=self.model,
                input=batch
            )

            if result.status_code == 200:
                # 按原始顺序提取向量
                embeddings_map = {}
                for emb_data in result.output['embeddings']:
                    idx = emb_data.get('text_index', len(embeddings_map))
                    embeddings_map[idx] = emb_data['embedding']

                # 按顺序添加
                for idx in sorted(embeddings_map.keys()):
                    all_embeddings.append(embeddings_map[idx])
            else:
                raise Exception(f"批量 Embedding 失败：{result.code}")

        return all_embeddings

    def similarity(self, text1, text2):
        """
        计算两个文本的相似度

        参数:
            text1: 文本 1
            text2: 文本 2
        返回:
            余弦相似度 (0-1)
        """
        # 生成向量
        vec1 = self.embed(text1)
        vec2 = self.embed(text2)

        # 计算余弦相似度
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(a * a for a in vec2) ** 0.5

        return dot / (norm1 * norm2)


def demo_embedding_tool():
    """
    演示 Embedding 工具类
    """
    print()
    print("=" * 60)
    print("示例 3: Embedding 工具类封装")
    print("=" * 60)

    # 注意：实际运行需要有效的 API Key
    # 这里演示 API 设计

    print("工具类 API 设计：")
    print("-" * 50)

    # 模拟演示（不实际调用 API）
    print("""
代码示例:

# 1. 初始化工具类
emb = AliyunEmbedding(model='text-embedding-v3')

# 2. 生成单个向量
vector = emb.embed("人工智能简介")
print(f"向量维度：{len(vector)}")

# 3. 批量生成
texts = ["文本 1", "文本 2", "文本 3"]
vectors = emb.embed_batch(texts)
print(f"生成了 {len(vectors)} 个向量")

# 4. 计算相似度
sim = emb.similarity("机器学习", "深度学习")
print(f"相似度：{sim:.4f}")
""")

    # 语义相似度示例
    print("\n语义相似度示例（模拟结果）：")
    print("-" * 50)

    pairs = [
        ("机器学习", "深度学习", 0.85),
        ("机器学习", "人工智能", 0.78),
        ("机器学习", "红烧肉", 0.12),
        ("自然语言处理", "文本分析", 0.82),
        ("计算机视觉", "图像处理", 0.88),
        ("Python 编程", "红烧肉做法", 0.08),
    ]

    for text1, text2, sim in pairs:
        bar = "█" * int(sim * 20)
        print(f"  {text1} ↔ {text2}")
        print(f"  相似度：{bar} {sim:.2f}")
        print()


# =============================================================================
# 示例 4: 相似度计算应用
# =============================================================================

def similarity_applications():
    """
    Embedding 相似度的应用场景
    """
    print("=" * 60)
    print("示例 4: 相似度计算应用")
    print("=" * 60)

    print("""
┌─────────────────────────────────────────────────────────┐
│ Embedding 相似度应用场景                                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ 1. 语义搜索                                             │
│    用户查询 → 生成向量 → 匹配最相似的文档               │
│                                                         │
│ 2. 文本去重                                             │
│    计算文档间相似度 > 阈值则判定为重复                   │
│                                                         │
│ 3. 推荐系统                                             │
│    用户历史喜欢的内容 → 找相似的新内容                  │
│                                                         │
│ 4. 问答匹配                                             │
│    用户问题 → 匹配 FAQ 库中最相似的问题                  │
│                                                         │
│ 5. 聚类分析                                             │
│    将相似的文本聚在一起，发现主题分组                   │
│                                                         │
└─────────────────────────────────────────────────────────┘

💡 相似度阈值参考:

| 场景 | 推荐阈值 | 说明 |
|------|----------|------|
| 严格去重 | >0.95 | 几乎相同的内容 |
| 语义匹配 | >0.75 | 语义高度相关 |
| 主题聚合 | >0.60 | 同一主题内容 |
| 宽泛相关 | >0.40 | 有一定关联 |

📊 余弦相似度说明:

cos(θ) = A·B / (‖A‖ × ‖B‖)

- 范围：[-1, 1]
- 1: 方向完全相同（语义完全一致）
- 0: 正交（语义无关）
- -1: 方向相反（语义对立）

注意：text-embedding-v3 输出已归一化，
分母为 1，可直接用点积计算相似度。
""")


# =============================================================================
# 示例 5: 错误处理与最佳实践
# =============================================================================

def error_handling_and_best_practices():
    """
    错误处理与最佳实践
    """
    print()
    print("=" * 60)
    print("示例 5: 错误处理与最佳实践")
    print("=" * 60)

    print("""
┌─────────────────────────────────────────────────────────┐
│ API 调用错误处理                                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ 常见错误码：                                            │
│   400: 请求参数错误（检查输入格式）                    │
│   401: 认证失败（检查 API Key）                        │
│   429: 请求频率超限（降低调用频率）                    │
│   500: 服务器内部错误（稍后重试）                      │
│                                                         │
│ 重试策略：                                              │
│   - 网络错误：指数退避重试（1s, 2s, 4s, 8s）           │
│   - 429 错误：等待后重试                                │
│   - 500 错误：重试 3 次后放弃                            │
│                                                         │
└─────────────────────────────────────────────────────────┘

💡 最佳实践:

1. API Key 管理
   - 使用环境变量，不要硬编码
   - 定期轮换 Key
   - 设置使用限额告警

2. 成本控制
   - 本地缓存已生成的向量
   - 批量调用比单次调用更经济
   - 监控调用量

3. 性能优化
   - 批量调用（最多支持 2000 条/批）
   - 异步并发调用（提高吞吐量）
   - 长文本预处理（分段后取平均）

4. 向量存储
   - 及时存储生成的向量
   - 使用向量数据库（如 Milvus）
   - 建立向量索引加速检索
""")

    # 重试示例代码
    print("\n重试策略代码示例：")
    print("-" * 50)
    print("""
import time
from dashscope import TextEmbedding

def embed_with_retry(text, max_retries=3):
    for attempt in range(max_retries):
        try:
            result = TextEmbedding.call(
                model='text-embedding-v3',
                input=text
            )
            if result.status_code == 200:
                return result.output['embeddings'][0]['embedding']
            elif result.status_code == 429:
                wait_time = 2 ** attempt  # 指数退避
                time.sleep(wait_time)
            else:
                raise Exception(f"API 错误：{result.code}")
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)

# 使用
vector = embed_with_retry("人工智能简介")
""")


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  Embedding 示例 - 阿里云百炼 Embedding API")
    print("  说明：学习如何调用阿里云百炼的文本向量化 API")
    print("=" * 70 + "\n")

    print("【说明】")
    print("  阿里云百炼提供 text-embedding-v3 模型")
    print("  支持中英文，维度 1024/1536 可选")
    print()

    print("【前置条件】")
    print("  1. 注册阿里云账号")
    print("  2. 开通百炼服务：https://bailian.console.aliyun.com/")
    print("  3. 获取 API Key")
    print("  4. 安装 SDK: pip install dashscope")
    print()

    # 运行示例
    print("-" * 60)

    # 注意：实际运行需要有效的 API Key
    # 以下示例会演示 API 设计，但需要配置后才能实际调用
    print("提示：以下示例需要配置 API Key 才能实际运行\n")

    basic_embedding_with_sdk()
    print()

    batch_embedding()
    print()

    demo_embedding_tool()
    print()

    similarity_applications()
    print()

    error_handling_and_best_practices()

    print()
    print("=" * 70)
    print("  阿里云百炼 Embedding API 学习完成！")
    print("  接下来可以学习：")
    print("  - 03_local_embedding.py（本地 Embedding 模型）")
    print("  - 04_embedding_comparison.py（模型对比）")
    print("=" * 70 + "\n")
