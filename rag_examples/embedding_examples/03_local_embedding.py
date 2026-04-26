# =============================================================================
# 本地 Embedding 模型调用
# =============================================================================
#  
# 用途：教学演示 - 使用 sentence-transformers 调用本地 Embedding 模型
#
# 核心概念：
#   - sentence-transformers 库使用
#   - 中文 Embedding 模型
#   - 本地部署 vs API 调用
# =============================================================================

# -----------------------------------------------------------------------------
# 运行前检查
# -----------------------------------------------------------------------------
# 1. 已安装依赖：pip install sentence-transformers
# 2. 首次运行会下载模型（约 100-500MB）
# 3. 需要约 1GB 内存
# -----------------------------------------------------------------------------

# 设置 UTF-8 编码（Windows 专用）
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')



# =============================================================================
# 第一部分：本地 Embedding 模型简介
# =============================================================================
"""
本地 Embedding 模型 vs API 调用

📊 对比

| 维度 | 本地模型 | API 调用 |
|------|----------|----------|
| 成本 | 一次下载，免费使用 | 按调用收费 |
| 速度 | 本地推理，无网络延迟 | 网络传输延迟 |
| 隐私 | 数据不出本地 | 数据发送到云端 |
| 维护 | 需要自己管理模型 | 无需管理 |
| 效果 | 取决于选择的模型 | 通常是最新最强 |

💡 何时选择本地模型？
   ✓ 数据敏感，不能上传到云端
   ✓ 调用量大，API 成本过高
   ✓ 需要离线运行
   ✓ 低延迟要求

⚠️ 何时选择 API？
   ✓ 不想管理模型文件
   ✓ 需要最新最强的模型
   ✓ 调用量小，成本可接受
   ✓ 追求最简单的使用方式
"""


# =============================================================================
# 示例 1: 基础调用（sentence-transformers）
# =============================================================================

def basic_embedding_with_sentence_transformers():
    """
    使用 sentence-transformers 生成向量
    """
    print("=" * 60)
    print("示例 1: 基础调用（sentence-transformers）")
    print("=" * 60)

    try:
        from sentence_transformers import SentenceTransformer

        # 文本
        text = "人工智能是计算机科学的一个分支"

        print(f"输入文本：{text}")
        print(f"文本长度：{len(text)} 字符\n")

        # 加载模型（首次运行会下载）
        print("正在加载模型...")
        print("  模型：bge-large-zh-v1.5（中文）")
        print("  说明：首次运行需要下载，约 1-2 分钟")
        print("  后续运行会从缓存加载，约 5-10 秒\n")

        # 中文推荐模型
        # bge-large-zh-v1.5: 效果好，1024 维
        # m3e-base: 平衡，768 维
        model = SentenceTransformer('BAAI/bge-large-zh-v1.5')

        print("✓ 模型加载完成\n")

        # 生成向量
        print("正在生成向量...")
        embedding = model.encode(text)

        print(f"✓ 生成成功！")
        print(f"\n向量信息：")
        print(f"  维度：{len(embedding)}")
        print(f"  数据类型：{embedding.dtype}")
        print(f"  前 10 元素：{embedding[:10]}")
        print(f"  向量范数：{sum(x*x for x in embedding)**0.5:.4f}")

        return model

    except ImportError:
        print("✗ 需要安装：pip install sentence-transformers")
        print("\n建议安装命令：")
        print("  pip install sentence-transformers -i https://pypi.tuna.tsinghua.edu.cn/simple")
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
        from sentence_transformers import SentenceTransformer

        # 批量文本
        texts = [
            "机器学习是人工智能的核心技术",
            "深度学习使用神经网络进行表征学习",
            "自然语言处理让计算机理解人类语言",
            "计算机视觉用于图像识别和分析",
            "推荐系统根据用户行为推荐内容",
        ]

        print(f"批量处理 {len(texts)} 条文本：\n")
        for i, text in enumerate(texts, 1):
            print(f"  [{i}] {text}")
        print()

        # 加载模型
        model = SentenceTransformer('BAAI/bge-large-zh-v1.5')

        # 批量生成
        print("正在批量生成向量...")
        embeddings = model.encode(texts, show_progress_bar=True)

        print(f"\n✓ 批量生成完成！")
        print(f"\n向量信息：")
        print(f"  数量：{len(embeddings)}")
        print(f"  维度：{embeddings.shape[1]}")
        print(f"  数据类型：{embeddings.dtype}")

        # 显示每个向量的前 5 个元素
        print(f"\n向量预览（前 5 元素）：")
        for i, emb in enumerate(embeddings):
            print(f"  [{i+1}] {emb[:5]}")

    except ImportError:
        print("✗ 需要安装：pip install sentence-transformers")
    except Exception as e:
        print(f"✗ 错误：{e}")


# =============================================================================
# 示例 3: 相似度计算应用
# =============================================================================

def similarity_calculation():
    """
    使用 Embedding 计算文本相似度
    """
    print()
    print("=" * 60)
    print("示例 3: 相似度计算应用")
    print("=" * 60)

    try:
        from sentence_transformers import SentenceTransformer, util
        import torch

        # 加载模型
        print("加载模型...")
        model = SentenceTransformer('BAAI/bge-large-zh-v1.5')

        # 文本对
        text_pairs = [
            ("机器学习", "深度学习"),
            ("机器学习", "人工智能"),
            ("机器学习", "红烧肉"),
            ("自然语言处理", "文本分析"),
            ("计算机视觉", "图像处理"),
            ("Python 编程", "红烧肉做法"),
        ]

        print("\n计算文本对的相似度：\n")
        print("-" * 60)

        for text1, text2 in text_pairs:
            # 生成向量
            emb1 = model.encode(text1, convert_to_tensor=True)
            emb2 = model.encode(text2, convert_to_tensor=True)

            # 计算余弦相似度
            sim = util.cos_sim(emb1, emb2)[0][0].item()

            # 可视化
            bar_len = int(sim * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)

            print(f"{text1} ↔ {text2}")
            print(f"  相似度：{bar} {sim:.3f}")
            print()

        print("💡 观察：")
        print("   - 语义相近的词（如机器学习 - 深度学习）相似度高")
        print("   - 语义无关的词（如机器学习 - 红烧肉）相似度低")

    except ImportError:
        print("✗ 需要安装：pip install sentence-transformers")
    except Exception as e:
        print(f"✗ 错误：{e}")


# =============================================================================
# 示例 4: 语义检索演示
# =============================================================================

def semantic_search_demo():
    """
    简单的语义检索演示
    """
    print()
    print("=" * 60)
    print("示例 4: 语义检索演示")
    print("=" * 60)

    try:
        from sentence_transformers import SentenceTransformer, util

        # 文档库
        documents = [
            "机器学习是人工智能的核心技术，通过数据训练模型。",
            "深度学习使用多层神经网络，是机器学习的重要分支。",
            "自然语言处理让计算机理解和生成人类语言。",
            "计算机视觉让计算机能够看懂图像和视频。",
            "推荐系统根据用户历史行为推荐相关内容。",
            "知识图谱用图结构存储和表示知识。",
            "大语言模型是基于海量文本训练的 AI 模型。",
            "强化学习通过奖励机制训练智能体做出决策。",
        ]

        print("文档库：")
        for i, doc in enumerate(documents, 1):
            print(f"  [{i}] {doc}")
        print()

        # 加载模型
        model = SentenceTransformer('BAAI/bge-large-zh-v1.5')

        # 预计算所有文档的向量
        print("预计算文档向量...")
        doc_embeddings = model.encode(documents, convert_to_tensor=True)
        print("✓ 预计算完成\n")

        # 测试查询
        queries = [
            "机器学习需要什么？",
            "AI 模型有哪些应用？"
        ]

        for query in queries:
            print(f"查询：'{query}'")
            print("-" * 50)

            # 生成查询向量
            query_embedding = model.encode(query, convert_to_tensor=True)

            # 计算相似度
            cos_scores = util.cos_sim(query_embedding, doc_embeddings)[0]

            # 排序，取 Top-3
            top_results = torch.topk(cos_scores, k=3)

            print(f"检索结果（Top-3）：\n")
            for i, (score, idx) in enumerate(zip(top_results.values[0], top_results.indices[0])):
                print(f"  [{i+1}] 相似度：{score:.4f}")
                print(f"      文档：{documents[idx]}")
            print()

    except ImportError:
        print("✗ 需要安装：pip install sentence-transformers")
    except Exception as e:
        print(f"✗ 错误：{e}")


# =============================================================================
# 示例 5: 常用中文 Embedding 模型
# =============================================================================

def chinese_embedding_models():
    """
    常用的中文 Embedding 模型介绍
    """
    print()
    print("=" * 60)
    print("示例 5: 常用中文 Embedding 模型")
    print("=" * 60)

    print("""
┌─────────────────────────────────────────────────────────┐
│ 常用中文 Embedding 模型（sentence-transformers 可加载）   │
├──────────────────────┬─────────┬─────────┬──────────────┤
│        模型           │  维度   │  大小   │    特点       │
├──────────────────────┼─────────┼─────────┼──────────────┤
│ bge-large-zh-v1.5    │ 1024    │ ~1.2GB  │ 效果最好     │
│ bge-base-zh-v1.5     │ 768     │ ~500MB  │ 平衡性能     │
│ bge-small-zh-v1.5    │ 512     │ ~100MB  │ 轻量快速     │
├──────────────────────┼─────────┼─────────┼──────────────┤
│ m3e-base             │ 768     │ ~500MB  │ MokaAI 出品  │
│ m3e-large            │ 1024    │ ~1.2GB  │ 大模型版     │
├──────────────────────┼─────────┼─────────┼──────────────┤
│ text2vec-base-chinese│ 768     │ ~500MB  │ 早期经典     │
└──────────────────────┴─────────┴─────────┴──────────────┘

💡 模型选择建议:

1. 追求效果
   → bge-large-zh-v1.5

2. 平衡性能
   → bge-base-zh-v1.5 或 m3e-base

3. 资源受限
   → bge-small-zh-v1.5

📦 安装方式:

```python
from sentence_transformers import SentenceTransformer

# 加载模型（首次自动下载）
model = SentenceTransformer('BAAI/bge-large-zh-v1.5')

# 或者使用其他模型
model = SentenceTransformer('m3e-base')
model = SentenceTransformer('bge-small-zh-v1.5')
```

🌐 模型来源:
   - HuggingFace: https://huggingface.co/BAAI
   - 国内镜像：https://www.modelscope.cn/
""")


# =============================================================================
# 示例 6: 本地模型最佳实践
# =============================================================================

def local_model_best_practices():
    """
    本地模型最佳实践
    """
    print()
    print("=" * 60)
    print("示例 6: 本地模型最佳实践")
    print("=" * 60)

    print("""
┌─────────────────────────────────────────────────────────┐
│ 本地 Embedding 模型最佳实践                               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ 1. 模型缓存                                             │
│    - 模型首次下载后缓存在 ~/.cache/torch/               │
│    - 可设置环境变量修改缓存位置                         │
│                                                         │
│ 2. 性能优化                                             │
│    - 使用 GPU 加速（如有）                              │
│    - 批量处理比单个处理更高效                           │
│    - 预计算文档向量，避免重复计算                       │
│                                                         │
│ 3. 内存管理                                             │
│    - 大模型需要约 1-2GB 内存                             │
│    - 使用 half 精度可减少一半内存                        │
│    - 及时释放不用的模型：del model                      │
│                                                         │
│ 4. 长文本处理                                           │
│    - 模型有最大长度限制（如 512 tokens）                │
│    - 长文本需要分段处理，然后取平均                     │
│                                                         │
└─────────────────────────────────────────────────────────┘

💡 代码技巧:

```python
# GPU 加速（如有）
model = SentenceTransformer('bge-large-zh-v1.5', device='cuda')

# Half 精度（节省内存）
model = SentenceTransformer('bge-large-zh-v1.5', device='cpu',
                           trust_remote_code=True)
model.half()

# 批量处理
embeddings = model.encode(large_text_list, batch_size=32,
                          show_progress_bar=True)

# 长文本分段
def encode_long_text(model, text, max_length=512):
    chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
    embeddings = model.encode(chunks)
    return embeddings.mean(axis=0)  # 取平均
```
""")


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  Embedding 示例 - 本地 Embedding 模型")
    print("  说明：使用 sentence-transformers 调用本地模型")
    print("=" * 70 + "\n")

    print("【说明】")
    print("  本地模型无需 API Key，下载后免费使用")
    print("  首次运行需要下载模型，约 1-2 分钟")
    print()

    print("【安装命令】")
    print("  pip install sentence-transformers")
    print()

    # 运行示例
    print("-" * 60)

    # 提示
    print("提示：以下示例需要安装 sentence-transformers\n")

    basic_embedding_with_sentence_transformers()
    print()

    batch_embedding()
    print()

    similarity_calculation()
    print()

    semantic_search_demo()
    print()

    chinese_embedding_models()
    print()

    local_model_best_practices()

    print()
    print("=" * 70)
    print("  本地 Embedding 模型学习完成！")
    print("  接下来可以学习：")
    print("  - 04_embedding_comparison.py（模型对比）")
    print("=" * 70 + "\n")
