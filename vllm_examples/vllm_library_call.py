# =============================================================================
# vLLM 库直接调用示例（非 HTTP）
# =============================================================================
#  
# 用途：教学演示 - 展示 vLLM 作为推理引擎库的直接用法
# =============================================================================

# -----------------------------------------------------------------------------
# 核心知识：vLLM 的两种调用方式
# -----------------------------------------------------------------------------
#
# vLLM 的特殊性：既可以作为服务运行，也可以作为库直接导入使用
#
# 方式对比：
# ┌─────────────────────────────────────────────────────────────┐
# │                    用户代码层                                │
# │  方式 1: HTTP 请求       ← 通过 API 调用远程服务              │
# │  方式 2: from vllm import LLM  ← 直接导入库，本地推理        │
# └─────────────────────────────────────────────────────────────┘
#
# 重要区别：
#
# HTTP 调用（vllm_basic_call.py）：
#   - 需要 vLLM 服务已启动（python -m vllm.entrypoints.api_server）
#   - 通过 OpenAI 兼容 API 通信
#   - 适合远程调用、生产部署
#
# 库直接调用（本文件）：
#   - 无需启动服务，直接在代码中初始化
#   - 直接调用底层推理引擎
#   - 适合本地测试、嵌入应用
#
# 调用层次结构：
# ┌─────────────────────────────────────────────────────────────┐
# │   from vllm import LLM    ← 直接导入 vLLM 库                │
# └─────────────────────────────────────────────────────────────┘
#                            ↓
# ┌─────────────────────────────────────────────────────────────┐
# │   llm = LLM(model="...")  ← 初始化推理引擎（加载模型）      │
# └─────────────────────────────────────────────────────────────┘
#                            ↓
# ┌─────────────────────────────────────────────────────────────┐
# │   outputs = llm.generate() ← 执行推理，返回结果             │
# └─────────────────────────────────────────────────────────────┘
#                            ↓
# ┌─────────────────────────────────────────────────────────────┐
# │   GPU/CPU 执行计算                                           │
# └─────────────────────────────────────────────────────────────┘
#
# 为什么用库直接调用？
# 1. 无 HTTP 开销，性能更高
# 2. 无需运行服务，部署简单
# 3. 可以直接访问底层 API
# 4. 适合嵌入 Python 应用
#
# 注意事项：
# - 需要足够的 GPU 显存加载模型
# - 模型会从 HuggingFace 下载（首次运行）
# - 需要安装 vllm 和 torch
#
# -----------------------------------------------------------------------------

# 设置 UTF-8 编码（Windows 专用）
import sys
import io
sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer,
    encoding='utf-8',
    errors='replace',
    line_buffering=True
)

# -----------------------------------------------------------------------------
# 安装依赖
# -----------------------------------------------------------------------------
# pip install vllm torch
#
# 注意：vLLM 需要 PyTorch 支持，首次运行会自动下载模型
# -----------------------------------------------------------------------------


# =============================================================================
# 示例 1: 最简单的文本生成
# =============================================================================
def basic_generation():
    """
    基础文本生成示例

    核心步骤：
    1. 初始化 LLM 对象（加载模型到显存）
    2. 调用 generate() 方法生成文本
    3. 解析输出结果

    与 HTTP 调用的区别：
    - 无需启动服务
    - 直接调用底层引擎
    - 性能更高
    """
    print("=" * 60)
    print("示例 1: 基础文本生成（库直接调用）")
    print("=" * 60)

    # 导入 vLLM 核心类
    from vllm import LLM, SamplingParams

    # 初始化 LLM 对象
    # model: 模型名称（HuggingFace 格式）
    # trust_remote_code: 允许运行远程代码（某些模型需要）
    llm = LLM(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        trust_remote_code=True
    )

    # 配置生成参数
    sampling_params = SamplingParams(
        max_tokens=200,       # 最大生成 token 数
        temperature=0.7,      # 温度（随机性）
        top_p=0.9             # 核采样参数
    )

    # 生成文本
    prompts = ["你好，请简单介绍一下你自己。"]
    outputs = llm.generate(prompts, sampling_params)

    # 输出结果
    for output in outputs:
        print(output.outputs[0].text)

    print()


# =============================================================================
# 示例 2: 批量生成（vLLM 优势功能）
# =============================================================================
def batch_generation():
    """
    批量生成示例

    vLLM 的核心优势：
    - PagedAttention 技术
    - 高效并行处理多个请求
    - 显存利用率更高

    适用场景：
    - 批量处理大量文本
    - 高并发推理服务
    """
    print("=" * 60)
    print("示例 2: 批量生成（vLLM 优势）")
    print("=" * 60)

    from vllm import LLM, SamplingParams

    llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct")

    sampling_params = SamplingParams(
        max_tokens=100,
        temperature=0.7
    )

    # 批量处理多个提示
    prompts = [
        "人工智能是什么？",
        "机器学习和深度学习有什么区别？",
        "什么是神经网络？"
    ]

    outputs = llm.generate(prompts, sampling_params)

    for i, output in enumerate(outputs):
        print(f"问题{i + 1}: {prompts[i]}")
        print(f"回答：{output.outputs[0].text}")
        print("-" * 40)

    print()


# =============================================================================
# 示例 3: 流式生成
# =============================================================================
def streaming_generation():
    """
    流式生成示例

    流式原理：
    - 使用 llm.stream() 方法
    - 逐个返回生成的 token
    - 形成打字机效果
    """
    print("=" * 60)
    print("示例 3: 流式生成（打字机效果）")
    print("=" * 60)

    from vllm import LLM, SamplingParams

    llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct")

    sampling_params = SamplingParams(
        max_tokens=200,
        temperature=0.7
    )

    prompt = "请用 100 字介绍春天。"

    # 使用流式生成
    stream = llm.stream(prompt, sampling_params)

    for output in stream:
        # 获取最新生成的 token
        token = output.outputs[0].text
        print(token, end='', flush=True)

    print("\n")


# =============================================================================
# 示例 4: 对话格式生成（ChatML）
# =============================================================================
def chat_generation():
    """
    对话格式生成示例

    使用 Chat 模板：
    - 将对话历史格式化为模型可理解的格式
    - 支持多轮对话上下文
    - 适用于 instruct 类模型
    """
    print("=" * 60)
    print("示例 4: 对话格式生成")
    print("=" * 60)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    # 加载 tokenizer 用于格式化对话
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

    llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct")

    sampling_params = SamplingParams(
        max_tokens=200,
        temperature=0.7
    )

    # 构建对话历史
    messages = [
        {"role": "system", "content": "你是一位有帮助的助手。"},
        {"role": "user", "content": "你好，我叫小明。"},
        {"role": "assistant", "content": "你好小明！很高兴认识你。"},
        {"role": "user", "content": "我今天心情不太好。"}
    ]

    # 使用 tokenizer 格式化为模型输入
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    outputs = llm.generate([prompt], sampling_params)

    print(outputs[0].outputs[0].text)
    print()


# =============================================================================
# 示例 5: 自定义生成参数
# =============================================================================
def generation_with_params():
    """
    自定义生成参数示例

    常见参数说明：
    - temperature: 随机性（0-1，越高越有创意）
    - max_tokens: 最大生成长度
    - top_p: 核采样
    - top_k: 只从前 k 个 token 采样
    - repetition_penalty: 重复惩罚
    - stop: 停止词
    """
    print("=" * 60)
    print("示例 5: 自定义生成参数")
    print("=" * 60)

    from vllm import LLM, SamplingParams

    llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct")

    # 配置各种参数
    sampling_params = SamplingParams(
        max_tokens=100,
        temperature=0.5,        # 较低温度，输出更稳定
        top_p=0.9,              # 从累积概率 90% 的词中采样
        top_k=50,               # 只从前 50 个 token 采样
        repetition_penalty=1.1, # 轻微惩罚重复内容
        stop=["。", "！", "？"]  # 遇到这些标点停止
    )

    prompt = "请用一句话解释机器学习。"

    outputs = llm.generate([prompt], sampling_params)

    print(f"输入：{prompt}")
    print(f"输出：{outputs[0].outputs[0].text}")
    print()


# =============================================================================
# 示例 6: 错误处理
# =============================================================================
def generation_with_error_handling():
    """
    带错误处理的生成示例

    常见错误：
    - 显存不足 (CUDA out of memory)
    - 模型下载失败
    - 模型加载失败
    """
    print("=" * 60)
    print("示例 6: 错误处理")
    print("=" * 60)

    try:
        from vllm import LLM, SamplingParams

        llm = LLM(
            model="Qwen/Qwen2.5-1.5B-Instruct",
            enforce_eager=True  # 使用 eager 模式，节省显存
        )

        sampling_params = SamplingParams(max_tokens=100)

        outputs = llm.generate(["测试消息"], sampling_params)
        print(outputs[0].outputs[0].text)

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("显存不足！请尝试：")
            print("  1. 使用更小的模型")
            print("  2. 减少 batch size")
            print("  3. 启用 GPU 共享")
        else:
            print(f"运行时错误：{e}")

    except Exception as e:
        print(f"未知错误：{e}")

    print()


# =============================================================================
# 主程序入口
# =============================================================================
if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  vLLM 库直接调用示例（非 HTTP）")
    print("  说明：直接导入 vLLM 库，无需启动服务")
    print("=" * 70 + "\n")

    print("【运行前检查】")
    print("  1. 已安装依赖：pip install vllm torch")
    print("  2. 有足够的 GPU 显存（至少 4GB）")
    print("  3. 首次运行会自动下载模型")
    print()

    print("【库调用 vs HTTP 调用】")
    print("  库调用：直接 import vllm，本地推理，性能更高")
    print("  HTTP:   需要服务运行，远程调用，适合部署")
    print()

    # 取消注释以运行相应示例
    # 注意：首次运行会下载模型，可能需要几分钟
    basic_generation()             # 示例 1: 基础生成
    # batch_generation()           # 示例 2: 批量生成
    # streaming_generation()       # 示例 3: 流式生成
    # chat_generation()            # 示例 4: 对话生成
    # generation_with_params()     # 示例 5: 自定义参数
    # generation_with_error_handling()  # 示例 6: 错误处理

    print("提示：取消注释相应的函数调用来运行示例。")
