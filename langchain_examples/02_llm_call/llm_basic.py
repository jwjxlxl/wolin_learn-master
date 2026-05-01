# =============================================================================
# 基础 LLM 调用
# =============================================================================
#  
# 用途：教学演示 - 使用 LangChain 调用 LLM
#
# 核心概念：
#   - LLM vs Chat Model 的区别
#     - LLM: 文本补全（像续写句子）
#     - Chat Model: 对话交互（像聊天）
#   - 本地模型 (Ollama) vs 云端 API (Qwen/DeepSeek)
# =============================================================================

# -----------------------------------------------------------------------------
# 运行前检查
# -----------------------------------------------------------------------------
# 1. 已安装 Ollama 服务（使用本地模型时）
# 2. 已下载模型：ollama pull qwen3.5:2b
# 3. 已安装依赖：pip install -r requirements.txt
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

# =============================================================================
# 第一部分：理解 LLM 和 Chat Model 的区别
# =============================================================================
"""
LLM（语言模型）vs Chat Model（对话模型）

想象两种不同的交流方式：

📝 LLM - 文本补全
   你："今天天气真好，我想出去"
   LLM："......散步。公园里的花开得很美。"
   （它把你的话当成一个句子的开头，帮你续写）

💬 Chat Model - 对话交互
   你："今天天气真好，我想出去"
   Chat Model："是啊，这么好的天气，出去走走吧！想去哪里？"
   （它理解你在说话，并给出回应）

结论：对话应用优先使用 Chat Model！
"""


# =============================================================================
# 第二部分：使用 Ollama 本地模型
# =============================================================================

def ollama_chat_example():
    """
    使用 Ollama 本地模型进行对话

    优点：
    - 免费，无需 API Key
    - 数据本地处理，隐私安全
    - 可离线使用

    缺点：
    - 需要本地 GPU 资源
    - 模型相对较小
    """
    print("=" * 60)
    print("Ollama 本地模型调用")
    print("=" * 60)

    from langchain_ollama import ChatOllama

    # 创建 Chat Model 实例
    # model: 指定使用的模型名称（需与 Ollama 中的一致）
    model = ChatOllama(model="qwen3.5:2b")

    # 调用模型
    # invoke: 最基础的调用方法，传入消息内容
    response = model.invoke("你好，请简单介绍一下你自己。")

    # 打印结果
    # response.content: 获取模型回复的文本内容
    print(f"AI 回复：{response.content}")
    print()


def ollama_with_messages_example():
    """
    使用多轮对话历史调用

    Chat Model 支持传入消息列表，包含角色信息
    """
    print("=" * 60)
    print("Ollama 多轮对话")
    print("=" * 60)

    from langchain_ollama import ChatOllama
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

    model = ChatOllama(model="qwen3.5:2b")

    # 构建消息列表
    messages = [
        # SystemMessage: 系统提示词，设定 AI 的角色和行为
        SystemMessage(content="你是一位友好的助手，说话简洁。"),

        # HumanMessage: 用户的消息
        HumanMessage(content="我叫小明，今年 10 岁。"),

        # AIMessage: AI 的回复（手动添加，用于模拟历史对话）
        AIMessage(content="你好小明！很高兴认识你。有什么问题吗？"),

        # 新的用户消息
        HumanMessage(content="你喜欢什么颜色？"),
    ]

    # 传入完整消息列表，AI 会根据上下文回复
    response = model.invoke(messages)

    print(f"AI 回复：{response.content}")
    print()


# =============================================================================
# 第三部分：使用云端 API（阿里云 Qwen）
# =============================================================================

def qwen_chat_example():
    """
    使用阿里云 Qwen API 进行对话

    优点：
    - 模型强大，效果好
    - 无需本地 GPU
    - 按使用付费

    缺点：
    - 需要网络
    - 数据上传到云端
    """
    print("=" * 60)
    print("阿里云 Qwen 调用")
    print("=" * 60)

    try:
        from langchain_openai import ChatOpenAI
        import os
        from dotenv import load_dotenv

        # 加载环境变量
        load_dotenv()

        # 检查 API Key
        api_key = os.getenv("ALIYUN_API_KEY")
        if not api_key or api_key == "sk-your-aliyun-api-key-here":
            print("未配置 API Key，跳过此示例")
            print("提示：在 .env 文件中配置 ALIYUN_API_KEY")
            print()
            return

        # 创建 Chat Model 实例
        model = ChatOpenAI(
            model="qwen-plus",              # 模型名称
            api_key=api_key,                # API Key
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 阿里云兼容模式地址
        )

        # 调用模型
        response = model.invoke("你好，请简单介绍一下你自己。")

        print(f"AI 回复：{response.content}")
        print()

    except ImportError:
        print("未安装 langchain-openai")
        print("提示：pip install langchain-openai")
        print()


# =============================================================================
# 第四部分：使用 DeepSeek API
# =============================================================================

def deepseek_chat_example():
    """
    使用 DeepSeek API 进行对话

    DeepSeek 特点：
    - 性价比高
    - 代码生成能力强（deepseek-coder）
    - 对话模型效果好（deepseek-chat）
    """
    print("=" * 60)
    print("DeepSeek 调用")
    print("=" * 60)

    try:
        from langchain_openai import ChatOpenAI
        import os
        from dotenv import load_dotenv

        load_dotenv()

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key or api_key == "sk-your-deepseek-api-key-here":
            print("未配置 API Key，跳过此示例")
            print("提示：在 .env 文件中配置 DEEPSEEK_API_KEY")
            print()
            return

        # DeepSeek 也兼容 OpenAI API 格式
        model = ChatOpenAI(
            model="deepseek-chat",          # 对话模型
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )

        response = model.invoke("你好，请简单介绍一下你自己。")

        print(f"AI 回复：{response.content}")
        print()

    except ImportError:
        print("未安装 langchain-openai")
        print("提示：pip install langchain-openai")
        print()


# =============================================================================
# 第五部分：统一封装（推荐写法）
# =============================================================================

def get_model(provider="ollama"):
    """
    根据服务商返回对应的模型实例

    这样写的好处：
    - 方便切换模型
    - 代码集中管理配置
    """
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model="qwen3.5:2b")

    elif provider == "qwen":
        from langchain_openai import ChatOpenAI
        import os
        from dotenv import load_dotenv

        load_dotenv()
        return ChatOpenAI(
            model="qwen-plus",
            api_key=os.getenv("ALIYUN_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    elif provider == "deepseek":
        from langchain_openai import ChatOpenAI
        import os
        from dotenv import load_dotenv

        load_dotenv()
        return ChatOpenAI(
            model="deepseek-chat",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )

    else:
        raise ValueError(f"不支持的服务商：{provider}")


def unified_chat_example():
    """
    使用统一封装的模型调用

    推荐在实际项目中使用这种方式
    """
    print("=" * 60)
    print("统一封装调用（推荐）")
    print("=" * 60)

    # 切换模型只需改这一行
    model = get_model("ollama")
    # model = get_model("qwen")       # 切换到 Qwen
    # model = get_model("deepseek")   # 切换到 DeepSeek

    response = model.invoke("你好，请简单介绍一下你自己。")
    print(f"AI 回复：{response.content}")
    print()


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  LangChain 基础 LLM 调用")
    print("  说明：LLM vs Chat Model 的区别")
    print("=" * 70 + "\n")

    print("【运行前检查】")
    print("  1. 已安装 Ollama 服务（使用本地模型时）")
    print("  2. 已下载模型：ollama pull qwen3.5:2b")
    print("  3. 已安装依赖：pip install -r requirements.txt")
    print()

    # 运行示例
    ollama_chat_example()
    # ollama_with_messages_example()

    # 云端 API 示例（有 API Key 可取消注释）
    # qwen_chat_example()
    # deepseek_chat_example()

    unified_chat_example()

    print("=" * 70)
    print("  接下来学习：chat_model.py（深入理解消息类型）")
    print("=" * 70 + "\n")
