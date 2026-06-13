# =============================================================================
# LLM 基础调用 — 用 LangChain 调用各种大模型
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 用本地 Ollama 模型进行单轮和多轮对话
#   ✅ 理解 SystemMessage / HumanMessage / AIMessage 三种角色
#   ✅ 封装统一的模型切换函数（本地 ↔ 云端自由切换）
# =============================================================================

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)


# =============================================================================
# 示例 1: 本地 Ollama 基础调用
# =============================================================================

def ollama_basic():
    """
    使用 Ollama 本地模型进行单轮对话。

    核心概念：
    - ChatOllama: LangChain 用于连接 Ollama 的接口
    - model.invoke("问题"): 发送问题，等待完整回复
    - response.content: 取出纯文本内容

    优点：免费、离线可用、数据不离开本机
    缺点：需要本地 GPU/内存，模型能力有限
    """
    from langchain_ollama import ChatOllama

    print(f"\n-- 示例 1: 本地 Ollama 调用")

    model = ChatOllama(model="qwen3.5:2b")
    response = model.invoke("你好，请用一句话介绍你自己。")
    print(f"回复: {response.content}")


# =============================================================================
# 示例 2: 多轮对话（带消息历史）
# =============================================================================

def ollama_multiturn():
    """
    使用消息列表实现多轮对话。

    核心概念：
    - SystemMessage: 设定 AI 的角色和行为准则（像导演给演员剧本）
    - HumanMessage:  用户的输入
    - AIMessage:     AI 的回复（手动添加到历史中，让 AI "记住"之前说过的话）

    为什么需要消息列表？
    LLM 本身是"无状态"的——每次调用都是全新的对话。
    要让它"记住"上下文，必须把历史消息一起传给它。
    """
    from langchain_ollama import ChatOllama
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

    print(f"\n-- 示例 2: 多轮对话（带消息历史）")

    model = ChatOllama(model="qwen3.5:2b")

    messages = [
        SystemMessage(content="你是一位友好的助手，说话简洁。"),
        HumanMessage(content="我叫小明，今年 10 岁。"),
        AIMessage(content="你好小明！很高兴认识你。有什么问题吗？"),
        HumanMessage(content="你还记得我叫什么吗？"),
    ]

    response = model.invoke(messages)
    print(f"回复: {response.content}")


# =============================================================================
# 示例 3: 统一模型封装（本地 ↔ 云端自由切换）
# =============================================================================

def get_model(provider: str = "ollama"):
    """
    根据服务商名称返回对应的模型实例。

    好处：切换模型只改一行参数，不需要到处改代码。
    """
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model="qwen3.5:2b")

    elif provider == "qwen":
        import os
        from dotenv import load_dotenv
        from langchain_openai import ChatOpenAI
        load_dotenv()
        return ChatOpenAI(
            model="qwen-plus",
            api_key=os.getenv("ALIYUN_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    elif provider == "deepseek":
        import os
        from dotenv import load_dotenv
        from langchain_openai import ChatOpenAI
        load_dotenv()
        return ChatOpenAI(
            model="deepseek-chat",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )

    else:
        raise ValueError(f"不支持的服务商: {provider}")


def unified_model_demo():
    """
    演示统一封装的使用方式——切换模型只需改一个字符串。

    这个模式在实际项目中非常实用：
    - 开发阶段用 Ollama（免费、快速迭代）
    - 上线阶段切换到云端 API（能力更强）
    - 不需要改动业务逻辑代码
    """
    print(f"\n-- 示例 3: 统一模型封装")

    # 切换模型只需改这一行: "ollama" / "qwen" / "deepseek"
    model = get_model("ollama")
    response = model.invoke("你好，请用一句话介绍你自己。")
    print(f"回复: {response.content}")

    # 切换到 Qwen 云端模型（需要配置 .env 中的 ALIYUN_API_KEY）
    # model = get_model("qwen")


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n>>> 02_llm_call — LLM 基础调用\n")

    ollama_basic()
    ollama_multiturn()
    unified_model_demo()

    # 接下来学习: chat_model.py（消息类型详解）
