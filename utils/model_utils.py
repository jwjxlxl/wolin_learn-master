"""
项目共享模型工具模块

提供统一的模型获取接口：
  - get_qwen_client():  阿里云 Qwen 云端模型（需要 API Key）
  - get_model():        通用模型获取（默认 Ollama 本地，可选云端）
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


def get_qwen_client(model_name="qwen-plus"):
    """
    获取阿里云通义千问 (Qwen) 大模型客户端实例。

    Args:
        model_name: 模型名称，默认为 qwen-plus

    Returns:
        ChatOpenAI 实例，如果配置无效则返回 None
    """
    load_dotenv()

    api_key = os.getenv("ALIYUN_API_KEY")
    if not api_key or api_key == "sk-your-aliyun-api-key-here":
        print("警告：未配置有效的 ALIYUN_API_KEY")
        print("提示：请在 .env 文件中配置你的阿里云 DashScope API Key")
        return None

    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )


def get_model(use_cloud: bool = False, model_name: str = None):
    """
    获取模型实例（默认 Ollama 本地，与 langchain_examples 一致）。

    Args:
        use_cloud: True=使用阿里云 Qwen（需 ALIYUN_API_KEY）
                   False=使用本地 Ollama qwen3.5:2b（默认）
        model_name: 自定义模型名（Ollama 默认 qwen3.5:2b，云端默认 qwen-plus）

    Returns:
        ChatModel 实例，如果不可用则返回 None
    """
    if use_cloud:
        return get_qwen_client(model_name or "qwen-plus")

    # 默认：Ollama 本地模型
    try:
        from langchain_ollama import ChatOllama
        ollama_model = model_name or "qwen3.5:2b"
        return ChatOllama(model=ollama_model)
    except ImportError:
        print("警告：未安装 langchain-ollama，请运行：pip install langchain-ollama")
        print("  或使用云端 API：get_model(use_cloud=True)")
        return None
