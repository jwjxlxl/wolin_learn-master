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


def get_model(provider: str = None):
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
