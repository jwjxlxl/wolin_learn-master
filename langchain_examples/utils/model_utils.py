"""统一的模型客户端获取工具

提供 Qwen（阿里云 DashScope）客户端的统一创建逻辑，
所有 Agent 示例文件通过此模块获取模型实例，避免重复配置。
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


def get_qwen_client(model_name: str = "qwen-plus"):
    """获取阿里云通义千问 (Qwen) 大模型客户端实例。

    自动从 .env 文件读取 API Key，无需在每个示例文件中重复配置。

    Args:
        model_name: 模型名称，可选 qwen-plus / qwen-turbo / qwen-max 等
                    默认 qwen-plus（性价比最高的版本）

    Returns:
        ChatOpenAI 实例，如果未配置 API Key 则返回 None
    """
    load_dotenv()

    api_key = os.getenv("ALIYUN_API_KEY")
    if not api_key or api_key == "sk-your-aliyun-api-key-here":
        print("⚠️ 未配置有效的 ALIYUN_API_KEY")
        print("  提示：复制 .env.example 为 .env，填写你的阿里云 DashScope API Key")
        return None

    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
