import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


def get_qwen_client(model_name="qwen-plus"):
    """
    获取阿里云通义千问 (Qwen) 大模型客户端实例

    Args:
        model_name (str): 模型名称，默认为 qwen-plus

    Returns:
        ChatOpenAI: LangChain 的聊天模型实例，如果配置无效则返回 None
    """
    # 加载环境变量
    load_dotenv()

    # 检查是否有 API Key
    api_key = os.getenv("ALIYUN_API_KEY")
    if not api_key or api_key == "sk-your-aliyun-api-key-here":
        print("警告：未配置有效的 ALIYUN_API_KEY")
        print("提示：请在 .env 文件中配置你的阿里云 DashScope API Key")
        return None

    # 创建并返回云端模型实例
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
