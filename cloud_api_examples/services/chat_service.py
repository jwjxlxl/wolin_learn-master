# =============================================================================
# Chat 服务模块
# =============================================================================
#  
# 用途：封装通用的 Chat 调用逻辑，支持流式和非流式输出
# =============================================================================

import os
import json
from typing import List, Optional, AsyncGenerator, Union
from openai import OpenAI


# =============================================================================
# 消息类型定义
# =============================================================================

class ChatMessage(dict):
    """
    单条消息
    支持 role: system, user, assistant
    """
    def __init__(self, role: str, content: str):
        super().__init__({"role": role, "content": content})

    @classmethod
    def system(cls, content: str) -> "ChatMessage":
        return cls("system", content)

    @classmethod
    def user(cls, content: str) -> "ChatMessage":
        return cls("user", content)

    @classmethod
    def assistant(cls, content: str) -> "ChatMessage":
        return cls("assistant", content)


# =============================================================================
# 云 API 客户端管理
# =============================================================================

class ChatClientManager:
    """管理不同云服务的 OpenAI 客户端"""

    _clients: dict = {}

    @classmethod
    def register(cls, name: str, client: OpenAI):
        """注册一个客户端"""
        cls._clients[name.lower()] = client

    @classmethod
    def get(cls, model: str) -> OpenAI:
        """根据模型名称获取客户端"""
        model_lower = model.lower()

        # 按优先级匹配
        for name, client in cls._clients.items():
            if name in model_lower:
                return client

        # 默认返回第一个客户端
        if cls._clients:
            return list(cls._clients.values())[0]
        raise ValueError("未注册任何客户端")

    @classmethod
    def list_models(cls) -> List[dict]:
        """列出所有可用模型"""
        return [{"id": name, "object": "model"} for name in cls._clients.keys()]


# 初始化默认客户端
def init_default_clients():
    """初始化默认的云服务客户端"""
    from dotenv import load_dotenv
    load_dotenv()

    # 阿里云百炼 (Qwen)
    ChatClientManager.register("qwen", OpenAI(
        api_key=os.getenv("ALIYUN_API_KEY", "sk-placeholder"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    ))

    # DeepSeek
    ChatClientManager.register("deepseek", OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY", "sk-placeholder"),
        base_url="https://api.deepseek.com"
    ))

    # vLLM 本地
    ChatClientManager.register("vllm", OpenAI(
        api_key="not-needed",
        base_url="http://localhost:8000/v1"
    ))

    # 注册别名
    ChatClientManager.register("aliyun", ChatClientManager._clients.get("qwen"))
    ChatClientManager.register("dashscope", ChatClientManager._clients.get("qwen"))
    ChatClientManager.register("local", ChatClientManager._clients.get("vllm"))


# =============================================================================
# Chat 服务核心类
# =============================================================================

class ChatService:
    """
    Chat 服务类

    用法:
        # 简单调用（仅 user 消息）
        ChatService.chat(model="qwen-plus", question="你好")

        # 完整调用（支持 system 和多轮对话）
        ChatService.chat(
            model="qwen-plus",
            messages=[
                ChatMessage.system("你是一个助手"),
                ChatMessage.user("你好"),
                ChatMessage.assistant("你好！有什么可以帮你？"),
                ChatMessage.user("继续"),
            ]
        )

        # 流式调用
        async for chunk in ChatService.chat_stream(...):
            print(chunk["content"])
    """

    @staticmethod
    def get_client(model: str) -> OpenAI:
        """获取模型对应的客户端"""
        return ChatClientManager.get(model)

    @staticmethod
    def chat(
        model: str,
        question: Optional[str] = None,
        messages: Optional[List[Union[ChatMessage, dict]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.9,
    ) -> str:
        """
        非流式 Chat 调用

        Args:
            model: 模型名称
            question: 用户问题（可选，与 messages 二选一）
            messages: 消息列表（可选，支持 system 和多轮对话）
            temperature: 温度参数
            max_tokens: 最大 token 数
            top_p: Top-p 采样

        Returns:
            模型回复内容（字符串）
        """
        client = ChatService.get_client(model)

        # 构建消息列表
        if messages:
            msg_list = [dict(m) if isinstance(m, ChatMessage) else m for m in messages]
        elif question:
            msg_list = [{"role": "user", "content": question}]
        else:
            raise ValueError("必须提供 question 或 messages 参数")

        response = client.chat.completions.create(
            model=model,
            messages=msg_list,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p
        )

        return response.choices[0].message.content

    @staticmethod
    async def chat_stream(
        model: str,
        question: Optional[str] = None,
        messages: Optional[List[Union[ChatMessage, dict]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.9,
    ) -> AsyncGenerator[dict, None]:
        """
        流式 Chat 调用

        Args:
            model: 模型名称
            question: 用户问题（可选）
            messages: 消息列表（可选）
            temperature: 温度参数
            max_tokens: 最大 token 数
            top_p: Top-p 采样

        Yields:
            {"content": str} 每次 yield 一个内容块
        """
        client = ChatService.get_client(model)

        # 构建消息列表
        if messages:
            msg_list = [dict(m) if isinstance(m, ChatMessage) else m for m in messages]
        elif question:
            msg_list = [{"role": "user", "content": question}]
        else:
            raise ValueError("必须提供 question 或 messages 参数")

        stream = client.chat.completions.create(
            model=model,
            messages=msg_list,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield {"content": chunk.choices[0].delta.content}

        yield {"done": True}


# =============================================================================
# 快捷函数（无需类即可调用）
# =============================================================================

def chat(
    model: str,
    question: Optional[str] = None,
    messages: Optional[List[Union[ChatMessage, dict]]] = None,
    **kwargs
) -> str:
    """快捷 chat 函数"""
    return ChatService.chat(model, question=question, messages=messages, **kwargs)


async def chat_stream(
    model: str,
    question: Optional[str] = None,
    messages: Optional[List[Union[ChatMessage, dict]]] = None,
    **kwargs
) -> AsyncGenerator[dict, None]:
    """快捷流式 chat 函数"""
    async for chunk in ChatService.chat_stream(model, question=question, messages=messages, **kwargs):
        yield chunk


# =============================================================================
# 初始化
# =============================================================================

init_default_clients()

__all__ = [
    "ChatService",
    "ChatClientManager",
    "ChatMessage",
    "chat",
    "chat_stream",
]
