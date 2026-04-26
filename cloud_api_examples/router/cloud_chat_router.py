# =============================================================================
# Cloud Chat 路由模块 - OpenAI 风格
# =============================================================================
#  
# 用途：封装统一的 Chat 接口路由，支持流式输出，兼容 OpenAI API 格式
# =============================================================================

import sys
sys.path.append("/")

import json
from typing import List, Optional, AsyncGenerator

from fastapi import APIRouter, FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# 导入统一响应类和 Chat 服务
from utils.http_utils import ok, fail, StatusCodes
from cloud_api_examples.services.chat_service import ChatService, ChatMessage


# -----------------------------------------------------------------------------
# 数据模型
# -----------------------------------------------------------------------------

class ChatMessageRequest(BaseModel):
    """单条消息"""
    role: str = Field(..., description="消息角色：system, user, assistant")
    content: str = Field(..., description="消息内容")


class ChatRequest(BaseModel):
    """请求体 - 支持 question 和 messages 两种模式"""
    model: str = Field(default="qwen-plus", description="模型名称")
    question: Optional[str] = Field(default=None, description="用户问题（简单模式）")
    messages: Optional[List[ChatMessageRequest]] = Field(default=None, description="消息列表（多轮对话模式）")
    temperature: Optional[float] = Field(default=0.7, ge=0, le=2, description="温度参数")
    max_tokens: Optional[int] = Field(default=1024, ge=1, description="最大生成 token 数")
    stream: Optional[bool] = Field(default=False, description="是否流式输出")
    top_p: Optional[float] = Field(default=0.9, ge=0, le=1, description="Top-p 采样")


class ChatResponse(BaseModel):
    """响应体 - 只返回答案内容"""
    answer: str


# -----------------------------------------------------------------------------
# 路由定义
# -----------------------------------------------------------------------------

router = APIRouter(prefix="/v1", tags=["Cloud Chat API"])



@router.post("/chat/completions")
async def chat_completions(request: ChatRequest):
    """
    Chat Completion 接口

    两种调用模式:

    1. 简单模式（仅 user 消息）:
    ```json
    {
        "model": "qwen-plus",
        "question": "你好"
    }
    ```

    2. 多轮对话模式（支持 system 和历史对话）:
    ```json
    {
        "model": "qwen-plus",
        "messages": [
            {"role": "system", "content": "你是一个助手"},
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！有什么可以帮你？"},
            {"role": "user", "content": "继续"}
        ]
    }
    ```

    返回:
    - 非流式：HttpResponse{data: {answer: "答案内容"}}
    - 流式：SSE 格式
    """
    try:
        if request.stream:
            return StreamingResponse(
                generate_stream_response(request),
                media_type="text/event-stream"
            )
        else:
            # 构建消息列表
            if request.messages:
                messages = [ChatMessage(m.role, m.content) for m in request.messages]
                answer = ChatService.chat(request.model, messages=messages)
            elif request.question:
                answer = ChatService.chat(request.model, question=request.question)
            else:
                return fail(message="必须提供 question 或 messages 参数", code=StatusCodes.BAD_REQUEST)

            return ok(data={"answer": answer})

    except Exception as e:
        return fail(message=str(e), code=StatusCodes.INTERNAL_ERROR)


async def generate_stream_response(request: ChatRequest) -> AsyncGenerator[str, None]:
    """生成流式响应数据"""
    try:
        # 构建消息列表
        if request.messages:
            messages = [ChatMessage(m.role, m.content) for m in request.messages]
        elif request.question:
            messages = [ChatMessage.user(request.question)]
        else:
            yield f"data: {json.dumps({'error': '必须提供 question 或 messages 参数'}, ensure_ascii=False)}\n\n"
            return

        async for chunk in ChatService.chat_stream(
            request.model,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p
        ):
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"


# -----------------------------------------------------------------------------
# 路由注册函数
# -----------------------------------------------------------------------------

def register_cloud_chat_routes(app: FastAPI, prefix: str = "/v1"):
    """
    将 Cloud Chat 路由注册到 FastAPI 应用

    Args:
        app: FastAPI 应用实例
        prefix: 路由前缀，默认 "/v1"
    """
    cloud_router = APIRouter(prefix=prefix, tags=["Cloud Chat API"])

    for route in router.routes:
        cloud_router.routes.append(route)

    app.include_router(cloud_router)


# 导出
__all__ = ["router", "register_cloud_chat_routes", "ChatRequest", "ChatMessageRequest", "ChatService", "ChatMessage"]
