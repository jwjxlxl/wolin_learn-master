# =============================================================================
# HTTP 工具模块
# =============================================================================
#  
# 用途：封装统一的 HTTP 响应格式
# =============================================================================

from typing import Optional, Any, Generic, TypeVar
from pydantic import BaseModel, Field

T = TypeVar("T")


class HttpResponse(BaseModel, Generic[T]):
    """
    统一 HTTP 响应类

    所有接口返回统一格式:
    {
        "code": 200,          # 状态码，200 表示成功，其他表示错误
        "message": "success", # 响应消息（失败时为错误信息）
        "data": {...}         # 响应数据，成功时返回
    }
    """
    code: int = Field(default=200, description="状态码，200 表示成功")
    message: str = Field(default="success", description="响应消息")
    data: Optional[T] = Field(default=None, description="响应数据")

    @classmethod
    def success(cls, data: Any = None, message: str = "success", code: int = 200) -> "HttpResponse":
        """成功响应"""
        return cls(code=code, message=message, data=data)

    @classmethod
    def error(cls, message: str = "error", code: int = 500) -> "HttpResponse":
        """错误响应"""
        return cls(code=code, message=message, data=None)


# =============================================================================
# 快捷响应函数
# =============================================================================

def ok(data: Any = None, message: str = "success") -> HttpResponse:
    """成功响应快捷方式"""
    return HttpResponse.success(data=data, message=message)


def fail(message: str = "error", code: int = 500) -> HttpResponse:
    """失败响应快捷方式"""
    return HttpResponse.error(message=message, code=code)


# =============================================================================
# 常用状态码常量
# =============================================================================

class StatusCodes:
    """常用状态码常量"""
    SUCCESS = 200
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    INTERNAL_ERROR = 500
    SERVICE_UNAVAILABLE = 503


__all__ = ["HttpResponse", "ok", "fail", "StatusCodes"]
