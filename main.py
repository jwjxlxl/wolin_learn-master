# =============================================================================
# Main - FastAPI 应用入口
# =============================================================================
#  
# 用途：创建 FastAPI 应用实例，注册各模块路由
# =============================================================================

import uvicorn
from fastapi import FastAPI

# =============================================================================
# 创建 FastAPI 应用实例
# =============================================================================

app = FastAPI(
    title="Wolin Learn API",
    description="LLM 学习与实验 API 聚合服务",
    version="1.0.0"
)


# =============================================================================
# 路由注册
# =============================================================================

# 根路由
@app.get("/")
async def root():
    """根路径，返回 API 信息"""
    return {
        "service": "Wolin Learn API",
        "version": "1.0.0",
        "modules": {
            "cloud_chat": "/v1 (Cloud Chat API - OpenAI 风格)"
        },
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}


# 注册 Cloud Chat 路由（OpenAI 风格，支持流式输出）
from cloud_api_examples.router.cloud_chat_router import register_cloud_chat_routes

register_cloud_chat_routes(app, prefix="/v1")


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Wolin Learn API 服务")
    print("=" * 60)
    print("服务地址：http://localhost:8080")
    print("API 文档：http://localhost:8080/docs")
    print("Cloud Chat: POST /v1/chat/completions")
    print("模型列表：GET /v1/models")
    print("=" * 60)
    print()

    uvicorn.run(app, host="0.0.0.0", port=8099)
