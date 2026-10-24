# =============================================================================
# 本地 Ollama 模型访问示例（qwen3.5:2b）
# =============================================================================
# 用途：演示如何通过 HTTP 接口访问本地 Ollama 部署的模型
#
# Ollama 默认服务地址：http://localhost:11434
# 常用接口：
#   - POST /api/chat      多轮对话（推荐，支持 messages 结构）
#   - POST /api/generate  单次文本生成
#   - GET  /api/tags      列出本地已拉取的模型
#
# 说明：base_url 使用环境变量，方便后续切换为 Nginx 反向代理地址，
#       例如：export OLLAMA_BASE_URL=http://localhost:8080 （Nginx 代理端口）
# =============================================================================

import os
import requests

# 服务地址：优先读环境变量，方便切换到 Nginx 代理；默认为本地直连
# OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# 模型名称：默认 qwen3.5:2b
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:2b")

OLLAMA_BASE_URL = "http://127.0.0.1"


def call_ollama_chat(prompt, model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL,
                     temperature=0.7):
    """
    访问本地 Ollama 的 /api/chat 对话接口（非流式）。

    Args:
        prompt: 用户输入的文本
        model: 模型名称，默认 qwen3.5:2b
        base_url: Ollama 服务地址（或 Nginx 代理地址）
        temperature: 温度，控制输出随机性（0-1）

    Returns:
        模型回复文本；请求失败时返回 None
    """
    url = f"{base_url}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": temperature},
    }

    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]

    except requests.exceptions.ConnectionError:
        print(f"连接失败：无法访问 {base_url}，请确认 Ollama 已启动（ollama serve）")
    except requests.exceptions.Timeout:
        print("请求超时，请稍后重试或调大 timeout")
    except Exception as e:
        print(f"请求失败：{e}")
    return None


# =============================================================================
# 示例
# =============================================================================
if __name__ == "__main__":

    reply = call_ollama_chat("你好，请用一句话介绍你自己。")
    if reply:
        print(f"模型回复：{reply}\n")
