"""通过共享 get_model("qwen") 获取在线客户端，适配 DeepEval 的裁判接口。"""

import math
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# 直接运行案例文件时也能导入根目录 utils；系统环境变量优先。
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
load_dotenv(ROOT_DIR / ".env")
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("DEEPEVAL_DISABLE_PROGRESS_BAR", "1")

from deepeval.models import DeepEvalBaseLLM
from utils.model_utils import get_model


class QwenJudge(DeepEvalBaseLLM):
    """仅将 LangChain 的同步/异步调用转成 DeepEval 所需的文本或结构化结果。"""

    def __init__(self, client):
        self.client = client
        super().__init__(model=client.model_name)

    def load_model(self):
        return self.client

    def generate(self, prompt, schema=None):
        if schema is not None:
            return self.model.with_structured_output(schema, method="function_calling").invoke(prompt)
        return self.model.invoke(prompt).content

    async def a_generate(self, prompt, schema=None):
        if schema is not None:
            return await self.model.with_structured_output(schema, method="function_calling").ainvoke(prompt)
        return (await self.model.ainvoke(prompt)).content

    def get_model_name(self):
        return self.model.model_name

    def supports_log_probs(self):
        return False


def get_judge():
    """配置有误时给出中文指引并返回 None，调用方停止评分。"""
    api_key = os.getenv("ALIYUN_API_KEY", "").strip()
    if not api_key or api_key in {"your_aliyun_api_key_here", "sk-your-aliyun-api-key-here"}:
        print("已跳过评分：请在根目录 .env 配置 ALIYUN_API_KEY；可从阿里云百炼控制台获取。")
        return None
    try:
        timeout = float(os.getenv("DEEPEVAL_TIMEOUT_SECONDS", "120"))
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError
    except ValueError:
        print("配置错误：DEEPEVAL_TIMEOUT_SECONDS 必须是有限正数。")
        return None
    client = get_model("qwen", temperature=0, timeout=timeout, max_retries=1)
    return QwenJudge(client)
