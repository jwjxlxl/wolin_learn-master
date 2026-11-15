"""离线回归：真实 SDK 配合预设裁判响应，验证调用、门禁和文件启动。"""

import asyncio
import importlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

import judge as judge_module  # 先加载配置，关闭 SDK 遥测。
from judge import QwenJudge, get_judge

EXAMPLES = Path(__file__).resolve().parents[1] / "deepeval_examples"


@pytest.fixture
def scripted(monkeypatch, tmp_path):
    """预设响应只验证 SDK 协议，不代表模型具有真实判断能力。"""
    monkeypatch.chdir(tmp_path)
    calls = Counter()

    def generate(prompt, schema=None):
        name = schema.__name__
        calls[name] += 1
        payloads = {
            "Steps": {"steps": ["核对参考答案", "检查新增事实"]},
            "ReasonScore": {"score": 10 if calls[name] == 1 else 0, "reason": "接口测试响应"},
            "Truths": {"truths": ["保修 18 个月"]},
            "Claims": {"claims": ["保修 18 个月"]},
            "Statements": {"statements": ["保修 18 个月"]},
            "ExtractedPII": {"extracted_pii": ["student@example.com"] if "student@example.com" in prompt else []},
        }
        if name == "Verdicts":
            payload = {"verdicts": [{"verdict": "yes", "reason": "接口测试响应"}]}
        elif name.endswith("ScoreReason"):
            payload = {"reason": "接口测试：student@example.com"}
        else:
            payload = payloads[name]
        return schema.model_validate(payload)

    class FakeClient:
        model_name = "脚本裁判"

        def with_structured_output(self, schema, method):
            assert method == "function_calling"

            async def a_invoke(prompt):
                return generate(prompt, schema)

            return SimpleNamespace(invoke=lambda prompt: generate(prompt, schema), ainvoke=a_invoke)

        def invoke(self, prompt):
            return SimpleNamespace(content="文本响应")

        async def ainvoke(self, prompt):
            return self.invoke(prompt)

    return QwenJudge(FakeClient()), calls


@pytest.mark.parametrize("lesson", ["01_first_evaluation", "02_rag_evaluation"])
def test_scoring_calls_sdk_without_reports(lesson, scripted, tmp_path, capsys):
    judge, calls = scripted
    assert importlib.import_module(lesson).main(judge) == 0
    assert not list(tmp_path.glob("reports/*.json"))
    if lesson.startswith("01"):
        assert calls["Steps"] == 1 and calls["ReasonScore"] == 2
        output = capsys.readouterr().out
        assert "是否通过：True" in output and "是否通过：False" in output
    else:
        assert calls["Claims"] == 3 and calls["Statements"] == 3


@pytest.mark.parametrize("version, expected", [("fixed", 0), ("broken", 1)])
def test_privacy_assertion_and_redacted_log(version, expected, scripted, tmp_path, monkeypatch, capsys):
    module = importlib.import_module("03_safety_gate")
    monkeypatch.setattr(module, "REPORT_DIR", tmp_path / "reports")
    judge, calls = scripted
    assert module.main(version, judge) == expected
    text = next((tmp_path / "reports").glob("*.json")).read_text(encoding="utf-8")
    report = json.loads(text)
    assert report["exit_code"] == expected
    assert report["score"] == (1 if version == "fixed" else 0)
    assert report["evaluation_id"] and "trace_id" not in report
    assert "student@example.com" not in text
    assert "student@example.com" not in capsys.readouterr().out
    assert calls["ExtractedPII"] == 1 and calls["ReasonScore"] == 0


@pytest.mark.parametrize("mode", ["timeout", "invalid_score", "metric_error"])
def test_execution_errors_never_pass(mode, scripted, tmp_path, monkeypatch):
    module = importlib.import_module("03_safety_gate")
    monkeypatch.setattr(module, "REPORT_DIR", tmp_path / "reports")

    def failed_assert(case, metrics, **kwargs):
        if mode == "timeout":
            raise TimeoutError("不应写入报告的请求原文")
        metrics[0].score = float("nan") if mode == "invalid_score" else 1
        metrics[0].error = "不应写入报告的请求原文" if mode == "metric_error" else None

    monkeypatch.setattr(module, "assert_test", failed_assert)
    assert module.main(judge=scripted[0]) == 2
    text = next((tmp_path / "reports").glob("*.json")).read_text(encoding="utf-8")
    assert json.loads(text)["status"] == "错误"
    assert "请求原文" not in text and "NaN" not in text


@pytest.mark.parametrize("lesson", ["01_first_evaluation", "02_rag_evaluation", "03_safety_gate"])
@pytest.mark.parametrize("location", ["repository", "examples"])
def test_direct_file_launch_without_key(lesson, location):
    # 空环境变量覆盖 .env，保证子进程不会使用本机密钥或发起模型请求。
    env = {**os.environ, "ALIYUN_API_KEY": "",
           "DEEPEVAL_TIMEOUT_SECONDS": "10", "PYTHONIOENCODING": "utf-8"}
    cwd = EXAMPLES.parent if location == "repository" else EXAMPLES
    script = f"deepeval_examples/{lesson}.py" if location == "repository" else f"{lesson}.py"
    result = subprocess.run([sys.executable, script], cwd=cwd, env=env,
                            capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 2
    assert "ALIYUN_API_KEY" in result.stdout and "Traceback" not in result.stderr


@pytest.mark.parametrize("value", ["abc", "0", "nan"])
def test_invalid_timeout_stops_scoring(value, monkeypatch, capsys):
    monkeypatch.setenv("ALIYUN_API_KEY", "test-placeholder")
    monkeypatch.setenv("DEEPEVAL_TIMEOUT_SECONDS", value)
    assert get_judge() is None
    assert "DEEPEVAL_TIMEOUT_SECONDS" in capsys.readouterr().out


def test_shared_qwen_factory_creates_one_client(monkeypatch):
    import langchain_openai

    clients = []

    def create_client(**options):
        client = SimpleNamespace(model_name=options["model"], options=options)
        clients.append(client)
        return client

    # 运行真实 get_model("qwen") 分支，只替换最底层客户端构造，确保未绕过共享工厂。
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", create_client)
    monkeypatch.setenv("DEEPEVAL_TIMEOUT_SECONDS", "37")
    monkeypatch.setenv("ALIYUN_API_KEY", "test-placeholder")
    judge = get_judge()
    assert isinstance(judge, QwenJudge)
    assert len(clients) == 1 and judge.model is clients[0]
    assert judge.get_model_name() == "qwen-plus"
    assert clients[0].options == {
        "model": "qwen-plus", "api_key": "test-placeholder",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "temperature": 0, "timeout": 37, "max_retries": 1,
    }


def test_adapter_supports_sync_async_and_structured_output(scripted):
    class ReasonScore(BaseModel):
        score: int
        reason: str

    judge, _ = scripted
    assert judge.generate("评分", ReasonScore).score == 10
    assert asyncio.run(judge.a_generate("评分", ReasonScore)).score == 0
    assert judge.generate("问候") == "文本响应"
    assert asyncio.run(judge.a_generate("问候")) == "文本响应"


@pytest.mark.parametrize("key", ["", "your_aliyun_api_key_here", "sk-your-aliyun-api-key-here"])
def test_missing_or_placeholder_key_skips_client(key, monkeypatch, capsys):
    def unexpected_client(*args, **kwargs):
        pytest.fail("未配置有效密钥时不应创建在线客户端")

    monkeypatch.setenv("ALIYUN_API_KEY", key)
    monkeypatch.setattr(judge_module, "get_model", unexpected_client)
    assert get_judge() is None
    assert "ALIYUN_API_KEY" in capsys.readouterr().out
