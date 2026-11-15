"""无需云端密钥的 Langfuse 教学案例回归测试。"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langfuse import Langfuse
from langfuse.openai import OpenAI as TracedOpenAI
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from langfuse_demo import (
    first_trace,
    prompt_experiment,
    qa_agent_with_tracing as agent,
    rag_with_tracing,
)


@pytest.mark.parametrize("module", [first_trace, rag_with_tracing, prompt_experiment])
def test_cloud_examples_skip_model_without_key(module, monkeypatch, capsys):
    client = MagicMock()
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")
    monkeypatch.setattr(module, "setup_console_encoding", lambda: None)
    monkeypatch.setattr(module, "get_langfuse_client", lambda: client)
    monkeypatch.setattr(sys, "argv", [module.__name__])
    missing_key = Mock(side_effect=RuntimeError("缺少模型密钥"))
    if module is first_trace:
        monkeypatch.setattr(module, "require_setting", missing_key)
    else:
        monkeypatch.setattr(module, "create_qwen_client", missing_key)

    module.main()
    assert "跳过模型调用" in capsys.readouterr().out
    client.flush.assert_called_once()


def test_manual_observation_is_ended():
    client = Mock()
    span = client.start_observation.return_value
    assert first_trace.normalize_manually(client, "  Trace  ") == "Trace"
    span.end.assert_called_once()


def test_first_trace_shows_one_trace_with_three_child_types(monkeypatch, capsys):
    exporter = InMemorySpanExporter()
    public_key = "pk-lf-first-trace-integration-test"
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", public_key)
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.delenv("LANGFUSE_TRACING_ENABLED", raising=False)
    client = Langfuse(
        public_key=public_key,
        secret_key="sk-lf-test",
        base_url="http://127.0.0.1:9",
        span_exporter=exporter,
        tracing_enabled=True,
    )

    def respond(_request):
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "qwen-plus",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Trace 记录一次请求。"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            },
        )

    monkeypatch.setattr(first_trace, "setup_console_encoding", lambda: None)
    monkeypatch.setattr(first_trace, "get_langfuse_client", lambda: client)
    monkeypatch.setattr(first_trace, "require_setting", lambda _name: "test-key")
    monkeypatch.setattr(
        first_trace,
        "OpenAI",
        lambda **_kwargs: TracedOpenAI(
            api_key="test-key",
            base_url="https://example.test/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(respond)),
        ),
    )
    monkeypatch.setattr(
        client,
        "get_trace_url",
        lambda *, trace_id: f"https://example.test/traces/{trace_id}",
    )

    first_trace.main()
    output = capsys.readouterr().out
    spans = exporter.get_finished_spans()
    root = next(span for span in spans if span.parent is None)
    children = [span for span in spans if span.parent is not None]
    assert len(spans) == 4
    assert {span.name for span in children} == {
        "OpenAI-generation", "normalize-question", "normalize-manual"
    }
    assert all(span.context.trace_id == root.context.trace_id for span in children)
    assert all(span.parent.span_id == root.context.span_id for span in children)
    assert f"Trace ID：{root.context.trace_id:032x}" in output
    assert f"根节点 ID：{root.context.span_id:016x}" in output
    assert f"https://example.test/traces/{root.context.trace_id:032x}" in output


def test_rag_creates_one_trace_with_retrieval_and_generation(monkeypatch):
    monkeypatch.delenv("LANGFUSE_TRACING_ENABLED", raising=False)
    exporter = InMemorySpanExporter()
    client = Langfuse(
        public_key="pk-lf-rag-example-test",
        secret_key="sk-lf-test",
        base_url="http://127.0.0.1:9",
        span_exporter=exporter,
        tracing_enabled=True,
    )
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="资料 [trace]"))],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3),
    )
    model = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=Mock(return_value=completion)))
    )

    result = rag_with_tracing.run_rag(
        "Trace 记录什么？", langfuse_client=client, model_client=model
    )
    client.flush()
    spans = exporter.get_finished_spans()
    assert result.source_ids == ("trace",)
    assert result.trace_id
    assert len(spans) == 3
    assert len({span.context.trace_id for span in spans}) == 1
    assert len([span for span in spans if span.parent]) == 2


def test_initialize_creates_one_prompt_pair_and_dataset():
    client = Mock()
    prompt_experiment.initialize_demo(client)
    assert [
        call.kwargs["labels"] for call in client.create_prompt.call_args_list
    ] == [["baseline"], ["with-citation", "production"]]
    items = client.create_dataset_item.call_args_list
    assert [call.kwargs["input"] for call in items] == list(prompt_experiment.QUESTIONS)
    assert len({call.kwargs["id"] for call in items}) == len(items)


def test_online_score_uses_same_citation_rule(monkeypatch):
    client = Mock()
    result = rag_with_tracing.RagResult("资料 [span]", ("span",), "trace-123")
    run_rag = Mock(return_value=result)
    monkeypatch.setattr(prompt_experiment, "run_rag", run_rag)

    assert prompt_experiment.run_prompt_and_score(client, object()) == result
    assert run_rag.call_args.kwargs["prompt"] is client.get_prompt.return_value
    assert client.create_score.call_args.kwargs["trace_id"] == "trace-123"
    assert client.create_score.call_args.kwargs["value"] == 1.0


def test_experiment_compares_two_prompts_on_same_rag_task(monkeypatch):
    prompts = {"baseline": object(), "with-citation": object()}
    client = Mock()
    client.get_prompt.side_effect = lambda _name, *, label: prompts[label]
    model = object()
    calls = []

    def run_rag(question, *, langfuse_client, model_client, prompt):
        calls.append((question, prompt))
        answer = "资料 [trace]" if prompt is prompts["with-citation"] else "资料"
        return rag_with_tracing.RagResult(answer, ("trace",), "trace-123")

    monkeypatch.setattr(prompt_experiment, "run_rag", run_rag)

    class Dataset:
        def __init__(self):
            self.scores = []

        def run_experiment(self, **arguments):
            question = "Trace 记录什么？"
            answer = arguments["task"](item=SimpleNamespace(input=question))
            score = arguments["evaluators"][0](input=question, output=answer)
            self.scores.append(score.value)
            return SimpleNamespace(format=lambda: "完成")

    dataset = Dataset()
    client.get_dataset.return_value = dataset
    results = prompt_experiment.run_experiments(client, model)
    assert set(results) == set(prompts)
    assert calls == [
        ("Trace 记录什么？", prompts["baseline"]),
        ("Trace 记录什么？", prompts["with-citation"]),
    ]
    assert dataset.scores == [0.0, 1.0]


def test_agent_uses_knowledge_tool_and_keeps_history():
    class ScriptedModel:
        def __init__(self):
            self.saw_previous_tool = False

        def bind_tools(self, tools, *, tool_choice):
            assert [tool.name for tool in tools] == ["search_knowledge"]
            assert tool_choice == "required"
            return self

        def invoke(self, messages):
            last = messages[-1]
            if isinstance(last, ToolMessage):
                return AIMessage(content=f"资料：{last.content}")
            if "刚才" in last.content:
                self.saw_previous_tool = any(
                    isinstance(message, ToolMessage) for message in messages
                )
            return AIMessage(
                content="",
                tool_calls=[
                    {"name": "search_knowledge", "args": {"query": "span"}, "id": "call-1"}
                ],
            )

    model = ScriptedModel()
    graph = agent.build_qa_agent(model=model)
    first, history = agent.run_agent_turn(graph, "什么是 Span？", [], {})
    second, history = agent.run_agent_turn(graph, "刚才的 Span 是什么？", history, {})
    assert "Span 表示" in first and "Span 表示" in second
    assert model.saw_previous_tool
    assert sum(isinstance(message, HumanMessage) for message in history) == 2
    assert sum(isinstance(message, ToolMessage) for message in history) == 2


def test_agent_groups_two_traces_in_one_session():
    class ScriptedGraph:
        def invoke(self, state, config):
            return {"messages": [*state["messages"], AIMessage(content="课程回答")]}

    exporter = InMemorySpanExporter()
    client = Langfuse(
        public_key="pk-lf-agent-example-test",
        secret_key="sk-lf-test",
        base_url="http://127.0.0.1:9",
        span_exporter=exporter,
        tracing_enabled=True,
    )
    agent.qa_agent_with_tracing(client, ScriptedGraph(), object())
    client.flush()
    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    assert len({span.context.trace_id for span in spans}) == 2
    assert len({span.attributes["session.id"] for span in spans}) == 1


def test_agent_flushes_after_failure(monkeypatch):
    client = Mock()
    monkeypatch.setattr(agent, "setup_console_encoding", lambda: None)
    monkeypatch.setattr(agent, "require_setting", lambda _name: "test-key")
    monkeypatch.setattr(agent, "get_langfuse_client", lambda: client)
    monkeypatch.setattr(agent, "build_qa_agent", Mock(side_effect=RuntimeError("模型失败")))
    with pytest.raises(RuntimeError, match="模型失败"):
        agent.main()
    client.flush.assert_called_once()
