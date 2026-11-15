"""第二课：给一个最小 RAG 的检索与生成分别埋点。

运行方式：python -m langfuse_demo.rag_with_tracing
检索使用内置课程资料，便于把注意力放在 Langfuse Trace 上。
真实项目可将 search_documents 替换为 Milvus 检索。
"""

from dataclasses import dataclass

from langfuse import propagate_attributes

from langfuse_demo.common import (
    MODEL_NAME,
    create_qwen_client,
    get_langfuse_client,
    setup_console_encoding,
)


DOCUMENTS = (
    {
        "id": "trace",
        "keywords": ("trace", "链路"),
        "content": "Trace 记录一次用户请求的完整执行过程，由多个观察节点组成。",
    },
    {
        "id": "span",
        "keywords": ("span", "步骤"),
        "content": "Span 表示普通执行步骤；检索和工具调用也可分别标记为 Retriever、Tool。",
    },
    {
        "id": "generation",
        "keywords": ("generation", "模型", "token"),
        "content": "Generation 记录一次模型调用，可包含模型名与 Token 用量。",
    },
)

LOCAL_PROMPT = (
    "请仅根据资料回答问题；找不到依据时明确说不知道。"
    "回答时在相关句子后标注资料编号，例如 [trace]。\n"
    "资料：\n{context}\n问题：{question}"
)


@dataclass(frozen=True)
class RagResult:
    """一次问答的结果和对应的 Trace 标识。"""

    answer: str
    source_ids: tuple[str, ...]
    trace_id: str | None


def search_documents(question: str, top_k: int = 2) -> list[dict]:
    """用可重复的关键词检索代替向量库，保留检索分数。"""
    lowered = question.lower()
    matches = []
    for document in DOCUMENTS:
        score = sum(keyword.lower() in lowered for keyword in document["keywords"])
        if score:
            matches.append({**document, "score": score})
    return sorted(matches, key=lambda document: -document["score"])[:top_k]


def format_context(documents: list[dict]) -> str:
    """让模型和 Trace 都能看见资料编号及内容。"""
    if not documents:
        return "未检索到相关资料。"
    return "\n".join(f"[{document['id']}] {document['content']}" for document in documents)


def run_rag(
    question: str,
    *,
    langfuse_client,
    model_client,
    prompt=None,
) -> RagResult:
    """将检索标为 Retriever、模型调用标为 Generation。"""
    with langfuse_client.start_as_current_observation(
        as_type="span", name="rag-qa", input={"question": question}
    ) as root_span:
        with propagate_attributes(user_id="student-demo", tags=["langfuse-demo", "rag"]):
            trace_id = langfuse_client.get_current_trace_id()
            with langfuse_client.start_as_current_observation(
                as_type="retriever",
                name="keyword-retrieval",
                input={"query": question, "top_k": 2},
            ) as retrieval:
                documents = search_documents(question)
                retrieval.update(
                    output=[
                        {
                            "id": document["id"],
                            "score": document["score"],
                            "content": document["content"],
                        }
                        for document in documents
                    ]
                )

            context = format_context(documents)
            compiled_prompt = (
                prompt.compile(question=question, context=context)
                if prompt is not None
                else LOCAL_PROMPT.format(question=question, context=context)
            )
            messages = [{"role": "user", "content": compiled_prompt}]
            with langfuse_client.start_as_current_observation(
                as_type="generation",
                name="rag-answer",
                model=MODEL_NAME,
                input=messages,
                prompt=prompt,
            ) as generation:
                completion = model_client.chat.completions.create(
                    model=MODEL_NAME, messages=messages
                )
                answer = completion.choices[0].message.content or ""
                usage = completion.usage
                if usage is not None:
                    generation.update(
                        output=answer,
                        usage_details={
                            "input": usage.prompt_tokens,
                            "output": usage.completion_tokens,
                        },
                    )
                else:
                    generation.update(output=answer)

            source_ids = tuple(document["id"] for document in documents)
            root_span.update(output={"answer": answer, "sources": source_ids})
    return RagResult(answer=answer, source_ids=source_ids, trace_id=trace_id)


def main() -> None:
    """运行带检索与生成分层的问答。"""
    setup_console_encoding()
    try:
        langfuse_client = get_langfuse_client()
    except RuntimeError as exc:
        print(f"请先配置 Langfuse 项目密钥后重试：{exc}")
        return
    try:
        try:
            model_client = create_qwen_client()
        except RuntimeError as exc:
            print(f"跳过模型调用：{exc}")
            return
        result = run_rag(
            "Trace 和 Span 有什么区别？",
            langfuse_client=langfuse_client,
            model_client=model_client,
        )
    finally:
        langfuse_client.flush()
    print(f"\n-- 示例 3: RAG 分层追踪\n回答：{result.answer}")
    print(f"检索资料：{', '.join(result.source_ids)}")
    print("请在 Trace 中比较 Retriever 与 Generation 的输入、输出和耗时。")


if __name__ == "__main__":
    main()
