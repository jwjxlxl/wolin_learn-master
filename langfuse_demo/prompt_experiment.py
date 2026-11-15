"""第三课：用平台 Prompt、Trace Score 和 Dataset 实验形成改进闭环。

首次运行：python -m langfuse_demo.prompt_experiment --init
查看单次评分：python -m langfuse_demo.prompt_experiment
比较两版提示词：python -m langfuse_demo.prompt_experiment --experiment
实验使用同一份两题数据集，共调用四次云端模型。
引用规则分只检查格式，不代表回答的事实准确性或整体质量。
"""

import argparse
import uuid

from langfuse import Evaluation

from langfuse_demo.common import create_qwen_client, get_langfuse_client, setup_console_encoding
from langfuse_demo.rag_with_tracing import RagResult, run_rag, search_documents


PROMPT_NAME = "langfuse-demo-rag-answer"
DATASET_NAME = "langfuse-demo-observability"
QUESTIONS = ("Trace 记录什么？", "Span 代表什么？")
PROMPT_BASELINE = (
    "请仅依据下列资料简洁回答问题，不要补充资料外的事实。\n"
    "资料：\n{{context}}\n问题：{{question}}"
)
PROMPT_WITH_CITATION = (
    "请仅依据下列资料回答问题，不要补充资料外的事实。"
    "引用资料时写出编号，例如 [trace]。\n"
    "资料：\n{{context}}\n问题：{{question}}"
)


def initialize_demo(langfuse_client) -> None:
    """一次创建两版同名提示词和平台题目集。"""
    langfuse_client.create_prompt(
        name=PROMPT_NAME, type="text", prompt=PROMPT_BASELINE, labels=["baseline"]
    )
    langfuse_client.create_prompt(
        name=PROMPT_NAME,
        type="text",
        prompt=PROMPT_WITH_CITATION,
        labels=["with-citation", "production"],
    )
    langfuse_client.create_dataset(
        name=DATASET_NAME,
        description="课程示例：比较两版提示词的资料引用格式",
    )
    for index, question in enumerate(QUESTIONS, start=1):
        langfuse_client.create_dataset_item(
            dataset_name=DATASET_NAME,
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{DATASET_NAME}:{index}")),
            input=question,
        )


def citation_score(answer: str, source_ids: tuple[str, ...]) -> float:
    """只检查回答是否引用了本次检索资料的编号。"""
    return float(any(f"[{source_id}]" in answer for source_id in source_ids))


def run_prompt_and_score(langfuse_client, model_client) -> RagResult:
    """使用 production 提示词回答，并把引用规则分挂到 Trace。"""
    prompt = langfuse_client.get_prompt(PROMPT_NAME, label="production")
    result = run_rag(
        "Span 代表什么？",
        langfuse_client=langfuse_client,
        model_client=model_client,
        prompt=prompt,
    )
    if result.trace_id is None:
        raise RuntimeError("没有取得 Trace ID，无法关联评分")
    langfuse_client.create_score(
        trace_id=result.trace_id,
        name="has_citation",
        value=citation_score(result.answer, result.source_ids),
        comment="仅检查资料编号引用，不判断回答质量",
    )
    return result


def rule_evaluator(*, input, output, **_kwargs) -> Evaluation:
    """在 Dataset 实验中复用同一项引用规则。"""
    source_ids = tuple(document["id"] for document in search_documents(input))
    return Evaluation(
        name="has_citation",
        value=citation_score(output, source_ids),
        comment="仅检查资料编号引用，不判断回答质量",
    )


def run_experiments(langfuse_client, model_client) -> dict:
    """用同一 RAG 流程和题目集对比两版平台提示词。"""
    dataset = langfuse_client.get_dataset(DATASET_NAME)
    results = {}
    for label in ("baseline", "with-citation"):
        prompt = langfuse_client.get_prompt(PROMPT_NAME, label=label)

        def answer_item(*, item, selected_prompt=prompt, **_kwargs) -> str:
            return run_rag(
                item.input,
                langfuse_client=langfuse_client,
                model_client=model_client,
                prompt=selected_prompt,
            ).answer

        results[label] = dataset.run_experiment(
            name=f"langfuse-demo-{label}",
            description="同一 RAG 流程比较提示词版本的引用格式",
            task=answer_item,
            evaluators=[rule_evaluator],
            max_concurrency=1,
        )
    return results


def main() -> None:
    """按命令参数初始化资源、查看单次评分或运行版本实验。"""
    setup_console_encoding()
    parser = argparse.ArgumentParser(description="Langfuse 提示词与实验示例")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--init", action="store_true", help="创建两版提示词与数据集")
    actions.add_argument("--experiment", action="store_true", help="运行数据集版本对比")
    arguments = parser.parse_args()

    try:
        langfuse_client = get_langfuse_client()
    except RuntimeError as exc:
        print(f"请先配置 Langfuse 项目密钥后重试：{exc}")
        return

    try:
        if arguments.init:
            initialize_demo(langfuse_client)
            print(f"\n已创建提示词 {PROMPT_NAME} 的两个版本和数据集 {DATASET_NAME}")
            return

        try:
            model_client = create_qwen_client()
        except RuntimeError as exc:
            print(f"跳过模型调用：{exc}")
            return

        if arguments.experiment:
            results = run_experiments(langfuse_client, model_client)
            print("\n-- 示例 5: Dataset 上的提示词版本对比")
            for label, result in results.items():
                print(f"\n提示词标签：{label}\n{result.format()}")
            print("请比较引用规则分；这项分数不能证明整体回答质量更好。")
        else:
            result = run_prompt_and_score(langfuse_client, model_client)
            print(f"\n-- 示例 4: 平台提示词与 Trace 评分\n回答：{result.answer}")
            print("请在 Trace 和 Prompt 页面查看版本关联及引用规则分。")
    finally:
        langfuse_client.flush()


if __name__ == "__main__":
    main()
