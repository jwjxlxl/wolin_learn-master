"""第一课：在同一条 Trace 中比较自动模型追踪与函数级埋点。

在项目根目录安装依赖：pip install -r requirements.txt
运行方式：python -m langfuse_demo.first_trace
配置 Langfuse 项目密钥后即可运行函数埋点；模型调用还需 ALIYUN_API_KEY。
RAG 示例会继续演示适合嵌套步骤的上下文管理器。
"""

import os

from langfuse import observe
from langfuse.openai import OpenAI

from langfuse_demo.common import (
    MODEL_NAME,
    QWEN_BASE_URL,
    get_langfuse_client,
    require_setting,
    setup_console_encoding,
)


@observe(name="normalize-question")
def normalize_question(question: str) -> str:
    """装饰器自动记录函数输入、输出和耗时。"""
    return question.strip()


def normalize_manually(langfuse_client, question: str) -> str:
    """需要手动控制观察节点时，创建后必须显式结束。"""
    span = langfuse_client.start_observation(
        as_type="span", name="normalize-manual", input=question
    )
    try:
        answer = question.strip()
        span.update(output=answer)
        return answer
    finally:
        span.end()


def main() -> None:
    """打印 Trace ID、父子路径与两种函数埋点方式。"""
    setup_console_encoding()
    try:
        langfuse_client = get_langfuse_client()
    except RuntimeError as exc:
        print(f"请先配置 Langfuse 项目密钥后重试：{exc}")
        return

    question = "  什么是 Trace？  "
    model_called = False
    try:
        with langfuse_client.start_as_current_observation(
            as_type="span", name="first-trace-demo", input={"question": question}
        ) as root_span:
            trace_id = langfuse_client.get_current_trace_id()
            root_id = langfuse_client.get_current_observation_id()

            print("\n-- 示例 1: 自动追踪模型调用")
            try:
                api_key = require_setting("ALIYUN_API_KEY")
            except RuntimeError as exc:
                print(f"跳过模型调用：{exc}")
                answer = None
            else:
                # 包装客户端会自动把模型调用记录为当前父 Span 下的 Generation。
                # 指定项目公钥，确保多个 Langfuse 客户端并存时仍关联到当前 Trace。
                model_client = OpenAI(api_key=api_key, base_url=QWEN_BASE_URL)
                completion = model_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": question.strip()}],
                    langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
                )
                answer = completion.choices[0].message.content or ""
                model_called = True
                print(f"回答：{answer}")

            print("\n-- 示例 2: 同一问题的两种函数埋点")
            decorated = normalize_question(
                question, langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY")
            )
            manual = normalize_manually(langfuse_client, question)
            print(f"装饰器结果：{decorated}")
            print(f"手动埋点结果：{manual}")
            root_span.update(output={"answer": answer, "normalized_question": decorated})
    finally:
        langfuse_client.flush()

    print("\n-- 在 Langfuse 中查看追踪路径")
    print(f"Trace ID：{trace_id}")
    print(f"根节点 ID：{root_id}")
    print("first-trace-demo（父 Span：一次演示）")
    if model_called:
        print("├─ OpenAI-generation（Generation：包装客户端自动记录模型调用）")
    print("├─ normalize-question（Span：@observe 自动记录函数输入和输出）")
    print("└─ normalize-manual（Span：start_observation 创建，update 后需 end）")
    try:
        trace_url = langfuse_client.get_trace_url(trace_id=trace_id)
    except Exception:
        trace_url = None
    if trace_url:
        print(f"在浏览器打开：{trace_url}")
    else:
        print("请在 Langfuse 的 Traces 页面按 Trace ID 查找这次运行。")


if __name__ == "__main__":
    main()
