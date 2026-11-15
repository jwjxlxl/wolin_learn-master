"""案例 2：RAG 哪里出了错？只用忠实度、回答相关性两把尺子。"""

from judge import get_judge
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase


def main(judge=None):
    judge = judge or get_judge()
    if judge is None:
        return 2
    # 固定资料和回答，专注评估，不必先部署向量数据库。
    warranty = "本产品自购买之日起保修 18 个月。"
    payment = "本商城支持微信支付和支付宝。"
    examples = [
        ("资料和回答都正确", warranty, warranty),
        ("找对资料却编造答案", "本产品提供终身免费保修。", warranty),
        ("照抄了不相关的资料", payment, payment),
    ]
    metrics = [
        # 忠实度：回答有没有得到检索资料的支持？
        FaithfulnessMetric(model=judge, threshold=0.7, async_mode=False),
        # 回答相关性：回答有没有回应用户的问题？
        AnswerRelevancyMetric(model=judge, threshold=0.7, async_mode=False),
    ]
    for index, (name, answer, context) in enumerate(examples, 1):
        print(f"\n-- 示例 {index}: {name}")
        case = LLMTestCase(
            input="本产品保修期多久？", actual_output=answer,
            retrieval_context=[context],  # 实际检索资料，不是参考答案。
        )
        for metric in metrics:
            metric.measure(case)
            print(f"{metric.__name__}：分数={metric.score}，阈值={metric.threshold}")
            print(f"理由：{metric.reason}；是否通过：{metric.is_successful()}")
    print("学习重点：忠于资料不代表答对问题；结合两项评分与理由定位故障。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
