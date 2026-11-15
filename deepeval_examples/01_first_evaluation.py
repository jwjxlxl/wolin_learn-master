"""案例 1：给答案打分。记住三个概念：考题、评分标准、及格线。"""

from judge import get_judge  # 先加载模型配置；本文件可直接运行。
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams


def main(judge=None):
    judge = judge or get_judge()
    if judge is None:
        return 2
    answers = [
        ("正确回答", "本产品从购买之日起提供 18 个月保修。"),
        ("编造回答", "本产品提供终身免费保修。"),
    ]
    # 只创建一次：两份答卷复用同一套评分步骤，便于公平比较。
    metric = GEval(
        name="答案正确性", model=judge, async_mode=False,
        criteria="以 Expected Output（参考答案）为事实依据，核对 Actual Output（实际回答）。"
                 "保修期限与起算时间均一致给满分；与参考答案冲突或编造服务承诺给零分。"
                 "允许同义表达，用中文解释理由。",
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
        threshold=0.8,  # 及格线：分数达到 0.8 才通过。
    )
    for index, (name, answer) in enumerate(answers, 1):
        print(f"\n-- 示例 {index}: {name}")
        case = LLMTestCase(
            input="本产品保修期多久？", actual_output=answer,
            expected_output="自购买之日起保修 18 个月。",
        )
        metric.measure(case)  # 把考题交给裁判。
        print(f"分数：{metric.score}；及格线：{metric.threshold}")
        print(f"理由：{metric.reason}")
        print(f"是否通过：{metric.is_successful()}")
    # 负例低分是演示效果；真正的测试断言留到案例 03。
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
