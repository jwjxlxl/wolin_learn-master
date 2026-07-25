"""
Tree of Thoughts (ToT) 教学示例 (树状思维)
==========================================
背景: Princeton & DeepMind 《Tree of Thoughts》(2023)

核心思想:
  不沿单一线索推理，而是生成多条思路分支，像树一样展开，
  再通过评估机制选出最佳分支，剪枝淘汰劣解。

场景例子:
  解数独时，Agent 尝试多个候选填法（分支A/B/C），逐步排除错误分支，
  最终选出唯一解。

适合: 复杂规划、解谜、创作类任务
"""

import logging
import re

from dotenv import load_dotenv

from agent_examples.qwen import call_llm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
load_dotenv()


# ──────────────────── Tree of Thoughts Agent ────────────────────

def tot_solve(
    problem: str,
    n_branches: int = 3,
    depth: int = 3,
    beam_width: int = 1,
) -> str:
    """
    Tree of Thoughts: 多分支展开 + 评估剪枝

    流程:
      1. Generate:  为当前状态生成 n_branches 条思路分支
      2. Evaluate:  用 LLM 对全部分支统一排序，映射为差异化分数
      3. Select:    保留得分最高的 beam_width 条分支（剪枝）
      4. 在保留的分支上继续下一层思考，直到达到 depth
      5. 最后从所有叶子中选出最佳思路，给出最终答案

    参数:
      problem:    要解决的问题
      n_branches: 每层生成的思路分支数
      depth:      树的深度（推理轮次）
      beam_width: 每层保留的分支数（Beam Search）

    返回:
      最佳答案
    """
    # 每条路径用 (thought_path, score) 表示
    # thought_path 是字符串，记录推理轨迹
    active_paths: list[tuple[str, float]] = [("", 0.0)]

    for d in range(1, depth + 1):
        logger.info(f"=== 第 {d} 层 (当前活跃路径数: {len(active_paths)}) ===")

        candidates: list[tuple[str, float]] = []

        for path, path_score in active_paths:
            # --- Generate: 生成多条分支 ---
            context = f"问题: {problem}"
            if path:
                context += f"\n\n已有推理: {path}"

            branch_response = call_llm([
                {"role": "system", "content": (
                    f"你是一个善于多角度思考的助手。请针对以下问题，"
                    f"提出 {n_branches} 条不同的思路/方案。"
                    f"每条思路以 '---' 分隔。"
                )},
                {"role": "user", "content": context},
            ])

            # 按 --- 分割多条分支
            branches = [
                b.strip()
                for b in branch_response.split("---")
                if b.strip()
            ]
            # 保底：如果 LLM 没有用 --- 分隔，尝试按编号分割
            if len(branches) < 2:
                branches = [
                    b.strip()
                    for b in re.split(r"\d+[\.、:]\s*", branch_response)
                    if b.strip()
                ]
            branches = branches[:n_branches]

            logger.info(f"  生成了 {len(branches)} 条分支")
            logger.info(f"  分支内容: {branches}")

            # --- Evaluate: 对所有分支统一排序打分 ---
            # 一次性让 LLM 对全部分支排序，避免逐条打分的趋同性
            branches_text = "\n".join(
                f"[{idx}] {b}" for idx, b in enumerate(branches, 1)
            )
            rank_response = call_llm([
                {"role": "system", "content": (
                    "你是一个严格的评估助手。请对以下思路从优到劣排序。\n"
                    "输出格式: 数字列表，如 [2, 1, 3] 表示第2条最好、第1条次之、第3条最差。\n"
                    "只输出数字列表，不要其他内容。"
                )},
                {"role": "user", "content": (
                    f"问题: {problem}\n"
                    f"已有推理: {path}\n"
                    f"待排序的思路:\n{branches_text}\n"
                    f"请排序:"
                )},
            ])

            # 提取排序列表
            rank_match = re.findall(r"\d+", rank_response)
            ranking = [int(x) for x in rank_match]
            # 验证：必须包含所有分支编号
            if set(ranking) != set(range(1, len(branches) + 1)):
                ranking = list(range(1, len(branches) + 1))  # 降级为默认顺序

            # 按排名映射差异化分数 (10分制)
            score_map = {1: 10, 2: 7, 3: 5, 4: 3, 5: 2}
            for rank_idx, branch_num in enumerate(ranking):
                branch = branches[branch_num - 1]
                score = score_map.get(rank_idx + 1, 20)

                new_path = f"{path}\n[Layer {d}] {branch}" if path else f"[Layer {d}] {branch}"
                new_total_score = path_score + score
                candidates.append((new_path, new_total_score))
                logger.info(f"    分支[{branch_num}] 排名: 第{rank_idx + 1}名, 得分: {score} -> 累计: {new_total_score}")

        # --- Select: 保留 top-k (beam_width) ---
        candidates.sort(key=lambda x: x[1], reverse=True)
        active_paths = candidates[:beam_width]

        logger.info(f"  剪枝后保留 {len(active_paths)} 条路径\n")

    # ── 最终：从最佳路径生成答案 ──
    best_path, best_score = active_paths[0]
    logger.info(f"最佳路径得分: {best_score}")
    logger.info(f"最佳路径:\n{best_path}\n")

    # 打印所有思路分支
    logger.info("=== 所有思路分支 ===")
    for i, (path, score) in enumerate(candidates, 1):
        logger.info(f"[分支 {i}] 得分: {score}\n{path}\n")

    final_answer = call_llm([
        {"role": "system", "content": "你是最终回答助手。请根据以下推理过程，给出简洁、准确的答案。"},
        {"role": "user", "content": (
            f"问题: {problem}\n\n"
            f"推理过程:\n{best_path}\n\n"
            f"请给出最终答案:"
        )},
    ])

    return final_answer


if __name__ == "__main__":
    print("=" * 60)
    print("Tree of Thoughts 演示")
    print("=" * 60)

    problem = (
        "公司预算砍了30%，但需要在三个月内上线一个新功能来留住用户。"
        "请提出三种不同的应对策略，并说明每种的具体执行方案。"
    )
    print(f"问题: {problem}")
    result = tot_solve(problem, n_branches=4, depth=3, beam_width=1)
    print(f"最终答案: {result}")
