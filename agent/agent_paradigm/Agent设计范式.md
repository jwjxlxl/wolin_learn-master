# Agent主流设计范式
针对主流智能体开发范式中的**单智能体**，同样存在较多的设计范式。
<img src='./images/主流智能体开发范式.png' width=60% height=60%>

## Chain of Thought(思维链)
- **提出背景：** Google Research 在2022年发表的论文《Chain-of-Thought Prompting Elicits Reasoning in Large Language Models》。
- **核心思想：** 让模型在回答前，把推理过程一步步写出来。不是一口气报出答案，而是把整个推理过程展示出来。


### 本质揭示
> CoT 没有产生意识，也没有赋予模型真正的逻辑推理引擎。它只是利用了 Transformer 架构的**自回归特性**（Autoregressive Nature），通过延长生成序列，将**高难度的单步预测转化为低难度的多步预测**，从而涌现（Emerge）出了看似智能的推理能力。

### 代码示例
``` python
    # --- 第一步: 逐步推理 ---
    thinking = call_llm([
        {"role": "system", "content": "You are a helpful assistant. Think step by step."},
        {"role": "user", "content": f"Q: {question}\nA: Let's think step by step."},
    ])
    logger.info(f"[ZeroShot-Thinking]\n{thinking}\n")

    # --- 第二步: 总结答案 ---
    answer = call_llm([
        {"role": "system", "content": "Based on the reasoning above, give a concise final answer."},
        {"role": "user", "content": (
            f"Question: {question}\n\n"
            f"Reasoning:\n{thinking}\n\n"
            f"Final answer:"
        )},
    ])
```

***

## Self-Ask(自问自答)[问题拆解]
- **提出背景：** Microsoft Research 在 2022 年的研究工作 《Self-Ask with Search》
- **核心思想:** 让模型在回答时学会"反问自己"，把大问题拆成多个小问题，然后逐个回答。
- **场景例子:** 问2016年奥斯卡最佳男主角的年龄是多少?Self-Ask会先问:2016年奥斯卡最佳男主是谁?(答:李奥纳多·狄卡比奥)，再问他当时多大?(答:41岁)，最后组合答案。
这种方式特别适合事实链路长的问题。

### 提示词示例

``` python
SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions by breaking them down into smaller sub-questions when needed.

You MUST respond in JSON format with the following structure:

{
    "need_followup": true or false,
    "followup_question": "the sub-question to answer next (only when need_followup is true)",
    "final_answer": "the final combined answer (only when need_followup is false)"
}

Here are two examples:

---
Example 1:
Question: Who is older, Taylor Swift or Justin Bieber?

Response:
{
    "need_followup": true,
    "followup_question": "When was Taylor Swift born?",
    "final_answer": null
}

(Intermediate answer: Taylor Swift was born on December 13, 1989.)

Response:
{
    "need_followup": true,
    "followup_question": "When was Justin Bieber born?",
    "final_answer": null
}

(Intermediate answer: Justin Bieber was born on March 1, 1994.)

Response:
{
    "need_followup": false,
    "followup_question": null,
    "final_answer": "Taylor Swift is older."
}

---
Example 2:
Question: What is the population of the city where the founder of Microsoft was born?

Response:
{
    "need_followup": true,
    "followup_question": "Where was the founder of Microsoft born?",
    "final_answer": null
}

(Intermediate answer: Bill Gates, founder of Microsoft, was born in Seattle, Washington.)

Response:
{
    "need_followup": true,
    "followup_question": "What is the population of Seattle?",
    "final_answer": null
}

(Intermediate answer: The population of Seattle is approximately 740,000 (as of 2023).)

Response:
{
    "need_followup": false,
    "followup_question": null,
    "final_answer": "Approximately 740,000."
}
"""
```

***

## ReAct (推理 + 行动)
<img src='./images/ReAct示意图.jpeg' height=500 width=300 >

- **提出背景:** Princeton 与 Google Research 在 2022 年论文 《ReAct: Synergizing Reasoning
and Acting in Language Models》交互的闭环。
- **核心思想:** 在推理(Reasoning)和外部行动(Acting，比如调用搜索引擎或API)之间交
替进行。ReAct比CoT、Self-Ask更全能，原因在于它不仅是推理模式，还内建了与外部世界
- **场景例子:** 问杭州昨天的天气?ReAct会先想:"我不知道昨天的天气，需要查询"，然后执
行"调用天气API"，再推理并回答。这让Agent既有思维，又能动手。

### 代码示例
```python
def react(question, tools, max_steps=5):
    """ReAct: 推理和行动交替进行"""
    messages = [{"role": "system", "content": "你是一个智能助手。回答时先思考(Thought)，如果需要外部信息则调用工具(Action)，观察结果(Observation)，再继续推理。"}]
    messages.append({"role": "user", "content": f"问题: {question}"})

    for step in range(max_steps):
        # 1. 推理：模型决定下一步做什么
        response = call_llm(messages)
        messages.append({"role": "assistant", "content": response})

        thought = extract_thought(response)   # 提取 "Thought: ..."
        action  = extract_action(response)    # 提取 "Action: search_weather"

        if not action:
            # 模型认为不需要工具，直接返回答案
            return response

        # 2. 行动：执行工具
        observation = tools[action]()

        # 3. 观察结果：把工具返回的信息加入上下文
        messages.append({"role": "user", "content": f"Observation: {observation}"})

    return response
```

***

## Plan-and-Execute (计划与执行)
- **提出背景:** 出现在2023年前后的Agent应用开发框架实践(如LangChain社区)
- **核心思想:** 把任务拆成两个阶段，先生成计划(Planning)，再逐步执行(Execution)。
- **场景例子:** 假设你让Agent写一篇"新能源车的市场调研报告"，它不会直接生成报告，而是先拟定计划:收集销量数据，分析政策趋势，总结消费者反馈，撰写结论。然后逐条执行。适合多步骤、需长时间任务的场景。

### 代码示例
```python
def plan_and_execute(task, max_iterations=10):
    """Plan-and-Execute: 先计划，再执行"""

    # 第一阶段：生成计划
    plan = call_llm([
        {"role": "system", "content": "你是一个项目规划师。请将任务拆解为有序的步骤列表，每一步都有明确的动作和预期输出。"},
        {"role": "user", "content": f"任务: {task}\n\n请拆解为步骤列表，返回 JSON 格式: [{{\"step\": 1, \"action\": \"...\"}}, ...]"}
    ])

    steps = parse_json(plan)
    results = []

    # 第二阶段：逐条执行
    for step_info in steps:
        step_num = step_info["step"]
        action   = step_info["action"]

        # 执行当前步骤，传入之前的结果
        prev_results = "\n".join(results)
        result = call_llm([
            {"role": "system", "content": f"当前是第 {step_num} 步: {action}\n请基于之前的结果完成本步骤。"},
            {"role": "user", "content": f"之前的结果:\n{prev_results}"}
        ])
        results.append(result)

    return results
```

***

## Tree of Thoughts(TOT, 树状思维)
- **提出背景:** Princeton 和 DeepMind 在 2023 年的论文《Tree of Thoughts: Deliberate Problem Solving with Large Language Models》 .
- **核心思想:** 不是单线思维，而是生成多条思路分支，像树一样展开，再通过评估机制选出最佳分支。
- **场景例子:** 解一道数独时，Agent会尝试多个候选解法(分支A、B、C)，逐步排除错误分支，最终选出唯一解。适合复杂规划和解谜任务。

### 代码示例
```python
def tree_of_thoughts(problem, n_branches=3, max_depth=3):
    """Tree of Thoughts: 多思路展开，评估选优"""

    def evaluate(thought):
        """评估函数：让模型判断思路的质量"""
        score = call_llm([
            {"role": "system", "content": "请评估以下思路的质量，给出 1-10 分，并简要说明理由。返回 JSON: {\"score\": 7, \"reason\": \"...\"}"},
            {"role": "user", "content": f"问题: {problem}\n\n思路: {thought}"}
        ])
        return parse_json(score)["score"]

    def expand(current_path, depth):
        if depth >= max_depth:
            return current_path

        # 生成多条分支
        branches = call_llm([
            {"role": "system", "content": f"请为以下问题生成 {n_branches} 条不同的解决思路。返回 JSON 数组。"},
            {"role": "user", "content": f"问题: {problem}\n当前已走到: {current_path}"}
        ])

        # 评估每条分支，选最优的
        best_branch = None
        best_score  = -1
        for branch in parse_json(branches):
            score = evaluate(branch)
            if score > best_score:
                best_score  = score
                best_branch = branch

        # 沿着最优分支继续深入
        return expand(current_path + " → " + best_branch, depth + 1)

    return expand("", 0)
```

***

## Reflexion/Iterative Refinement (反思与迭代优化)
- **提出背景:** 2023 年文 《Reflexion: Language Agents with Verbal Reinforcement
Learning》.
- **核心思想:** Agent具备自我纠错的能力，犯错后会总结失败原因，再带着反思尝试下一次。
- **场景例子:** 让Agent写一段Python代码，如果第一次运行报错，它会读报错信息，反思"函数参数写错了"，然后自动修正并重试。适合代码生成、流程执行类场景。

### 代码示例
```python
def reflexion(task, max_retries=3):
    """Reflexion: 试错 → 反思 → 重试"""
    memories = []  # 反思记忆

    for attempt in range(max_retries):
        # 1. 尝试执行任务
        result = call_llm([
            {"role": "system", "content": f"请完成以下任务: {task}"},
            # 如果有之前的反思，也传进去
            *[{"role": "user", "content": f"之前尝试失败的原因: {m}"} for m in memories],
        ])

        # 2. 验证结果（需要外部验证器，如测试用例、编译器、评估函数）
        feedback = verify(result)

        if feedback.success:
            return result

        # 3. 反思失败原因
        reflection = call_llm([
            {"role": "system", "content": "上次尝试失败了。请分析原因，总结成一条经验教训。"},
            {"role": "user", "content": f"任务: {task}\n你的回答: {result}\n验证结果: {feedback.message}"}
        ])
        memories.append(reflection)

    # 超过最大重试次数，返回最后一次结果
    return result
```

***

## Role-playing Agents (角色扮演式智能体或者说是多智能体协作)
- **提出背景:**源自AutoGPT、ChatDev、CAMEL等社区项目。
- **核心思想:**把任务拆分给不同角色的Agent，每个Agent都有专属职责，通过对话协作完成任务。
- **场景例子:**一个软件开发任务里，有产品经理Agent写需求文档，程序员Agent写代码，测试Agent写测试用例。它们像团队一样协作。适合复杂系统开发或跨职能协同。

### 代码示例
```python
def role_playing(task, roles):
    """
    角色扮演式多智能体协作
    roles: 字典，key=角色名, value=角色描述
    """
    messages_history = []

    # 每个角色按顺序发言
    for role_name, role_desc in roles.items():
        # 获取之前的讨论内容作为上下文
        context = "\n".join(messages_history)

        # 当前角色发言
        response = call_llm([
            {"role": "system", "content": f"你是{role_name}。你的职责：{role_desc}。请基于之前的讨论，给出你的意见或方案。"},
            {"role": "user", "content": f"任务: {task}\n\n之前的讨论:\n{context}"},
        ])

        messages_history.append(f"[{role_name}]: {response}")

    # 最后让所有角色投票或总结
    summary = call_llm([
        {"role": "system", "content": "你是一个会议总结助手。请汇总以上各角色的讨论，给出最终方案。"},
        {"role": "user", "content": "\n".join(messages_history)},
    ])

    return summary

# 使用示例
role_playing(
    task="开发一个在线商城系统",
    roles={
        "产品经理": "负责需求分析和功能优先级排序",
        "架构师":   "负责技术选型和系统架构设计",
        "开发工程师": "负责实现方案和技术细节",
        "测试工程师": "负责风险评估和测试策略",
    }
)
```

这些认知框架，其实构成了Agent世界里的思维模式库:

***

## 七大范式对比速查表

| 范式 | 核心动作 | 类比 | 适用场景 | 优势 | 局限 |
|------|---------|------|---------|------|------|
| **CoT（思维链）** | 一步步想，再答 | 草稿纸上先列算式，再写答案 | 数学题、逻辑推理、需要解释的问题 | 简单、零额外成本 | 只能推理，不能调用外部工具 |
| **Self-Ask（自问自答）** | 拆问题，逐个子问题求解 | 连环追问"先搞清楚A，再查B，最后合起来" | 事实查询、多跳推理、需要搜索的长链路问题 | 拆解精准、适合搜索引擎结合 | 需要多轮 API 调用，速度慢 |
| **ReAct（推理+行动）** | 想一步→做一步→看结果→再想 | 边查资料边写作业 | 需要外部工具的场景（搜索、API、数据库） | 既有思维又能动手，最全能 | Token 消耗大、上下文膨胀快 |
| **Plan-and-Execute（计划与执行）** | 先做完整计划，再逐条执行 | 先写项目计划书，再按计划推进 | 多步骤、长时间运行的复杂任务 | 全局视野、不易跑偏 | 计划可能不完美，执行中可能需要调整 |
| **ToT（树状思维）** | 生成多条思路，评估后选最优 | 写论文先列 3 个大纲方向，选了再展开 | 解谜、规划、需要多方案对比的任务 | 避免单线思维的死胡同 | 计算成本高，需要多次生成和评估 |
| **Reflexion（反思迭代）** | 试→失败→总结→重试 | 代码报错后看错误信息，改了再跑 | 代码生成、需要验证结果的任务 | 自我纠错、越做越好 | 需要可验证的反馈信号（测试/评估器） |
| **Role-playing（角色扮演）** | 不同角色各司其职，协作完成 | 真实团队开会：PM 出需求、开发写代码、测试找 bug | 复杂系统开发、跨领域综合任务 | 分工明确、各司其职 | 多 Agent 通信复杂、调试困难 |

***

## 复杂场景实战：范式选择与组合

### 场景一：市场调研报告生成

> **任务**：让 AI 写一份"新能源车行业 2025 年市场调研报告"，要求：收集最新销量数据、分析政策趋势、总结消费者反馈、撰写结论。

**适用范式**：Plan-and-Execute + ReAct + CoT

**思路**：
- Plan-and-Execute 先拟定计划（搜数据→分析政策→总结反馈→撰写报告）
- 执行步骤中，搜索数据和政策时用 ReAct（需要调用搜索引擎 API）
- 数据分析步骤中用 CoT（让模型逐步推理数据背后的逻辑）

```python
def market_research_report():
    # 阶段1: 生成计划
    plan = call_llm([{
        "role": "system", "content": "将'新能源车行业2025年市场调研报告'拆解为有序步骤。"
    }, {
        "role": "user", "content": "请返回 JSON: [{\"step\": 1, \"action\": \"...\"}, ...]"
    }])

    results = []
    for step in parse_json(plan):
        if "搜索" in step["action"] or "查询" in step["action"]:
            # 步骤需要外部数据 → 用 ReAct
            result = react(step["action"], tools={"search": search_api, "wikipedia": wiki_api})
        else:
            # 纯推理步骤 → 用 CoT
            result = call_llm([{
                "role": "system", "content": "Think step by step.",
            }, {
                "role": "user", "content": f"任务: {step['action']}\n之前的结果: {results}"
            }])
        results.append(result)

    # 最终汇总
    return call_llm([{
        "role": "system", "content": "请基于以上各步骤结果，撰写完整的市场调研报告。"
    }, {
        "role": "user", "content": "\n".join(results)
    }])
```

---

### 场景二：数学竞赛题求解

> **任务**：让 AI 解一道 IMO 级别的数学证明题，要求给出完整证明过程并验证正确性。

**适用范式**：ToT + Reflexion + CoT

**思路**：
- ToT 生成多条证明思路分支（反证法、归纳法、构造法……）
- 每条分支内部用 CoT 展开详细推理
- Reflexion 用于验证证明——如果发现逻辑漏洞，反思后换另一条分支

```python
def solve_math_proof(problem):
    # ToT: 生成多条证明思路
    approaches = call_llm([{
        "role": "system", "content": "请为以下数学题生成 3 种不同的证明思路。返回 JSON 数组。"
    }, {
        "role": "user", "content": f"题目: {problem}"
    }])

    best_proof = None
    memories = []

    for approach in parse_json(approaches):
        # CoT: 沿着这条思路展开详细推理
        proof = call_llm([{
            "role": "system", "content": "Think step by step. 给出完整的数学证明过程。"
        }, {
            "role": "user", "content": f"题目: {problem}\n证明思路: {approach}\n{memories}"
        }])

        # Reflexion: 验证证明是否正确
        feedback = verify_proof(proof)  # 调用验证器（如符号计算引擎）

        if feedback.valid:
            best_proof = proof
            break
        else:
            # 反思为什么这条路走不通
            reflection = call_llm([{
                "role": "system", "content": "以上证明失败了，请分析原因。"
            }, {
                "role": "user", "content": f"题目: {problem}\n思路: {approach}\n证明: {proof}\n错误: {feedback.message}"
            }])
            memories.append(f"之前失败的思路: {approach}，原因: {reflection}")

    return best_proof or "所有思路均未能完成证明"
```

---

### 场景三：智能客服系统

> **任务**：构建一个电商 AI 客服，需要：识别用户意图（退货/咨询/投诉）、查询订单信息、给出解决方案、复杂情况转人工。

**适用范式**：ReAct + Self-Ask + Role-playing

**思路**：
- ReAct 是主体框架——用户提出问题 → 想（需要什么信息）→ 做（查订单/查政策）→ 回答
- Self-Ask 处理复杂问题——如"我买的鞋子尺码不对能换吗？"先拆成"查订单→查鞋子尺码政策→判断是否符合退换条件"
- Role-playing 处理升级场景——复杂投诉转由"投诉处理专家"角色 Agent 接手

```python
def smart_customer_service(user_input):
    # Self-Ask: 意图识别与拆解
    intent = call_llm([{
        "role": "system", "content": "分析用户意图，返回 JSON: {\"intent\": \"退货|咨询|投诉\", \"needs_order\": true/false, \"sub_questions\": [...]}"
    }, {
        "role": "user", "content": f"用户说: {user_input}"
    }])

    intent_data = parse_json(intent)

    # 如果需要订单信息，用 ReAct 查询
    if intent_data["needs_order"]:
        order_info = react("查询用户最新订单信息", tools={"query_order": order_api})
    else:
        order_info = None

    # 根据意图选择处理角色
    if intent_data["intent"] == "投诉":
        # Role-playing: 转投诉处理专家
        response = call_llm([{
            "role": "system", "content": "你是投诉处理专家。你的职责是安抚用户情绪、理解问题、给出补偿方案。"
        }, {
            "role": "user", "content": f"用户: {user_input}\n订单: {order_info}"
        }])
    else:
        # 常规 ReAct 流程
        response = react(user_input, tools={
            "query_order": order_api,
            "query_policy": policy_api,
        })

    return response
```

---

### 场景四：自动化代码生成与调试

> **任务**：让 AI 根据需求描述生成完整的 Python 模块，包含函数实现、单元测试，并自动修复测试失败的问题。

**适用范式**：Reflexion + CoT + Plan-and-Execute

**思路**：
- Plan-and-Execute 先把需求拆解为"设计接口→实现函数→编写测试→运行验证"
- 函数实现步骤用 CoT 逐步推导逻辑
- 测试失败后用 Reflexion 自动修复

```python
def auto_code_generation(requirement):
    # 阶段1: 计划
    plan = [
        {"step": 1, "action": "设计模块接口和数据结构"},
        {"step": 2, "action": "实现核心函数"},
        {"step": 3, "action": "编写单元测试"},
        {"step": 4, "action": "运行测试并修复失败用例"},
    ]

    code = ""
    tests = ""
    memories = []

    for step in plan:
        if step["step"] <= 2:
            # 设计和实现 → CoT
            code = call_llm([{
                "role": "system", "content": "Think step by step. 请先分析需求，再给出代码实现。"
            }, {
                "role": "user", "content": f"需求: {requirement}\n步骤: {step['action']}\n{memories}"
            }])
        elif step["step"] == 3:
            # 写测试
            tests = call_llm([{
                "role": "system", "content": "请为以下代码编写完整的单元测试，使用 pytest 框架。"
            }, {
                "role": "user", "content": f"代码:\n{code}"
            }])
        else:
            # Reflexion: 运行测试 → 失败 → 反思 → 修复
            for retry in range(3):
                result = run_tests(tests, code)
                if result.all_passed:
                    break
                # 反思修复
                code = call_llm([{
                    "role": "system", "content": "测试失败了。请分析错误信息，修复代码。"
                }, {
                    "role": "user", "content": f"代码:\n{code}\n测试:\n{tests}\n错误: {result.errors}"
                }])
                memories.append(f"第{retry+1}次修复: 针对错误 {result.errors}")

    return code, tests
```

---

### 场景五：软件团队自动化开发

> **任务**：给定一个产品需求文档（PRD），让 AI 团队自动完成从需求评审到代码提交的全流程。

**适用范式**：Role-playing + Plan-and-Execute + Reflexion

**思路**：
- Role-playing 是主体——产品经理、架构师、开发、测试各司其职
- Plan-and-Execute 在每个角色内部使用——先规划自己要做什么，再执行
- Reflexion 在开发和测试环节使用——代码跑不通时自动修复

```python
def ai_dev_team(prd):
    roles = {
        "产品经理": {
            "职责": "审阅 PRD，输出功能需求清单和优先级排序",
            "范式": "plan_and_execute",  # 拆解需求
        },
        "架构师": {
            "职责": "根据需求清单设计系统架构，输出技术方案文档",
            "范式": "plan_and_execute",
        },
        "开发工程师": {
            "职责": "根据技术方案实现代码",
            "范式": "reflexion",  # 写代码 → 测试 → 修复
        },
        "测试工程师": {
            "职责": "审查代码质量，编写集成测试，评估风险",
            "范式": "reflexion",
        },
    }

    messages = []

    for role_name, config in roles.items():
        context = "\n".join(messages)

        # 每个角色使用自己的范式执行任务
        response = call_llm([{
            "role": "system", "content": f"你是{role_name}。你的职责：{config['职责']}。请基于之前的讨论完成你的工作。"
        }, {
            "role": "user", "content": f"产品需求:\n{prd}\n\n之前的讨论:\n{context}"
        }])

        messages.append(f"[{role_name}]: {response}")

    # 汇总所有产出
    final = call_llm([{
        "role": "system", "content": "请汇总以上各角色的产出，给出完整的项目交付清单。"
    }, {
        "role": "user", "content": "\n".join(messages)
    }])

    return final
```

***
