# 用三个案例理解 DeepEval

学习主线：**把回答变成测试用例，按标准评分，再把安全要求变成断言。**

运行代码只有四个文件。按 01 → 02 → 03 阅读，模型连接统一放在 `judge.py`。

| 文件 | 学什么 |
|---|---|
| `01_first_evaluation.py` | 用例、G-Eval 评分标准、阈值、评分理由 |
| `02_rag_evaluation.py` | 忠实于检索资料与回答相关的区别 |
| `03_safety_gate.py` | 隐私断言、失败退出、脱敏评估日志 |
| `judge.py` | 通过 `utils.model_utils.get_model("qwen")` 获取在线客户端，并适配 DeepEval 接口 |

三个案例用固定问答作为待测数据，评分由真实模型完成，无需搭建 RAG 服务或数据库。实际接入业务时，将 `actual_output` 换成应用生成的回答。

## 安装与运行

在项目根目录打开 PowerShell，首次运行时安装依赖：

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
if (!(Test-Path .env)) { Copy-Item .env.example .env }
```

已有项目环境可直接复用，无需重建。三个案例统一使用共享 `get_model("qwen")` 返回的在线客户端，默认模型为 `qwen-plus`。在根目录 `.env` 中设置：

```dotenv
ALIYUN_API_KEY=填入自己的密钥
# 可选：单次模型请求超时（秒）
DEEPEVAL_TIMEOUT_SECONDS=120
```

已有项目根目录密钥时无需重复填写。只读取根目录 `.env`，系统环境变量优先。依赖包写在根目录 `requirements.txt`；环境变量写在 `.env`，两种文件不能混用。

`get_model("qwen")` 返回 LangChain 客户端；`judge.py` 中的 `QwenJudge` 只补齐 DeepEval 要求的同步、异步及结构化输出接口，不重新创建客户端。模型名称、密钥和服务地址统一由共享工厂管理，不再使用 `DEEPEVAL_JUDGE_PROVIDER`、`DEEPEVAL_JUDGE_MODEL` 切换裁判。

直接按文件运行：

```powershell
.venv/Scripts/python.exe deepeval_examples/01_first_evaluation.py
.venv/Scripts/python.exe deepeval_examples/02_rag_evaluation.py
.venv/Scripts/python.exe deepeval_examples/03_safety_gate.py --version broken
$LASTEXITCODE
.venv/Scripts/python.exe deepeval_examples/03_safety_gate.py --version fixed
$LASTEXITCODE
```

也可在 IDE 中直接运行案例文件。已移除统一的 `--lesson`、`--preview` 入口；没有模型时先阅读文件中的固定用例。

## 观察什么

**01：同一道题、同一套标准，对比两个答案。** 正确的“18 个月保修”应高分，编造的“终身保修”应低分。直接阅读 `LLMTestCase → GEval → measure() → score/reason/is_successful()`；两个答案复用同一套评分步骤。

**02：两个指标互相补充。** `retrieval_context` 是实际检索资料，不是参考答案。

| 回答情形 | 忠实度：资料支持吗？ | 相关性：回应问题了吗？ |
|---|---|---|
| 资料和回答都是保修 18 个月 | 高 | 高 |
| 资料写 18 个月，回答说终身保修 | 低 | 仍可能高 |
| 问保修，回答照抄支付方式 | 仍可能高 | 低 |

以上是教学预期，实际分数由裁判产生。两项指标帮助分析答案，不能单独衡量检索召回是否完整。

**03：安全要求成为测试断言。** broken 披露虚构邮箱，fixed 拒绝披露。原生 `assert_test()` 完成评分与断言，不先调用 `measure()` 重复评分。DeepEval **4.3.3** 的 `PIILeakageMetric` 高分表示更安全。退出码 0 表示通过、1 表示安全不通过、2 表示配置或评估错误；CI 应拦截所有非零退出码。01、02 的负例低分属于正常演示。

第三课自动在根目录 `reports/deepeval/` 保存 JSON，记录分数、理由、阈值、耗时、退出码和 `evaluation_id`，并隐藏其中的邮箱。这体现评估过程的可观测性，属于评估日志，不是应用调用链追踪。报告目录可在不需要历史结果时清理，下次运行会重新创建。邮箱脱敏仅覆盖本课样本；SDK 自身的缓存或已配置的云端报告不经过此处理。

## 验证

```powershell
.venv/Scripts/python.exe -m pytest tests/test_deepeval_examples.py -q
```

离线测试使用预设裁判响应检查 SDK 调用、安全断言、日志脱敏和直接运行入口，不验证真实模型的判断能力。真实评分用上面的四条案例命令验证；Qwen 调用会产生 API 费用。理由保留裁判原文，可能是英文。

实测中，裁判曾把拒绝句里的“私人邮箱”字段名称误当作个人信息，也曾给出与安全分方向不一致的解释。因此仍需结合实际回答人工复核，不能把一次用例通过等同于完整的安全保障。

本模块只维护 `.py`，不生成或同步 `.ipynb`。

教学目录只有上述四个 `.py` 和 README。根目录 `.venv/` 是可重建的运行环境，`reports/deepeval/` 是生成结果；`tests/test_deepeval_examples.py` 是维护用源码，保留用于回归验证，不要求学员阅读。
