# =============================================================================
# graph_rag_demo — GraphRAG 六阶段流水线（真实 LLM + Function Calling + LangGraph）
# =============================================================================
# 用途：结合 Neo4j 知识图谱 + 大模型回答复杂问题，把「查图谱」拆成六个阶段，
#       用真实 LLM 驱动实体抽取 / Cypher 生成 / 答案生成，
#       用 Function Calling 调用 Neo4j 查询工具，用 LangGraph 串联整个流水线。
# 难度：⭐⭐⭐⭐⭐（5 星）
#
# 六阶段流水线（一一对应下方 StateGraph 的六个节点）：
#   1. 实体抽取    extract_entities —— LLM 从问题中抽出图谱相关实体
#   2. 生成 Cypher  generate_cypher  —— LLM 依实体 + Schema 生成 Cypher，并调用工具
#   3. Neo4j 执行   execute_query    —— 读取模型的 tool_calls，真正执行 Cypher 查询
#   4. 组装上下文   assemble_context —— 图谱结果整理成自然语言事实
#   5. 构建 Prompt  build_prompt      —— 事实 + 问题 + 防幻觉约束
#   6. 生成答案    generate_answer   —— LLM 基于事实回答，不编造
#
# 三种技术各自的落地位置：
#   - 真实 LLM       ：阶段 1 / 2 / 6 都是调用 get_model() 得到的大模型
#   - Function Calling：query_graph 用 @tool 声明；阶段 2 用 bind_tools 绑定，
#                       模型发出 tool_calls，阶段 3 读取并执行
#   - LangGraph      ：StateGraph 把六个阶段串成一张图，状态在节点间流转
#
# 依赖：
#   - Neo4j：项目根 .env 中的 NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD
#   - LLM  ：默认 Ollama 本地 qwen3.5:2b（无需 API Key）；
#            可把 DEFAULT_LLM_PROVIDER 改成 "qwen" 走云端（需 ALIYUN_API_KEY）
# =============================================================================

import json
import os
import sys
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from neo4j import GraphDatabase

from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from utils.model_utils import get_model

# 显式加载项目根目录的 .env：本文件位于 rag_examples/06_rag_advanced/ 下，
# 而 rag_examples/ 下也有一个 .env（不含 Neo4j 配置），直接 load_dotenv()
# 会就近加载到它。故向上定位到项目根，读取含 NEO4J_* 配置的 .env。
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

# Windows 控制台编码
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# 默认 LLM 服务商：None = Ollama 本地 qwen3.5:2b（无需 API Key）
#                  "qwen" = 阿里云 qwen-plus（需 ALIYUN_API_KEY，Function Calling 更稳）
DEFAULT_LLM_PROVIDER = "qwen"

# ---- 图谱 Schema 描述（供 LLM 生成 Cypher 时参考）----
SCHEMA_DESC = """知识图谱 Schema：
- 节点标签 Concept，属性：name（概念名，唯一）、category（分类）、description（一句话描述）
- 关系类型 RELATED，属性：type（关系名，例如「基于」「使用」「优化」）
- 关系总是从 Concept 指向 Concept"""

# ---- 种子数据：概念 + 关系 ----
CONCEPTS = [
    {"name": "Transformer", "category": "模型架构",
     "description": "Attention Is All You Need 提出的架构，一眼看全局、自动划重点"},
    {"name": "BERT", "category": "模型架构",
     "description": "Google 的 Encoder-only 模型，前后都看，像做完形填空"},
    {"name": "GPT", "category": "模型架构",
     "description": "OpenAI 的 Decoder-only 模型，逐字生成"},
    {"name": "LLaMA", "category": "模型架构",
     "description": "Meta 的开源大模型，任何人都能下载"},
    {"name": "RAG", "category": "知识增强",
     "description": "检索增强生成——AI 先查资料再回答，开卷考试"},
    {"name": "Fine-tuning", "category": "知识增强",
     "description": "微调——用专业数据再训练，通才变专家"},
    {"name": "RLHF", "category": "模型优化",
     "description": "人类反馈强化学习——人类给回答打分，教 AI 好好说话"},
    {"name": "Agent", "category": "Agent架构",
     "description": "能自主感知环境、规划行动并完成复杂任务的 AI"},
    {"name": "MCP", "category": "Agent架构",
     "description": "Model Context Protocol——AI 的万能 USB-C 接口"},
    {"name": "Function Calling", "category": "Agent架构",
     "description": "让 AI 调用外部工具，从聊天变成能干活"},
    {"name": "Prompt Engineering", "category": "提示交互",
     "description": "提示词工程——怎么提问，AI 才给你好答案"},
    {"name": "Chain of Thought", "category": "推理范式",
     "description": "思维链——让 AI 一步步写出推理过程"},
    {"name": "Embedding", "category": "知识增强",
     "description": "把文字变成数字向量，语义相近的数字相近"},
    {"name": "Vector Database", "category": "知识增强",
     "description": "向量数据库——按语义相似度找东西"},
    {"name": "Token", "category": "模型基础",
     "description": "AI 读写的最小单位，像积木块"},
    {"name": "Hallucination", "category": "模型基础",
     "description": "幻觉——AI 一本正经地胡说八道"},
    {"name": "Context Window", "category": "模型基础",
     "description": "上下文窗口——AI 一次能记住的最大文字量"},
]

RELATIONS = [
    ("BERT", "基于", "Transformer"),
    ("GPT", "基于", "Transformer"),
    ("LLaMA", "基于", "Transformer"),
    ("RAG", "补充", "Fine-tuning"),
    ("RAG", "使用", "Embedding"),
    ("RAG", "使用", "Vector Database"),
    ("RLHF", "优化", "GPT"),
    ("Agent", "使用", "Prompt Engineering"),
    ("Agent", "使用", "Function Calling"),
    ("Agent", "使用", "MCP"),
    ("Chain of Thought", "提升", "Prompt Engineering"),
    ("Hallucination", "被缓解", "RAG"),
    ("Context Window", "限制", "Token"),
    ("Embedding", "用于", "Vector Database"),
]


class GraphRAGSystem:
    """GraphRAG 系统 — 用知识图谱增强 LLM 的检索和推理能力

    内部用 LangGraph 把「实体抽取 → 生成 Cypher → 执行查询 →
    组装上下文 → 构建 Prompt → 生成答案」六个阶段串成一张图。
    """

    def __init__(self, neo4j_uri=None, neo4j_user=None, neo4j_password=None,
                 provider=DEFAULT_LLM_PROVIDER):
        # 1. 连接 Neo4j（配置从 .env 读取，不硬编码密码）
        self.uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
        self.password = neo4j_password or os.getenv("NEO4J_PASSWORD", "")
        self.database = os.getenv("NEO4J_DATABASE", "neo4j")
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        self._init_graph()

        # 2. 初始化 LLM 与查询工具
        self.llm = get_model(provider)
        self.query_graph_tool = self._make_query_graph_tool()
        self.llm_with_tools = self.llm.bind_tools([self.query_graph_tool])

        # 3. 构建六阶段流水线
        self.graph = self._build_graph()

    def close(self):
        """关闭驱动，释放连接资源"""
        self.driver.close()

    # ------------------------------------------------------------------
    # 图数据初始化
    # ------------------------------------------------------------------
    def _init_graph(self):
        """幂等写入种子数据：已存在的概念/关系复用，缺失的才创建。

        注意：不清空已有数据，避免误删其他数据；如需彻底重置请自行删除。
        """
        with self.driver.session(database=self.database) as session:
            for c in CONCEPTS:
                session.run(
                    "MERGE (c:Concept {name: $name}) "
                    "SET c.category = $category, c.description = $description",
                    **c,
                )

            for from_name, rel_type, to_name in RELATIONS:
                session.run(
                    "MATCH (a:Concept {name: $from_name}) "
                    "MATCH (b:Concept {name: $to_name}) "
                    "MERGE (a)-[r:RELATED {type: $rel_type}]->(b)",
                    from_name=from_name, to_name=to_name, rel_type=rel_type,
                )

        print(f"✅ 知识图谱就绪（{len(CONCEPTS)} 个概念，{len(RELATIONS)} 条关系）\n")

    def _get_concept_names(self):
        """返回图谱中所有概念名（供实体抽取阶段参考）。"""
        with self.driver.session(database=self.database) as session:
            return [r["name"] for r in session.run(
                "MATCH (c:Concept) RETURN c.name AS name ORDER BY name")]

    # ------------------------------------------------------------------
    # Function Calling 工具：查询 Neo4j
    # ------------------------------------------------------------------
    def _make_query_graph_tool(self):
        """把 Neo4j 只读查询封装成 @tool，供模型通过 Function Calling 调用。"""

        @tool
        def query_graph(cypher: str) -> str:
            """在 Neo4j 知识图谱上执行一条只读 Cypher 查询（MATCH ... RETURN），返回结果。

            参数 cypher：Cypher 查询语句，实体名用单引号字面量（如 'BERT'），不要用 $参数。
            """
            return self._execute_cypher(cypher)

        return query_graph

    def _execute_cypher(self, cypher: str) -> str:
        """执行只读 Cypher 并返回结果文本（带安全校验，防写入/注入）。"""
        cypher = cypher.strip().strip("`").strip()
        upper = cypher.upper()

        # 只允许只读查询
        if not upper.startswith("MATCH") or " RETURN " not in upper:
            return "[错误] 只允许 MATCH ... RETURN 形式的只读查询"
        for kw in (" DELETE", " DETACH", " CREATE", " MERGE", " SET ", " REMOVE",
                   " DROP", " CALL "):
            if kw in upper:
                return f"[错误] 查询包含被禁止的写操作关键词：{kw.strip()}"

        try:
            with self.driver.session(database=self.database) as session:
                records = [r.data() for r in session.run(cypher)]
            if not records:
                return "（查询无结果）"
            return json.dumps(records, ensure_ascii=False, indent=2)
        except Exception as e:
            return f"[查询失败] {e}"

    def _fallback_query(self, entities):
        """兜底查询：当模型没生成可用 Cypher 时，按实体查直接关系。"""
        if not entities:
            return "（未识别到图谱实体）"
        results = []
        with self.driver.session(database=self.database) as session:
            for name in entities:
                rows = [r.data() for r in session.run(
                    "MATCH (c:Concept {name: $name})-[r:RELATED]->(o:Concept) "
                    "RETURN c.name AS c, r.type AS t, o.name AS o",
                    name=name)]
                results.append({"实体": name, "关系": rows})
        return json.dumps(results, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # LangGraph 六阶段流水线
    # ------------------------------------------------------------------
    def _build_graph(self):
        # 状态：六个阶段的中间产物都在这里流转
        class GraphRAGState(TypedDict):
            question: str
            entities: list
            cypher: str
            graph_result: str
            graph_context: str
            final_prompt: str
            answer: str
            messages: Annotated[list, add_messages]

        # —— 阶段 1：实体抽取（LLM）——
        def extract_entities(state: GraphRAGState):
            question = state["question"]
            concepts = self._get_concept_names()
            prompt = (
                f"你是实体抽取助手。下面是一个知识图谱中的概念清单：\n{concepts}\n\n"
                f"用户问题：{question}\n\n"
                f"请从概念清单中找出用户问题涉及的概念（问题中提到的、或语义相关的概念名）。"
                f"只输出一个 JSON 数组，例如：[\"BERT\", \"Transformer\"]，不要输出其他内容。"
            )
            entities = []
            try:
                resp = self.llm.invoke(prompt)
                entities = self._parse_entities(resp.content, concepts)
            except Exception:
                entities = []
            # 兜底：LLM 失败时退回子串匹配
            if not entities:
                entities = [c for c in concepts if c.lower() in question.lower()]
            print(f"  [阶段1 实体抽取] {entities}")
            return {"entities": entities}

        # —— 阶段 2：生成 Cypher 并调用工具（LLM + Function Calling）——
        def generate_cypher(state: GraphRAGState):
            system = (
                f"你是 Neo4j 知识图谱查询助手，必须通过调用工具来查询图谱。\n\n"
                f"{SCHEMA_DESC}\n\n"
                f"规则：\n"
                f"1. 只能调用 query_graph 工具执行只读 Cypher（MATCH ... RETURN）。\n"
                f"2. Cypher 中实体名用单引号字面量（如 'BERT'），不要用 $参数。\n"
                f"3. 查关系示例："
                f"MATCH (c:Concept {{name: 'BERT'}})-[r:RELATED]->(o:Concept) "
                f"RETURN c.name, r.type, o.name\n"
                f"4. 查描述示例："
                f"MATCH (c:Concept {{name: 'BERT'}}) RETURN c.name, c.category, c.description\n\n"
                f"请先根据实体和问题生成合适的 Cypher，然后调用 query_graph 工具执行，"
                f"不要直接回答。"
            )
            human = (
                f"实体：{state['entities']}\n"
                f"用户问题：{state['question']}\n\n"
                f"请生成 Cypher 并调用 query_graph 工具。"
            )
            resp = self.llm_with_tools.invoke([SystemMessage(content=system),
                                               HumanMessage(content=human)])
            return {"messages": [resp]}

        # —— 阶段 3：执行查询（读取 tool_calls，真正查 Neo4j）——
        def execute_query(state: GraphRAGState):
            last = state["messages"][-1]
            tool_messages = []
            cypher_used = ""

            if getattr(last, "tool_calls", None):
                for tc in last.tool_calls:
                    if tc["name"] == "query_graph":
                        cypher_used = tc["args"].get("cypher", "")
                        content = self._execute_cypher(cypher_used)
                        # 模型生成的 Cypher 无效时，退回兜底查询
                        if content.startswith("[错误") or content.startswith("[查询失败"):
                            content = self._fallback_query(state["entities"])
                        tool_messages.append(
                            ToolMessage(content=content, tool_call_id=tc["id"]))

            # 模型没调用工具时的兜底
            if not tool_messages:
                content = self._fallback_query(state["entities"])
                tool_messages.append(ToolMessage(content=content, tool_call_id="fallback"))

            print(f"  [阶段3 执行查询] cypher = {cypher_used or '（兜底）'}")
            return {"messages": tool_messages,
                    "cypher": cypher_used,
                    "graph_result": tool_messages[-1].content}

        # —— 阶段 4：组装图谱上下文（代码）——
        def assemble_context(state: GraphRAGState):
            graph_context = (
                "以下是知识图谱中的相关事实（JSON 形式）：\n"
                f"{state['graph_result']}"
            )
            return {"graph_context": graph_context}

        # —— 阶段 5：构建最终 Prompt（代码，含防幻觉约束）——
        def build_prompt(state: GraphRAGState):
            final_prompt = (
                f"你是一个严谨的 AI 技术专家。请严格根据下面的知识图谱事实回答用户问题。\n\n"
                f"【知识图谱事实】\n{state['graph_context']}\n\n"
                f"【用户问题】\n{state['question']}\n\n"
                f"【回答要求】\n"
                f"1. 只能依据上面的事实回答，禁止使用外部知识或编造事实。\n"
                f"2. 如果事实不足以回答，直接说明「根据图谱信息无法回答」，不要猜测。\n"
                f"3. 回答要通俗、准确。"
            )
            return {"final_prompt": final_prompt}

        # —— 阶段 6：生成答案（LLM）——
        def generate_answer(state: GraphRAGState):
            resp = self.llm.invoke(state["final_prompt"])
            return {"answer": resp.content}

        # 把六个节点串成图
        graph = (
            StateGraph(GraphRAGState)
            .add_node("extract_entities", extract_entities)
            .add_node("generate_cypher", generate_cypher)
            .add_node("execute_query", execute_query)
            .add_node("assemble_context", assemble_context)
            .add_node("build_prompt", build_prompt)
            .add_node("generate_answer", generate_answer)
            .add_edge(START, "extract_entities")
            .add_edge("extract_entities", "generate_cypher")
            .add_edge("generate_cypher", "execute_query")
            .add_edge("execute_query", "assemble_context")
            .add_edge("assemble_context", "build_prompt")
            .add_edge("build_prompt", "generate_answer")
            .add_edge("generate_answer", END)
            .compile()
        )
        return graph

    # ------------------------------------------------------------------
    # 解析辅助
    # ------------------------------------------------------------------
    def _parse_entities(self, text, concepts):
        """把 LLM 输出解析成合法实体名列表（过滤掉不在图谱里的名字）。"""
        text = (text or "").strip()
        if text.startswith("```"):
            text = text.strip("`").strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [x for x in data if isinstance(x, str) and x in concepts]
            if isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, list):
                        return [x for x in v if isinstance(x, str) and x in concepts]
        except Exception:
            pass
        # 兜底：直接从文本里找出现过的概念名
        return [c for c in concepts if c.lower() in text.lower()]

    # ------------------------------------------------------------------
    # 对外入口
    # ------------------------------------------------------------------
    def answer_question(self, question: str, verbose: bool = True) -> str:
        """GraphRAG 完整流程入口：跑一遍六阶段流水线，返回最终答案。"""
        if verbose:
            print(f"❓ 用户问题：{question}\n")

        result = self.graph.invoke({"question": question, "messages": []})

        if verbose:
            print(f"🕸️ 图谱结果：\n{result.get('graph_result', '')}\n")

        return result.get("answer", "")


# =============================================================================
# 主程序
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  GraphRAG — 六阶段流水线（真实 LLM + Function Calling + LangGraph）")
    print("=" * 70 + "\n")

    try:
        rag = GraphRAGSystem()

        questions = [
            "BERT 是什么？它和 Transformer 有什么关系？",
            "GPT 和 BERT 有什么区别？它们的底层架构是什么？",
            "Agent 需要用到哪些技术？",
            "RAG 和 Fine-tuning 有什么区别？",
        ]

        for q in questions:
            print("=" * 70)
            answer = rag.answer_question(q)
            print(f"\n💡 回答：{answer}\n")

        rag.close()
        print("=" * 70)
        print("✅ GraphRAG 六阶段流水线演示完成")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ 运行失败：{e}")
        print("  提示：LLM 需要 Ollama 运行中（ollama pull qwen3.5:2b）；")
        print("       或把 DEFAULT_LLM_PROVIDER 改为 \"qwen\" 走云端（需 ALIYUN_API_KEY）。")
