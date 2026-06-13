#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
langgraph_examples 专用 .py → .ipynb 转换脚本

将教学 .py 文件转换为 Jupyter Notebook 格式。
规则：
  - # ==== 分隔符 → markdown 标题
  - 多行三引号块 → markdown 概念说明
  - Windows 编码样板自动剥离
  - 排除纯概念文件和工具文件

用法：
  python convert_py_to_ipynb.py              # 转换全部教学文件
  python convert_py_to_ipynb.py --dry-run    # 仅列出将转换的文件
"""

import os
import re
import json
import sys

# 项目根目录（向上两级从脚本位置）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 需要转换的文件（相对路径）
TEACHING_FILES = [
    "01_introduction/simple_graph.py",
    "02_state_and_branching/conditional_branch.py",
    "03_agent_loop/agent_react.py",
    "03_agent_loop/create_agent_demo.py",
    "04_workflows/prompt_chain.py",
    "04_workflows/routing.py",
    "04_workflows/parallelization.py",
    "04_workflows/evaluator_optimizer.py",
    "05_practical/search_qa_agent.py",
]

# 不需要转换的文件
EXCLUDE = [
    "what_is_langgraph.py",  # 纯概念文件
    "utils/graph_helpers.py",  # 共享工具
    "__init__.py",
    "convert_py_to_ipynb.py",
]


def strip_encoding_boilerplate(code: str) -> str:
    """移除 Windows 编码样板代码（Jupyter 不需要）"""
    # 匹配: import sys\nimport io\nif sys.stdout.encoding ... \n    sys.stdout = io.TextIOWrapper(...)
    pattern = (
        r"import sys\n"
        r"(?:import io\n)?"  # io 导入可能存在
        r"(?:# .*\n)?"  # 可选注释
        r"if sys\.stdout\.encoding[^\n]*:\n"
        r"    sys\.stdout = io\.TextIOWrapper\([^)]*\)\n"
    )
    code = re.sub(pattern, "", code, flags=re.DOTALL)
    return code.lstrip("\n")


def parse_file_to_cells(filepath: str) -> list[dict]:
    """解析 .py 文件为 notebook 单元格列表

    策略：
      1. 按 "# ====..." 行切分文件为多个 segment
      2. 每个 segment 的第一个非空行如果是 "# <文字>" 则当作标题
      3. 其余内容：纯 docstring → markdown，有代码 → code cell
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # 剥离编码样板
    content = strip_encoding_boilerplate(content)

    # 按分隔行切分：匹配 "# " 开头 + 大量 "=" 的行
    SEP = re.compile(r'^# ={20,}\s*$', re.MULTILINE)

    # 找到所有分隔符位置
    splits = list(SEP.finditer(content))

    if not splits:
        # 没有分隔符，整个文件作为一个代码块
        return [{"cell_type": "code", "source": [content.strip()]}]

    cells = []
    pending_title = None

    # 处理每对分隔符之间的内容
    for idx in range(len(splits)):
        start = splits[idx].end()
        end = splits[idx + 1].start() if idx + 1 < len(splits) else len(content)
        segment = content[start:end].strip()

        if not segment:
            continue

        # 尝试提取标题：segment 的第一行如果是 "# 文字内容"（不含 ====），则为标题
        seg_lines = segment.split("\n")
        first_line = seg_lines[0].strip()
        is_title = False

        if first_line.startswith("#") and "===" not in first_line:
            # 检查是不是一个独立的标题行（下一行是空行或直接就是内容）
            potential_title = first_line.lstrip("#").strip()
            if potential_title and len(seg_lines) >= 1:
                # 如果整个 segment 只有这一行注释 → 这是一个 section 标题
                rest = "\n".join(seg_lines[1:]).strip()
                if not rest or rest.startswith('"""') or rest.startswith("'''"):
                    # 标题行后跟 docstring 或没有内容 → 这是标题
                    pending_title = potential_title
                    segment = rest  # 把 docstring 留给下一段处理
                    is_title = True

        if is_title and not segment:
            continue

        # 处理 segment 内容（可能带有 pending_title）
        _process_segment(segment, pending_title, cells)
        if is_title and segment:
            pending_title = None  # 标题已消费

    return cells


def _process_segment(segment: str, title: str | None, cells: list):
    """处理一个 segment：判断是 markdown 还是 code"""
    segment = segment.strip()
    if not segment:
        if title:
            cells.append({"cell_type": "markdown", "source": [f"## {title}"]})
        return

    # 如果整个 segment 是一个 docstring（用于概念说明）
    if (segment.startswith('"""') or segment.startswith("'''")) and \
       (segment.endswith('"""') or segment.endswith("'''")):
        inner = segment[3:-3].strip()
        if inner:
            has_code = bool(re.search(
                r'\b(def|class|if|for|while|return|import|from|print|try|with)\b',
                inner
            ))
            if not has_code and len(inner) > 10:
                # 纯概念说明 → markdown
                source = f"## {title}\n\n{inner}" if title else inner
                cells.append({"cell_type": "markdown", "source": [source]})
                return

    # 默认：代码 cell
    if title:
        cells.append({"cell_type": "markdown", "source": [f"## {title}"]})

    cells.append({"cell_type": "code", "source": [segment]})


def cleanup_notebook(cells: list[dict]) -> list[dict]:
    """后处理：清理空代码 cell、优化 cell 边界"""
    cleaned = []

    for cell in cells:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"]).strip()
            if not source:
                continue  # 跳过空代码 cell
            # 检查是否只剩 __main__ 检查行
            lines = source.split("\n")
            if len(lines) == 1 and lines[0].strip().startswith("if __name__"):
                continue
            # 检查是否只剩注释行
            non_comment = [l for l in lines if l.strip() and not l.strip().startswith("#")]
            if not non_comment:
                cleaned.append({
                    "cell_type": "markdown",
                    "source": ["\n".join(
                        l.strip().lstrip("# ").strip()
                        for l in lines if l.strip().startswith("#")
                    )],
                })
                continue

        cleaned.append(cell)

    return cleaned


def write_notebook(cells: list[dict], output_path: str):
    """写入 .ipynb JSON 文件"""
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    for i, cell in enumerate(cells):
        nb_cell = {
            "id": f"cell-{i}",
            "cell_type": cell["cell_type"],
            "metadata": {},
            "source": cell["source"],
        }
        if cell["cell_type"] == "code":
            nb_cell["outputs"] = []
            nb_cell["execution_count"] = None
        notebook["cells"].append(nb_cell)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)

    # 验证：检查写入的 JSON 是否有效
    with open(output_path, "r", encoding="utf-8") as f:
        json.load(f)


def convert_one(py_path: str) -> str:
    """转换单个 .py 文件，返回输出路径"""
    py_full = os.path.join(SCRIPT_DIR, py_path)

    if not os.path.exists(py_full):
        print(f"  ⚠️ 文件不存在: {py_path}")
        return None

    ipynb_path = py_full.replace(".py", ".ipynb")

    print(f"  转换: {py_path} → {os.path.basename(ipynb_path)}")

    cells = parse_file_to_cells(py_full)
    cells = cleanup_notebook(cells)

    code_count = sum(1 for c in cells if c["cell_type"] == "code")
    md_count = sum(1 for c in cells if c["cell_type"] == "markdown")
    print(f"    {len(cells)} 个单元格（{md_count} markdown + {code_count} code）")

    write_notebook(cells, ipynb_path)
    return ipynb_path


def main():
    dry_run = "--dry-run" in sys.argv

    print("=" * 60)
    print("langgraph_examples .py → .ipynb 转换")
    print("=" * 60)
    print()

    if dry_run:
        print("【预览模式】将转换以下文件：")
        for f in TEACHING_FILES:
            print(f"  - {f}")
        print(f"\n共 {len(TEACHING_FILES)} 个文件")
        return

    converted = []
    for py_file in TEACHING_FILES:
        result = convert_one(py_file)
        if result:
            converted.append(result)

    print(f"\n完成：{len(converted)}/{len(TEACHING_FILES)} 个 notebook 生成成功")

    # 快速验证
    print("\n【验证】")
    for ipynb_path in converted:
        try:
            with open(ipynb_path, "r", encoding="utf-8") as f:
                nb = json.load(f)
            cells = nb.get("cells", [])
            code_cells = [c for c in cells if c["cell_type"] == "code"]
            md_cells = [c for c in cells if c["cell_type"] == "markdown"]
            empty_code = [
                i for i, c in enumerate(code_cells)
                if not "".join(c["source"]).strip()
            ]
            status = "OK" if not empty_code else f"WARN: {len(empty_code)} empty code cells"
            print(f"  {os.path.basename(ipynb_path)}: {len(md_cells)} md + {len(code_cells)} code [{status}]")
        except Exception as e:
            print(f"  {os.path.basename(ipynb_path)}: ERROR: {e}")


if __name__ == "__main__":
    main()
