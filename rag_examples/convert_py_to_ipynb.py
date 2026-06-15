# -*- coding: utf-8 -*-
"""
批量将 Python 文件转换为 Jupyter Notebook 格式
转换规则：
1. 文件顶部的文件头注释 -> Markdown 单元
2. 每个函数定义 -> 一个完整代码单元（含 docstring）
"""

import json
import os
import re


def parse_py_file(file_path):
    """解析 Python 文件，提取文件头和函数定义"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')
    cells = []

    # ── 1. 提取文件头部注释（# ==== 和 # 说明行） → Markdown ──
    header_lines = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("if __name__"):
            break
        if stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
            header_lines.append(line)
        elif header_lines and not stripped:
            # 注释块后的第一个空行 → 头部结束
            header_lines.append(line)
        elif header_lines and stripped:
            # 遇到非注释非空行（如 import）→ 头部结束
            break
        elif not stripped and not header_lines:
            # 文件开头的空行，跳过
            continue

    if header_lines:
        header_text = '\n'.join(header_lines)
        # 去掉第一组 # ==== 行
        header_text = re.sub(r'^#.*?\n#.*?\n#.*?\n', '', header_text, count=1)
        header_text = convert_py_comments_to_markdown(header_text)
        if header_text.strip():
            cells.append({
                'cell_type': 'markdown',
                'metadata': {},
                'source': [header_text]
            })

    # ── 2. 收集模块级 import 行 → 第一个代码单元 ──
    import_lines = []
    func_start_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('def '):
            func_start_idx = i
            break
        # 收集 import / from-import 行
        if stripped.startswith('import ') or stripped.startswith('from '):
            import_lines.append(line)
        # 也收集紧随 import 后的空行（不超过1个）
        elif import_lines and not stripped and (not import_lines or import_lines[-1].strip()):
            # 只在 import 块内收集一个空行分隔
            if len(import_lines) == 0 or import_lines[-1].strip():
                import_lines.append(line)
            else:
                break
        elif import_lines and stripped:
            # 非空非 import 行 → import 块结束
            break

    if import_lines:
        code_text = '\n'.join(import_lines).rstrip('\n')
        if code_text.strip():
            cells.append(make_code_cell(code_text))

    # ── 3. 提取函数定义 — 每个函数一个完整代码单元 ──
    current_lines = []

    for i in range(func_start_idx, len(lines)):
        line = lines[i]
        stripped = line.strip()

        # 遇到 if __name__ 停止
        if stripped.startswith("if __name__"):
            if current_lines:
                code_text = '\n'.join(current_lines).rstrip('\n')
                if code_text.strip():
                    cells.append(make_code_cell(code_text))
                current_lines = []
            break

        # 新函数开始 — 保存上一个
        if stripped.startswith('def '):
            if current_lines:
                code_text = '\n'.join(current_lines).rstrip('\n')
                if code_text.strip():
                    cells.append(make_code_cell(code_text))
            current_lines = [line]
            continue

        # 收集函数代码行
        if current_lines:
            current_lines.append(line)

    # 保存最后一个函数
    if current_lines:
        code_text = '\n'.join(current_lines).rstrip('\n')
        if code_text.strip():
            cells.append(make_code_cell(code_text))

    return cells


def make_code_cell(code_text):
    """创建标准代码单元"""
    # 去掉尾部连续空行
    code_text = code_text.rstrip('\n')
    return {
        'cell_type': 'code',
        'metadata': {},
        'source': [code_text],
        'outputs': [],
        'execution_count': None
    }


def convert_py_comments_to_markdown(text):
    """将 Python 注释转换为 Markdown 格式"""
    lines = text.split('\n')
    result = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('# ---'):
            result.append('---')
        elif stripped.startswith('# ='):
            match = re.match(r'# (=+)\s*(.*?)\s*(=+)', stripped)
            if match:
                title = match.group(2).strip()
                level = min(len(match.group(1)), 6)
                result.append('#' * level + ' ' + title)
            else:
                result.append(stripped.lstrip('# '))
        elif stripped.startswith('# '):
            result.append(stripped[2:])
        elif stripped.startswith('#'):
            result.append(stripped[1:].lstrip())
        elif stripped == '':
            result.append('')
        else:
            result.append(stripped)

    return '\n'.join(result)


def convert_py_docstring_to_markdown(text):
    """将 Python 文档字符串转换为 Markdown（保留，供外部调用）"""
    text = text.strip()
    if text.startswith('"""'):
        text = text[3:]
    if text.endswith('"""'):
        text = text[:-3]
    if text.startswith("'''"):
        text = text[3:]
    if text.endswith("'''"):
        text = text[:-3]

    lines = text.split('\n')
    result = []
    for line in lines:
        stripped = line.strip()
        if any(stripped.startswith(c) for c in ['📊', '💡', '📦', '🔑', '⚠️']):
            result.append('**' + stripped + '**')
        else:
            result.append(stripped)
    return '\n'.join(result)


def create_ipynb(cells, output_path):
    """创建 Jupyter Notebook 文件"""
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)


def convert_py_to_ipynb(py_path, ipynb_path):
    """转换单个 Python 文件为 Jupyter Notebook"""
    print(f"转换: {py_path} -> {ipynb_path}")
    cells = parse_py_file(py_path)
    if cells:
        create_ipynb(cells, ipynb_path)
        print(f"  ✓ 成功创建 {ipynb_path} ({len(cells)} 个单元)")
    else:
        print(f"  ✗ 未能解析文件内容")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    files_to_convert = [
        # 01_milvus_basics
        ("01_milvus_basics/01_connect_milvus.py", "01_milvus_basics/01_connect_milvus.ipynb"),
        ("01_milvus_basics/02_create_collection.py", "01_milvus_basics/02_create_collection.ipynb"),
        ("01_milvus_basics/03_insert_data.py", "01_milvus_basics/03_insert_data.ipynb"),
        ("01_milvus_basics/04_create_index.py", "01_milvus_basics/04_create_index.ipynb"),
        # 02_document_chunking
        ("02_document_chunking/01_fixed_chunking.py", "02_document_chunking/01_fixed_chunking.ipynb"),
        ("02_document_chunking/02_sliding_window.py", "02_document_chunking/02_sliding_window.ipynb"),
        ("02_document_chunking/03_ai_chunking.py", "02_document_chunking/03_ai_chunking.ipynb"),
        ("02_document_chunking/04_summary_chunking.py", "02_document_chunking/04_summary_chunking.ipynb"),
        ("02_document_chunking/05_chunking_comparison.py", "02_document_chunking/05_chunking_comparison.ipynb"),
        # 03_retrieval_methods
        ("03_retrieval_methods/01_scalar_query.py", "03_retrieval_methods/01_scalar_query.ipynb"),
        ("03_retrieval_methods/02_vector_search.py", "03_retrieval_methods/02_vector_search.ipynb"),
        ("03_retrieval_methods/03_keyword_search.py", "03_retrieval_methods/03_keyword_search.ipynb"),
        ("03_retrieval_methods/04_hybrid_search.py", "03_retrieval_methods/04_hybrid_search.ipynb"),
        ("03_retrieval_methods/05_rerank.py", "03_retrieval_methods/05_rerank.ipynb"),
        # 04_rag_api
        ("04_rag_api/rag_qna_api.py", "04_rag_api/rag_qna_api.ipynb"),
        ("04_rag_api/rag_retrieval_api.py", "04_rag_api/rag_retrieval_api.ipynb"),
        # 05_rag_pipeline
        ("05_rag_pipeline/rag_full_pipeline.py", "05_rag_pipeline/rag_full_pipeline.ipynb"),
        ("05_rag_pipeline/rag_minimal.py", "05_rag_pipeline/rag_minimal.ipynb"),
        # 06_rag_advanced
        ("06_rag_advanced/01_hybrid_search_advanced.py", "06_rag_advanced/01_hybrid_search_advanced.ipynb"),
        ("06_rag_advanced/02_dual_collection_design.py", "06_rag_advanced/02_dual_collection_design.ipynb"),
        ("06_rag_advanced/03_from_mock_to_real.py", "06_rag_advanced/03_from_mock_to_real.ipynb"),
        # 06_rag_evaluation
        ("06_rag_evaluation/01_rag_evaluation.py", "06_rag_evaluation/01_rag_evaluation.ipynb"),
        # embedding_examples
        ("embedding_examples/01_embedding_basics.py", "embedding_examples/01_embedding_basics.ipynb"),
        ("embedding_examples/02_aliyun_embedding.py", "embedding_examples/02_aliyun_embedding.ipynb"),
        ("embedding_examples/03_local_embedding.py", "embedding_examples/03_local_embedding.ipynb"),
        ("embedding_examples/04_embedding_comparison.py", "embedding_examples/04_embedding_comparison.ipynb"),
    ]

    print("=" * 60)
    print("开始转换 Python 文件为 Jupyter Notebook")
    print("=" * 60)
    print()

    success_count = 0
    for py_file, ipynb_file in files_to_convert:
        py_path = os.path.join(base_dir, py_file)
        ipynb_path = os.path.join(base_dir, ipynb_file)
        if os.path.exists(py_path):
            convert_py_to_ipynb(py_path, ipynb_path)
            success_count += 1
        else:
            print(f"文件不存在: {py_path}")

    print()
    print("=" * 60)
    print(f"转换完成！成功转换 {success_count}/{len(files_to_convert)} 个文件")
    print("=" * 60)


if __name__ == '__main__':
    main()
