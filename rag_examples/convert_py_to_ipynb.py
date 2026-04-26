# -*- coding: utf-8 -*-
"""
批量将 Python 文件转换为 Jupyter Notebook 格式
转换规则：
1. 文件顶部的多行文档注释 -> Markdown 单元
2. 每个函数定义 -> 代码单元
"""

import json
import os
import re
import sys

def parse_py_file(file_path):
    """
    解析 Python 文件，提取各个部分
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    # 分离文件
    cells = []
    
    # 1. 提取文件开头的注释块（文档部分）
    header_lines = []
    in_header = True
    header_started = False
    
    for i, line in enumerate(lines):
        # 跳过顶部的 if __name__ == '__main__' 部分及其之前的空行
        if line.strip().startswith("if __name__") or line.strip().startswith('if __name__'):
            # 找到主程序入口，收集之前的部分
            break
        
        # 收集文件头部注释
        if in_header:
            # 跳过顶部的导入语句和空行
            stripped = line.strip()
            if stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
                header_started = True
                header_lines.append(line)
            elif header_started and not stripped:
                # 头部注释后遇到空行，结束头部
                if len(header_lines) > 0:
                    in_header = False
            elif header_started:
                header_lines.append(line)
            elif stripped and not stripped.startswith('#'):
                # 非注释非空行，说明头部结束
                in_header = False
    
    # 添加头部注释为 Markdown 单元
    if header_lines:
        # 清理注释头部
        header_text = '\n'.join(header_lines)
        # 移除顶部可能的多余内容
        header_text = re.sub(r'^#.*?\n#.*?\n#.*?\n', '', header_text, count=1)  # 移除第一组 # ==== 头部
        # 转换 # 注释为 Markdown
        header_text = convert_py_comments_to_markdown(header_text)
        if header_text.strip():
            cells.append({
                'cell_type': 'markdown',
                'metadata': {},
                'source': [header_text]
            })
    
    # 2. 提取函数定义
    current_cell_lines = []
    current_cell_type = 'code'  # 默认是代码单元
    in_function = False
    function_docstring = []
    in_docstring = False
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # 跳过主程序入口及其后面的内容
        if stripped.startswith("if __name__") or stripped.startswith('if __name__'):
            # 保存当前代码块
            if current_cell_lines:
                code_text = '\n'.join(current_cell_lines)
                cells.append({
                    'cell_type': 'code',
                    'metadata': {},
                    'source': [code_text],
                    'outputs': [],
                    'execution_count': None
                })
                current_cell_lines = []
            break
        
        # 检测函数定义
        if stripped.startswith('def '):
            # 保存之前的代码块
            if current_cell_lines:
                code_text = '\n'.join(current_cell_lines)
                cells.append({
                    'cell_type': 'code',
                    'metadata': {},
                    'source': [code_text],
                    'outputs': [],
                    'execution_count': None
                })
                current_cell_lines = []
            
            # 开始新的函数
            current_cell_lines = [line]
            in_function = True
            
            # 检查函数后面是否有文档字符串
            # 简单处理：直接添加整个函数作为代码单元
            continue
        
        # 检测文档字符串（多行字符串）
        if '"""' in stripped or "'''" in stripped:
            if stripped.count('"""') == 2:
                # 单行文档字符串
                continue
            elif '"""' in stripped:
                quote = '"""'
            else:
                quote = "'''"
            
            if not in_docstring and stripped.startswith(quote):
                in_docstring = True
                function_docstring = [line]
            elif in_docstring:
                if quote in stripped:
                    in_docstring = False
                    function_docstring.append(line)
                    # 将文档字符串转换为 Markdown
                    doc_text = '\n'.join(function_docstring)
                    doc_text = convert_py_docstring_to_markdown(doc_text)
                    if doc_text.strip():
                        cells.append({
                            'cell_type': 'markdown',
                            'metadata': {},
                            'source': [doc_text]
                        })
                    function_docstring = []
            else:
                function_docstring.append(line)
            continue
        
        # 在函数内收集代码
        if in_function or current_cell_lines:
            # 检查是否是函数块的结束（新的函数定义或类定义）
            if stripped.startswith('def ') or stripped.startswith('class ') or stripped.startswith('if __name__'):
                # 保存当前代码块
                if current_cell_lines:
                    code_text = '\n'.join(current_cell_lines)
                    cells.append({
                        'cell_type': 'code',
                        'metadata': {},
                        'source': [code_text],
                        'outputs': [],
                        'execution_count': None
                    })
                    current_cell_lines = []
                
                if stripped.startswith('def '):
                    current_cell_lines = [line]
                    in_function = True
                else:
                    in_function = False
            else:
                current_cell_lines.append(line)
                # 如果是空行太多，可能需要分割
                if len(current_cell_lines) > 50:
                    # 检查是否需要分割
                    pass
    
    # 保存最后的代码块
    if current_cell_lines:
        code_text = '\n'.join(current_cell_lines)
        cells.append({
            'cell_type': 'code',
            'metadata': {},
            'source': [code_text],
            'outputs': [],
            'execution_count': None
        })
    
    return cells


def convert_py_comments_to_markdown(text):
    """
    将 Python 注释转换为 Markdown 格式
    """
    lines = text.split('\n')
    result = []
    
    in_code_block = False
    
    for line in lines:
        stripped = line.strip()
        
        # 处理代码块标记
        if stripped.startswith('```'):
            if not in_code_block:
                in_code_block = True
            else:
                in_code_block = False
            result.append(stripped)
        elif in_code_block:
            result.append(stripped)
        elif stripped.startswith('# ---'):
            result.append('---')
        elif stripped.startswith('# ='):
            # # ====== 标题
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
    """
    将 Python 文档字符串转换为 Markdown
    """
    # 移除三引号
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
        if stripped.startswith('📊') or stripped.startswith('💡') or stripped.startswith('📦') or stripped.startswith('🔑') or stripped.startswith('⚠️'):
            result.append('**' + stripped + '**')
        elif stripped.startswith('- '):
            result.append(stripped)
        elif stripped == '':
            result.append('')
        elif re.match(r'^\d+\.', stripped):  # 编号列表
            result.append(stripped)
        else:
            result.append(stripped)
    
    return '\n'.join(result)


def create_ipynb(cells, output_path):
    """
    创建 Jupyter Notebook 文件
    """
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
    """
    转换单个 Python 文件为 Jupyter Notebook
    """
    print(f"转换: {py_path} -> {ipynb_path}")
    
    cells = parse_py_file(py_path)
    
    if cells:
        create_ipynb(cells, ipynb_path)
        print(f"  ✓ 成功创建 {ipynb_path} ({len(cells)} 个单元)")
    else:
        print(f"  ✗ 未能解析文件内容")


def main():
    # 定义要转换的文件列表
    base_dir = r"c:\沃林AI课程\AI0226_0309课程\wolin_learn-master\rag_examples"
    
    files_to_convert = [
        # 01_milvus_basics
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
        ("05_rag_pipeline/rag_step_by_step.py", "05_rag_pipeline/rag_step_by_step.ipynb"),
        
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
