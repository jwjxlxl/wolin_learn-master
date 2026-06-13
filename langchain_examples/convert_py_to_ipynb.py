# -*- coding: utf-8 -*-
"""
批量将 langchain_examples 的 Python 教学文件转换为 Jupyter Notebook 格式。

转换规则：
1. 剥离 Windows 终端编码样板代码（sys.stdout = io.TextIOWrapper(...)）
2. 文件顶部 # === 注释块 → Markdown 单元（标题 + 学习目标）
3. 模块级文档字符串（三引号块）→ Markdown 单元（概念讲解）
4. import 语句 → 第一个代码单元
5. 每个 # === 分隔的示例函数 → Markdown 标题 + 代码单元
6. if __name__ == '__main__' → 末尾代码单元

排除：
- utils/ 工具模块
- 学习路线.py 元信息文件
- human_in_the_loop.py（交互式，保留 .py）
"""

import json
import os
import re


def strip_encoding_boilerplate(content: str) -> str:
    """移除 Windows 终端编码样板代码（Jupyter 不需要）。"""
    # 匹配整个样板代码块：
    #   [可选注释行]
    #   import sys
    #   import io
    #   [可选空行]
    #   sys.stdout = io.TextIOWrapper(...)
    # 使用 DOTALL 让 . 匹配换行符
    pattern = (
        r'(?:# [^\n]*编码[^\n]*\n)?'   # 可选：注释行（含"编码"）
        r'import sys\n'
        r'import io\n'
        r'\n?'                          # 可选空行
        r'sys\.stdout = io\.TextIOWrapper\(.+?\)\n\n?'  # TextIOWrapper 调用
    )
    content = re.sub(pattern, '', content, flags=re.DOTALL)
    return content


def extract_header_markdown(lines: list, start: int) -> tuple:
    """
    提取文件顶部的 # === 注释块，转为 Markdown。

    Returns:
        (markdown_text, next_line_index)
    """
    header_lines = []
    i = start
    in_header = False

    while i < len(lines):
        stripped = lines[i].strip()

        if stripped.startswith('# ===') or stripped.startswith('# ---'):
            # 分隔线本身 → 跳过，但标记已进入 header
            in_header = True
            i += 1
            continue

        if stripped.startswith('# '):
            # 去掉 # 前缀，保留中文内容
            header_lines.append(stripped[2:])
            in_header = True
            i += 1
        elif stripped.startswith('#'):
            header_lines.append(stripped[1:].lstrip())
            in_header = True
            i += 1
        elif stripped == '' and in_header:
            # header 结束
            break
        elif stripped == '':
            i += 1
        else:
            # 遇到非注释非空行，header 结束
            break

    if header_lines:
        return '\n'.join(header_lines), i
    return '', start


def extract_docstring(lines: list, start: int) -> tuple:
    """
    提取模块级文档字符串（三引号块），转为 Markdown。

    Returns:
        (markdown_text, next_line_index)
    """
    # 跳过空行
    while start < len(lines) and lines[start].strip() == '':
        start += 1

    if start >= len(lines):
        return '', start

    stripped = lines[start].strip()
    if not (stripped.startswith('"""') or stripped.startswith("'''")):
        return '', start

    quote = '"""' if stripped.startswith('"""') else "'''"

    # 单行文档字符串
    if len(stripped) > 3 and stripped.endswith(quote) and stripped.count(quote) == 2:
        text = stripped[3:-3].strip()
        return text, start + 1

    # 多行文档字符串
    doc_lines = [stripped[3:]] if len(stripped) > 3 else []
    i = start + 1
    while i < len(lines):
        if quote in lines[i]:
            # 结束行
            end_part = lines[i].split(quote)[0]
            if end_part.strip():
                doc_lines.append(end_part)
            break
        doc_lines.append(lines[i])
        i += 1

    return '\n'.join(doc_lines).strip(), i + 1


def extract_imports(lines: list, start: int) -> tuple:
    """
    提取 import 语句块。

    Returns:
        (import_code_string, next_line_index)
    """
    import_lines = []
    i = start

    while i < len(lines):
        stripped = lines[i].strip()

        if stripped.startswith('from ') or stripped.startswith('import '):
            import_lines.append(lines[i])
            i += 1
        elif stripped == '' and import_lines:
            # 空行后可能还有 import
            i += 1
            # 检查是否还有更多 import
            lookahead = i
            while lookahead < len(lines) and lines[lookahead].strip() == '':
                lookahead += 1
            if lookahead < len(lines) and (lines[lookahead].strip().startswith('from ') or
                                            lines[lookahead].strip().startswith('import ')):
                import_lines.append('')
                continue
            else:
                break
        elif stripped == '':
            i += 1
        else:
            break

    return '\n'.join(import_lines), i


def split_code_by_functions(code: str) -> list:
    """
    将一段代码按 def/class 边界拆分。

    返回: [(title, code_block), ...]
    title 从函数/类名推断，如 "without_langchain — 演示不用 LangChain 写 AI 应用"
    """
    if not code.strip():
        return []

    lines = code.split('\n')
    blocks = []
    current = []
    current_title = ''

    for i, line in enumerate(lines):
        stripped = line.strip()

        # 遇到 def/class 定义 —— 保存前一个块，开始新块
        if (stripped.startswith('def ') or stripped.startswith('class ')) and \
           (not line.startswith(' ') and not line.startswith('\t')):
            # 保存上一个块
            if current:
                block_code = '\n'.join(current).strip()
                if block_code:
                    blocks.append((current_title, block_code))
                    current_title = ''

            current = [line]

            # 从函数名生成标题
            match = re.match(r'(def|class)\s+(\w+)', stripped)
            if match:
                name = match.group(2)
                # 将 snake_case 转为可读标题
                readable = ' '.join(w.capitalize() if i == 0 else w
                                    for i, w in enumerate(name.split('_')))
                current_title = readable

            # 预览下一行是否有多行 docstring 的开始
            if i + 1 < len(lines):
                next_stripped = lines[i + 1].strip()
                if next_stripped.startswith('"""') or next_stripped.startswith("'''"):
                    # 读取 docstring 第一行作为标题补充
                    doc_line = next_stripped.strip('"').strip("'").strip()
                    if doc_line:
                        current_title = doc_line[:60]
            continue

        current.append(line)

    # 保存最后一个块
    if current:
        block_code = '\n'.join(current).strip()
        if block_code:
            blocks.append((current_title, block_code))

    return blocks


def find_section_boundaries(lines: list, start: int) -> list:
    """
    从 start 开始，找到由 # === 分隔的各段落边界。

    返回: [(section_type, title, code), ...]
    section_type: 'section' (有标题), 'code' (无标题), 'main' (__main__ 块)
    """
    sections = []
    i = start

    while i < len(lines):
        stripped = lines[i].strip()

        # 跳过空行
        if stripped == '':
            i += 1
            continue

        # 遇到 if __name__ —— 最终块
        if stripped.startswith('if __name__'):
            code_lines = []
            while i < len(lines):
                code_lines.append(lines[i])
                i += 1
            sections.append(('main', '', '\n'.join(code_lines)))
            break

        # 遇到 # === 分隔符
        if stripped.startswith('# ===') or stripped.startswith('# ---'):
            # 提取标题
            title = ''
            match = re.match(r'# [=\-]+(.+?)[=\-]*$', stripped)
            if match:
                title = match.group(1).strip()
            else:
                title = stripped.lstrip('# =').strip()

            i += 1

            # 跳过此分隔符后的空行
            while i < len(lines) and lines[i].strip() == '':
                i += 1

            # 收集代码块（到下一个 # === 或 if __name__ 或文件末尾）
            code_lines = []
            while i < len(lines):
                s = lines[i].strip()
                if s.startswith('# ===') or s.startswith('# ---') or s.startswith('if __name__'):
                    break
                code_lines.append(lines[i])
                i += 1

            code = '\n'.join(code_lines).strip()
            if code:
                # 检查代码块中是否有多个函数——如有则拆分
                sub_blocks = split_code_by_functions(code)
                if len(sub_blocks) > 1:
                    for sub_title, sub_code in sub_blocks:
                        sections.append(('section', sub_title or title, sub_code))
                elif sub_blocks:
                    sections.append(('section', title or sub_blocks[0][0], sub_blocks[0][1]))
                else:
                    sections.append(('section', title, code))

            continue

        # 非分隔符开头的代码 → 游离代码块
        code_lines = []
        while i < len(lines):
            s = lines[i].strip()
            if s.startswith('# ===') or s.startswith('# ---') or s.startswith('if __name__'):
                break
            if s == '' and code_lines and i + 1 < len(lines) and \
               lines[i + 1].strip().startswith(('# ===', 'if __name__')):
                break
            code_lines.append(lines[i])
            i += 1

        code = '\n'.join(code_lines).strip()
        if code:
            # 拆分游离代码中的函数
            sub_blocks = split_code_by_functions(code)
            if len(sub_blocks) > 1:
                for sub_title, sub_code in sub_blocks:
                    sections.append(('section', sub_title, sub_code))
            else:
                sections.append(('code', '', code))

    return sections


def parse_file_to_cells(file_path: str) -> list:
    """
    用状态机解析 .py 文件，转为 notebook cells。

    状态: HEADER → IMPORTS → DOCSTRING → BODY → MAIN
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 剥离编码样板
    content = strip_encoding_boilerplate(content)
    lines = content.split('\n')

    cells = []
    i = 0
    n = len(lines)

    # ---- 状态: HEADER - 收集顶部 # 注释 ----
    header_lines = []
    while i < n:
        stripped = lines[i].strip()
        if stripped.startswith('# ===') or stripped.startswith('# ---'):
            i += 1
            continue  # 跳过分隔线本身
        if stripped.startswith('# '):
            header_lines.append(stripped[2:])
            i += 1
        elif stripped.startswith('#') and not stripped.startswith('##'):
            header_lines.append(stripped[1:].lstrip())
            i += 1
        elif stripped == '':
            i += 1
        else:
            break  # 遇到非注释非空行，header 结束

    if header_lines:
        cells.append({
            'cell_type': 'markdown',
            'metadata': {},
            'source': ['\n'.join(header_lines)]
        })

    # ---- 状态: IMPORTS - 收集 import 语句 ----
    import_lines = []
    while i < n:
        stripped = lines[i].strip()
        if stripped.startswith('from ') or stripped.startswith('import '):
            import_lines.append(lines[i])
            i += 1
        elif stripped == '':
            # 空行：检查后面是否还有 import
            lookahead = i + 1
            while lookahead < n and lines[lookahead].strip() == '':
                lookahead += 1
            if lookahead < n and (lines[lookahead].strip().startswith('from ') or
                                   lines[lookahead].strip().startswith('import ')):
                import_lines.append('')
                i += 1
            else:
                i += 1
                break
        else:
            break

    if import_lines:
        cells.append({
            'cell_type': 'code',
            'metadata': {},
            'source': ['\n'.join(import_lines)],
            'outputs': [],
            'execution_count': None
        })

    # ---- 状态: DOCSTRING - 跳过空行找模块文档字符串 ----
    while i < n and lines[i].strip() == '':
        i += 1

    if i < n:
        stripped = lines[i].strip()
        if (stripped.startswith('"""') or stripped.startswith("'''")) and \
           not (stripped.startswith('def ') or stripped.startswith('class ')):
            quote = '"""' if stripped.startswith('"""') else "'''"
            # 单行文档字符串
            if len(stripped) > 6 and stripped.endswith(quote) and stripped.count(quote) == 2:
                doc_text = stripped[3:-3].strip()
                if doc_text:
                    cells.append({
                        'cell_type': 'markdown',
                        'metadata': {},
                        'source': [doc_text]
                    })
                i += 1
            else:
                # 多行文档字符串
                doc_body = [stripped[3:]] if len(stripped) > 3 else []
                i += 1
                while i < n:
                    if quote in lines[i]:
                        end_part = lines[i].split(quote)[0]
                        if end_part.strip():
                            doc_body.append(end_part)
                        i += 1
                        break
                    doc_body.append(lines[i])
                    i += 1
                doc_text = '\n'.join(doc_body).strip()
                if doc_text:
                    cells.append({
                        'cell_type': 'markdown',
                        'metadata': {},
                        'source': [doc_text]
                    })

    # ---- 状态: BODY - 处理 # === 分隔的段落和函数 ----
    # 收集从当前位置到 if __name__ 之间的所有行
    body_lines = []
    main_block = None

    while i < n:
        stripped = lines[i].strip()
        if stripped.startswith('if __name__'):
            # 收集 main 块
            main_lines = []
            while i < n:
                main_lines.append(lines[i])
                i += 1
            main_block = '\n'.join(main_lines)
            break
        body_lines.append(lines[i])
        i += 1

    # 解析 body 段落
    body_sections = parse_body_sections(body_lines)
    for sec_type, title, code in body_sections:
        if sec_type == 'section':
            if code:
                # 正常 section: 标题 + 代码
                cells.append({
                    'cell_type': 'markdown',
                    'metadata': {},
                    'source': [f'## {title}']
                })
                cells.append({
                    'cell_type': 'code',
                    'metadata': {},
                    'source': [code],
                    'outputs': [],
                    'execution_count': None
                })
            else:
                # 裸文档字符串: 内容本身就是 markdown
                cells.append({
                    'cell_type': 'markdown',
                    'metadata': {},
                    'source': [title]
                })
        elif sec_type == 'code':
            cells.append({
                'cell_type': 'code',
                'metadata': {},
                'source': [code],
                'outputs': [],
                'execution_count': None
            })

    # ---- 状态: MAIN - 添加 __main__ 块 ----
    if main_block:
        cells.append({
            'cell_type': 'code',
            'metadata': {},
            'source': [main_block],
            'outputs': [],
            'execution_count': None
        })

    return cells


def parse_body_sections(body_lines: list) -> list:
    """
    解析 BODY 部分的行，按 # === 分隔符和 def/class 边界拆分。

    策略：
    1. 遇到 # === 分隔线 → 提取标题，刷新当前 section
    2. 遇到 # 注释（非分隔线）→ 收集为 pending_title
    3. 遇到 def/class → 使用 pending_title 作为标题，函数体作为代码
    4. 其他代码 → 收集到当前代码块
    5. 跳过只有注释、没有实际代码的块

    返回: [(type, title, code), ...]
    """
    if not body_lines:
        return []

    result = []
    pending_title = ''     # 从 # 注释累积的标题
    current_code = []      # 当前代码行
    current_section_title = ''  # 从最后一个 # === 提取的标题

    def flush_section():
        """将当前的 pending_title + current_code 输出为一个 section。"""
        nonlocal pending_title, current_code, current_section_title

        # 分离纯注释行和实际代码
        comment_lines = []
        code_lines = []
        for line in current_code:
            s = line.strip()
            if s.startswith('#') and not s.startswith('# ===') and not s.startswith('# ---'):
                comment_lines.append(s[1:].lstrip())
            elif s:
                code_lines.append(line)
            else:
                code_lines.append(line)

        # 清理空行
        while code_lines and code_lines[0].strip() == '':
            code_lines.pop(0)
        while code_lines and code_lines[-1].strip() == '':
            code_lines.pop(-1)

        code_str = '\n'.join(code_lines).strip() if code_lines else ''

        # 检测裸文档字符串（没有函数/类的纯 docstring）
        if code_str and (code_str.startswith('"""') or code_str.startswith("'''")):
            # 尝试提取文档字符串内容
            quote = '"""' if code_str.startswith('"""') else "'''"
            inner = code_str[len(quote):]
            if inner.endswith(quote):
                inner = inner[:-len(quote)]
            inner = inner.strip()
            # 如果 code_str 只是文档字符串（不含任何 def/class/if），转为 markdown
            has_code = bool(re.search(r'\b(def|class|if|for|while|return|import|from|print|try|with)\b', inner))
            if not has_code and len(inner) > 10:
                # 这是一个纯概念解释文档字符串 → Markdown
                if pending_title:
                    inner = pending_title + '\n' + inner
                    pending_title = ''
                result.append(('section', inner, ''))  # title=内容, code=空 → 只生成 markdown
                current_code = []
                current_section_title = ''
                return

        if not code_str:
            # 没有实际代码 → 把注释累积为标题
            if comment_lines:
                if pending_title:
                    pending_title += '\n' + '\n'.join(comment_lines)
                else:
                    pending_title = '\n'.join(comment_lines)
            current_code = []
            return

        # 构建标题：优先 pending_title > current_section_title > 函数 docstring
        title = pending_title or current_section_title

        # 如果没有好的标题，从函数 docstring 或函数名推断
        if not title:
            for idx, line in enumerate(code_lines):
                m = re.match(r'(def|class)\s+(\w+)', line.strip())
                if m:
                    # 尝试从紧随的 docstring 获取更好的标题
                    for j in range(idx + 1, min(idx + 6, len(code_lines))):
                        ds = code_lines[j].strip()
                        if ds.startswith('"""') or ds.startswith("'''"):
                            inner = ds.strip('"').strip("'").strip()
                            if inner and not inner.startswith('Args') and not inner.startswith('Returns'):
                                title = inner[:80]
                            else:
                                # """ 独占一行，内容在下一行
                                if j + 1 < len(code_lines):
                                    next_line = code_lines[j + 1].strip()
                                    if next_line and not next_line.startswith('#'):
                                        title = next_line[:80]
                            break
                        if ds and not ds.startswith('#') and not ds.startswith('@'):
                            # 非注释非装饰器行（不是 docstring）
                            break
                    if not title:
                        title = ' '.join(w.capitalize() if i == 0 else w
                                        for i, w in enumerate(m.group(2).split('_')))
                    break

        if title:
            result.append(('section', title, code_str))
        else:
            result.append(('code', '', code_str))

        pending_title = ''
        current_code = []
        current_section_title = ''

    for line in body_lines:
        stripped = line.strip()

        # # === 分隔线 → 刷新前一个 section，提取新标题
        if stripped.startswith('# ===') or stripped.startswith('# ---'):
            flush_section()
            title_match = re.match(r'# [=\-]+\s*(.+?)\s*[=\-]*$', stripped)
            if title_match:
                t = title_match.group(1).strip()
                # 过滤纯符号的"标题"
                if not re.match(r'^[=\-]+$', t):
                    current_section_title = t
            continue

        # # 注释行（非分隔线）→ 如果当前没有代码，累积为标题
        if stripped.startswith('#'):
            # 跳过纯注释指令
            if re.match(r'^# [=\-]{10,}', stripped):
                continue
            if current_code:
                # 已有代码，这条注释属于代码的一部分
                current_code.append(line)
            else:
                # 还没有代码，累积为 pending_title
                text = stripped[1:].lstrip()
                if text and not text.startswith('===') and not text.startswith('---'):
                    if pending_title:
                        pending_title += ' — ' + text
                    else:
                        pending_title = text
            continue

        # 遇到 def/class 定义
        if (stripped.startswith('def ') or stripped.startswith('class ')) and \
           not line.startswith(' ') and not line.startswith('\t'):
            # 如果已有代码（前一个函数），先刷新
            if current_code and any(
                l.strip().startswith('def ') or l.strip().startswith('class ')
                for l in current_code if not l.startswith(' ')
            ):
                flush_section()
            current_code.append(line)
            continue

        # 普通代码行
        current_code.append(line)

    # 刷新最后一个 section
    flush_section()

    return result


def create_ipynb(cells: list, output_path: str):
    """将 cells 列表写入 .ipynb 文件。"""
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


def convert_one(py_path: str, ipynb_path: str) -> bool:
    """转换单个文件。返回是否成功。"""
    print(f"转换: {os.path.basename(py_path)} → {os.path.basename(ipynb_path)}")
    try:
        cells = parse_file_to_cells(py_path)
        if cells:
            create_ipynb(cells, ipynb_path)
            print(f"  ✓ {len(cells)} 个单元")
            return True
        else:
            print(f"  ✗ 未能解析出任何单元")
            return False
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        return False


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 排除列表
    exclude = {
        'utils/model_utils.py',
        'utils/__init__.py',
        '学习路线.py',
        'convert_py_to_ipynb.py',            # 本脚本自身
        '09_agent/human_in_the_loop.py',     # 交互式，保留 .py
    }

    # 自动发现所有 .py 文件
    py_files = []
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if not f.endswith('.py'):
                continue
            rel_path = os.path.relpath(os.path.join(root, f), base_dir).replace('\\', '/')
            if rel_path in exclude:
                print(f"跳过: {rel_path}")
                continue
            py_files.append(rel_path)

    print("=" * 60)
    print(f"开始转换 {len(py_files)} 个 Python 文件为 Jupyter Notebook")
    print("=" * 60)
    print()

    success_count = 0
    for rel_path in sorted(py_files):
        py_path = os.path.join(base_dir, rel_path)
        ipynb_path = os.path.join(base_dir, rel_path.replace('.py', '.ipynb'))

        if convert_one(py_path, ipynb_path):
            success_count += 1
        print()

    print("=" * 60)
    print(f"转换完成！成功 {success_count}/{len(py_files)} 个文件")
    print("=" * 60)


if __name__ == '__main__':
    main()
