# =============================================================================
# Few-Shot Prompt 示例
# =============================================================================
#  
# 用途：教学演示 - 使用 Few-Shot Learning 让模型理解格式要求
#
# 核心概念：
#   - Few-Shot = "给例子"
#   - 如何让模型理解你的格式要求
#   - 就像教小孩做题一样
# =============================================================================

# -----------------------------------------------------------------------------
# 运行前检查
# -----------------------------------------------------------------------------
# 1. 已安装 Ollama 服务
# 2. 已下载模型：ollama pull qqwen3.5:9b
# -----------------------------------------------------------------------------

# 设置 UTF-8 编码（Windows 专用）
import sys
import io
sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer,
    encoding='utf-8',
    errors='replace',
    line_buffering=True
)


# =============================================================================
# 第一部分：理解 Few-Shot Learning
# =============================================================================
"""
什么是 Few-Shot Prompting？

📖 定义
   Few-Shot = "几个例子"
   给模型几个示例，让它模仿格式回答问题

🎯 为什么要给例子？

   不用 Few-Shot:
   你："把下面这些话分类为正面或负面"
   AI："好的，这句话是..." (可能不理解你想要的格式)

   用 Few-Shot:
   你："把下面这些话分类为正面或负面"
       "例子 1: '今天天气真好' → 正面"
       "例子 2: '我心情不好' → 负面"
       "例子 3: '这部电影很棒' → 正面"
       "现在请分类：'这个餐厅很难吃'"
   AI："负面" (明白了格式和要求)

💡 生活化比喻
   Few-Shot 就像教小孩做题：
   "看，这道题是这样做的..." (给例子)
   "现在你来做这道..." (让小孩模仿)
"""


# =============================================================================
# 示例 1: 最简单的 Few-Shot
# =============================================================================

def simplest_few_shot():
    """
    最简单的 Few-Shot 示例

    先给几个例子，再问新问题
    """
    print("=" * 60)
    print("示例 1: 最简单的 Few-Shot")
    print("=" * 60)

    from langchain_core.prompts import PromptTemplate
    from langchain_ollama import ChatOllama

    # 创建 Few-Shot 模板
    prompt = PromptTemplate.from_template("""
请模仿下面的例子，完成最后一题。

例子 1:
问：法国首都是哪里？
答：巴黎

例子 2:
问：美国首都是哪里？
答：华盛顿

例子 3:
问：中国首都是哪里？
答：北京

问题：
问：日本首都是哪里？
答：""")

    model = ChatOllama(model="qwen3.5:9b")
    response = model.invoke(prompt.format())
    print(f"AI 回复：{response.content}")
    print()


# =============================================================================
# 示例 2: 情感分类（实用场景）
# =============================================================================

def sentiment_classification():
    """
    使用 Few-Shot 进行情感分类

    让模型理解分类格式
    """
    print("=" * 60)
    print("示例 2: 情感分类（Few-Shot）")
    print("=" * 60)

    from langchain_core.prompts import PromptTemplate
    from langchain_ollama import ChatOllama

    # Few-Shot 模板：给几个分类例子
    prompt = PromptTemplate.from_template("""
请判断以下评论的情感倾向（正面/负面）。

例子：
评论："这家餐厅的食物很好吃，服务也很棒！"
情感：正面

评论："等了一个小时才上菜，味道还很咸。"
情感：负面

评论："环境不错，但是价格太贵了。"
情感：负面

评论："电影特效很震撼，剧情也很感人。"
情感：正面

现在请判断：
评论："{review}"
情感：""")

    model = ChatOllama(model="qwen3.5:2b")

    # 测试不同评论
    reviews = [
        "这个产品太棒了，我非常喜欢！",
        "完全不值这个价，浪费钱。",
        "快递很快，但包装有破损。"
    ]

    for review in reviews:
        formatted = prompt.format(review=review)
        response = model.invoke(formatted)
        print(f"评论：{review}")
        print(f"情感：{response.content}")
        print()

'''
"query": "深圳是一个经济特区"，
"out_put": "深圳市"

"西安是一个历史悠久的"
"out_put": "西安市"
'''

'''
"query": "北京今天在下雨"
"out_put": "北京市"
'''

# =============================================================================
# 示例 3: 格式化输出
# =============================================================================

def formatted_output():
    """
    使用 Few-Shot 让模型输出特定格式

    比如 JSON、CSV 等结构化格式
    """
    print("=" * 60)
    print("示例 3: 格式化输出（Few-Shot）")
    print("=" * 60)

    from langchain_core.prompts import PromptTemplate
    from langchain_ollama import ChatOllama

    # Few-Shot 模板：教模型输出 JSON 格式
    prompt = PromptTemplate.from_template("""
请提取文本中的人名、年龄和职业，输出为 JSON 格式。

例子 1:
文本："小明今年 25 岁，是一名软件工程师。"
输出：
{{
  "name": "小明",
  "age": 25,
  "job": "软件工程师"
}}

例子 2:
文本："李华医生，30 岁，在协和医院工作。"
输出：
{{
  "name": "李华",
  "age": 30,
  "job": "医生"
}}

现在请处理：
文本："{text}"
输出：""")

    model = ChatOllama(model="qwen3.5:2b")

    text = "王芳今年 28 岁，她是一名小学老师。"
    formatted = prompt.format(text=text)
    response = model.invoke(formatted)
    print(f"文本：{text}")
    print(f"提取结果：{response.content}")
    print()


# =============================================================================
# 示例 4: 语言风格模仿
# =============================================================================

def style_imitation():
    """
    使用 Few-Shot 让模型模仿特定语言风格
    """
    print("=" * 60)
    print("示例 4: 语言风格模仿（Few-Shot）")
    print("=" * 60)

    from langchain_core.prompts import PromptTemplate
    from langchain_ollama import ChatOllama

    # Few-Shot 模板：教模型用古风语言
    prompt = PromptTemplate.from_template("""
请用古风语言回复。

例子：
问：你好吗？
答：甚好，甚好。

问：今天天气如何？
答：今日天朗气清，风和日丽。

问：你吃饭了吗？
答：已用过膳，多谢关心。

问："{question}"
答：""")

    model = ChatOllama(model="qwen3.5:2b")

    questions = [
        "你在做什么？",
        "你喜欢什么颜色？"
    ]

    for question in questions:
        formatted = prompt.format(question=question)
        response = model.invoke(formatted)
        print(f"问：{question}")
        print(f"答：{response.content}")
        print()


# =============================================================================
# 示例 5: 使用 FewShotPromptTemplate（LangChain 封装）
# =============================================================================

def using_few_shot_prompt_template():
    """
    使用 LangChain 的 FewShotPromptTemplate

    更优雅地管理示例
    """
    print("=" * 60)
    print("示例 5: FewShotPromptTemplate（LangChain 封装）")
    print("=" * 60)

    from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
    from langchain_ollama import ChatOllama

    # 1. 准备示例列表
    examples = [
        {
            "question": "法国首都是哪里？",
            "answer": "巴黎"
        },
        {
            "question": "美国首都是哪里？",
            "answer": "华盛顿"
        },
        {
            "question": "中国首都是哪里？",
            "answer": "北京"
        }
    ]

    # 2. 创建示例模板（定义每个例子怎么格式化）
    example_prompt = PromptTemplate.from_template(
        "问：{question}\n答：{answer}"
    )

    # 3. 创建 FewShotPromptTemplate
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,              # 示例列表
        example_prompt=example_prompt,  # 示例模板
        prefix="请模仿下面的例子回答问题：",
        suffix="问题：{question}\n答：",
        input_variables=["question"]    # 输入变量
    )

    # 4. 使用
    model = ChatOllama(model="qwen3.5:2b")

    # 会自动把所有示例拼接成 prompt
    formatted = few_shot_prompt.format(question="日本首都是哪里？")
    print("生成的完整 Prompt:")
    print("-" * 40)
    print(formatted)
    print("-" * 40)
    print()

    response = model.invoke(formatted)
    print(f"AI 回复：{response.content}")
    print()


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  Few-Shot Prompt 示例")
    print("  说明：给例子让模型模仿")
    print("=" * 70 + "\n")

    print("【运行前检查】")
    print("  1. 已安装 Ollama 服务")
    print("  2. 已下载模型：ollama pull qwen3.5:2b")
    print()

    # 运行示例
    # simplest_few_shot()
    # sentiment_classification()
    # formatted_output()
    # style_imitation()
    using_few_shot_prompt_template()

    print("=" * 70)
    print("  接下来学习：pipeline_prompt.py（Pipeline 组合）")
    print("=" * 70 + "\n")
