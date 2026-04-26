# =============================================================================
# 函数封装的演进：从零散代码到可复用函数
# =============================================================================
#  
# 用途：教学演示 - 引导学生理解函数封装的必要性
# =============================================================================

# -----------------------------------------------------------------------------
# 重要：设置 UTF-8 编码和输出缓冲
# -----------------------------------------------------------------------------
# 这段代码解决两个问题：
#
# 1. UTF-8 编码：Windows 命令行默认使用 GBK 编码，无法显示中文
# 2. 输出缓冲：line_buffering=True 确保 print() 立即显示，而非程序结束时才输出
# -----------------------------------------------------------------------------
import sys
import io
sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer,
    encoding='utf-8',
    errors='replace',
    line_buffering=True  # 关键！让学生看到程序的实时执行进度
)

import ollama

# =============================================================================
# 第 1 步：重复的代码（让学生观察问题）
# =============================================================================
# 场景：我们需要多次调用大语言模型
#
# 🤔 请学生观察：这三段代码有什么相同点？有什么不同点？
# =============================================================================

def demo_repeated_code():
    """
    演示：重复的代码

    这是新手常写的代码风格——复制粘贴
    """
    print("=" * 70)
    print("【第 1 步】观察：重复的代码")
    print("=" * 70)

    # --- 第 1 次调用 ---
    print("\n>>> 第 1 次调用：打招呼")
    response1 = ollama.chat(
        model='qwen3:4b',
        messages=[{'role': 'user', 'content': '你好'}]
    )
    print(f"回复：{response1['message']['content'][:50]}...")

    # --- 第 2 次调用 ---
    print("\n>>> 第 2 次调用：问天气")
    response2 = ollama.chat(
        model='qwen3:4b',
        messages=[{'role': 'user', 'content': '如何判断明天会不会下雨？'}]
    )
    print(f"回复：{response2['message']['content'][:50]}...")

    # --- 第 3 次调用 ---
    print("\n>>> 第 3 次调用：问数学")
    response3 = ollama.chat(
        model='qwen3:4b',
        messages=[{'role': 'user', 'content': '1+1 等于几？'}]
    )
    print(f"回复：{response3['message']['content'][:50]}...")

    print("\n" + "-" * 70)
    print("💡 观察与思考：")
    print("   1. 这三段代码哪里长得一样？")
    print("   2. 哪里不一样？")
    print("   3. 如果要调用 100 次，要复制粘贴多少次？")
    print("   4. 如果想改模型名称，要改多少处？")
    print("-" * 70)
    print()


# =============================================================================
# 第 2 步：最简单的封装（提取共同模式）
# =============================================================================
# 🤔 引导：既然只有消息内容在变，为什么不把它变成参数呢？
# =============================================================================

def ask(message):
    """
    最简单的封装：只接受消息内容

    参数：
        message (str): 用户的问题
    """
    response = ollama.chat(
        model='qwen3:4b',
        messages=[{'role': 'user', 'content': message}]
    )
    return response['message']['content']


def demo_simple_encapsulation():
    """
    演示：最简单的封装
    """
    print("=" * 70)
    print("【第 2 步】封装：最简单的 ask 函数")
    print("=" * 70)

    print("\n>>> 代码变成这样：")
    print("    def ask(message):")
    print("        response = ollama.chat(...)")
    print("        return response['message']['content']")
    print()

    print(">>> 使用时：")
    result1 = ask('你好')
    print(f"  ask('你好') = {result1[:50]}...")

    result2 = ask('如何判断明天会不会下雨？')
    print(f"  ask('如何判断...') = {result2[:50]}...")

    print("\n" + "-" * 70)
    print("💡 好处：")
    print("   ✓ 代码变少了")
    print("   ✓ 想改模型名称，只需要改 1 处")
    print()
    print("🤔 新问题：")
    print("   1. 如果想用别的模型怎么办？")
    print("   2. 如果想多轮对话怎么办？")
    print("-" * 70)
    print()


# =============================================================================
# 第 3 步：添加更多参数（应对变化）
# =============================================================================
# 🤔 引导：模型名称也在变，为什么不也把它变成参数？
# =============================================================================

def ask_v2(message, model='qwen3:4b'):
    """
    增强版：支持指定模型

    参数：
        message (str): 用户的问题
        model (str): 模型名称，默认 qwen3:4b
    """
    response = ollama.chat(
        model=model,
        messages=[{'role': 'user', 'content': message}]
    )
    return response['message']['content']


def demo_more_parameters():
    """
    演示：添加更多参数
    """
    print("=" * 70)
    print("【第 3 步】增强：支持指定模型")
    print("=" * 70)

    print("\n>>> 代码：")
    print("    def ask_v2(message, model='qwen3:4b'):")
    print("        # 模型名称也变成参数")
    print()

    print(">>> 使用：")
    result1 = ask_v2('你好', model='qwen3:4b')
    print(f"  ask_v2('你好', model='qwen3:4b') = {result1[:50]}...")

    # 默认参数可以省略
    result2 = ask_v2('1+1=?')
    print(f"  ask_v2('1+1=?') [省略 model] = {result2[:50]}...")

    print("\n" + "-" * 70)
    print("💡 知识点：")
    print("   - 默认参数：调用时可以省略")
    print("   - 常用模型设默认值，特殊需求可覆盖")
    print("-" * 70)
    print()


# =============================================================================
# 第 4 步：支持多轮对话（消息历史）
# =============================================================================
# 🤔 引导：大模型不记得之前的对话，因为我们每次都只发一条消息
#      为什么不把整个对话历史都传给它？
# =============================================================================

def ask_v3(messages, model='qwen3:4b'):
    """
    支持多轮对话：传入完整的消息历史

    参数：
        messages (list): 消息列表，格式：
            [{'role': 'user', 'content': '...'},
             {'role': 'assistant', 'content': '...'}]
        model (str): 模型名称
    """
    response = ollama.chat(model=model, messages=messages)
    return response['message']['content']


def demo_multi_turn():
    """
    演示：多轮对话
    """
    print("=" * 70)
    print("【第 4 步】进阶：支持多轮对话")
    print("=" * 70)

    print("\n>>> 问题：之前的 ask 函数无法进行多轮对话")
    print(">>> 解决：传入完整的消息历史")
    print()

    # 构建对话历史
    messages = [
        {'role': 'user', 'content': '我叫小明，今年 10 岁'}
    ]

    print(">>> 第 1 轮：")
    reply1 = ask_v3(messages)
    print(f"  用户：我叫小明，今年 10 岁")
    print(f"  助手：{reply1[:50]}...")

    # 把助手的回复也加入历史
    messages.append({'role': 'assistant', 'content': reply1})

    # 第 2 轮
    messages.append({'role': 'user', 'content': '我叫什么名字？'})

    print("\n>>> 第 2 轮：")
    reply2 = ask_v3(messages)
    print(f"  用户：我叫什么名字？")
    print(f"  助手：{reply2[:80]}...")

    print("\n" + "-" * 70)
    print("💡 关键：")
    print("   - messages 列表保存完整对话历史")
    print("   - 每次调用后，把新消息追加到列表")
    print("-" * 70)
    print()


# =============================================================================
# 第 5 步：添加错误处理（生产级代码）
# =============================================================================
# 🤔 引导：如果 Ollama 服务没启动怎么办？模型不存在怎么办？
#      为什么不把错误处理也封装进函数？
# =============================================================================

def ask_v4(message, model='qwen3:4b'):
    """
    带错误处理的版本

    参数：
        message (str): 用户的问题
        model (str): 模型名称

    返回：
        str: 助手回复，出错时返回错误信息
    """
    try:
        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': message}]
        )
        return response['message']['content']

    except ollama.ResponseError as e:
        if e.status_code == 404:
            return f"错误：模型 '{model}' 不存在"
        return f"Ollama 错误：{e}"

    except ConnectionError:
        return "错误：无法连接到 Ollama 服务"

    except Exception as e:
        return f"未知错误：{e}"


def demo_error_handling():
    """
    演示：带错误处理
    """
    print("=" * 70)
    print("【第 5 步】健壮：添加错误处理")
    print("=" * 70)

    print("\n>>> 代码结构：")
    print("    try:")
    print("        response = ollama.chat(...)")
    print("        return response['message']['content']")
    print("    except ollama.ResponseError as e:")
    print("        return f'错误：{e}'")
    print()

    print(">>> 正常调用：")
    result = ask_v4('你好')
    print(f"  ask_v4('你好') = {result[:50]}...")

    print("\n>>> 错误情况（模型不存在）：")
    result = ask_v4('你好', model='nonexistent-model')
    print(f"  ask_v4('你好', model='nonexistent') = {result}")

    print("\n" + "-" * 70)
    print("💡 好处：")
    print("   ✓ 程序不会崩溃")
    print("   ✓ 调用者不需要写重复的 try-except")
    print("   ✓ 错误信息更友好")
    print("-" * 70)
    print()


# =============================================================================
# 第 6 步：终极版本（灵活配置 + 流式输出）
# =============================================================================
# 🤔 引导：有些时候我们想要流式输出，有些时候想要设置温度参数
#      难道要为每个参数都写一个函数？
#      用 **kwargs 可以把"可能变化的参数"一起打包传入！
# =============================================================================

def ask_v5(message, model='qwen3:4b', stream=False, **options):
    """
    终极版本：支持所有 ollama.chat 的参数

    参数：
        message (str): 用户的问题
        model (str): 模型名称，默认 qwen3:4b
        stream (bool): 是否流式输出，默认 False
        **options: 其他参数（如 temperature, num_predict 等）

    返回:
        str: 助手回复
    """
    try:
        # 构建基础消息
        messages = [{'role': 'user', 'content': message}]

        # 如果有额外参数，传入 options
        if options:
            response = ollama.chat(
                model=model,
                messages=messages,
                stream=stream,
                options=options  # 温度、最大长度等参数
            )
        else:
            response = ollama.chat(
                model=model,
                messages=messages,
                stream=stream
            )

        # 流式输出需要特殊处理
        if stream:
            result = ''
            for chunk in response:
                content = chunk['message']['content']
                print(content, end='', flush=True)
                result += content
            print()  # 换行
            return result
        else:
            return response['message']['content']

    except ollama.ResponseError as e:
        return f"错误：{e}"
    except Exception as e:
        return f"未知错误：{e}"


def demo_advanced_features():
    """
    演示：终极版本的功能
    """
    print("=" * 70)
    print("【第 6 步】终极：流式输出 + 参数配置")
    print("=" * 70)

    print("\n>>> 流式输出（像 ChatGPT 一样逐字显示）：")
    print("  ask_v5('写一首诗', stream=True)")
    print("  >>> ", end='')
    ask_v5('写一首关于春天的五言绝句', stream=True)

    print("\n>>> 配置模型参数：")
    print("  ask_v5('1+1=?', temperature=0.1)  # 低温，更确定")
    result = ask_v5('用一句话解释量子力学', temperature=0.3, num_predict=50)
    print(f"  >>> {result}")

    print("\n" + "-" * 70)
    print("💡 **kwargs 的妙用：")
    print("   - 不需要为每个可能的参数写死函数签名")
    print("   - 调用者可以传入任何 ollama.chat 支持的参数")
    print("   - 灵活性极大提高")
    print("-" * 70)
    print()


# =============================================================================
# 第 7 步：封装成类（管理对话状态）
# =============================================================================
# 🤔 引导：每次都要手动维护 messages 列表，太麻烦了
#      为什么不创建一个类，让它自动记住对话历史？
# =============================================================================

class ChatBot:
    """
    聊天机器人封装

    自动维护对话历史，支持重置对话、流式输出等
    """

    def __init__(self, model='qwen3:4b'):
        """
        初始化机器人

        参数：
            model (str): 使用的模型名称
        """
        self.model = model
        self.messages = []  # 对话历史

    def ask(self, message, stream=False, **options):
        """
        发送消息并获取回复

        参数：
            message (str): 用户消息
            stream (bool): 是否流式输出
            **options: 其他参数

        返回:
            str: 助手回复
        """
        # 添加用户消息到历史
        self.messages.append({'role': 'user', 'content': message})

        try:
            if options:
                response = ollama.chat(
                    model=self.model,
                    messages=self.messages,
                    stream=stream,
                    options=options
                )
            else:
                response = ollama.chat(
                    model=self.model,
                    messages=self.messages,
                    stream=stream
                )

            if stream:
                result = ''
                for chunk in response:
                    content = chunk['message']['content']
                    print(content, end='', flush=True)
                    result += content
                print()
                # 流式输出后也要添加历史
                self.messages.append({'role': 'assistant', 'content': result})
                return result
            else:
                reply = response['message']['content']
                # 添加助手回复到历史
                self.messages.append({'role': 'assistant', 'content': reply})
                return reply

        except Exception as e:
            # 出错时移除刚才添加的用户消息
            self.messages.pop()
            return f"错误：{e}"

    def reset(self):
        """清空对话历史"""
        self.messages = []
        print("对话已重置")

    def get_history(self):
        """获取对话历史"""
        return self.messages

    def print_history(self):
        """打印对话历史"""
        print("=" * 50)
        print("对话历史：")
        for msg in self.messages:
            role = "用户" if msg['role'] == 'user' else "助手"
            print(f"  {role}: {msg['content'][:60]}...")
        print("=" * 50)


def demo_chatbot_class():
    """
    演示：封装成类
    """
    print("=" * 70)
    print("【第 7 步】抽象：封装成 ChatBot 类")
    print("=" * 70)

    print("\n>>> 创建机器人：")
    bot = ChatBot(model='qwen3:4b')
    print(f"  bot = ChatBot(model='qwen3:4b')")

    print("\n>>> 第 1 轮：")
    reply = bot.ask('我叫小明，记住了')
    print(f"  用户：我叫小明，记住了")
    print(f"  助手：{reply[:50]}...")

    print("\n>>> 第 2 轮（不需要传历史，自动记住）：")
    reply = bot.ask('我叫什么名字？')
    print(f"  用户：我叫什么名字？")
    print(f"  助手：{reply[:50]}...")

    print("\n>>> 查看历史：")
    bot.print_history()

    print("\n>>> 重置对话：")
    bot.reset()
    bot.ask('你好')  # 新对话

    print("\n" + "-" * 70)
    print("💡 封装成类的好处：")
    print("   ✓ 状态（对话历史）自动维护")
    print("   ✓ 相关功能组织在一起（高内聚）")
    print("   ✓ 可以创建多个机器人实例")
    print("   ✓ 可以扩展更多功能（保存历史、加载历史等）")
    print("-" * 70)
    print()


# =============================================================================
# 主程序：完整的封装演进演示
# =============================================================================

if __name__ == '__main__':
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "函数封装的演进：从零散到复用" + " " * 15 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    print("本演示将引导学生思考：")
    print("  1. 为什么要封装函数？")
    print("  2. 如何从重复代码中发现模式？")
    print("  3. 如何设计易用的函数接口？")
    print("  4. 如何处理错误和边界情况？")
    print("  5. 何时使用类来封装状态？")
    print()
    print("按 Enter 键继续每一步，或取消注释运行特定步骤")
    print("-" * 70)
    print()

    # 取消注释以运行相应演示
    # demo_repeated_code()           # 第 1 步：观察重复代码
    # demo_simple_encapsulation()    # 第 2 步：最简单封装
    # demo_more_parameters()         # 第 3 步：添加参数
    # demo_multi_turn()              # 第 4 步：多轮对话
    # demo_error_handling()          # 第 5 步：错误处理
    # demo_advanced_features()       # 第 6 步：终极版本
    # demo_chatbot_class()           # 第 7 步：封装成类

    print("提示：取消注释相应的函数调用来运行演示。")
    print()
    print("建议教学顺序：按 1→7 逐步演示，每步都让学生先思考问题")
