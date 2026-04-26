# =============================================================================
# 第一个 LangChain 程序
# =============================================================================
#  
# 用途：教学演示 - 5 分钟内让学员体验成功感
#
# 核心概念：
#   - 最简单的 Chain 示例
#   - 一行代码调用 LLM
#   - 看到输出结果的成就感
# =============================================================================

# -----------------------------------------------------------------------------
# 运行前检查
# -----------------------------------------------------------------------------
# 1. 已安装 Ollama 服务
# 2. 已下载模型：ollama pull qwen3:4b
# 3. Ollama 服务正在运行
#
# 如果只想用云端 API，参考代码中的注释部分
# -----------------------------------------------------------------------------

# 设置 UTF-8 编码（Windows 专用）
import sys
import io

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer,
    encoding='utf-8',
    errors='replace',
    line_buffering=True
)

# -----------------------------------------------------------------------------
# 安装依赖
# -----------------------------------------------------------------------------
# pip install langchain-ollama python-dotenv
# -----------------------------------------------------------------------------

from langchain_ollama import ChatOllama


# =============================================================================
# 示例 1: 最简单的调用（3 行代码）
# =============================================================================

def simplest_call():
    """
    最简单的 LangChain 调用

    只需 3 行代码，就能让 AI 回答你的问题！
    """
    print("=" * 60)
    print("示例 1: 最简单的调用（3 行代码）")
    print("=" * 60)

    # 第 1 行：创建模型实例
    model = ChatOllama(model="qwen3:4b")

    # 第 2 行：调用模型
    response = model.invoke("你好，请用一句话介绍你自己。")

    # 第 3 行：打印结果
    print(response.content)
    print()


# =============================================================================
# 示例 2: 使用云端 API（阿里云 Qwen）
# =============================================================================

def cloud_api_call():
    """
    使用云端 API 调用

    如果你有阿里云 API Key，可以用更强大的模型
    """
    print("=" * 60)
    print("示例 2: 使用云端 API（阿里云 Qwen）")
    print("=" * 60)

    try:
        # 导入云端模型
        from langchain_openai import ChatOpenAI
        import os
        from dotenv import load_dotenv

        # 加载环境变量
        load_dotenv()

        # 检查是否有 API Key
        api_key = os.getenv("ALIYUN_API_KEY")
        if not api_key or api_key == "sk-your-aliyun-api-key-here":
            print("未配置 API Key，跳过此示例")
            print("提示：在 .env 文件中配置 ALIYUN_API_KEY")
            print()
            return

        # 创建云端模型实例
        cloud_model = ChatOpenAI(
            model="qwen-plus",
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )



        # 调用模型
        response = cloud_model.invoke("你好，请用一句话介绍你自己。")

        from langchain.agents import create_agent
        from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse, before_agent
        from langchain.agents.middleware import AgentState, ModelRequest, ModelResponse, dynamic_prompt
        from langgraph.runtime import Runtime

        def get_weather(city: str) -> str:
            """获取指定城市的天气。 """
            return f"{city}总是阳光明媚！"


        @before_agent
        def dynamic_model_selection(state: AgentState, runtime: Runtime) -> ModelResponse:
            """根据对话复杂性选择模型。"""
            print("hello !!!")
            # print(request.state.get('messages'))
            # message_count = len(request.state["messages"])
            #
            # if message_count > 2:
            #     # 对较长的对话使用高级模型
            #     print("对话轮数 超过两轮， 请分发到 大参数的LLM")
            # else:
            #     print("对话轮数 超过两轮， 请分发到 小参数的LLM")
            #
            # request.model = cloud_model
            return None

        agent = create_agent(
            model=cloud_model,
            tools=[get_weather],
            middleware=[dynamic_model_selection],
            system_prompt=SystemMessage("你是一个有用的助手"),
            checkpointer=InMemorySaver(),
        )

        # 运行代理
        response = agent.invoke(
            {"messages": [HumanMessage("旧金山的天气怎么样")]},
            {"configurable": {"thread_id": "1"}},
        )

        print(response.get('messages')[-1].content)
        response = agent.invoke(
            {"messages": [HumanMessage("我的上一个问题是什么")]},
            {"configurable": {"thread_id": "1"}},
        )

        print(response.get('messages')[-1].content)

        print()

    except ImportError:
        print("未安装 langchain-openai，跳过此示例")
        print("提示：pip install langchain-openai")
        print()


# =============================================================================
# 示例 3: 流式输出（打字机效果）
# =============================================================================

def streaming_call():
    """
    流式输出

    像打字机一样逐字显示，提升用户体验
    """
    print("=" * 60)
    print("示例 3: 流式输出（打字机效果）")
    print("=" * 60)

    # 创建模型实例
    model = ChatOllama(model="qwen3:4b")

    # 流式调用
    stream = model.stream("请用 50 字左右介绍人工智能。")

    # 逐字打印
    for chunk in stream:
        print(chunk.content, end='', flush=True)

    print("\n")


# =============================================================================
# 示例 4: 带简单错误处理
# =============================================================================

def call_with_error_handling():
    """
    带错误处理的调用

    实际应用中应该处理可能的异常
    """
    print("=" * 60)
    print("示例 4: 带错误处理的调用")
    print("=" * 60)

    model = ChatOllama(model="qwen3:4b")

    try:
        response = model.invoke("测试消息")
        print(response.content)

    except Exception as e:
        print(f"调用失败：{e}")
        print()
        print("可能的原因：")
        print("  1. Ollama 服务未启动（运行 'ollama serve'）")
        print("  2. 模型未下载（运行 'ollama pull qwen3:4b'）")
        print("  3. 网络连接问题")

    print()


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  第一个 LangChain 程序")
    print("  说明：5 分钟内体验成功感")
    print("=" * 70 + "\n")

    print("【运行前检查】")
    print("  1. 已安装 Ollama 服务")
    print("  2. 已下载模型：ollama pull qwen3:4b")
    print("  3. Ollama 服务正在运行")
    print()

    # 运行示例
    # simplest_call()
    cloud_api_call()      # 有 API Key 可取消注释
    # streaming_call()
    # call_with_error_handling()  # 需要测试错误处理可取消注释

    print("=" * 70)
    print("  恭喜！你已经完成了第一个 LangChain 程序！")
    print("  接下来学习：02_llm_call/llm_basic.py")
    print("=" * 70 + "\n")
