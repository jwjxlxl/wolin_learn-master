# =============================================================================
# Celery 异步任务队列 — 入门与实战
# =============================================================================
#
# 学完本文件你将能够：
#   ✅ 理解 Celery 的核心概念：Broker、Worker、Task、Result Backend
#   ✅ 搭建最小可用的 Celery 项目结构
#   ✅ 编写异步任务、并行任务、任务链、错误重试
#   ✅ 理解真实 Celery + Redis 项目的启动方式
#   ✅ 不安装 Redis 也能通过模拟模式理解核心流程
#
# 运行前检查：
#   方式 1（入门推荐）：不安装任何依赖，直接运行本文件，进入模拟演示模式
#   方式 2（真实项目）：安装 Redis + celery + redis + gevent 后，按示例 4 的项目结构运行 Worker
# =============================================================================

import sys
import os
import io
import time
import threading
from typing import Callable

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# =============================================================================
# 核心概念：什么是 Celery？
# =============================================================================
"""
Celery 是一个分布式异步任务队列系统。

核心概念：
  Task（任务）    — 你要异步执行的函数，用 @app.task 装饰器标记
  Broker（中间件） — 任务队列的消息中转站，常用 Redis 或 RabbitMQ
  Worker（工人）   — 从 Broker 取任务并执行的进程
  Result Backend  — 存储任务执行结果的数据库

工作流程：
  你调用 task.delay() → 消息发给 Broker → Worker 从 Broker 取消息 →
  执行任务 → 结果存入 Backend

生活化比喻：
  你去餐厅点外卖（调用任务）：
    Task     = 你要做的菜
    Broker   = 接单系统（美团/饿了么平台）
    Worker   = 后厨的厨师
    Backend  = 订单完成记录
  你不需要等菜做完（不阻塞），可以去做别的事，做完通知你

什么时候用：
  ✅ 耗时操作：发送邮件、生成报表、视频转码
  ✅ 并行处理：批量调用接口、批量处理文件
  ✅ 任务编排：先抓取数据 → 再分析 → 最后发送通知
  ✅ 失败重试：外部接口偶发失败时自动重试
"""


# =============================================================================
# 模式 1: 模拟运行（无需 Redis，开箱即懂）
# =============================================================================
# 这个模式用纯 Python 模拟了 Celery 的核心行为：
#   - @task 装饰器把函数注册为任务
#   - .delay() 异步调用（不阻塞主线程）
#   - .get() 获取结果
#   目的是让你先理解 API 用法，再去看真实 Celery 时不陌生。

class MockAsyncResult:
    """模拟 Celery AsyncResult，用于 .delay() 后的 .get() 获取结果。"""

    def __init__(self, task_id: str, func: Callable, args: tuple, kwargs: dict):
        self.task_id = task_id
        self._func = func
        self._args = args
        self._kwargs = kwargs
        self._result = None
        self._done = False
        self._started = threading.Event()

        # 在后台线程执行，模拟异步
        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()

    def _run(self):
        self._started.set()
        try:
            self._result = self._func(*self._args, **self._kwargs)
        except Exception as e:
            self._result = e
        finally:
            self._done = True

    def get(self, timeout=None):
        """阻塞等待结果，模拟 result.get()。"""
        if not self._started.wait(timeout=timeout):
            raise TimeoutError("任务超时")
        if timeout:
            deadline = time.time() + timeout
            while not self._done:
                if time.time() > deadline:
                    raise TimeoutError("任务超时")
                time.sleep(0.05)
        else:
            while not self._done:
                time.sleep(0.05)
        if isinstance(self._result, Exception):
            raise self._result
        return self._result

    def ready(self):
        """检查任务是否完成。"""
        return self._done


class MockCeleryApp:
    """
    模拟 Celery 应用，实现最核心的 API：
      - @app.task 装饰器
      - task.delay() 异步调用
      - task.apply_async() 带参数调用
    """

    def __init__(self, name: str):
        self.name = name
        self.tasks: dict[str, Callable] = {}
        self._task_counter = 0

    def task(self, maybe_func=None, name: str = None, bind: bool = False):
        """
        装饰器：将函数注册为任务。

        支持两种语法：
            @app.task          # 无括号
            @app.task()        # 有括号
            @app.task(name="my_task")  # 带参数

        等价于：
            def my_func(x):
                return x * 2
            my_task = app.task()(my_func)
        """
        def decorator(func):
            task_name = name or f"{self.name}.{func.__name__}"
            wrapped = MockTask(func, task_name, self)
            self.tasks[task_name] = wrapped
            return wrapped

        if maybe_func is not None and callable(maybe_func):
            # @app.task 形式（无括号）
            return decorator(maybe_func)
        else:
            # @app.task() 或 @app.task(name="...") 形式
            return decorator


class MockTask:
    """包装后的任务对象，提供 .delay() 和 .apply_async() 方法。"""

    def __init__(self, func: Callable, name: str, app: MockCeleryApp):
        self.func = func
        self.name = name
        self.__name__ = func.__name__
        self._app = app
        self.__doc__ = func.__doc__

    def delay(self, *args, **kwargs):
        """异步调用任务，返回 AsyncResult 对象。"""
        self._app._task_counter += 1
        task_id = f"task-{self._app._task_counter}"
        print(f"    📤 任务 [{self.name}] 已发送到队列 (ID: {task_id})")
        return MockAsyncResult(task_id, self.func, args, kwargs)

    def apply_async(self, args=None, kwargs=None, countdown=None, eta=None, **options):
        """带高级参数的异步调用。"""
        self._app._task_counter += 1
        task_id = f"task-{self._app._task_counter}"

        if countdown:
            print(f"    ⏰ 任务 [{self.name}] 将在 {countdown} 秒后执行")
            time.sleep(countdown)  # 模拟延迟
        elif eta:
            wait_time = max(0, (eta - time.time()))
            if wait_time > 0:
                print(f"    ⏰ 任务 [{self.name}] 将在指定时间执行（等待 {wait_time:.1f}s）")
                time.sleep(wait_time)

        print(f"    📤 任务 [{self.name}] 已发送 (ID: {task_id})")
        return MockAsyncResult(task_id, self.func, args or (), kwargs or {})

    def apply(self, args=None, kwargs=None):
        """同步调用（立即执行，用于对比）。"""
        result = self.func(*(args or ()), **(kwargs or {}))
        print(f"    ✅ 任务 [{self.name}] 同步执行完成")
        return result

    def __call__(self, *args, **kwargs):
        """直接调用 = 同步执行。"""
        return self.func(*args, **kwargs)


# 创建模拟 Celery 应用
mock_app = MockCeleryApp("celery_demo")


# ---- 定义示例任务 ----

@mock_app.task
def add(x, y):
    """两数相加（模拟耗时操作）。"""
    time.sleep(0.5)  # 模拟耗时
    return x + y


@mock_app.task
def generate_report(name: str, items: int):
    """生成报表（模拟耗时操作）。"""
    print(f"    📊 正在生成「{name}」报表，共 {items} 条数据...")
    time.sleep(1)
    return {"report_name": name, "item_count": items, "status": "completed"}


@mock_app.task
def send_email(to: str, subject: str, body: str):
    """发送邮件（模拟耗时操作）。"""
    print(f"    📧 正在发送邮件到 {to}...")
    print(f"       主题: {subject}")
    time.sleep(0.5)
    return {"to": to, "subject": subject, "sent": True}

# =============================================================================
# 示例 1: 最基础的异步任务 — 异步 vs 同步对比
# =============================================================================

def demo_basic_async():
    """
    对比同步和异步执行的区别。

    同步：程序被阻塞，必须等任务完成才能继续
    异步：程序立即返回，可以做其他事，需要结果时再 .get()
    """
    print(f"\n-- 示例 1: 最基础的异步任务 — 同步 vs 异步")

    # --- 同步方式（会阻塞）---
    print("\n  【同步调用 add.apply()】")
    start = time.time()
    result = add.apply(args=(3, 5))
    print(f"  结果: {result}")
    print(f"  耗时: {time.time() - start:.2f}s（程序被阻塞了）")

    # --- 异步方式（不阻塞）---
    print("\n  【异步调用 add.delay()】")
    start = time.time()
    async_result = add.delay(3, 5)
    print(f"  ⚡ delay() 立即返回，程序可以继续做别的事...")
    print(f"  ⚡ 正在做其他工作（模拟 0.2s）...")
    time.sleep(0.2)  # 模拟主线程继续工作

    # 需要结果时才等待
    result = async_result.get()
    print(f"  结果: {result}")
    print(f"  总耗时: {time.time() - start:.2f}s（异步不阻塞主线程）")


# =============================================================================
# 示例 2: 多个任务并行执行
# =============================================================================

def demo_parallel_tasks():
    """
    同时发送多个任务到队列，并行执行。

    真实场景：批量发送邮件通知、批量处理图片、批量生成报表等。
    """
    print(f"\n-- 示例 2: 多个任务并行执行")

    print("\n  【同时发送 3 个邮件任务】")
    start = time.time()

    # 同时发出 3 个任务
    results = [
        send_email.delay("alice@example.com", "通知1", "你好 Alice！"),
        send_email.delay("bob@example.com", "通知2", "你好 Bob！"),
        send_email.delay("charlie@example.com", "通知3", "你好 Charlie！"),
    ]

    print(f"  ✅ 3 个任务已全部入队，主线程未被阻塞")
    print(f"  💡 如果同步执行需要 {0.5 * 3:.1f}s，异步只需等最慢的那个...")

    # 等待所有结果
    for r in results:
        output = r.get()
        print(f"  ✅ 完成: {output}")

    print(f"  总耗时: {time.time() - start:.2f}s")


# =============================================================================
# 示例 3: 任务链（Chain）— 上一步的输出是下一步的输入
# =============================================================================

def demo_task_chain():
    """
    任务链：多个任务按顺序执行，前一个的结果作为下一个的输入。

    真实场景：
      抓取网页数据 → 提取关键信息 → 生成摘要 → 发送邮件

    在真实 Celery 中用 celery.chain() 实现：
        chain(step1.s() | step2.s() | step3.s())().get()

    这里用模拟方式展示同样的概念。
    """
    print(f"\n-- 示例 3: 任务链（Chain）")

    # 定义链式任务
    @mock_app.task
    def fetch_data(url):
        """模拟从 URL 获取数据。"""
        time.sleep(0.3)
        return {"url": url, "content": "这是一段从网页中提取的重要数据", "size": 1024}

    @mock_app.task
    def extract_info(data):
        """从数据中提取关键信息。"""
        time.sleep(0.3)
        return {"keywords": ["重要", "数据", "提取"], "summary": data.get("content", "")[:20]}

    @mock_app.task
    def send_summary(info, to_email):
        """将摘要发送给指定邮箱。"""
        time.sleep(0.3)
        return {"sent_to": to_email, "summary": info.get("summary", ""), "status": "done"}

    print("\n  【任务链：抓取数据 → 提取信息 → 发送摘要】")
    start = time.time()

    # 模拟链式执行
    step1_result = fetch_data.apply(args=("https://example.com/article",))
    print(f"    步骤 1 结果: {step1_result}")

    step2_result = extract_info.apply(args=(step1_result,))
    print(f"    步骤 2 结果: {step2_result}")

    step3_result = send_summary.apply(args=(step2_result, "admin@example.com"))
    print(f"    步骤 3 结果: {step3_result}")

    print(f"  总耗时: {time.time() - start:.2f}s")
    print(f"  ✅ 三个步骤顺序执行，每一步的输出是下一步的输入")


# =============================================================================
# 示例 4: 真实 Celery 项目的文件结构
# =============================================================================

def demo_real_project_structure():
    """
    展示真实 Celery 项目的文件结构和关键代码。

    这个示例不执行代码，而是展示你需要创建的文件。
    """
    print(f"\n-- 示例 4: 真实 Celery 项目的文件结构")

    print("""
  一个典型 Celery + Redis 项目的文件结构：

  my_project/
  ├── celery_app.py          # Celery 应用配置
  ├── tasks.py               # 任务定义
  ├── worker.py              # Worker 启动入口
  └── client.py              # 客户端（发送任务的脚本）

  ──────────────────────────────────────────
  【celery_app.py】— 创建 Celery 实例

      from celery import Celery

      app = Celery(
          'my_project',
          broker='redis://localhost:6379/0',      # Redis 作为消息队列
          backend='redis://localhost:6379/1',     # Redis 存储结果
      )

      # 可选配置
      app.conf.update(
          task_serializer='json',
          result_serializer='json',
          accept_content=['json'],
          timezone='Asia/Shanghai',
          enable_utc=True,

          # 定时任务配置
          beat_schedule={
              'daily-report': {
                  'task': 'tasks.generate_daily_report',
                  'schedule': timedelta(hours=24),  # 每 24 小时
                  'args': ('日报',),
              },
          },
      )

  ──────────────────────────────────────────
  【tasks.py】— 定义任务

      from celery_app import app
      import time

      @app.task(bind=True, max_retries=3)
      def add(self, x, y):
          \"\"\"两数相加。\"\"\"
          time.sleep(0.5)
          return x + y

      @app.task(bind=True, max_retries=3)
      def send_email(self, to, subject, body):
          \"\"\"发送邮件，失败自动重试。\"\"\"
          try:
              # 发送邮件逻辑...
              return {"sent": True}
          except Exception as exc:
              # 失败后 60 秒重试
              raise self.retry(exc=exc, countdown=60)

      @app.task
      def generate_daily_report(report_name):
          \"\"\"定时任务：生成日报。\"\"\"
          return {"report": report_name, "status": "done"}

  ──────────────────────────────────────────
  【启动 Worker】— 在终端运行

      # Windows（gevent 模式，因为 Windows 不支持 prefork）
      celery -A celery_app worker --pool=gevent --loglevel=info

      # Linux / Mac
      celery -A celery_app worker --loglevel=info

      # 启动定时任务调度器（Beat）
      celery -A celery_app beat --loglevel=info

  ──────────────────────────────────────────
  【client.py】— 客户端发送任务

      from tasks import add, send_email, generate_daily_report

      # 异步调用（不阻塞）
      result = add.delay(3, 5)
      print(f"任务已发送，ID: {result.id}")

      # 等待结果（会阻塞直到完成）
      print(f"结果: {result.get(timeout=10)}")

      # 检查状态
      if result.ready():
          print("任务已完成！")
      else:
          print("任务还在执行中...")

      # 延迟调用
      send_email.apply_async(
          args=["user@example.com", "Hello", "这是一封测试邮件"],
          countdown=300,  # 5 分钟后执行
      )

      # 任务链
      from celery import chain
      result = chain(
          fetch_data.s("https://example.com"),
          extract_info.s(),
          send_summary.s("admin@example.com")
      )()
  """)


# =============================================================================
# 示例 5: 错误处理与重试
# =============================================================================

def demo_error_handling():
    """
    Celery 任务中的错误处理策略。

    真实 Celery 中：
      - @app.task(bind=True, max_retries=3) 自动重试
      - self.retry(exc=exc, countdown=60) 延迟重试
      - acks_late=True 失败后重新入队
    """
    print(f"\n-- 示例 5: 错误处理与重试")

    @mock_app.task
    def flaky_task(name: str):
        """模拟一个可能失败的任务。"""
        import random
        print(f"    🔄 正在执行 {name}...")
        time.sleep(0.3)
        # 模拟偶发失败
        if random.random() < 0.3:
            raise ConnectionError(f"连接失败: {name}")
        return f"{name} 执行成功"

    @mock_app.task
    def robust_task(name: str, max_retries: int = 3):
        """带重试的稳健任务。"""
        for attempt in range(1, max_retries + 1):
            try:
                result = flaky_task.apply(args=(name,))
                return result
            except Exception as e:
                print(f"    ⚠️  第 {attempt} 次尝试失败: {e}")
                if attempt == max_retries:
                    raise
                time.sleep(0.2)  # 等待后重试
        return None

    print("\n  【模拟失败 → 自动重试 → 最终成功】")
    for i in range(3):
        try:
            result = robust_task(f"任务-{i+1}")
            print(f"  ✅ {result}")
        except Exception as e:
            print(f"  ❌ 最终失败: {e}")


# =============================================================================
# 主入口
# =============================================================================
if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  Celery 异步任务队列 — 入门与实战")
    print("  模拟模式运行（无需 Redis），聚焦核心概念和真实项目结构")
    print("=" * 70 + "\n")

    print("💡 提示：当前运行在模拟模式，使用纯 Python 模拟 Celery 行为。")
    print("   要运行真实 Celery，请安装 Redis 并参考示例 4 的项目结构:")
    print("   pip install celery redis gevent")
    print("   celery -A celery_app worker --pool=gevent --loglevel=info")

    # 建议按顺序学习：1 → 2 → 3 → 4 → 5
    # demo_basic_async()
    # demo_parallel_tasks()
    # demo_task_chain()
    demo_real_project_structure()
    # demo_error_handling()

    print("\n" + "=" * 70)
    print("  学习路线总结")
    print("=" * 70)
    print("""
  1. 核心概念：Task / Broker / Worker / Backend
  2. 基础用法：@app.task 装饰 + .delay() 异步调用
  3. 并行执行：多个 .delay() 同时发出，.get() 等待结果
  4. 任务链：chain() 顺序执行，上一步输出 = 下一步输入
  5. 项目落地：Celery + Redis 的文件结构和启动命令
  6. 错误处理：max_retries / self.retry()
  """)
    print("=" * 70 + "\n")
