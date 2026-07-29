import redis
import time
from utils.model_utils import get_model
# 连接到 Redis（Docker 容器中的 Redis 通过 localhost:6379 访问）
r = redis.Redis(host='192.168.142.128', port=6379, password="123456", decode_responses=True)

# 测试连接
print(r.ping())  # True → 连接成功


def get_llm_response(prompt: str) -> str:
    """获取 LLM 回答——有缓存就返回缓存，没有就调 API"""
    start_time = time.time()
    # 1. 先去缓存查
    cache_key = f"llm_cache:{prompt}"
    cached = r.get(cache_key)

    if cached:
        print(f"✅ 命中缓存：{prompt}")
        return cached

    # 2. 缓存没有，模拟调用 LLM API（实际使用时替换为真实的 API 调用）
    print(f"🔄 调用 API：{prompt}")
    model = get_model("qwen")
    response = model.invoke(prompt)

    # 3. 存到 Redis，24 小时（86400 秒）过期
    r.set(cache_key, response.content, ex=86400)

    return response.content

if __name__ == '__main__':

    print(r.get("status"))
