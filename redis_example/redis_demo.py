# pip install redis
import time

import redis
from utils.model_utils import get_model

'''
    在Python中，操作Redis，需要建立一个到Redis服务器的连接。
'''

# 连接到 Redis（Docker 容器中的 Redis 通过 localhost:6379 访问）
r = redis.Redis(host='192.168.142.128', port=6379, decode_responses=True, password='redis123456')

'''
    bytes
    序列化和反序列化
    在Python中，Redis的键和值都是字符串。
    如果要存储的数据不是字符串，则需要先进行序列化（如JSON），然后再存储。
    反序列化时，需要先从Redis中获取字符串，然后再反序列化（如JSON）。
    
    序列化和反序列化在各种语言中都非常常见，是数据交换的基本方式。
    网络传输的数据，也需要序列化和反序列化。
    磁盘存储的数据，也需要序列化和反序列化。
'''

def get_llm_response(prompt: str) -> str:
    """获取 LLM 回答——有缓存就返回缓存，没有就调 API"""

    # 1. 先去缓存查
    cache_key = f"llm_cache:{prompt}"
    cached = r.get(cache_key)

    if cached:
        print(f"✅ 命中缓存：{prompt}")
        return cached

    # 2. 缓存没有，模拟调用 LLM API（实际使用时替换为真实的 API 调用）
    print(f"🔄 调用 API：{prompt}")
    client = get_model('qwen')
    response = client.invoke(prompt)

    # 3. 存到 Redis，24 小时（86400 秒）过期
    r.set(cache_key, response.content, ex=86400)

    return response.content

# 测试连接
# print(r.ping())  # True → 连接成功

if __name__ == '__main__':
    # r.set('class_name', 'AI0522', 50)
    print(r.get('class_name'))
    r.delete('class_name')
    print(r.get('class_name'))

    # start = time.time()
    # res = get_llm_response('深度学习是什么')
    # print(res)
    # end = time.time()
    # print(f"耗时: {end - start:.2f}秒")
    # r.delete("llm_cache:深度学习是什么")
    # '''
    #     Redis 缓存的作用之外，还可以作为Agent短期记忆的存储
    #     过期时间
    #     长期记忆：MySQL
    # '''
