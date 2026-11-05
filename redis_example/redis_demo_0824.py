# pip install redis 安装SDK
import redis

from utils import get_model
import time

# 实例化一个到Redis的客户端
r1 = redis.Redis(host="192.168.142.128", port=6379, password="123456")

# 从Redis中读数据
# print(r1.get("name"))

# 调用大模型回答用户的问题
def query(question: str):
    '''
       如果Redis有相同的问题，直接从Redis中读取答案
       如果Redis没有，调用大模型生成答案，并写入Redis
    '''

    prefix="que_"
    key = prefix + question
    if r1.get(key):
        return r1.get(key).decode("utf-8")

    # 实例化LLM的客户端
    model = get_model("qwen")

    response = model.invoke(question)


    # 如果Redis中没有相同的问题，模型回答后写入Redis作为缓存
    r1.set(key, response.content, ex=60*60*24)
    return response.content


if __name__ == '__main__':
    # 向Redis中存数据
    # r1.set("n1", "姚斌")
    # print(r1.get("n1").decode("utf-8"))

    # 存数据并且设定过期时间
    # r1.set("n2", 555, 50)
    # print(r1.get("n2"))

    # 更新Redis中的数据, 如果key存在就是更新数据
    # r1.set("n2", "laksdlfsjdklj")

    # 过期时间到了会自动删除
    # r1.delete("name")

    # 记录一下模型调用花费的时间
    start_time = time.time()
    print(query("介绍一下LLM，不超过100个字"))
    print(f"回答问题花费时间: {time.time() - start_time}秒")
