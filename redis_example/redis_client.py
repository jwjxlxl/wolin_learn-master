import json
import redis
from redis.exceptions import RedisError
from typing import Optional, Any, Callable


class RedisClient:
    """生产级 Redis 客户端封装工具类"""

    def __init__(self, host='localhost', port=6379, db=0, password=None,
                 max_connections=20, decode_responses=True):
        """
        初始化 Redis 连接池
        """
        self.pool = redis.ConnectionPool(
            host=host,
            port=port,
            db=db,
            password=password,
            max_connections=max_connections,
            decode_responses=decode_responses,
            socket_timeout=5,
            socket_connect_timeout=5,
            health_check_interval=30
        )
        self.client = redis.Redis(connection_pool=self.pool)

    def set_json(self, key: str, value: Any, ex: int = None) -> bool:
        """存储 JSON 对象"""
        try:
            self.client.set(key, json.dumps(value, ensure_ascii=False), ex=ex)
            return True
        except RedisError as e:
            print(f"[Redis Error] set_json failed for key '{key}': {e}")
            return False

    def get_json(self, key: str) -> Optional[Any]:
        """获取 JSON 对象，不存在或异常返回 None"""
        try:
            data = self.client.get(key)
            return json.loads(data) if data else None
        except (RedisError, json.JSONDecodeError) as e:
            print(f"[Redis Error] get_json failed for key '{key}': {e}")
            return None

    def get_or_set(self, key: str, func: Callable, ex: int = 60) -> Optional[Any]:
        """
        缓存穿透保护：存在直接返回，不存在则调用 func 获取并缓存
        :param key: 缓存键
        :param func: 获取数据的回调函数
        :param ex: 过期时间（秒）
        """
        cached = self.get_json(key)
        if cached is not None:
            return cached

        # 缓存不存在，查询数据源
        data = func()
        if data is not None:
            self.set_json(key, data, ex)
        return data

    def delete(self, key: str) -> bool:
        """删除指定键"""
        try:
            self.client.delete(key)
            return True
        except RedisError as e:
            print(f"[Redis Error] delete failed for key '{key}': {e}")
            return False

    def exists(self, key: str) -> bool:
        """判断 key 是否存在"""
        try:
            return bool(self.client.exists(key))
        except RedisError as e:
            print(f"[Redis Error] exists check failed for key '{key}': {e}")
            return False


# ================= 使用示例 =================
if __name__ == '__main__':
    # 1. 初始化客户端
    redis_cli = RedisClient(host='192.168.142.128',password='123456')

    # 2. 写入字典数据（自动转为JSON），过期时间300秒
    user_data = {"name": "Alice", "age": 25, "tags": ["python", "redis"]}
    redis_cli.set_json("user:1001", user_data, ex=300)

    # 3. 读取数据
    print(redis_cli.get_json("user:1001"))


    # 输出: {'name': 'Alice', 'age': 25, 'tags': ['python', 'redis']}

    # 4. 缓存穿透保护示例
    def fetch_from_db():
        print(">>> 缓存未命中，正在从数据库查询...")
        return {"id": 1002, "name": "Bob"}


    # 第一次调用会触发 fetch_from_db
    print(redis_cli.get_or_set("user:1002", fetch_from_db, ex=60))
    # 第二次调用直接返回缓存，不再打印查询日志
    print(redis_cli.get_or_set("user:1002", fetch_from_db, ex=60))

    '''
        Redis 中缓存穿透：
        当一个 key 在 Redis 中不存在，而数据库中存在时，就会发生缓存穿透。
        这种情况会导致大量请求直接访问数据库，导致数据库压力过大。
        解决方案：
        1. 使用布隆过滤器（Bloom Filter）来过滤不存在的 key，减少对数据库的访问。
        2. 使用空值（如 "N/A"）来缓存不存在的 key，避免缓存击穿。
        
        Redis 缓存雪崩：
        当缓存服务器宕机或大量缓存失效时，会导致大量请求直接访问数据库，导致数据库压力过大。
        解决方案：
        1. 使用 Redis 集群或主从架构，避免单点故障。
        2. 设置随机的过期时间，避免缓存集中失效。
        3. 使用 Redis 持久化机制，如 RDB 或 AOF，避免缓存数据丢失。
    '''