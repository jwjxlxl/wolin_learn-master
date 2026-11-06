import os
from minio import Minio

client = Minio(
    "192.168.142.128:9100",        # Endpoint 端点
    access_key="minioadmin", # 账号
    secret_key="minioadmin", # 密码
    secure=False,            # 本机 http 用 False；线上 https 用 True
)

# 验证连接：列出所有桶
buckets = client.list_buckets()
for b in buckets:
    print(f"📦 {b.name}（创建于 {b.creation_date}）")


def upload_file(bucket_name: str, object_name: str, file_path: str) -> None:
    """上传本地文件到 MinIO 的指定桶中。

    :param bucket_name: 桶名称
    :param object_name: 对象名（桶内的路径，如 "docs/report.pdf"）
    :param file_path: 本地文件路径
    """
    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)
        print(f"桶 {bucket_name} 不存在，已自动创建")
    client.fput_object(bucket_name, object_name, file_path)
    print(f"上传成功：{file_path} → {bucket_name}/{object_name}")


def download_file(bucket_name: str, object_name: str, file_path: str) -> None:
    """从 MinIO 下载对象到本地文件。

    :param bucket_name: 桶名称
    :param object_name: 对象名
    :param file_path: 本地保存路径
    """
    client.fget_object(bucket_name, object_name, file_path)
    print(f"下载成功：{bucket_name}/{object_name} → {file_path}")


def list_objects(bucket_name: str, prefix: str = "") -> list:
    """查看桶中指定目录下的所有对象（支持递归）。

    :param bucket_name: 桶名称
    :param prefix: 目录前缀，如 "docs/"，空字符串表示整个桶
    :return: 对象名列表
    """
    objects = list(client.list_objects(bucket_name, prefix=prefix, recursive=True))
    if not objects:
        print(f"桶 {bucket_name} 中前缀 '{prefix}' 下没有对象")
        return []
    print(f"桶 {bucket_name} 中 '{prefix}' 下的内容：")
    for obj in objects:
        size_kb = obj.size / 1024
        print(f"  📄 {obj.object_name}（{size_kb:.2f} KB，修改于 {obj.last_modified}）")
    return [obj.object_name for obj in objects]


# ===== 使用示例 =====
if __name__ == "__main__":
    bucket = "ai0824"

    # 1. 上传文件（把当前脚本自身作为示例文件上传）
    upload_file(bucket, "demo/minio_demo.py", os.path.abspath(__file__))

    # 2. 查看目录下的内容
    list_objects(bucket, prefix="demo/")

    # 3. 下载文件到本地
    download_file(bucket, "demo/minio_demo.py", "downloaded_minio_demo.py")
