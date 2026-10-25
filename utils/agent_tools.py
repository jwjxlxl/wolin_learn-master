"""Agent 共享工具

提供各 Agent 示例复用的工具函数，避免在多个教学文件中重复定义。
"""

from langchain_core.tools import tool


@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气。"""
    weather_db = {"北京": "晴，25°C", "上海": "多云，28°C", "广州": "小雨，30°C"}
    return weather_db.get(city, f"暂无 {city} 的天气数据")
