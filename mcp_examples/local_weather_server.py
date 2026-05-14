"""
本地 MCP 天气服务器 - 教学演示用

这是一个简单的 MCP 服务器，提供天气查询工具。
通过 stdio 方式与客户端通信。

启动方式：python local_weather_server.py
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("LocalWeather")


@mcp.tool()
def get_weather(city: str) -> str:
    """查询指定城市的天气信息。

    Args:
        city: 城市名称，如"北京"、"上海"、"深圳"
    """
    weather_db = {
        "北京": "晴，25°C，空气质量良好",
        "上海": "多云，28°C，有轻微雾霾",
        "广州": "小雨，30°C，注意携带雨具",
        "深圳": "晴，29°C，紫外线较强",
        "杭州": "多云，26°C，适宜出行",
        "成都": "阴，22°C，湿度较大",
    }
    return weather_db.get(city, f"暂无 {city} 的天气数据")


@mcp.tool()
def get_air_quality(city: str) -> str:
    """查询指定城市的空气质量。

    Args:
        city: 城市名称，如"北京"、"上海"
    """
    aqi_db = {
        "北京": "AQI 65 - 良好",
        "上海": "AQI 82 - 轻度污染",
        "广州": "AQI 45 - 优",
        "深圳": "AQI 50 - 优",
        "杭州": "AQI 60 - 良好",
        "成都": "AQI 90 - 轻度污染",
    }
    return aqi_db.get(city, f"暂无 {city} 的空气质量数据")


if __name__ == "__main__":
    mcp.run()
