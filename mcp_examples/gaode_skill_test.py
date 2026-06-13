"""
高德地图综合服务 Skill 测试案例

参照高德官方 Skill 文档：https://lbs.amap.com/api/skill/ready-to-use/summary
实现 6 大场景：关键词搜索、周边搜索、路径规划、天气查询、POI 详情搜索、旅游规划

运行前准备：
1. 高德开放平台注册账号：https://lbs.amap.com/
2. 创建应用 → 添加 Key，服务平台选择 "Web服务"
3. 将 Key 设置到 .env 文件的 AMAP_KEY 变量中

依赖：pip install requests python-dotenv
"""

import sys
import os
import io
import json
import urllib.parse
from datetime import datetime

# UTF-8 编码
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding and sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.model_utils import get_qwen_client

from dotenv import load_dotenv
import requests
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

# ── 加载 API Key ────────────────────────────────────────────────
load_dotenv()
AMAP_KEY = os.getenv("AMAP_KEY")
AMAP_BASE = "https://restapi.amap.com/v3"

# ── 高德 API 封装 ───────────────────────────────────────────────

def _amap_get(path: str, params: dict) -> dict:
    """调用高德 Web Service API。"""
    params["key"] = AMAP_KEY
    params["output"] = "JSON"
    resp = requests.get(f"{AMAP_BASE}/{path}", params=params, timeout=10)
    return resp.json()


# =============================================================================
# Tool 1: 地理编码（地址 → 经纬度）
# =============================================================================
@tool
def amap_geo(address: str) -> str:
    """地理编码：将地址转换为经纬度坐标。"""
    data = _amap_get("geocode/geo", {"address": address})
    if data.get("status") != "1" or not data.get("geocodes"):
        return f"未找到地址「{address}」的坐标信息"
    loc = data["geocodes"][0]["location"]  # 经度,纬度
    formatted = data["geocodes"][0]["formatted_address"]
    return f"地址: {formatted}\n坐标: {loc} (经度,纬度)"


# =============================================================================
# Tool 2: 逆地理编码（经纬度 → 地址）
# =============================================================================
@tool
def amap_regeo(location: str) -> str:
    """逆地理编码：将经纬度坐标转换为地址信息。参数 location 格式：经度,纬度。"""
    data = _amap_get("geocode/regeo", {"location": location})
    if data.get("status") != "1" or not data.get("regeocode"):
        return f"未找到坐标 {location} 的地址信息"
    addr = data["regeocode"]["formatted_address"]
    return f"坐标 {location} 对应地址: {addr}"


# =============================================================================
# Tool 3: POI 关键词搜索
# =============================================================================
@tool
def amap_poi_search(keywords: str, city: str = "", types: str = "") -> str:
    """POI 关键词搜索：搜索地点、商家、景点等。

    Args:
        keywords: 搜索关键词（如"美食"、"酒店"、"天安门"）
        city: 城市名称或编码（如"北京"、"010"），可选
        types: POI 类型编码，可选
    """
    params = {"keywords": keywords, "city": city, "types": types, "offset": 10}
    data = _amap_get("place/text", params)
    if data.get("status") != "1" or not data.get("pois"):
        return f"未找到「{keywords}」相关地点"

    pois = data["pois"][:5]
    lines = [f"搜索「{keywords}」共找到 {data['count']} 条结果，前 {len(pois)} 条："]
    for i, p in enumerate(pois, 1):
        name = p.get("name", "")
        addr = p.get("address", "无地址")
        loc = p.get("location", "")
        lines.append(f"  {i}. {name} | {addr} | 坐标: {loc}")
    return "\n".join(lines)


# =============================================================================
# Tool 4: POI 周边搜索
# =============================================================================
@tool
def amap_around_search(keywords: str, location: str, radius: int = 1000) -> str:
    """周边搜索：在指定坐标点周边搜索 POI。

    Args:
        keywords: 搜索关键词（如"美食"、"酒店"）
        location: 中心点坐标，格式：经度,纬度
        radius: 搜索半径（米），默认 1000 米
    """
    params = {"keywords": keywords, "location": location, "radius": str(radius), "offset": 10}
    data = _amap_get("place/around", params)
    if data.get("status") != "1" or not data.get("pois"):
        return f"在坐标 {location} 周边 {radius} 米内未找到「{keywords}」"

    pois = data["pois"][:5]
    lines = [f"在坐标 {location} 周边 {radius} 米内搜索「{keywords}」，找到 {data['count']} 条："]
    for i, p in enumerate(pois, 1):
        name = p.get("name", "")
        addr = p.get("address", "无地址")
        dist = p.get("distance", "?")
        lines.append(f"  {i}. {name} | {addr} | 距离: {dist}米")
    return "\n".join(lines)


# =============================================================================
# Tool 5: 天气查询
# =============================================================================
@tool
def amap_weather(city: str) -> str:
    """查询指定城市的实时天气信息。

    Args:
        city: 城市名称（如"北京"、"深圳"）或城市 adcode
    """
    data = _amap_get("weather/weatherInfo", {"city": city, "extensions": "all"})
    if data.get("status") != "1" or not data.get("forecasts"):
        return f"未找到「{city}」的天气信息"

    fc = data["forecasts"][0]
    city_name = fc.get("city", city)
    lines = [f"【{city_name}】天气预报："]
    for day in fc.get("casts", [])[:3]:
        lines.append(
            f"  {day['date']} ({day['week']}) | {day['dayweather']} | "
            f"{day['nighttemp']}°C ~ {day['daytemp']}°C | {day['daywind']}风 {day['daypower']}级"
        )
    return "\n".join(lines)


# =============================================================================
# Tool 6: 路径规划（步行/驾车/骑行/公交）
# =============================================================================
@tool
def amap_route(origin: str, destination: str, route_type: str = "walking", city: str = "") -> str:
    """路径规划：规划不同出行方式的路线。

    Args:
        origin: 起点坐标，格式：经度,纬度
        destination: 终点坐标，格式：经度,纬度
        route_type: 路线类型，可选值：walking(步行)、driving(驾车)、riding(骑行)、transit(公交)
        city: 城市名称（公交路线必填）
    """
    routes = {
        "walking": ("direction/walking", {"origin": origin, "destination": destination}),
        "driving": ("direction/driving", {"origin": origin, "destination": destination}),
        "riding": ("direction/bicycling", {"origin": origin, "destination": destination}),
        "transit": ("direction/transit/integrated", {"origin": origin, "destination": destination, "city": city}),
    }
    if route_type not in routes:
        return f"不支持的路线类型: {route_type}，可选: walking, driving, riding, transit"

    path, params = routes[route_type]
    data = _amap_get(path, params)
    type_names = {"walking": "步行", "driving": "驾车", "riding": "骑行", "transit": "公交"}

    if data.get("status") != "1":
        return f"路径规划失败: {data.get('info', '未知错误')}"

    if route_type == "transit":
        result = data.get("result", {})
        routes_data = result.get("transits", [])
    else:
        result = data.get("result", {})
        routes_data = result.get("paths", [])

    if not routes_data:
        return "未找到可用路线"

    r = routes_data[0]
    distance = int(r.get("distance", 0))
    duration = int(r.get("duration", 0))
    lines = [
        f"【{type_names.get(route_type, route_type)}路线】",
        f"  距离: {distance / 1000:.1f} 公里 | 预计耗时: {duration // 60} 分钟",
    ]

    if route_type == "transit" and r.get("segments"):
        lines.append("  换乘方案：")
        for seg in r["segments"][:3]:
            bus = seg.get("bus", {})
            for line_info in bus.get("buslines", []):
                lines.append(f"    - {line_info.get('name', '')} ({line_info.get('departure_stop', '')} → {line_info.get('arrival_stop', '')})")
            if seg.get("walking"):
                walk = seg["walking"]
                lines.append(f"    - 步行 {int(walk.get('distance', 0))} 米")
    elif r.get("steps"):
        lines.append("  路线概要：")
        for step in r["steps"][:5]:
            instruction = step.get("instruction", "")
            dist = int(step.get("distance", 0))
            if instruction:
                lines.append(f"    - {instruction}（{dist}米）")

    return "\n".join(lines)


# =============================================================================
# Tool 7: 高德地图搜索链接生成
# =============================================================================
@tool
def amap_search_url(query: str, location: str = "") -> str:
    """生成高德地图搜索链接，可直接在浏览器中打开查看地图。

    Args:
        query: 搜索关键词（如"美食"、"天安门"）
        location: 可选，中心点坐标（经度,纬度），带坐标可定位更精确
    """
    if location:
        lon, lat = location.split(",")
        url = (
            f"https://ditu.amap.com/search?query={query}"
            f"&query_type=RQBXY&longitude={lon}&latitude={lat}&range=1000"
        )
    else:
        url = f"https://www.amap.com/search?query={urllib.parse.quote(query)}"
    return f"🔍 高德地图搜索链接：\n{url}\n点击链接即可查看详情。"


# =============================================================================
# Skill: 智能旅游规划（组合 Tool）
# =============================================================================
@tool
def amap_travel_planner(city: str, interests: str = "景点,美食", route_type: str = "walking") -> str:
    """智能旅游规划：搜索城市兴趣点并规划游览路线。

    Args:
        city: 城市名称（如"北京"、"杭州"）
        interests: 兴趣类型，逗号分隔（如"景点,美食,酒店"）
        route_type: 路线类型：walking(步行)、driving(驾车)、riding(骑行)
    """
    # 先用地理编码获取城市中心坐标
    geo_data = _amap_get("geocode/geo", {"address": city})
    if geo_data.get("status") != "1" or not geo_data.get("geocodes"):
        return f"未找到城市「{city}」的位置信息"

    location = geo_data["geocodes"][0]["location"]

    # 搜索各类兴趣点
    all_pois = []
    types = [t.strip() for t in interests.split(",")]
    for t in types:
        poi_data = _amap_get("place/text", {"keywords": t, "city": city, "offset": 3})
        if poi_data.get("status") == "1" and poi_data.get("pois"):
            for p in poi_data["pois"][:2]:
                all_pois.append({"name": p["name"], "location": p["location"]})

    if not all_pois:
        return f"在「{city}」未找到相关兴趣点"

    # 生成搜索链接
    search_url = f"https://www.amap.com/search?query={urllib.parse.quote(city + '旅游')}"

    lines = [f"🗺️ 【{city}】旅游规划（兴趣：{interests}）", f"  推荐景点："]
    for i, p in enumerate(all_pois, 1):
        lines.append(f"    {i}. {p['name']}（坐标: {p['location']}）")

    lines.append(f"\n  🔍 查看完整地图: {search_url}")

    # 如果有多个景点，规划路线
    if len(all_pois) >= 2:
        lines.append(f"\n  景点间路线（{route_type}）：")
        for i in range(min(len(all_pois) - 1, 3)):
            origin = all_pois[i]["location"]
            dest = all_pois[i + 1]["location"]
            route_data = _amap_get(
                "direction/walking",
                {"origin": origin, "destination": dest},
            )
            if route_data.get("status") == "1" and route_data.get("result", {}).get("paths"):
                dist = int(route_data["result"]["paths"][0].get("distance", 0))
                dur = int(route_data["result"]["paths"][0].get("duration", 0))
                lines.append(f"    {all_pois[i]['name']} → {all_pois[i+1]['name']}: {dist/1000:.1f}km，约{dur//60}分钟")

    return "\n".join(lines)


# =============================================================================
# 创建高德地图 Skill Agent
# =============================================================================

def create_amap_skill_agent():
    """创建高德地图综合服务 Agent。"""
    if not AMAP_KEY:
        print("【跳过】未配置 AMAP_KEY")
        print("  请在 .env 文件中添加高德 Web Service Key：")
        print("  AMAP_KEY=你的高德Key")
        print()
        print("  获取 Key：https://lbs.amap.com/ → 创建应用 → 添加 Key → 选 Web服务")
        return None

    model = get_qwen_client()
    if model is None:
        print("【跳过】未配置阿里云 API Key，无法运行此示例")
        return None

    all_tools = [
        amap_geo,
        amap_regeo,
        amap_poi_search,
        amap_around_search,
        amap_weather,
        amap_route,
        amap_search_url,
        amap_travel_planner,
    ]

    agent = create_agent(
        model,
        tools=all_tools,
        system_prompt=(
            "你是高德地图智能助手，拥有丰富的地图服务能力。"
            "根据用户需求，选择合适的工具来回答：\n"
            "1. 【地理编码】amap_geo - 将地址转为经纬度\n"
            "2. 【逆地理编码】amap_regeo - 将经纬度转为地址\n"
            "3. 【POI 搜索】amap_poi_search - 关键词搜索地点\n"
            "4. 【周边搜索】amap_around_search - 在指定坐标周边搜索\n"
            "5. 【天气查询】amap_weather - 查询城市天气\n"
            "6. 【路径规划】amap_route - 步行/驾车/骑行/公交路线\n"
            "7. 【地图链接】amap_search_url - 生成高德地图搜索链接\n"
            "8. 【旅游规划】amap_travel_planner - 智能旅游规划\n"
            "\n"
            "注意：\n"
            "- 用户问地址/经纬度时用 amap_geo 或 amap_regeo\n"
            "- 用户搜地点时用 amap_poi_search\n"
            "- 用户搜周边时用 amap_around_search（需要经纬度，可先用 amap_geo 获取）\n"
            "- 用户问天气时用 amap_weather\n"
            "- 用户问路线时用 amap_route（需要经纬度，可先用 amap_geo 获取）\n"
            "- 用户要旅游规划时用 amap_travel_planner\n"
            "- 需要地图链接时用 amap_search_url\n"
            "回答时要简洁、实用、有条理。"
        ),
    )
    return agent


# =============================================================================
# 运行测试
# =============================================================================

def run_test():
    """运行高德地图 Skill Agent 测试。"""
    print("\n" + "=" * 70)
    print("  高德地图综合服务 Skill 测试")
    print("=" * 70 + "\n")

    agent = create_amap_skill_agent()
    if agent is None:
        return

    # 测试问题列表（覆盖官方 Skill 的 6 大场景）
    test_questions = [
        # 场景一：明确关键词搜索（POI 搜索）
        "帮我搜索一下北京的美食",
        # 场景二：基于位置的周边搜索
        "搜索西直门周边的餐厅",
        # 场景三：地址转经纬度
        "深圳市龙岗区富通海智科技园的经纬度是多少？",
        # 场景四：天气查询
        "今天深圳的天气怎么样？",
        # 场景五：路径规划
        "帮我规划从天安门到故宫的步行路线",
        # 场景六：智能旅游规划
        "帮我规划一下杭州一日游，主要看景点和美食",
    ]

    for q in test_questions:
        print(f"{'─' * 50}")
        print(f"【用户】{q}")
        try:
            r = agent.invoke({"messages": [HumanMessage(content=q)]})
            print(f"【回答】{r['messages'][-1].content}")
        except Exception as e:
            print(f"【错误】{e}")
        print()

    # 交互模式
    print(f"{'═' * 50}")
    print("进入交互模式（输入 'quit' 退出）")
    print(f"{'═' * 50}\n")

    while True:
        try:
            user_input = input("【用户】").strip()
            if not user_input:
                continue
            if user_input.lower() in ('quit', 'exit', 'q'):
                print("再见！")
                break
            r = agent.invoke({"messages": [HumanMessage(content=user_input)]})
            print(f"【回答】{r['messages'][-1].content}")
            print()
        except (KeyboardInterrupt, EOFError):
            print("\n再见！")
            break


if __name__ == '__main__':
    run_test()
