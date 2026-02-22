from langchain_core.tools import tool

_WEATHER_DATA = {
    "北京": {"temperature": 5, "condition": "晴", "humidity": 30, "wind": "北风3级",
             "suggestion": "天气寒冷干燥，建议穿羽绒服、围巾和手套，注意保暖防风。"},
    "上海": {"temperature": 12, "condition": "多云", "humidity": 65, "wind": "东南风2级",
             "suggestion": "气温适中偏凉，建议穿薄外套或风衣，可搭配毛衣。"},
    "广州": {"temperature": 22, "condition": "阴", "humidity": 80, "wind": "南风2级",
             "suggestion": "温暖湿润，穿长袖衬衫或薄外套即可，建议随身带伞。"},
    "深圳": {"temperature": 23, "condition": "多云转晴", "humidity": 75, "wind": "东南风3级",
             "suggestion": "天气温暖，穿T恤或薄长袖即可，户外注意防晒。"},
    "成都": {"temperature": 14, "condition": "阴天", "humidity": 70, "wind": "微风",
             "suggestion": "阴冷潮湿，建议穿厚外套或夹克，注意保暖。"},
    "杭州": {"temperature": 10, "condition": "小雨", "humidity": 85, "wind": "东风2级",
             "suggestion": "有小雨，建议穿防水外套，随身带雨伞。穿毛衣搭配风衣为佳。"},
    "武汉": {"temperature": 8, "condition": "晴转多云", "humidity": 50, "wind": "北风2级",
             "suggestion": "早晚温差大，建议穿大衣或厚外套，中午可适当减衣。"},
    "西安": {"temperature": 3, "condition": "晴", "humidity": 25, "wind": "西北风3级",
             "suggestion": "天气寒冷，建议穿棉衣或羽绒服，戴帽子和手套。"},
}


@tool
def weather_query(city: str) -> str:
    """Query the weather information for a Chinese city.

    Args:
        city: The name of the city in Chinese, e.g. "北京"
    """
    for city_name, data in _WEATHER_DATA.items():
        if city_name in city:
            return (
                f"🌤 {city_name}天气信息:\n"
                f"  温度: {data['temperature']}°C\n"
                f"  天气: {data['condition']}\n"
                f"  湿度: {data['humidity']}%\n"
                f"  风力: {data['wind']}\n"
                f"  穿衣建议: {data['suggestion']}"
            )

    return f"抱歉，暂无 {city} 的天气数据。目前支持查询的城市有：{', '.join(_WEATHER_DATA.keys())}"
