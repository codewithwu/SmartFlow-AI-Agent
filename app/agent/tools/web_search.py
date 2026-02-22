from langchain_core.tools import tool

_SEARCH_DATABASE = {
    "退货政策": [
        {
            "title": "电商退货政策全解析",
            "snippet": "根据消费者权益保护法，网购商品可在收到之日起7天内无理由退货。退货商品应保持完好，不影响二次销售。",
            "url": "https://example.com/return-policy",
        },
    ],
    "人工智能": [
        {
            "title": "2024年人工智能发展趋势",
            "snippet": "大语言模型持续进化，多模态AI成为主流。AI Agent技术快速发展，在企业自动化中扮演重要角色。",
            "url": "https://example.com/ai-trends",
        },
    ],
    "Python编程": [
        {
            "title": "Python 3.12 新特性一览",
            "snippet": "Python 3.12带来了更好的错误信息、性能提升、以及新的类型标注语法。",
            "url": "https://example.com/python312",
        },
    ],
}

_DEFAULT_RESULTS = [
    {
        "title": "搜索结果",
        "snippet": "暂未找到完全匹配的结果，以下是相关信息的摘要。请尝试使用更具体的关键词搜索。",
        "url": "https://example.com/search",
    },
]


@tool
def web_search(query: str) -> str:
    """Search the web for information on a given query.

    Args:
        query: The search query string.
    """
    results = None
    for keyword, data in _SEARCH_DATABASE.items():
        if keyword in query:
            results = data
            break

    if results is None:
        results = _DEFAULT_RESULTS

    output_parts = [f"搜索关键词: {query}\n"]
    for i, r in enumerate(results, 1):
        output_parts.append(f"{i}. **{r['title']}**\n   {r['snippet']}\n   来源: {r['url']}")
    return "\n".join(output_parts)
