from langchain_core.tools import tool

_SALES_DATA = {
    "2024-01": {"total": 1250000, "orders": 3200, "top_product": "智能手表Pro", "growth": 12.5},
    "2024-02": {"total": 980000, "orders": 2500, "top_product": "无线耳机X1", "growth": -5.2},
    "2024-03": {"total": 1450000, "orders": 3800, "top_product": "智能手表Pro", "growth": 48.0},
    "2024-04": {"total": 1320000, "orders": 3500, "top_product": "蓝牙音箱S3", "growth": -9.0},
    "2024-05": {"total": 1580000, "orders": 4100, "top_product": "智能手表Pro", "growth": 19.7},
    "2024-06": {"total": 1680000, "orders": 4300, "top_product": "无线耳机X2", "growth": 6.3},
    "2024-07": {"total": 1420000, "orders": 3600, "top_product": "蓝牙音箱S3", "growth": -15.5},
    "2024-08": {"total": 1550000, "orders": 4000, "top_product": "智能手表Ultra", "growth": 9.2},
    "2024-09": {"total": 1890000, "orders": 4800, "top_product": "智能手表Ultra", "growth": 21.9},
    "2024-10": {"total": 2100000, "orders": 5500, "top_product": "智能手表Ultra", "growth": 11.1},
    "2024-11": {"total": 2850000, "orders": 7200, "top_product": "智能手表Ultra", "growth": 35.7},
    "2024-12": {"total": 3200000, "orders": 8500, "top_product": "智能手表Ultra", "growth": 12.3},
}

_ORDER_DATA = {
    "ORD-2024-001": {"status": "已完成", "product": "智能手表Pro", "amount": 1299, "date": "2024-10-15"},
    "ORD-2024-002": {"status": "待发货", "product": "无线耳机X2", "amount": 599, "date": "2024-11-20"},
    "ORD-2024-003": {"status": "已退货", "product": "蓝牙音箱S3", "amount": 399, "date": "2024-11-05"},
    "ORD-2024-004": {"status": "运输中", "product": "智能手表Ultra", "amount": 2499, "date": "2024-12-01"},
}


@tool
def database_query(query_type: str, params: str = "") -> str:
    """Query the business database for sales data or order information.

    Args:
        query_type: Type of query. Use "sales" for sales data (params should be month like "2024-11"),
                    "order" for order lookup (params should be order ID like "ORD-2024-001"),
                    or "summary" for annual summary (no params needed).
        params: Query parameters depending on query_type.
    """
    if query_type == "sales":
        month = params.strip()
        if month in _SALES_DATA:
            d = _SALES_DATA[month]
            return (
                f"📊 {month} 销售数据:\n"
                f"  总销售额: ¥{d['total']:,.0f}\n"
                f"  订单数量: {d['orders']}\n"
                f"  热销商品: {d['top_product']}\n"
                f"  环比增长: {d['growth']:+.1f}%"
            )
        return f"未找到 {month} 的销售数据。可用月份: {', '.join(sorted(_SALES_DATA.keys()))}"

    elif query_type == "order":
        order_id = params.strip().upper()
        if order_id in _ORDER_DATA:
            d = _ORDER_DATA[order_id]
            return (
                f"📦 订单 {order_id} 信息:\n"
                f"  商品: {d['product']}\n"
                f"  金额: ¥{d['amount']}\n"
                f"  状态: {d['status']}\n"
                f"  日期: {d['date']}"
            )
        return f"未找到订单 {order_id}。可查询的订单: {', '.join(_ORDER_DATA.keys())}"

    elif query_type == "summary":
        total_sales = sum(d["total"] for d in _SALES_DATA.values())
        total_orders = sum(d["orders"] for d in _SALES_DATA.values())
        best_month = max(_SALES_DATA.items(), key=lambda x: x[1]["total"])
        return (
            f"📈 2024年度销售汇总:\n"
            f"  年度总销售额: ¥{total_sales:,.0f}\n"
            f"  年度总订单数: {total_orders}\n"
            f"  最佳月份: {best_month[0]} (¥{best_month[1]['total']:,.0f})\n"
            f"  月均销售额: ¥{total_sales / len(_SALES_DATA):,.0f}"
        )

    return f"不支持的查询类型: {query_type}。支持的类型: sales, order, summary"
