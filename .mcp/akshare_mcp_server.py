#!/usr/bin/env python3
"""
AKShare MCP Server — 中国金融/商品/宏观数据
用于电力物资需求预测的外部影响因子采集:
  - 上期所铜/铝期货结算价 (原材料价格因子 RMPF)
  - 全社会用电量 (工业用电需求因子 IEDF)
  - 新能源发电装机容量 (新能源并网容量因子 NEIF)
"""
import json
import sys
from datetime import datetime


def handle_request(request):
    """Minimal MCP stdio handler"""
    method = request.get("method", "")
    req_id = request.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "akshare-mcp",
                    "version": "0.1.0"
                }
            }
        }

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "shfe_futures_daily",
                        "description": "获取上期所(SHFE)铜/铝期货主力合约日结算价。用于原材料价格因子(RMPF)。返回日期、收盘价、结算价。",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "symbol": {
                                    "type": "string",
                                    "enum": ["cu", "al"],
                                    "description": "品种: cu=铜, al=铝"
                                },
                                "start_date": {
                                    "type": "string",
                                    "description": "开始日期 YYYYMMDD"
                                },
                                "end_date": {
                                    "type": "string",
                                    "description": "结束日期 YYYYMMDD"
                                }
                            },
                            "required": ["symbol"]
                        }
                    },
                    {
                        "name": "china_electricity_consumption",
                        "description": "获取中国全社会用电量月度数据。用于工业用电需求因子(IEDF)。返回月度用电量(亿千瓦时)及同比增速。",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "start_year": {"type": "string", "description": "开始年份, 如 '2020'"},
                                "end_year": {"type": "string", "description": "结束年份, 如 '2026'"}
                            }
                        }
                    },
                    {
                        "name": "china_power_generation_capacity",
                        "description": "获取中国新能源发电装机容量(风电+光伏)。用于新能源并网容量因子(NEIF)。返回累计装机容量(万千瓦)。",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "start_year": {"type": "string"},
                                "end_year": {"type": "string"}
                            }
                        }
                    }
                ]
            }
        }

    if method == "tools/call":
        tool_name = request["params"]["name"]
        args = request["params"].get("arguments", {})

        try:
            import akshare as ak
        except ImportError:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{
                        "type": "text",
                        "text": "Error: akshare not installed. Run: pip install akshare"
                    }]
                }
            }

        if tool_name == "shfe_futures_daily":
            symbol = args.get("symbol", "cu")
            symbol_map = {"cu": "CU", "al": "AL"}
            try:
                df = ak.futures_shfe_daily(symbol=symbol_map[symbol])
                result = df.tail(30).to_json(orient="records", force_ascii=False)
            except Exception as e:
                result = f"Error fetching SHFE data: {e}"

        elif tool_name == "china_electricity_consumption":
            try:
                df = ak.macro_china_society_electricity()
                result = df.tail(36).to_json(orient="records", force_ascii=False)
            except Exception as e:
                result = f"Error fetching electricity data: {e}"

        elif tool_name == "china_power_generation_capacity":
            try:
                df = ak.energy_china()
                result = df.to_json(orient="records", force_ascii=False)
            except Exception as e:
                result = f"Error fetching energy data: {e}"

        else:
            result = f"Unknown tool: {tool_name}"

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{"type": "text", "text": str(result)}]
            }
        }

    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Unknown method: {method}"}}


def main():
    import sys
    sys.stderr = open("D:/Users/dell/PycharmProjects/vmd-catboost/.mcp/akshare_mcp.log", "a")
    for line in sys.stdin:
        try:
            request = json.loads(line.strip())
            response = handle_request(request)
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()
        except json.JSONDecodeError:
            continue


if __name__ == "__main__":
    main()
