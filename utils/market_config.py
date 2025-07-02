"""
Configuration for supported markets
"""

from datetime import datetime

MARKET_CONFIG = {
    "dax": {
        "type": "index",
        "point_value": 25,
        "name": "DAX",
        "num_walks": 8,
        "start_date": datetime(2007, 9, 18),
        "end_date": datetime(2017, 5, 29),
    },
    "sp500": {
        "type": "index",
        "point_value": 50,
        "name": "S&P 500",
        "num_walks": 8,
        "start_date": datetime(2007, 9, 18),
        "end_date": datetime(2017, 5, 29),
    },
    "msft": {
        "type": "stock",
        "name": "MSFT",
        "num_walks": 8,
        "start_date": datetime(2007, 9, 18),
        "end_date": datetime(2017, 5, 29),
    }
}

def get_market_config(market: str) -> dict:
    """
    Get configuration for a specific market
    
    Args:
        market: Market name (dax or sp500)
        
    Returns:
        dict: Market configuration
        
    Raises:
        ValueError: If market is not supported
    """
    if market not in MARKET_CONFIG:
        raise ValueError(f"Market '{market}' is not supported. Supported markets: {', '.join(MARKET_CONFIG.keys())}")
    return MARKET_CONFIG[market]

def get_market_type(market: str) -> str:
    market_config = get_market_config(market)
    return market_config.get("type")

def get_market_point_value(market: str) -> int:
    market_config = get_market_config(market)
    return market_config.get("point_value")