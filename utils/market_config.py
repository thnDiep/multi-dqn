"""
Configuration for supported markets
"""

from datetime import datetime

MARKET_CONFIG = {
    "dax": {
        "name": "DAX",
        "num_walks": 8,
        "start_date": datetime(2007, 9, 18),
        "end_date": datetime(2017, 5, 29),
    },
    "sp500": {
        "name": "S&P 500",
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