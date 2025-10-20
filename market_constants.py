"""
market_constants.py - Market Configuration Constants

Defines constants for market symbols and configuration used across the application.
This ensures consistency between production code and tests.
"""

# The 5 fixed stocks supported by the live market interface
MARKET_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

# Market metadata - additional configuration for the live market
MARKET_INFO = {
    'exchange': 'NASDAQ',
    'currency': 'USD',
    'timezone': 'America/New_York',
    'market_hours': {
        'open': '09:30',
        'close': '16:00'
    },
    'api_provider': 'Finnhub'
}
