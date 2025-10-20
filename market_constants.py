"""
market_constants.py - Market Configuration Constants

Defines constants for market symbols and configuration used across the application.
This ensures consistency between production code and tests.
"""

# The 5 fixed stocks supported by the live market interface
MARKET_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

# Market metadata
MARKET_INFO = {
    'exchange': 'NASDAQ',
    'symbol_count': len(MARKET_SYMBOLS),
    'symbols': MARKET_SYMBOLS
}
