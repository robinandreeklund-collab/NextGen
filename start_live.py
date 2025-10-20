"""
start_live.py - Start NextGen Dashboard in Live Mode

Starts the dashboard and connects to live WebSocket data from Finnhub.
Requires valid Finnhub API key in environment.

Usage:
    python start_live.py
    
Then open http://localhost:8050 in your browser.
"""

from start_dashboard import NextGenDashboard


def main():
    """Start dashboard in live mode."""
    print("=" * 60)
    print("🚀 NextGen AI Trader - Live Mode")
    print("=" * 60)
    print()
    print("Starting dashboard with LIVE WebSocket data...")
    print("Connecting to Finnhub for real-time market data")
    print()
    print("⚠️  Warning: This connects to external WebSocket services")
    print("⚠️  Ensure you have a valid Finnhub API key configured")
    print()
    
    # Define the 5 fixed stocks supported by the market
    market_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    print(f"📊 Market symbols: {', '.join(market_symbols)}")
    print(f"📡 Will establish {len(market_symbols)} WebSocket connections")
    print()
    
    # Create dashboard in live mode with market symbols
    dashboard = NextGenDashboard(live_mode=True, market_symbols=market_symbols)
    
    # Run dashboard
    dashboard.run(host='0.0.0.0', port=8050, debug=False)


if __name__ == '__main__':
    main()
