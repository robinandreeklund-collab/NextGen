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
    print("⚡ Real-time WebSocket data - no artificial delays")
    print("   Agents react to market events as they occur")
    print()
    
    # Create dashboard in live mode (live_mode=True)
    # In live mode, tick_rate only affects agent processing loop
    # WebSocket data arrives in real-time regardless of tick_rate
    # tick_rate=0.1 allows agents to process data 10 times per second
    dashboard = NextGenDashboard(live_mode=True, tick_rate=0.1)
    
    # Run dashboard
    dashboard.run(host='0.0.0.0', port=8050, debug=False)


if __name__ == '__main__':
    main()
