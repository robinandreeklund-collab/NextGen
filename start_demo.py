"""
start_demo.py - Start NextGen Dashboard in Demo Mode

Starts the dashboard with simulated/mock data for demonstration purposes.
No live WebSocket connection required.

Usage:
    python start_demo.py
    
Then open http://localhost:8050 in your browser.
"""

from start_dashboard import NextGenDashboard


def main():
    """Start dashboard in demo mode."""
    print("=" * 60)
    print("🎯 NextGen AI Trader - Demo Mode")
    print("=" * 60)
    print()
    print("Starting dashboard with simulated data...")
    print("No live WebSocket connection - using mock/replay data")
    print()
    print("⚡ Fast tick rate: 100ms (10 ticks/second)")
    print("   Provides high precision for RL agent training")
    print("   Modify tick_rate parameter to adjust speed")
    print()
    
    # Create dashboard in demo mode (live_mode=False)
    # tick_rate=0.1 (100ms) allows 10 updates/second for precise agent reactions
    # Set tick_rate=0 for maximum speed (no artificial delay)
    dashboard = NextGenDashboard(live_mode=False, tick_rate=0.1)
    
    # Run dashboard
    dashboard.run(host='0.0.0.0', port=8050, debug=False)


if __name__ == '__main__':
    main()
