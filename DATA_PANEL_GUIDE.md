# Data Panel - Extended Features Guide

## Overview

The Data Panel in the NextGen Dashboard has been extended with comprehensive insights into WebSocket connections, RL agent performance, symbol rotation history, and additional system metrics.

## Access

1. Start the dashboard: `python start_demo.py` (or `start_live.py`)
2. Open browser: `http://localhost:8050`
3. Click **"📡 Data"** in the left sidebar

## Sections

### 1. WebSocket Connections

Shows real-time status of all data streaming connections.

**Columns:**
- **Symbol** - The ticker symbol being tracked
- **Status** - Connection state (Active, Pending, Active (Sim))
- **Uptime** - How long the connection has been running
- **Last Data** - Timestamp of most recent data received
- **Frequency** - Data update rate (trades per minute)
- **Health** - Connection health indicator (Good, Waiting)

**Data Source:**
- **Demo mode**: `data_ingestion_sim.symbols` - Simulated connections
- **Live mode**: `data_ingestion.subscribed_symbols` - Real WebSocket connections

### 2. RL Agent Insights

Displays reinforcement learning agent performance and symbol prioritization.

**Columns:**
- **Rank** - Priority ranking (1 = highest priority)
- **Symbol** - Ticker symbol
- **Avg Reward** - Average reward from last 5 trading decisions
- **Trend** - Price trend indicator (▲ Rising / ▼ Falling / ● Neutral)
- **Change** - Price change percentage over last 10 data points
- **Priority** - RL-computed priority score

**Data Sources:**
- `orchestrator_metrics['rl_scores_history']` - Symbol priorities from RL
- `decision_history` - Trading rewards per symbol
- `price_history` - Price trends

**Use Cases:**
- Identify which symbols RL agents favor
- Track trading performance per symbol
- Monitor trend alignment with priorities

### 3. Symbol Rotation History

Records all symbol rotation events with timestamps and reasons.

**Columns:**
- **Time** - When rotation occurred
- **Dropped** - Symbols removed from active tracking
- **Duration** - How long dropped symbols were tracked
- **Added** - New symbols added to replace dropped ones
- **Reason** - Why rotation occurred (RL-driven, time-based, performance-based)

**Data Source:**
- `orchestrator_metrics['symbol_rotations']` - Rotation events from orchestrator

**Use Cases:**
- Understand symbol rotation patterns
- Identify underperforming symbols being dropped
- Track rotation frequency and reasons

### 4. Additional Metrics

Three metric cards showing system-wide statistics.

#### Data Flow Statistics
- **Total Data Points** - Cumulative price data across all symbols
- **Active Symbols** - Number of currently tracked symbols
- **Avg Points/Symbol** - Average data accumulation per symbol
- **Update Rate** - Agent tick rate (default: 100ms for fast reactions, configurable)

**Note:** The update rate shown is the agent processing tick rate, not the dashboard refresh rate. Lower values mean faster agent reactions and better precision for RL training.

#### Portfolio-Protected Symbols
- **Protected Count** - Number of symbols with open positions
- **Symbol List** - Up to 5 protected symbols shown
- **Protection Status** - 🔒 indicates symbol cannot be rotated out

**Why Protected?** Symbols with active portfolio positions cannot be removed from tracking to maintain position monitoring.

#### WebSocket Health Score
- **Health Percentage** - 0-100% system health indicator
- **Status** - Excellent (≥80%) / Good (≥50%) / Poor (<50%)
- **Mode** - Live or Demo mode indicator

**Calculation:**
- **Live mode**: (symbols_with_real_data / total_symbols) × 100
- **Demo mode**: Always 100% (simulated)

## Data Integrity

**All data comes from actual system modules - NO hardcoded values!**

✅ Real data sources:
- `finnhub_orchestrator` - Symbol rotations, RL scores, stream metrics
- `rl_controller` - Agent rewards and priorities
- `portfolio_manager` - Protected symbols (current positions)
- `data_ingestion` / `data_ingestion_sim` - WebSocket connection status
- `message_bus` - Real-time event tracking

❌ No mockup data:
- No random values in visualizations
- No static placeholder text (except during initialization)
- No UI-generated metrics

## Updates

The Data Panel has two separate update mechanisms:

1. **Dashboard GUI Refresh**: Every 2 seconds (via Dash `dcc.Interval`)
   - Updates all visual components
   - Pulls latest data from system state

2. **Agent Tick Rate**: Default 100ms (configurable)
   - Controls how fast agents process market data
   - Lower = faster reactions = better RL precision
   - Set to 0 for maximum speed (no artificial delay)
   - In live mode, WebSocket data arrives in real-time

**Configuring Tick Rate:**

```python
# Fast agent training (default)
dashboard = NextGenDashboard(live_mode=False, tick_rate=0.1)  # 100ms

# Maximum speed
dashboard = NextGenDashboard(live_mode=False, tick_rate=0)  # No delay

# Custom rate
dashboard = NextGenDashboard(live_mode=False, tick_rate=0.05)  # 50ms
```

Lower tick rates provide better precision for RL agent training by allowing them to react faster to market changes.

## Navigation

- **Start/Stop buttons** - Control simulation from sidebar
- **Symbol Mode toggle** - Switch between multi-symbol and single-symbol mode
- **Other panels** - Navigate using sidebar menu (Portfolio, Agents, DQN, etc.)

## Verification

Run the verification script to confirm all data sources:

```bash
python verify_data_panel.py
```

This will:
1. Start dashboard in demo mode
2. Run simulation for 5 seconds
3. Test each section individually
4. Verify data sources
5. Output summary of data flow

## Examples

### Viewing RL Agent Performance

1. Start simulation
2. Navigate to Data panel
3. Check **RL Agent Insights** section
4. Look for:
   - Top-ranked symbols (RL priorities)
   - Symbols with positive rewards (profitable trades)
   - Trend alignment (rising trends + high priority = strong candidate)

### Monitoring WebSocket Health

1. In Live mode: Check **WebSocket Connections** section
2. Verify all symbols show "Active" status
3. Confirm "Last Data" shows "Now"
4. Check **WebSocket Health Score** is above 80%

If health is low:
- Check network connection
- Verify Finnhub API key
- Review console for WebSocket errors

### Tracking Symbol Rotations

1. Navigate to Data panel
2. Check **Symbol Rotation History** section
3. Review recent rotations:
   - Which symbols were dropped and why?
   - What were their durations?
   - Which symbols replaced them?

This helps understand orchestrator behavior and optimization patterns.

## Troubleshooting

### "Loading..." messages persist
- **Cause**: Simulation not started or data not yet accumulated
- **Solution**: Click "▶ Start" button in sidebar and wait 5-10 seconds

### "No rotations yet" in rotation history
- **Cause**: Orchestrator hasn't performed first rotation
- **Solution**: Wait for rotation interval (default 300s = 5 minutes) or check orchestrator config

### Empty RL insights
- **Cause**: No RL scores computed yet
- **Solution**: Wait for RL agents to process market data and compute priorities

### WebSocket health = 0% in Live mode
- **Cause**: No real data received from Finnhub
- **Solution**: 
  - Verify API key is valid
  - Check network connectivity
  - Review console for WebSocket connection errors

## See Also

- [README.md](README.md) - General dashboard documentation
- [DASHBOARD_GUIDE.md](DASHBOARD_GUIDE.md) - Complete dashboard guide
- [WEBSOCKET_TEST_GUIDE.md](WEBSOCKET_TEST_GUIDE.md) - WebSocket testing
- Verification script: `verify_data_panel.py`
