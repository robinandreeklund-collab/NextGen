# Quick Start Guide: 8 Specialized Trading Agents

## Running the System

### Basic Simulation

```bash
# Run simulation with all 8 agents
python sim_test.py

# The agents will automatically:
# 1. Initialize with $10,000 capital each ($80,000 total)
# 2. Subscribe to market data and indicators
# 3. Analyze every 2 seconds (cooldown)
# 4. Generate votes (BUY/SELL/HOLD)
# 5. Publish to ensemble voting system
```

### Expected Output

```
🎯 8 SPECIALIZED TRADING AGENTS
════════════════════════════════════════════════════════════════════════════════

📊 Aggregated Agent Statistics:
   Total agents: 8
   Combined capital: $80000.00
   Combined portfolio value: $80245.50
   Combined trades: 47

🤖 Individual Agent Performance:
   🟢 momentum_agent          : Value: $10123.45 | ROI: +1.23% | Trades:   8 | Win Rate:  62.5%
   🔴 mean_reversion_agent    : Value:  $9876.32 | ROI: -1.24% | Trades:   6 | Win Rate:  50.0%
   🟢 trend_following_agent   : Value: $10345.67 | ROI: +3.46% | Trades:   9 | Win Rate:  66.7%
   ⚪ volatility_agent        : Value: $10000.00 | ROI: +0.00% | Trades:   0 | Win Rate:  50.0%
   🟢 breakout_agent          : Value: $10234.56 | ROI: +2.35% | Trades:   7 | Win Rate:  57.1%
   ⚪ swing_agent             : Value: $10012.34 | ROI: +0.12% | Trades:   3 | Win Rate:  66.7%
   🟢 arbitrage_agent         : Value: $10156.78 | ROI: +1.57% | Trades:   5 | Win Rate:  80.0%
   ⚪ sentiment_agent         : Value:  $9996.38 | ROI: -0.04% | Trades:   9 | Win Rate:  44.4%
```

## Key Observations

### Agent Behavior Patterns

1. **MomentumAgent**: Most active in trending markets
   - High trade frequency when RSI > 60 or < 40
   - Best in strong directional moves

2. **MeanReversionAgent**: Active at extremes
   - Trades only when RSI < 30 or > 70
   - Lower frequency but potentially higher win rate

3. **TrendFollowingAgent**: Patient trader
   - Waits for MACD > 1.0 or < -1.0
   - Longer holding periods

4. **VolatilityAgent**: Selective trader
   - Only trades when ATR > 5.0
   - Lower frequency, risk-adjusted positions

5. **BreakoutAgent**: Opportunistic
   - Requires both RSI and MACD alignment
   - Medium frequency

6. **SwingAgent**: Timing-focused
   - Looks for early swing signals
   - Medium-term holds

7. **ArbitrageAgent**: Quick trades
   - Rapid in/out on price spikes
   - Highest frequency potential

8. **SentimentAgent**: Fundamental-based
   - Follows analyst consensus
   - Stable, longer-term positions

### Voting Patterns

Typical vote distribution:
- **Bullish Market**: BUY=5, SELL=1, HOLD=2
- **Bearish Market**: BUY=1, SELL=5, HOLD=2
- **Neutral Market**: BUY=2, SELL=2, HOLD=4
- **Volatile Market**: More diverse, ensemble crucial

## Monitoring Agent Performance

### Real-time Monitoring

Watch the agent statistics section in sim_test.py output:
- Green emoji (🟢): Positive ROI
- Red emoji (🔴): Negative ROI
- White emoji (⚪): Neutral or minimal trades

### Key Metrics

1. **Portfolio Value**: Total value (capital + positions)
2. **ROI**: Return on investment percentage
3. **Trades**: Number of trades executed
4. **Win Rate**: Percentage of profitable trades

### Performance Tracking

```python
# Via message bus
message_bus.subscribe('agent_state', lambda state: {
    print(f"{state['agent_id']}: "
          f"ROI {state['roi']*100:.2f}%, "
          f"Win Rate {state['win_rate']*100:.1f}%")
})
```

## Common Scenarios

### Scenario 1: Strong Uptrend (AAPL rallying)

**Market Data**: RSI=70, MACD=2.5, ATR=3.0
**Expected Votes**:
- ✅ MomentumAgent: BUY (RSI > 60)
- ❌ MeanReversionAgent: SELL (RSI > 70, expecting reversal)
- ✅ TrendFollowingAgent: BUY (MACD > 1.0)
- ⚪ VolatilityAgent: HOLD (ATR < 5.0)
- ✅ BreakoutAgent: BUY (RSI + MACD combo)
- ⚪ SwingAgent: HOLD (RSI too high for entry)
- ⚪ ArbitrageAgent: HOLD (steady move, no spike)
- ✅ SentimentAgent: BUY (analyst consensus)

**Result**: 4 BUY, 1 SELL, 3 HOLD → **Ensemble Decision: BUY**

### Scenario 2: Oversold Bounce (AAPL dip)

**Market Data**: RSI=25, MACD=-0.5, ATR=2.0
**Expected Votes**:
- ❌ MomentumAgent: SELL (RSI < 40, bearish)
- ✅ MeanReversionAgent: BUY (RSI < 30, oversold)
- ⚪ TrendFollowingAgent: HOLD (MACD not extreme)
- ⚪ VolatilityAgent: HOLD (ATR too low)
- ⚪ BreakoutAgent: HOLD (no breakout signal)
- ⚪ SwingAgent: HOLD (waiting for upswing)
- ✅ ArbitrageAgent: BUY (if rapid drop)
- ⚪ SentimentAgent: HOLD (neutral)

**Result**: 2 BUY, 1 SELL, 5 HOLD → **Ensemble Decision: HOLD** (low confidence)

### Scenario 3: High Volatility (Market turbulence)

**Market Data**: RSI=55, MACD=0.3, ATR=8.0
**Expected Votes**:
- ⚪ MomentumAgent: HOLD (neutral RSI)
- ⚪ MeanReversionAgent: HOLD (not at extreme)
- ⚪ TrendFollowingAgent: HOLD (weak MACD)
- ✅ VolatilityAgent: BUY (ATR > 5.0, buying dip)
- ⚪ BreakoutAgent: HOLD (no clear signal)
- ⚪ SwingAgent: HOLD (mixed signals)
- ⚪ ArbitrageAgent: Potentially BUY/SELL (on spikes)
- ⚪ SentimentAgent: HOLD (neutral)

**Result**: 1-2 BUY, 0-1 SELL, 5-6 HOLD → **Ensemble Decision: HOLD**

## Customization

### Adjusting Agent Capital

```python
# In sim_test.py setup_modules()
self.specialized_agents = SpecializedAgentsCoordinator(
    self.message_bus, 
    initial_capital_per_agent=5000.0  # Change from 1000 to 5000
)
```

### Changing Cooldown Period

```python
# In specialized_agents.py SpecializedAgentsCoordinator.__init__()
self.analysis_cooldown = 5.0  # Change from 2.0 to 5.0 seconds
```

### Modifying Strategy Thresholds

```python
# In specialized_agents.py, for example MomentumAgent
def analyze_and_vote(self, symbol: str) -> Dict[str, Any]:
    # Change thresholds
    if rsi > 65:  # Changed from 60
        return {'action': 'BUY', ...}
    elif rsi < 35:  # Changed from 40
        return {'action': 'SELL', ...}
```

## Troubleshooting

### Issue: Agents not voting

**Check:**
1. Market data is being published
2. Indicator data is available
3. Cooldown period hasn't blocked voting
4. Agents are subscribed to message_bus

**Debug:**
```python
# Add to sim_test.py
message_bus.subscribe('decision_vote', 
    lambda v: print(f"Vote from {v['agent_id']}: {v['action']}"))
```

### Issue: All agents vote HOLD

**Causes:**
- Neutral market conditions
- Missing indicator data
- Thresholds too strict

**Solution:**
- Check indicator values in output
- Verify market data is updating
- Temporarily lower thresholds for testing

### Issue: Ensemble always chooses same action

**Causes:**
- All agents have similar strategies
- One agent dominates with high win rate

**Solution:**
- Ensure diverse indicator data
- Check vote weighting in ensemble
- Review individual agent logic

## Next Steps

1. **Monitor Performance**: Watch ROI and win rates
2. **Adjust Strategies**: Tune thresholds based on results
3. **Add Agents**: Create custom agents for your strategies
4. **Optimize Weights**: Adjust ensemble weights based on performance
5. **Risk Management**: Implement stop losses per agent
6. **Position Sizing**: Make quantities dynamic based on confidence

## Support

For issues or questions:
- See `docs/8_agents_implementation.md` for detailed documentation
- Review tests in `tests/test_specialized_agents.py`
- Check architecture diagram in `docs/8_agents_architecture.txt`
