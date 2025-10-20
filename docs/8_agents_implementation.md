# 8 Specialized Trading Agents - Implementation Documentation

## Overview

This document describes the implementation of 8 specialized trading agents integrated into the NextGen AI Trader system. Each agent operates independently with its own state management and contributes to ensemble-based decision making through a voting mechanism.

## Architecture

### Agent Design

Each agent inherits from `BaseSpecializedAgent` which provides:

- **State Management**: Independent tracking of capital, positions, and P&L
- **Market Data Subscription**: Automatic reception of market data and indicators
- **Vote Generation**: Analysis of market conditions and generation of trading votes
- **Performance Tracking**: Win rate, ROI, trade history

### The 8 Agents

1. **MomentumAgent** (`momentum_agent`)
   - **Strategy**: Follows strong price momentum using RSI
   - **Buy Signal**: RSI > 60 (bullish momentum)
   - **Sell Signal**: RSI < 40 (bearish momentum)
   - **Strength**: Captures trending moves early
   
2. **MeanReversionAgent** (`mean_reversion_agent`)
   - **Strategy**: Trades on reversals to mean
   - **Buy Signal**: RSI < 30 (oversold, expect bounce)
   - **Sell Signal**: RSI > 70 (overbought, expect correction)
   - **Strength**: Profits from extremes

3. **TrendFollowingAgent** (`trend_following_agent`)
   - **Strategy**: Follows established trends via MACD
   - **Buy Signal**: MACD histogram > 1.0 (strong uptrend)
   - **Sell Signal**: MACD histogram < -1.0 (strong downtrend)
   - **Strength**: Rides long-term trends

4. **VolatilityAgent** (`volatility_agent`)
   - **Strategy**: Exploits high volatility using ATR
   - **Buy Signal**: ATR > 5.0 + RSI < 45 (buy dip in volatile market)
   - **Sell Signal**: ATR > 5.0 + RSI > 55 (sell rally in volatile market)
   - **Strength**: Risk-adjusted trading in volatile conditions

5. **BreakoutAgent** (`breakout_agent`)
   - **Strategy**: Trades technical breakouts
   - **Buy Signal**: RSI > 65 + MACD > 0.5 (bullish breakout)
   - **Sell Signal**: RSI < 35 + MACD < -0.5 (bearish breakdown)
   - **Strength**: Captures momentum from breakouts

6. **SwingAgent** (`swing_agent`)
   - **Strategy**: Captures swing moves (2-5 days)
   - **Buy Signal**: RSI 40-50 + MACD > 0.3 (early upswing)
   - **Sell Signal**: RSI 50-60 + MACD < -0.3 (early downswing)
   - **Strength**: Medium-term swing timing

7. **ArbitrageAgent** (`arbitrage_agent`)
   - **Strategy**: Exploits rapid price changes
   - **Buy Signal**: Rapid price decrease > 2%
   - **Sell Signal**: Rapid price increase > 2%
   - **Strength**: Mean reversion on short-term deviations

8. **SentimentAgent** (`sentiment_agent`)
   - **Strategy**: Based on analyst consensus
   - **Buy Signal**: Analyst consensus = BUY/STRONG_BUY
   - **Sell Signal**: Analyst consensus = SELL
   - **Strength**: Fundamental and sentiment analysis

## Integration Points

### Message Bus Topics

**Subscriptions (Input):**
- `market_data`: Price and volume updates
- `indicator_data`: Technical and fundamental indicators

**Publications (Output):**
- `decision_vote`: Trading votes with action, confidence, reasoning
- `agent_state`: Agent state updates (capital, positions, performance)

### Voting System

Each agent publishes votes to the `decision_vote` topic which are then:

1. Collected by `VoteEngine`
2. Weighted by agent performance (`win_rate`)
3. Aggregated by `EnsembleCoordinator`
4. Combined into final ensemble decision

**Vote Format:**
```python
{
    'agent_id': 'momentum_agent',
    'symbol': 'AAPL',
    'action': 'BUY',  # BUY, SELL, or HOLD
    'quantity': 2,
    'confidence': 0.75,  # 0.0 to 1.0
    'reasoning': 'Strong bullish momentum (RSI: 68.0)',
    'agent_performance': 0.65,  # Win rate for weighting
    'timestamp': 1234567890.0
}
```

### Simulation Loop Integration

The `SpecializedAgentsCoordinator` is integrated into `sim_test.py`:

1. **Initialization**: Created in `setup_modules()` with 8 agents
2. **Auto-triggering**: Subscribes to `market_data` to automatically analyze
3. **Cooldown**: 2-second cooldown per symbol to avoid spam
4. **Statistics**: Aggregated stats displayed in simulation output

## Usage Examples

### Basic Initialization

```python
from modules.specialized_agents import SpecializedAgentsCoordinator
from modules.message_bus import MessageBus

message_bus = MessageBus()
coordinator = SpecializedAgentsCoordinator(
    message_bus=message_bus,
    initial_capital_per_agent=10000.0
)
```

### Manual Analysis

```python
# Trigger analysis for specific symbol
coordinator.analyze_and_vote_all('AAPL')
```

### Get Statistics

```python
stats = coordinator.get_aggregated_statistics()
print(f"Total value: ${stats['total_portfolio_value']:.2f}")
print(f"Total trades: {stats['total_trades']}")

for agent_stat in stats['agent_statistics']:
    print(f"{agent_stat['agent_id']}: ROI {agent_stat['roi']*100:.2f}%")
```

### Individual Agent Access

```python
# Access individual agents
momentum_agent = coordinator.agents[0]
print(f"Capital: ${momentum_agent.capital:.2f}")
print(f"Positions: {momentum_agent.positions}")
print(f"Win rate: {momentum_agent.get_win_rate()*100:.1f}%")
```

## Testing

### Test Coverage

The test suite (`tests/test_specialized_agents.py`) includes:

- **Base Agent Tests** (8 tests)
  - Initialization
  - Portfolio value calculation
  - Buy/sell execution
  - Insufficient funds/holdings handling
  - Win rate and ROI calculation

- **Individual Agent Tests** (24 tests)
  - Strategy-specific vote generation
  - Bullish/bearish/neutral signal detection
  - Edge cases for each strategy

- **Coordinator Tests** (5 tests)
  - Multi-agent initialization
  - Vote publishing
  - State publishing
  - Aggregated statistics

- **Integration Tests** (3 tests)
  - Vote format compatibility
  - Agent performance tracking
  - Message bus integration

**Total: 40+ tests**

### Running Tests

```bash
# Run all agent tests
pytest tests/test_specialized_agents.py -v

# Run specific test class
pytest tests/test_specialized_agents.py::TestMomentumAgent -v

# Run with coverage
pytest tests/test_specialized_agents.py --cov=modules.specialized_agents
```

## Performance Characteristics

### Resource Usage

- **Memory**: ~1MB per agent (8MB total for 8 agents)
- **CPU**: Minimal (<1% per agent during analysis)
- **State Size**: Scales with position count and trade history

### Scalability

- **Agents**: Easily extensible to N agents
- **Symbols**: Each agent can track multiple symbols independently
- **History**: Configurable with `deque(maxlen=...)` to prevent memory leaks

### Trade Execution

Each agent maintains its own positions:
- No shared state between agents
- Independent capital management
- Simulated execution (not real portfolio impact)
- P&L tracking per agent

## Best Practices

### Adding New Agents

1. Inherit from `BaseSpecializedAgent`
2. Implement `analyze_and_vote(symbol)` method
3. Define unique `agent_id`
4. Add to `SpecializedAgentsCoordinator.agents` list
5. Write tests for the new agent

Example:
```python
class CustomAgent(BaseSpecializedAgent):
    def __init__(self, message_bus, initial_capital=10000.0):
        super().__init__('custom_agent', message_bus, initial_capital)
    
    def analyze_and_vote(self, symbol: str) -> Dict[str, Any]:
        # Your strategy logic here
        return {
            'symbol': symbol,
            'action': 'BUY',
            'quantity': 1,
            'confidence': 0.8,
            'reasoning': 'Custom strategy signal'
        }
```

### Tuning Agent Strategies

Modify strategy parameters in each agent class:
- Threshold values (e.g., RSI levels)
- Position sizes
- Confidence calculations
- Reasoning logic

### Monitoring Agent Performance

Track via `agent_state` messages:
```python
message_bus.subscribe('agent_state', lambda state: 
    print(f"{state['agent_id']}: ROI {state['roi']*100:.2f}%")
)
```

## Known Limitations

1. **No Real Trading**: Agents simulate trades internally, don't affect main portfolio
2. **Simplified Indicators**: Basic indicator logic, not full TA library
3. **No Order Management**: No pending orders, immediate execution
4. **Fixed Position Sizes**: Hardcoded quantities per agent

## Future Enhancements

1. **Dynamic Position Sizing**: Based on volatility and confidence
2. **Risk Management**: Per-agent stop losses and take profits
3. **Performance-based Weighting**: Automatic adjustment of vote weights
4. **Strategy Optimization**: RL-based parameter tuning per agent
5. **Cross-agent Communication**: Agents share insights
6. **Portfolio Constraints**: Aggregate position limits across agents

## Troubleshooting

### Agents Not Voting

- Check that `market_data` and `indicator_data` are published
- Verify agent subscriptions are active
- Check cooldown timing (2 seconds per symbol)

### Incorrect Vote Actions

- Review indicator data format
- Check strategy logic in `analyze_and_vote()`
- Verify confidence calculations

### Performance Issues

- Limit number of symbols tracked
- Adjust cooldown period
- Reduce history buffer sizes

## References

- Main Module: `modules/specialized_agents.py`
- Tests: `tests/test_specialized_agents.py`
- Integration: `sim_test.py` (lines 51-54, 114-120, 192-200, 1032-1069)
- Documentation: `README.md` (8 Specialized Trading Agents section)
