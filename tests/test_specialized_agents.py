"""
Tests for Specialized Trading Agents

Testar:
- Initialization av alla 8 agenter
- State management (positions, capital)
- Vote generation för olika marknadsförhållanden
- Performance tracking
- Integration med ensemble voting system
"""

import pytest
from modules.specialized_agents import (
    MomentumAgent,
    MeanReversionAgent,
    TrendFollowingAgent,
    VolatilityAgent,
    BreakoutAgent,
    SwingAgent,
    ArbitrageAgent,
    SentimentAgent,
    SpecializedAgentsCoordinator
)
from modules.message_bus import MessageBus


class TestBaseSpecializedAgent:
    """Tests för base agent functionality"""
    
    def test_initialization(self):
        """Test agent initialization"""
        message_bus = MessageBus()
        agent = MomentumAgent(message_bus, initial_capital=10000.0)
        
        assert agent.agent_id == 'momentum_agent'
        assert agent.capital == 10000.0
        assert agent.initial_capital == 10000.0
        assert agent.positions == {}
        assert agent.total_trades == 0
        
    def test_portfolio_value_calculation(self):
        """Test portfolio value calculation"""
        message_bus = MessageBus()
        agent = MomentumAgent(message_bus, initial_capital=10000.0)
        
        # Initial value
        assert agent.get_portfolio_value() == 10000.0
        
        # After adding position
        agent.positions['AAPL'] = 10
        agent.market_data['AAPL'] = {'price': 150.0}
        
        expected = 10000.0 + (10 * 150.0)  # capital + position value
        assert agent.get_portfolio_value() == expected
        
    def test_buy_execution(self):
        """Test buy trade execution"""
        message_bus = MessageBus()
        agent = MomentumAgent(message_bus, initial_capital=10000.0)
        
        # Execute buy
        success = agent.execute_trade('AAPL', 'BUY', 10, 150.0)
        
        assert success == True
        assert agent.capital == 10000.0 - (10 * 150.0)
        assert agent.positions['AAPL'] == 10
        assert agent.position_prices['AAPL'] == 150.0
        assert agent.total_trades == 1
        
    def test_sell_execution(self):
        """Test sell trade execution"""
        message_bus = MessageBus()
        agent = MomentumAgent(message_bus, initial_capital=10000.0)
        
        # Setup position
        agent.execute_trade('AAPL', 'BUY', 10, 150.0)
        
        # Execute sell
        success = agent.execute_trade('AAPL', 'SELL', 10, 160.0)
        
        assert success == True
        assert agent.positions.get('AAPL', 0) == 0
        assert agent.total_trades == 2
        assert agent.total_pnl == 10 * (160.0 - 150.0)  # Profit
        assert agent.winning_trades == 1
        
    def test_insufficient_funds(self):
        """Test trade rejection due to insufficient funds"""
        message_bus = MessageBus()
        agent = MomentumAgent(message_bus, initial_capital=1000.0)
        
        # Try to buy more than affordable
        success = agent.execute_trade('AAPL', 'BUY', 100, 150.0)
        
        assert success == False
        assert agent.capital == 1000.0
        assert agent.positions == {}
        
    def test_insufficient_holdings(self):
        """Test trade rejection due to insufficient holdings"""
        message_bus = MessageBus()
        agent = MomentumAgent(message_bus, initial_capital=10000.0)
        
        # Try to sell without position
        success = agent.execute_trade('AAPL', 'SELL', 10, 150.0)
        
        assert success == False
        assert agent.capital == 10000.0
        
    def test_win_rate_calculation(self):
        """Test win rate calculation"""
        message_bus = MessageBus()
        agent = MomentumAgent(message_bus, initial_capital=10000.0)
        
        # Initial win rate
        assert agent.get_win_rate() == 0.5  # Neutral
        
        # After trades
        agent.winning_trades = 7
        agent.losing_trades = 3
        assert agent.get_win_rate() == 0.7
        
    def test_roi_calculation(self):
        """Test ROI calculation"""
        message_bus = MessageBus()
        agent = MomentumAgent(message_bus, initial_capital=10000.0)
        
        # Initial ROI
        assert agent.get_roi() == 0.0
        
        # After profitable trade
        agent.execute_trade('AAPL', 'BUY', 10, 100.0)
        agent.market_data['AAPL'] = {'price': 110.0}
        
        portfolio_value = agent.get_portfolio_value()
        expected_roi = (portfolio_value - 10000.0) / 10000.0
        assert agent.get_roi() == pytest.approx(expected_roi)


class TestMomentumAgent:
    """Tests för MomentumAgent"""
    
    def test_bullish_momentum_vote(self):
        """Test bullish momentum detection"""
        message_bus = MessageBus()
        agent = MomentumAgent(message_bus)
        
        # Setup bullish indicators
        agent.indicator_data['AAPL'] = {
            'technical': {'RSI': 65}
        }
        agent.market_data['AAPL'] = {'price': 150.0}
        
        vote = agent.analyze_and_vote('AAPL')
        
        assert vote['action'] == 'BUY'
        assert vote['confidence'] > 0.5
        assert 'momentum' in vote['reasoning'].lower()
        
    def test_bearish_momentum_vote(self):
        """Test bearish momentum detection"""
        message_bus = MessageBus()
        agent = MomentumAgent(message_bus)
        
        # Setup bearish indicators
        agent.indicator_data['AAPL'] = {
            'technical': {'RSI': 35}
        }
        
        vote = agent.analyze_and_vote('AAPL')
        
        assert vote['action'] == 'SELL'
        assert vote['confidence'] > 0.5
        
    def test_neutral_momentum_vote(self):
        """Test neutral momentum"""
        message_bus = MessageBus()
        agent = MomentumAgent(message_bus)
        
        # Setup neutral indicators
        agent.indicator_data['AAPL'] = {
            'technical': {'RSI': 50}
        }
        agent.market_data['AAPL'] = {'price': 150.0}
        
        vote = agent.analyze_and_vote('AAPL')
        
        assert vote['action'] == 'HOLD'


class TestMeanReversionAgent:
    """Tests för MeanReversionAgent"""
    
    def test_oversold_reversion(self):
        """Test oversold mean reversion signal"""
        message_bus = MessageBus()
        agent = MeanReversionAgent(message_bus)
        
        # Oversold
        agent.indicator_data['AAPL'] = {
            'technical': {'RSI': 25}
        }
        
        vote = agent.analyze_and_vote('AAPL')
        
        assert vote['action'] == 'BUY'
        assert vote['confidence'] > 0.5
        assert 'oversold' in vote['reasoning'].lower()
        
    def test_overbought_reversion(self):
        """Test overbought mean reversion signal"""
        message_bus = MessageBus()
        agent = MeanReversionAgent(message_bus)
        
        # Overbought
        agent.indicator_data['AAPL'] = {
            'technical': {'RSI': 75}
        }
        
        vote = agent.analyze_and_vote('AAPL')
        
        assert vote['action'] == 'SELL'
        assert vote['confidence'] > 0.5
        assert 'overbought' in vote['reasoning'].lower()


class TestTrendFollowingAgent:
    """Tests för TrendFollowingAgent"""
    
    def test_uptrend_detection(self):
        """Test uptrend detection"""
        message_bus = MessageBus()
        agent = TrendFollowingAgent(message_bus)
        
        # Strong uptrend
        agent.indicator_data['AAPL'] = {
            'technical': {
                'MACD': {'histogram': 2.0}
            }
        }
        
        vote = agent.analyze_and_vote('AAPL')
        
        assert vote['action'] == 'BUY'
        assert 'uptrend' in vote['reasoning'].lower()
        
    def test_downtrend_detection(self):
        """Test downtrend detection"""
        message_bus = MessageBus()
        agent = TrendFollowingAgent(message_bus)
        
        # Strong downtrend
        agent.indicator_data['AAPL'] = {
            'technical': {
                'MACD': {'histogram': -2.0}
            }
        }
        
        vote = agent.analyze_and_vote('AAPL')
        
        assert vote['action'] == 'SELL'
        assert 'downtrend' in vote['reasoning'].lower()


class TestVolatilityAgent:
    """Tests för VolatilityAgent"""
    
    def test_high_volatility_buy_dip(self):
        """Test buying dip in high volatility"""
        message_bus = MessageBus()
        agent = VolatilityAgent(message_bus)
        
        # High volatility, RSI dip
        agent.indicator_data['AAPL'] = {
            'technical': {
                'ATR': 6.0,
                'RSI': 40
            }
        }
        
        vote = agent.analyze_and_vote('AAPL')
        
        assert vote['action'] == 'BUY'
        assert 'volatility' in vote['reasoning'].lower()
        assert vote['quantity'] == 1  # Smaller position in volatile market
        
    def test_high_volatility_sell_rally(self):
        """Test selling rally in high volatility"""
        message_bus = MessageBus()
        agent = VolatilityAgent(message_bus)
        
        # High volatility, RSI rally
        agent.indicator_data['AAPL'] = {
            'technical': {
                'ATR': 7.0,
                'RSI': 60
            }
        }
        
        vote = agent.analyze_and_vote('AAPL')
        
        assert vote['action'] == 'SELL'
        assert 'volatility' in vote['reasoning'].lower()


class TestBreakoutAgent:
    """Tests för BreakoutAgent"""
    
    def test_bullish_breakout(self):
        """Test bullish breakout detection"""
        message_bus = MessageBus()
        agent = BreakoutAgent(message_bus)
        
        # Breakout conditions
        agent.indicator_data['AAPL'] = {
            'technical': {
                'RSI': 68,
                'MACD': {'histogram': 1.0}
            }
        }
        
        vote = agent.analyze_and_vote('AAPL')
        
        assert vote['action'] == 'BUY'
        assert 'breakout' in vote['reasoning'].lower()
        
    def test_bearish_breakdown(self):
        """Test bearish breakdown detection"""
        message_bus = MessageBus()
        agent = BreakoutAgent(message_bus)
        
        # Breakdown conditions
        agent.indicator_data['AAPL'] = {
            'technical': {
                'RSI': 32,
                'MACD': {'histogram': -1.0}
            }
        }
        
        vote = agent.analyze_and_vote('AAPL')
        
        assert vote['action'] == 'SELL'
        assert 'breakdown' in vote['reasoning'].lower()


class TestSwingAgent:
    """Tests för SwingAgent"""
    
    def test_early_upswing(self):
        """Test early upswing detection"""
        message_bus = MessageBus()
        agent = SwingAgent(message_bus)
        
        # Early upswing
        agent.indicator_data['AAPL'] = {
            'technical': {
                'RSI': 45,
                'MACD': {'histogram': 0.5}
            }
        }
        
        vote = agent.analyze_and_vote('AAPL')
        
        assert vote['action'] == 'BUY'
        assert 'swing' in vote['reasoning'].lower()
        
    def test_early_downswing(self):
        """Test early downswing detection"""
        message_bus = MessageBus()
        agent = SwingAgent(message_bus)
        
        # Early downswing
        agent.indicator_data['AAPL'] = {
            'technical': {
                'RSI': 55,
                'MACD': {'histogram': -0.5}
            }
        }
        
        vote = agent.analyze_and_vote('AAPL')
        
        assert vote['action'] == 'SELL'
        assert 'swing' in vote['reasoning'].lower()


class TestArbitrageAgent:
    """Tests för ArbitrageAgent"""
    
    def test_rapid_price_increase_arbitrage(self):
        """Test arbitrage on rapid price increase"""
        message_bus = MessageBus()
        agent = ArbitrageAgent(message_bus)
        
        # Simulate rapid price increase
        agent.market_data['AAPL'] = {'price': 100.0}
        agent.analyze_and_vote('AAPL')  # Build history
        
        agent.market_data['AAPL'] = {'price': 101.0}
        agent.analyze_and_vote('AAPL')
        
        agent.market_data['AAPL'] = {'price': 103.0}
        vote = agent.analyze_and_vote('AAPL')
        
        assert vote['action'] == 'SELL'
        assert 'arbitrage' in vote['reasoning'].lower()
        
    def test_rapid_price_decrease_arbitrage(self):
        """Test arbitrage on rapid price decrease"""
        message_bus = MessageBus()
        agent = ArbitrageAgent(message_bus)
        
        # Simulate rapid price decrease
        agent.market_data['AAPL'] = {'price': 100.0}
        agent.analyze_and_vote('AAPL')
        
        agent.market_data['AAPL'] = {'price': 99.0}
        agent.analyze_and_vote('AAPL')
        
        agent.market_data['AAPL'] = {'price': 97.5}
        vote = agent.analyze_and_vote('AAPL')
        
        assert vote['action'] == 'BUY'
        assert 'arbitrage' in vote['reasoning'].lower()


class TestSentimentAgent:
    """Tests för SentimentAgent"""
    
    def test_positive_analyst_sentiment(self):
        """Test positive analyst sentiment"""
        message_bus = MessageBus()
        agent = SentimentAgent(message_bus)
        
        # Strong buy consensus
        agent.indicator_data['AAPL'] = {
            'fundamental': {
                'AnalystRatings': {'consensus': 'STRONG_BUY'}
            }
        }
        
        vote = agent.analyze_and_vote('AAPL')
        
        assert vote['action'] == 'BUY'
        assert 'sentiment' in vote['reasoning'].lower()
        assert vote['confidence'] == 0.8
        
    def test_negative_analyst_sentiment(self):
        """Test negative analyst sentiment"""
        message_bus = MessageBus()
        agent = SentimentAgent(message_bus)
        
        # Sell consensus
        agent.indicator_data['AAPL'] = {
            'fundamental': {
                'AnalystRatings': {'consensus': 'SELL'}
            }
        }
        
        vote = agent.analyze_and_vote('AAPL')
        
        assert vote['action'] == 'SELL'
        assert 'sentiment' in vote['reasoning'].lower()
        
    def test_neutral_analyst_sentiment(self):
        """Test neutral analyst sentiment"""
        message_bus = MessageBus()
        agent = SentimentAgent(message_bus)
        
        # Hold consensus
        agent.indicator_data['AAPL'] = {
            'fundamental': {
                'AnalystRatings': {'consensus': 'HOLD'}
            }
        }
        
        vote = agent.analyze_and_vote('AAPL')
        
        assert vote['action'] == 'HOLD'


class TestSpecializedAgentsCoordinator:
    """Tests för SpecializedAgentsCoordinator"""
    
    def test_coordinator_initialization(self):
        """Test coordinator initialization"""
        message_bus = MessageBus()
        coordinator = SpecializedAgentsCoordinator(message_bus, initial_capital_per_agent=5000.0)
        
        assert len(coordinator.agents) == 8
        assert all(agent.initial_capital == 5000.0 for agent in coordinator.agents)
        
    def test_all_agents_present(self):
        """Test that all 8 agents are initialized"""
        message_bus = MessageBus()
        coordinator = SpecializedAgentsCoordinator(message_bus)
        
        agent_ids = [agent.agent_id for agent in coordinator.agents]
        
        expected_ids = [
            'momentum_agent',
            'mean_reversion_agent',
            'trend_following_agent',
            'volatility_agent',
            'breakout_agent',
            'swing_agent',
            'arbitrage_agent',
            'sentiment_agent'
        ]
        
        assert agent_ids == expected_ids
        
    def test_aggregated_statistics(self):
        """Test aggregated statistics"""
        message_bus = MessageBus()
        coordinator = SpecializedAgentsCoordinator(message_bus, initial_capital_per_agent=1000.0)
        
        stats = coordinator.get_aggregated_statistics()
        
        assert stats['total_capital'] == 8 * 1000.0
        assert stats['num_agents'] == 8
        assert len(stats['agent_statistics']) == 8
        
    def test_vote_publishing(self):
        """Test that agents publish votes to message bus"""
        message_bus = MessageBus()
        coordinator = SpecializedAgentsCoordinator(message_bus)
        
        # Setup vote collector
        votes_received = []
        message_bus.subscribe('decision_vote', lambda vote: votes_received.append(vote))
        
        # Setup market data
        for agent in coordinator.agents:
            agent.indicator_data['AAPL'] = {
                'technical': {
                    'RSI': 65,
                    'MACD': {'histogram': 1.0},
                    'ATR': 2.0
                },
                'fundamental': {
                    'AnalystRatings': {'consensus': 'BUY'}
                }
            }
            agent.market_data['AAPL'] = {'price': 150.0}
        
        # Trigger analysis
        coordinator.analyze_and_vote_all('AAPL')
        
        # Should have 8 votes (one from each agent)
        assert len(votes_received) == 8
        
    def test_state_publishing(self):
        """Test that agents publish state to message bus"""
        message_bus = MessageBus()
        coordinator = SpecializedAgentsCoordinator(message_bus)
        
        # Setup state collector
        states_received = []
        message_bus.subscribe('agent_state', lambda state: states_received.append(state))
        
        # Setup market data
        for agent in coordinator.agents:
            agent.indicator_data['AAPL'] = {
                'technical': {'RSI': 50, 'MACD': {'histogram': 0.0}, 'ATR': 2.0},
                'fundamental': {'AnalystRatings': {'consensus': 'HOLD'}}
            }
            agent.market_data['AAPL'] = {'price': 150.0}
        
        # Trigger analysis
        coordinator.analyze_and_vote_all('AAPL')
        
        # Should have 8 states (one from each agent)
        assert len(states_received) == 8
        
    def test_market_data_triggers_analysis(self):
        """Test that market data automatically triggers analysis"""
        message_bus = MessageBus()
        coordinator = SpecializedAgentsCoordinator(message_bus)
        
        votes_received = []
        message_bus.subscribe('decision_vote', lambda vote: votes_received.append(vote))
        
        # Setup indicator data first
        for agent in coordinator.agents:
            agent.indicator_data['AAPL'] = {
                'technical': {'RSI': 50, 'MACD': {'histogram': 0.0}, 'ATR': 2.0},
                'fundamental': {'AnalystRatings': {'consensus': 'HOLD'}}
            }
        
        # Publish market data (should trigger analysis)
        message_bus.publish('market_data', {'symbol': 'AAPL', 'price': 150.0})
        
        # Should receive votes from all agents
        assert len(votes_received) == 8


class TestVoteIntegration:
    """Tests för integration med vote_engine"""
    
    def test_vote_format_compatibility(self):
        """Test that agent votes are compatible with vote_engine"""
        message_bus = MessageBus()
        agent = MomentumAgent(message_bus)
        
        agent.indicator_data['AAPL'] = {
            'technical': {'RSI': 65}
        }
        agent.market_data['AAPL'] = {'price': 150.0}
        
        vote = agent.analyze_and_vote('AAPL')
        
        # Vote should have required fields
        assert 'symbol' in vote
        assert 'action' in vote
        assert 'confidence' in vote
        assert 'reasoning' in vote
        assert vote['action'] in ['BUY', 'SELL', 'HOLD']
        assert 0.0 <= vote['confidence'] <= 1.0
        
    def test_vote_includes_agent_performance(self):
        """Test that published votes include agent performance"""
        message_bus = MessageBus()
        agent = MomentumAgent(message_bus)
        
        # Setup some performance
        agent.winning_trades = 6
        agent.losing_trades = 4
        
        votes_received = []
        message_bus.subscribe('decision_vote', lambda vote: votes_received.append(vote))
        
        agent.indicator_data['AAPL'] = {
            'technical': {'RSI': 65}
        }
        agent.market_data['AAPL'] = {'price': 150.0}
        
        vote = agent.analyze_and_vote('AAPL')
        agent.publish_vote(vote)
        
        assert len(votes_received) == 1
        assert 'agent_performance' in votes_received[0]
        assert votes_received[0]['agent_performance'] == 0.6  # 6/10
