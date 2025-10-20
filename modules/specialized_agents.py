"""
specialized_agents.py - 8 Specialized Trading Agents

Description:
    Implements 8 distinct trading agents with unique strategies.
    Each agent manages its own state (positions, capital) and votes
    independently through the ensemble voting system.

Agents:
    1. MomentumAgent - Follows strong price momentum
    2. MeanReversionAgent - Trades on reversals to the mean
    3. TrendFollowingAgent - Identifies and follows trends
    4. VolatilityAgent - Exploits volatility
    5. BreakoutAgent - Trades on technical breakouts
    6. SwingAgent - Captures swing movements
    7. ArbitrageAgent - Seeks arbitrage opportunities
    8. SentimentAgent - Based on market sentiment and analyst data

Role:
    - Each agent analyzes market data independently
    - Manages its own state (positions, capital, performance)
    - Generates a vote (BUY/SELL/HOLD) with confidence
    - Publishes votes to the ensemble voting system
    
Inputs:
    - market_data: Price and volume data
    - indicator_data: Technical indicators
    - portfolio_status: Global portfolio status
    
Outputs:
    - agent_vote: Vote for the ensemble system
    - agent_state: Agent's state and performance

Publishes to message_bus:
    - agent_vote: To vote_engine
    - agent_state: For monitoring and analysis
    
Subscribes to:
    - market_data
    - indicator_data
    - portfolio_status
    
Used in Sprint: Custom (8 Trading Agents)
"""

from typing import Dict, Any, List, Optional
import numpy as np
from collections import deque
import time


class BaseSpecializedAgent:
    """Basklass för alla specialiserade agenter"""
    
    def __init__(self, agent_id: str, message_bus, initial_capital: float = 10000.0):
        """
        Initialiserar basagent
        
        Args:
            agent_id: Unik identifierare för agenten
            message_bus: Referens till central message_bus
            initial_capital: Startkapital för agenten
        """
        self.agent_id = agent_id
        self.message_bus = message_bus
        
        # State management
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.positions: Dict[str, int] = {}  # symbol -> quantity
        self.position_prices: Dict[str, float] = {}  # symbol -> avg entry price
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.performance_history = deque(maxlen=100)
        
        # Market data cache
        self.market_data: Dict[str, Any] = {}
        self.indicator_data: Dict[str, Any] = {}
        
        # Subscribe to market data
        self.message_bus.subscribe('market_data', self._on_market_data)
        self.message_bus.subscribe('indicator_data', self._on_indicator_data)
        
    def _on_market_data(self, data: Dict[str, Any]) -> None:
        """Callback för market data"""
        symbol = data.get('symbol')
        if symbol:
            self.market_data[symbol] = data
            
    def _on_indicator_data(self, data: Dict[str, Any]) -> None:
        """Callback för indicator data"""
        symbol = data.get('symbol')
        if symbol:
            self.indicator_data[symbol] = data
    
    def get_portfolio_value(self) -> float:
        """Beräknar totalt portföljvärde (cash + positions)"""
        total_value = self.capital
        
        for symbol, quantity in self.positions.items():
            if symbol in self.market_data:
                current_price = self.market_data[symbol].get('price', 0)
                total_value += quantity * current_price
                
        return total_value
    
    def get_position_pnl(self, symbol: str) -> float:
        """Beräknar orealiserad P&L för en position"""
        if symbol not in self.positions or self.positions[symbol] == 0:
            return 0.0
            
        quantity = self.positions[symbol]
        entry_price = self.position_prices.get(symbol, 0)
        current_price = self.market_data.get(symbol, {}).get('price', 0)
        
        return quantity * (current_price - entry_price)
    
    def execute_trade(self, symbol: str, action: str, quantity: int, price: float) -> bool:
        """
        Simulerar trade execution internt i agenten
        
        Args:
            symbol: Symbol att handla
            action: BUY eller SELL
            quantity: Antal
            price: Pris
            
        Returns:
            True om trade genomfördes, False annars
        """
        if action == 'BUY':
            cost = quantity * price
            if cost > self.capital:
                return False  # Insufficient funds
                
            self.capital -= cost
            self.positions[symbol] = self.positions.get(symbol, 0) + quantity
            
            # Update average entry price
            current_qty = self.positions[symbol] - quantity
            if current_qty > 0:
                old_avg = self.position_prices.get(symbol, price)
                new_avg = (old_avg * current_qty + price * quantity) / self.positions[symbol]
                self.position_prices[symbol] = new_avg
            else:
                self.position_prices[symbol] = price
                
            self.total_trades += 1
            return True
            
        elif action == 'SELL':
            if symbol not in self.positions or self.positions[symbol] < quantity:
                return False  # Insufficient holdings
                
            self.capital += quantity * price
            entry_price = self.position_prices.get(symbol, price)
            pnl = quantity * (price - entry_price)
            self.total_pnl += pnl
            
            if pnl > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
                
            self.positions[symbol] -= quantity
            if self.positions[symbol] == 0:
                del self.positions[symbol]
                if symbol in self.position_prices:
                    del self.position_prices[symbol]
                    
            self.total_trades += 1
            self.performance_history.append(pnl)
            return True
            
        return False
    
    def analyze_and_vote(self, symbol: str) -> Dict[str, Any]:
        """
        Analyserar symbol och genererar vote.
        Måste implementeras av subklasser.
        
        Args:
            symbol: Symbol att analysera
            
        Returns:
            Vote dictionary med action, confidence, reasoning
        """
        raise NotImplementedError("Subclass must implement analyze_and_vote")
    
    def publish_vote(self, vote: Dict[str, Any]) -> None:
        """Publicerar vote till ensemble system"""
        vote_data = {
            'agent_id': self.agent_id,
            'agent_performance': self.get_win_rate(),
            **vote,
            'timestamp': time.time()
        }
        self.message_bus.publish('decision_vote', vote_data)
    
    def publish_state(self) -> None:
        """Publicerar agent state för monitoring"""
        state = {
            'agent_id': self.agent_id,
            'capital': self.capital,
            'portfolio_value': self.get_portfolio_value(),
            'positions': self.positions.copy(),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.get_win_rate(),
            'total_pnl': self.total_pnl,
            'roi': self.get_roi(),
            'timestamp': time.time()
        }
        self.message_bus.publish('agent_state', state)
    
    def get_win_rate(self) -> float:
        """Beräknar win rate"""
        total = self.winning_trades + self.losing_trades
        if total == 0:
            return 0.5  # Neutral om ingen historik
        return self.winning_trades / total
    
    def get_roi(self) -> float:
        """Beräknar ROI"""
        if self.initial_capital == 0:
            return 0.0
        return (self.get_portfolio_value() - self.initial_capital) / self.initial_capital


class MomentumAgent(BaseSpecializedAgent):
    """Agent 1: Momentum - Följer stark prismomentum (Rate of Change)"""
    
    def __init__(self, message_bus, initial_capital: float = 10000.0):
        super().__init__('momentum_agent', message_bus, initial_capital)
        self.lookback_period = 10  # Periods för momentum-beräkning
        
    def analyze_and_vote(self, symbol: str) -> Dict[str, Any]:
        """Analyserar momentum och genererar vote"""
        indicators = self.indicator_data.get(symbol, {}).get('technical', {})
        
        # Använd RSI och ROC för momentum
        rsi = indicators.get('RSI', 50)
        
        # Beräkna price momentum om vi har historik
        market = self.market_data.get(symbol, {})
        current_price = market.get('price', 0)
        
        # Strong momentum signals
        if rsi > 60 and current_price > 0:
            # Bullish momentum
            confidence = min(0.9, (rsi - 50) / 50)
            return {
                'symbol': symbol,
                'action': 'BUY',
                'quantity': 2,
                'confidence': confidence,
                'reasoning': f'Strong bullish momentum (RSI: {rsi:.1f})'
            }
        elif rsi < 40:
            # Bearish momentum
            confidence = min(0.9, (50 - rsi) / 50)
            return {
                'symbol': symbol,
                'action': 'SELL',
                'quantity': 2,
                'confidence': confidence,
                'reasoning': f'Strong bearish momentum (RSI: {rsi:.1f})'
            }
        
        return {
            'symbol': symbol,
            'action': 'HOLD',
            'quantity': 0,
            'confidence': 0.5,
            'reasoning': 'Neutral momentum'
        }


class MeanReversionAgent(BaseSpecializedAgent):
    """Agent 2: Mean Reversion - Handlar på reversals till medelvärde"""
    
    def __init__(self, message_bus, initial_capital: float = 10000.0):
        super().__init__('mean_reversion_agent', message_bus, initial_capital)
        
    def analyze_and_vote(self, symbol: str) -> Dict[str, Any]:
        """Analyserar mean reversion och genererar vote"""
        indicators = self.indicator_data.get(symbol, {}).get('technical', {})
        
        rsi = indicators.get('RSI', 50)
        
        # Mean reversion på RSI extremer
        if rsi < 30:
            # Översåld - förvänta bounce
            confidence = min(0.9, (30 - rsi) / 30)
            return {
                'symbol': symbol,
                'action': 'BUY',
                'quantity': 2,
                'confidence': confidence,
                'reasoning': f'Oversold mean reversion signal (RSI: {rsi:.1f})'
            }
        elif rsi > 70:
            # Överköpt - förvänta correction
            confidence = min(0.9, (rsi - 70) / 30)
            return {
                'symbol': symbol,
                'action': 'SELL',
                'quantity': 2,
                'confidence': confidence,
                'reasoning': f'Overbought mean reversion signal (RSI: {rsi:.1f})'
            }
        
        return {
            'symbol': symbol,
            'action': 'HOLD',
            'quantity': 0,
            'confidence': 0.5,
            'reasoning': 'No mean reversion signal'
        }


class TrendFollowingAgent(BaseSpecializedAgent):
    """Agent 3: Trend Following - Identifierar och följer trender via MACD"""
    
    def __init__(self, message_bus, initial_capital: float = 10000.0):
        super().__init__('trend_following_agent', message_bus, initial_capital)
        
    def analyze_and_vote(self, symbol: str) -> Dict[str, Any]:
        """Analyserar trend och genererar vote"""
        indicators = self.indicator_data.get(symbol, {}).get('technical', {})
        
        macd_data = indicators.get('MACD', {})
        macd_histogram = macd_data.get('histogram', 0.0)
        
        # Trend följning via MACD
        if macd_histogram > 1.0:
            # Strong uptrend
            confidence = min(0.9, macd_histogram / 5.0)
            return {
                'symbol': symbol,
                'action': 'BUY',
                'quantity': 3,
                'confidence': confidence,
                'reasoning': f'Strong uptrend detected (MACD: {macd_histogram:.2f})'
            }
        elif macd_histogram < -1.0:
            # Strong downtrend
            confidence = min(0.9, abs(macd_histogram) / 5.0)
            return {
                'symbol': symbol,
                'action': 'SELL',
                'quantity': 3,
                'confidence': confidence,
                'reasoning': f'Strong downtrend detected (MACD: {macd_histogram:.2f})'
            }
        
        return {
            'symbol': symbol,
            'action': 'HOLD',
            'quantity': 0,
            'confidence': 0.5,
            'reasoning': 'No clear trend'
        }


class VolatilityAgent(BaseSpecializedAgent):
    """Agent 4: Volatility - Drar nytta av hög volatilitet via ATR"""
    
    def __init__(self, message_bus, initial_capital: float = 10000.0):
        super().__init__('volatility_agent', message_bus, initial_capital)
        
    def analyze_and_vote(self, symbol: str) -> Dict[str, Any]:
        """Analyserar volatilitet och genererar vote"""
        indicators = self.indicator_data.get(symbol, {}).get('technical', {})
        
        atr = indicators.get('ATR', 2.0)
        rsi = indicators.get('RSI', 50)
        
        # Handel på volatilitet - köp på dips i hög volatilitet
        if atr > 5.0:
            # High volatility
            if rsi < 45:
                # Buy the dip in volatile market
                confidence = min(0.8, atr / 10.0)
                return {
                    'symbol': symbol,
                    'action': 'BUY',
                    'quantity': 1,  # Mindre position pga risk
                    'confidence': confidence,
                    'reasoning': f'High volatility buy opportunity (ATR: {atr:.2f}, RSI: {rsi:.1f})'
                }
            elif rsi > 55:
                # Sell rally in volatile market
                confidence = min(0.8, atr / 10.0)
                return {
                    'symbol': symbol,
                    'action': 'SELL',
                    'quantity': 1,
                    'confidence': confidence,
                    'reasoning': f'High volatility sell opportunity (ATR: {atr:.2f}, RSI: {rsi:.1f})'
                }
        
        return {
            'symbol': symbol,
            'action': 'HOLD',
            'quantity': 0,
            'confidence': 0.5,
            'reasoning': 'Volatility not actionable'
        }


class BreakoutAgent(BaseSpecializedAgent):
    """Agent 5: Breakout - Handlar på tekniska breakouts"""
    
    def __init__(self, message_bus, initial_capital: float = 10000.0):
        super().__init__('breakout_agent', message_bus, initial_capital)
        
    def analyze_and_vote(self, symbol: str) -> Dict[str, Any]:
        """Analyserar breakouts och genererar vote"""
        indicators = self.indicator_data.get(symbol, {}).get('technical', {})
        
        rsi = indicators.get('RSI', 50)
        macd_data = indicators.get('MACD', {})
        macd_histogram = macd_data.get('histogram', 0.0)
        
        # Breakout detection - RSI över 65 med positiv MACD
        if rsi > 65 and macd_histogram > 0.5:
            # Bullish breakout
            confidence = min(0.9, (rsi - 50) / 50 * 0.7 + macd_histogram / 5.0 * 0.3)
            return {
                'symbol': symbol,
                'action': 'BUY',
                'quantity': 2,
                'confidence': confidence,
                'reasoning': f'Bullish breakout (RSI: {rsi:.1f}, MACD: {macd_histogram:.2f})'
            }
        elif rsi < 35 and macd_histogram < -0.5:
            # Bearish breakdown
            confidence = min(0.9, (50 - rsi) / 50 * 0.7 + abs(macd_histogram) / 5.0 * 0.3)
            return {
                'symbol': symbol,
                'action': 'SELL',
                'quantity': 2,
                'confidence': confidence,
                'reasoning': f'Bearish breakdown (RSI: {rsi:.1f}, MACD: {macd_histogram:.2f})'
            }
        
        return {
            'symbol': symbol,
            'action': 'HOLD',
            'quantity': 0,
            'confidence': 0.5,
            'reasoning': 'No breakout pattern'
        }


class SwingAgent(BaseSpecializedAgent):
    """Agent 6: Swing - Fångar swing-rörelser (2-5 dagar)"""
    
    def __init__(self, message_bus, initial_capital: float = 10000.0):
        super().__init__('swing_agent', message_bus, initial_capital)
        
    def analyze_and_vote(self, symbol: str) -> Dict[str, Any]:
        """Analyserar swing opportunities och genererar vote"""
        indicators = self.indicator_data.get(symbol, {}).get('technical', {})
        
        rsi = indicators.get('RSI', 50)
        macd_data = indicators.get('MACD', {})
        macd_histogram = macd_data.get('histogram', 0.0)
        
        # Swing trading - kombinera RSI och MACD för timing
        if 40 <= rsi <= 50 and macd_histogram > 0.3:
            # Early upswing
            confidence = 0.7
            return {
                'symbol': symbol,
                'action': 'BUY',
                'quantity': 2,
                'confidence': confidence,
                'reasoning': f'Early upswing detected (RSI: {rsi:.1f}, MACD: {macd_histogram:.2f})'
            }
        elif 50 <= rsi <= 60 and macd_histogram < -0.3:
            # Early downswing
            confidence = 0.7
            return {
                'symbol': symbol,
                'action': 'SELL',
                'quantity': 2,
                'confidence': confidence,
                'reasoning': f'Early downswing detected (RSI: {rsi:.1f}, MACD: {macd_histogram:.2f})'
            }
        
        return {
            'symbol': symbol,
            'action': 'HOLD',
            'quantity': 0,
            'confidence': 0.5,
            'reasoning': 'No swing signal'
        }


class ArbitrageAgent(BaseSpecializedAgent):
    """Agent 7: Arbitrage - Söker arbitragemöjligheter (simulated via volatility mismatch)"""
    
    def __init__(self, message_bus, initial_capital: float = 10000.0):
        super().__init__('arbitrage_agent', message_bus, initial_capital)
        self.price_history: Dict[str, deque] = {}
        
    def analyze_and_vote(self, symbol: str) -> Dict[str, Any]:
        """Analyserar arbitrage opportunities och genererar vote"""
        market = self.market_data.get(symbol, {})
        current_price = market.get('price', 0)
        
        # Track price history
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=5)
        self.price_history[symbol].append(current_price)
        
        if len(self.price_history[symbol]) < 3:
            return {
                'symbol': symbol,
                'action': 'HOLD',
                'quantity': 0,
                'confidence': 0.5,
                'reasoning': 'Building price history'
            }
        
        # Simplified arbitrage: detect rapid price changes that may revert
        prices = list(self.price_history[symbol])
        price_change = (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0
        
        if price_change > 0.02:  # 2% rapid increase
            # Price might be temporarily inflated
            confidence = min(0.8, abs(price_change) * 10)
            return {
                'symbol': symbol,
                'action': 'SELL',
                'quantity': 1,
                'confidence': confidence,
                'reasoning': f'Arbitrage opportunity - rapid price increase ({price_change*100:.2f}%)'
            }
        elif price_change < -0.02:  # 2% rapid decrease
            # Price might be temporarily deflated
            confidence = min(0.8, abs(price_change) * 10)
            return {
                'symbol': symbol,
                'action': 'BUY',
                'quantity': 1,
                'confidence': confidence,
                'reasoning': f'Arbitrage opportunity - rapid price decrease ({price_change*100:.2f}%)'
            }
        
        return {
            'symbol': symbol,
            'action': 'HOLD',
            'quantity': 0,
            'confidence': 0.5,
            'reasoning': 'No arbitrage opportunity'
        }


class SentimentAgent(BaseSpecializedAgent):
    """Agent 8: Sentiment - Baserat på marknadssentiment och analystdata"""
    
    def __init__(self, message_bus, initial_capital: float = 10000.0):
        super().__init__('sentiment_agent', message_bus, initial_capital)
        
    def analyze_and_vote(self, symbol: str) -> Dict[str, Any]:
        """Analyserar sentiment och genererar vote"""
        indicators = self.indicator_data.get(symbol, {})
        fundamental = indicators.get('fundamental', {})
        
        analyst_ratings = fundamental.get('AnalystRatings', {})
        analyst_consensus = analyst_ratings.get('consensus', 'HOLD')
        
        # Sentiment från analyst ratings
        if analyst_consensus in ['BUY', 'STRONG_BUY']:
            confidence = 0.8 if analyst_consensus == 'STRONG_BUY' else 0.7
            return {
                'symbol': symbol,
                'action': 'BUY',
                'quantity': 2,
                'confidence': confidence,
                'reasoning': f'Positive analyst sentiment: {analyst_consensus}'
            }
        elif analyst_consensus == 'SELL':
            confidence = 0.7
            return {
                'symbol': symbol,
                'action': 'SELL',
                'quantity': 2,
                'confidence': confidence,
                'reasoning': f'Negative analyst sentiment: {analyst_consensus}'
            }
        
        return {
            'symbol': symbol,
            'action': 'HOLD',
            'quantity': 0,
            'confidence': 0.5,
            'reasoning': f'Neutral analyst sentiment: {analyst_consensus}'
        }


class SpecializedAgentsCoordinator:
    """Koordinerar alla 8 specialiserade agenter"""
    
    def __init__(self, message_bus, initial_capital_per_agent: float = 10000.0):
        """
        Initialiserar coordinator för alla agenter
        
        Args:
            message_bus: Referens till central message_bus
            initial_capital_per_agent: Startkapital för varje agent
        """
        self.message_bus = message_bus
        
        # Skapa alla 8 agenter
        self.agents = [
            MomentumAgent(message_bus, initial_capital_per_agent),
            MeanReversionAgent(message_bus, initial_capital_per_agent),
            TrendFollowingAgent(message_bus, initial_capital_per_agent),
            VolatilityAgent(message_bus, initial_capital_per_agent),
            BreakoutAgent(message_bus, initial_capital_per_agent),
            SwingAgent(message_bus, initial_capital_per_agent),
            ArbitrageAgent(message_bus, initial_capital_per_agent),
            SentimentAgent(message_bus, initial_capital_per_agent)
        ]
        
        # Prenumerera på market_data för att trigga analyser
        self.message_bus.subscribe('market_data', self._on_market_data)
        
        # Track last analysis timestamp per symbol to avoid spam
        self.last_analysis: Dict[str, float] = {}
        self.analysis_cooldown = 2.0  # Sekunder mellan analyser per symbol
        
    def _on_market_data(self, data: Dict[str, Any]) -> None:
        """Callback för market data - triggar agent-analyser"""
        symbol = data.get('symbol')
        if not symbol:
            return
        
        # Check cooldown
        current_time = time.time()
        if symbol in self.last_analysis:
            if current_time - self.last_analysis[symbol] < self.analysis_cooldown:
                return  # Too soon
        
        self.last_analysis[symbol] = current_time
        
        # Låt alla agenter analysera och rösta
        self.analyze_and_vote_all(symbol)
    
    def analyze_and_vote_all(self, symbol: str) -> None:
        """Låter alla agenter analysera och rösta för en symbol"""
        for agent in self.agents:
            vote = agent.analyze_and_vote(symbol)
            agent.publish_vote(vote)
            agent.publish_state()
    
    def get_aggregated_statistics(self) -> Dict[str, Any]:
        """Hämtar aggregerad statistik för alla agenter"""
        total_capital = sum(agent.capital for agent in self.agents)
        total_portfolio_value = sum(agent.get_portfolio_value() for agent in self.agents)
        total_trades = sum(agent.total_trades for agent in self.agents)
        
        agent_stats = []
        for agent in self.agents:
            agent_stats.append({
                'agent_id': agent.agent_id,
                'capital': agent.capital,
                'portfolio_value': agent.get_portfolio_value(),
                'positions': len(agent.positions),
                'total_trades': agent.total_trades,
                'win_rate': agent.get_win_rate(),
                'roi': agent.get_roi(),
                'total_pnl': agent.total_pnl
            })
        
        return {
            'total_capital': total_capital,
            'total_portfolio_value': total_portfolio_value,
            'total_trades': total_trades,
            'num_agents': len(self.agents),
            'agent_statistics': agent_stats,
            'timestamp': time.time()
        }
