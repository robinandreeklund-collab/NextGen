"""
strategy_engine.py - Strategimotor för tradeförslag

Beskrivning:
    Genererar tradeförslag baserat på indikatorer, portföljstatus och RL-belöning.
    Använder tekniska och fundamentala indikatorer för att identifiera handelsmöjligheter.

Roll:
    - Analyserar indikatordata från indicator_registry
    - Genererar tradeförslag (BUY, SELL, HOLD) med motivering
    - Tränas av RL-controller för att optimera strategier
    - Tar emot feedback från execution och portfolio för att förbättra beslut

Inputs:
    - indicator_data: Dict - Tekniska, fundamentala och alternativa indikatorer
    - portfolio_status: Dict - Nuvarande portföljposition och kapital

Outputs:
    - trade_proposal: Dict - Förslag med action (BUY/SELL/HOLD), symbol, quantity, reasoning

Publicerar till message_bus:
    - decision_proposal: Trade-förslag till decision_engine

Prenumererar på (från functions.yaml):
    - indicator_data (från indicator_registry)
    - portfolio_status (från portfolio_manager)

Använder RL: Ja (från functions.yaml)
Tar emot feedback: Ja (från execution_engine, portfolio_manager)

Anslutningar (från flowchart.yaml - strategy_flow):
    Från: indicator_registry (indicator_data)
    Till: 
    - decision_engine (trade_proposal)
    - decision_simulator (för sandbox-testning)

RL-anslutning (från feedback_loop.yaml):
    - Tar emot agent_update från rl_controller
    - Belöning baserad på trade outcome och portfolio performance

Indikatorer från indicator_map.yaml för Sprint 1:
    Använder:
    - OHLC: price analysis, entry/exit signals
    - Volume: liquidity assessment
    - SMA: trend detection, smoothing
    - RSI: overbought/oversold detection

Indikatorer från indicator_map.yaml för Sprint 2:
    Använder:
    - MACD: momentum and trend strength
    - ATR: volatility-based risk adjustment (för quantity-justering)
    - Analyst Ratings: external confidence and sentiment

Används i Sprint: 1, 2
"""

from typing import Dict, Any


class StrategyEngine:
    """Genererar tradeförslag baserat på indikatorer och RL-agenter."""
    
    def __init__(self, message_bus):
        """
        Initialiserar strategimotorn.
        
        Args:
            message_bus: Referens till central message_bus
        """
        self.message_bus = message_bus
        self.current_indicators: Dict[str, Any] = {}
        self.portfolio_status: Dict[str, Any] = {}
        self.rl_agent = None  # Uppdateras från rl_controller
        self.rl_enabled = False
        self.agent_performance = 0.0
        
        # Prenumerera på indicator_data, portfolio_status och agent_update
        self.message_bus.subscribe('indicator_data', self._on_indicator_data)
        self.message_bus.subscribe('portfolio_status', self._on_portfolio_status)
        self.message_bus.subscribe('agent_update', self._on_agent_update)
    
    def _on_indicator_data(self, data: Dict[str, Any]) -> None:
        """
        Callback för nya indikatordata från indicator_registry.
        
        Args:
            data: Indikatordata för en symbol
        """
        symbol = data.get('symbol')
        self.current_indicators[symbol] = data
    
    def _on_portfolio_status(self, status: Dict[str, Any]) -> None:
        """
        Callback för portföljstatus från portfolio_manager.
        
        Args:
            status: Aktuell portföljstatus (kapital, positioner, etc.)
        """
        self.portfolio_status = status
    
    def _on_agent_update(self, update: Dict[str, Any]) -> None:
        """
        Callback för agent updates från rl_controller.
        
        Args:
            update: Agent update med nya parametrar
        """
        if update.get('module') == 'strategy_engine':
            self.rl_enabled = update.get('policy_updated', False)
            metrics = update.get('metrics', {})
            self.agent_performance = metrics.get('average_reward', 0.0)
    
    def generate_proposal(self, symbol: str) -> Dict[str, Any]:
        """
        Genererar ett tradeförslag för en symbol.
        
        Args:
            symbol: Aktiesymbol att analysera
            
        Returns:
            Dict med trade_proposal (action, symbol, quantity, reasoning)
        """
        indicators = self.current_indicators.get(symbol, {})
        
        # Sprint 2: Använd både regel-baserad logik och RL
        proposal = {
            'symbol': symbol,
            'action': 'HOLD',
            'quantity': 0,
            'reasoning': 'Väntar på signal',
            'confidence': 0.5,
            'rl_enabled': self.rl_enabled
        }
        
        if not indicators:
            return proposal
        
        technical = indicators.get('technical', {})
        fundamental = indicators.get('fundamental', {})
        
        # Hämta indikatorer
        rsi = technical.get('RSI', 50)
        macd_data = technical.get('MACD', {})
        macd_histogram = macd_data.get('histogram', 0.0)
        atr = technical.get('ATR', 2.0)
        
        analyst_ratings = fundamental.get('AnalystRatings', {})
        analyst_consensus = analyst_ratings.get('consensus', 'HOLD')
        
        # Kombinerad analys med flera indikatorer
        buy_signals = 0
        sell_signals = 0
        reasons = []
        
        # RSI-analys
        if rsi < 30:
            buy_signals += 2  # Stark köpsignal
            reasons.append(f'RSI översåld ({rsi:.1f})')
        elif rsi > 70:
            sell_signals += 2  # Stark säljsignal
            reasons.append(f'RSI överköpt ({rsi:.1f})')
        
        # MACD-analys (Sprint 2)
        if macd_histogram > 0.5:
            buy_signals += 1
            reasons.append(f'MACD positiv ({macd_histogram:.2f})')
        elif macd_histogram < -0.5:
            sell_signals += 1
            reasons.append(f'MACD negativ ({macd_histogram:.2f})')
        
        # Analyst Ratings-analys (Sprint 2)
        if analyst_consensus in ['BUY', 'STRONG_BUY']:
            buy_signals += 1
            reasons.append(f'Analystconsensus: {analyst_consensus}')
        elif analyst_consensus == 'SELL':
            sell_signals += 1
            reasons.append(f'Analystconsensus: {analyst_consensus}')
        
        # Fatta beslut baserat på signaler
        if buy_signals > sell_signals and buy_signals >= 2:
            # Justera quantity baserat på ATR (volatilitet)
            base_quantity = 10
            if atr > 5.0:  # Hög volatilitet
                quantity = max(5, int(base_quantity * 0.7))
                reasons.append(f'Reducerad position pga hög volatilitet (ATR: {atr:.1f})')
            else:
                quantity = base_quantity
            
            proposal['action'] = 'BUY'
            proposal['quantity'] = quantity
            proposal['reasoning'] = ', '.join(reasons)
            proposal['confidence'] = min(0.9, 0.5 + (buy_signals * 0.1))
            
        elif sell_signals > buy_signals and sell_signals >= 2:
            proposal['action'] = 'SELL'
            proposal['quantity'] = 10
            proposal['reasoning'] = ', '.join(reasons)
            proposal['confidence'] = min(0.9, 0.5 + (sell_signals * 0.1))
        
        # RL-justering (om aktiverad)
        if self.rl_enabled and self.agent_performance > 0:
            # Öka confidence om RL-agent presterar bra
            proposal['confidence'] = min(1.0, proposal['confidence'] * 1.1)
            proposal['reasoning'] += f' (RL-förstärkt, perf: {self.agent_performance:.2f})'
        
        return proposal
    
    def publish_proposal(self, proposal: Dict[str, Any]) -> None:
        """
        Publicerar tradeförslag till message_bus.
        
        Args:
            proposal: Tradeförslag att publicera
        """
        self.message_bus.publish('decision_proposal', proposal)
    
    def receive_feedback(self, feedback: Dict[str, Any]) -> None:
        """
        Tar emot feedback från execution och portfolio.
        
        Args:
            feedback: Feedback om trade outcome och performance
        """
        # Stub: Skulle användas för att justera strategier
        # I Sprint 3 kommer feedback_router att hantera detta
        pass
    
    def update_from_rl(self, agent_update: Dict[str, Any]) -> None:
        """
        Uppdaterar strategi baserat på RL-träning.
        
        Args:
            agent_update: Uppdatering från rl_controller
        """
        # Stub: Implementeras i Sprint 2
        pass

