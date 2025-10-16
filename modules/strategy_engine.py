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
        self.rl_agent = None  # Kommer från rl_controller i Sprint 2
        
        # Prenumerera på indicator_data och portfolio_status
        self.message_bus.subscribe('indicator_data', self._on_indicator_data)
        self.message_bus.subscribe('portfolio_status', self._on_portfolio_status)
    
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
    
    def generate_proposal(self, symbol: str) -> Dict[str, Any]:
        """
        Genererar ett tradeförslag för en symbol.
        
        Args:
            symbol: Aktiesymbol att analysera
            
        Returns:
            Dict med trade_proposal (action, symbol, quantity, reasoning)
        """
        indicators = self.current_indicators.get(symbol, {})
        
        # Stub: Enkel regelbaserad logik för Sprint 1
        # I Sprint 2 kommer RL-agent att användas här
        proposal = {
            'symbol': symbol,
            'action': 'HOLD',
            'quantity': 0,
            'reasoning': 'Väntar på signal',
            'confidence': 0.5
        }
        
        # Exempel på indikatoranalys (stub)
        if indicators:
            technical = indicators.get('technical', {})
            rsi = technical.get('RSI', 50)
            
            if rsi < 30:
                proposal['action'] = 'BUY'
                proposal['quantity'] = 10
                proposal['reasoning'] = f'RSI översåld ({rsi}), köpsignal'
                proposal['confidence'] = 0.75
            elif rsi > 70:
                proposal['action'] = 'SELL'
                proposal['quantity'] = 10
                proposal['reasoning'] = f'RSI överköpt ({rsi}), säljsignal'
                proposal['confidence'] = 0.75
        
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

