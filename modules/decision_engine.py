"""
decision_engine.py - Beslutsmotor för handelsbeslut

Beskrivning:
    Samlar insikter från strategi, risk och minne för att fatta slutgiltiga handelsbeslut.
    Kombinerar olika perspektiv och använder RL för att optimera beslutsprocessen.

Roll:
    - Samlar trade_proposal från strategy_engine
    - Integrerar risk_profile från risk_manager
    - Använder memory_insights från strategic_memory_engine
    - Fattar final_decision baserat på samtliga inputs
    - Tränas av RL-controller för beslutsoptimering

Inputs:
    - trade_proposal: Dict - Tradeförslag från strategy_engine
    - risk_profile: Dict - Riskbedömning från risk_manager
    - memory_insights: Dict - Historiska lärdomar från strategic_memory_engine

Outputs:
    - final_decision: Dict - Slutgiltigt handelsbeslut med confidence score

Publicerar till message_bus:
    - decision_vote: Beslut för vote_engine (används i Sprint 5)

Prenumererar på (från functions.yaml):
    - trade_proposal (från strategy_engine)
    - risk_profile (från risk_manager)
    - memory_insights (från strategic_memory_engine)

Använder RL: Ja (från functions.yaml)
Tar emot feedback: Ja (från execution_engine, consensus_engine)

Anslutningar (från flowchart.yaml - decision_flow):
    Från:
    - strategy_engine (trade_proposal)
    - risk_manager (risk_profile)
    - strategic_memory_engine (memory_insights)
    Till:
    - vote_engine (decision_vote)
    - strategic_memory_engine (för loggning)

RL-anslutning (från feedback_loop.yaml):
    - Tar emot agent_update från rl_controller
    - Belöning baserad på decision outcome och portfolio performance

Indikatorer från indicator_map.yaml:
    Använder:
    - RSI: overbought/oversold detection (från strategy)
    - ATR: volatility-based risk adjustment (från risk)

Används i Sprint: 1, 2, 4, 5
"""

from typing import Dict, Any, Optional


class DecisionEngine:
    """Fattar slutgiltiga handelsbeslut baserat på strategi, risk och minne."""
    
    def __init__(self, message_bus):
        """
        Initialiserar beslutsmotorn.
        
        Args:
            message_bus: Referens till central message_bus
        """
        self.message_bus = message_bus
        self.trade_proposals: Dict[str, Any] = {}
        self.risk_profiles: Dict[str, Any] = {}
        self.memory_insights: Optional[Dict[str, Any]] = None
        self.rl_agent = None  # Kommer från rl_controller i Sprint 2
        
        # Prenumerera på relevanta topics
        self.message_bus.subscribe('decision_proposal', self._on_trade_proposal)
        self.message_bus.subscribe('risk_profile', self._on_risk_profile)
        self.message_bus.subscribe('memory_insights', self._on_memory_insights)
    
    def _on_trade_proposal(self, proposal: Dict[str, Any]) -> None:
        """
        Callback för tradeförslag från strategy_engine.
        
        Args:
            proposal: Tradeförslag för en symbol
        """
        symbol = proposal.get('symbol')
        self.trade_proposals[symbol] = proposal
    
    def _on_risk_profile(self, profile: Dict[str, Any]) -> None:
        """
        Callback för riskprofil från risk_manager.
        
        Args:
            profile: Riskbedömning för en symbol
        """
        symbol = profile.get('symbol')
        self.risk_profiles[symbol] = profile
    
    def _on_memory_insights(self, insights: Dict[str, Any]) -> None:
        """
        Callback för minnesinsikter från strategic_memory_engine.
        
        Args:
            insights: Historiska lärdomar och mönster
        """
        self.memory_insights = insights
    
    def make_decision(self, symbol: str) -> Dict[str, Any]:
        """
        Fattar slutgiltigt beslut för en symbol baserat på alla inputs.
        
        Args:
            symbol: Aktiesymbol att fatta beslut för
            
        Returns:
            Dict med final_decision (action, symbol, quantity, confidence, reasoning)
        """
        proposal = self.trade_proposals.get(symbol, {})
        risk_profile = self.risk_profiles.get(symbol, {})
        
        # Stub: Enkel logik för Sprint 1
        # I Sprint 2 kommer RL-agent att användas här
        decision = {
            'symbol': symbol,
            'action': proposal.get('action', 'HOLD'),
            'quantity': proposal.get('quantity', 0),
            'confidence': 0.5,
            'reasoning': 'Baserat på strategi och risk'
        }
        
        # Justera baserat på risk
        risk_level = risk_profile.get('risk_level', 'MEDIUM')
        if risk_level == 'HIGH':
            # Minska quantity vid hög risk
            decision['quantity'] = int(decision['quantity'] * 0.5)
            decision['reasoning'] += f', reducerad position pga hög risk'
            decision['confidence'] *= 0.8
        elif risk_level == 'LOW':
            # Öka confidence vid låg risk
            decision['confidence'] *= 1.2
            decision['confidence'] = min(decision['confidence'], 1.0)
        
        return decision
    
    def publish_decision(self, decision: Dict[str, Any]) -> None:
        """
        Publicerar beslut till message_bus.
        
        Args:
            decision: Slutgiltigt beslut att publicera
        """
        # Publicera som decision_vote (för vote_engine i Sprint 5)
        self.message_bus.publish('decision_vote', decision)
        
        # Publicera också som final_decision (för execution i Sprint 1)
        self.message_bus.publish('final_decision', decision)
    
    def receive_feedback(self, feedback: Dict[str, Any]) -> None:
        """
        Tar emot feedback om beslut från execution och consensus.
        
        Args:
            feedback: Feedback om decision outcome
        """
        # Stub: Implementeras fullt ut i Sprint 3
        pass
    
    def update_from_rl(self, agent_update: Dict[str, Any]) -> None:
        """
        Uppdaterar beslutslogik baserat på RL-träning.
        
        Args:
            agent_update: Uppdatering från rl_controller
        """
        # Stub: Implementeras i Sprint 2
        pass

