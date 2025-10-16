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
    - final_decision: Slutgiltigt handelsbeslut för exekvering/loggning
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
        self.portfolio_status: Optional[Dict[str, Any]] = None
        self.rl_agent = None  # Uppdateras från rl_controller
        self.rl_enabled = False
        self.agent_performance = 0.0
        
        # Prenumerera på relevanta topics
        self.message_bus.subscribe('decision_proposal', self._on_trade_proposal)
        self.message_bus.subscribe('risk_profile', self._on_risk_profile)
        self.message_bus.subscribe('memory_insights', self._on_memory_insights)
        self.message_bus.subscribe('portfolio_status', self._on_portfolio_status)
        self.message_bus.subscribe('agent_update', self._on_agent_update)
    
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
    
    def _on_portfolio_status(self, status: Dict[str, Any]) -> None:
        """
        Callback för portfolio status från portfolio_manager.
        
        Args:
            status: Aktuell portföljstatus
        """
        self.portfolio_status = status
    
    def _on_agent_update(self, update: Dict[str, Any]) -> None:
        """
        Callback för agent updates från rl_controller.
        
        Args:
            update: Agent update med nya parametrar
        """
        if update.get('module') == 'decision_engine':
            self.rl_enabled = update.get('policy_updated', False)
            metrics = update.get('metrics', {})
            self.agent_performance = metrics.get('average_reward', 0.0)
    
    def make_decision(self, symbol: str, current_price: float = None) -> Dict[str, Any]:
        """
        Fattar slutgiltigt beslut för en symbol baserat på alla inputs.
        
        Args:
            symbol: Aktiesymbol att fatta beslut för
            current_price: Aktuellt pris för insufficient funds-check
            
        Returns:
            Dict med final_decision (action, symbol, quantity, confidence, reasoning)
        """
        proposal = self.trade_proposals.get(symbol, {})
        risk_profile = self.risk_profiles.get(symbol, {})
        
        # Sprint 2: Förbättrad beslutslogik med risk-justering och RL
        decision = {
            'symbol': symbol,
            'action': proposal.get('action', 'HOLD'),
            'quantity': proposal.get('quantity', 0),
            'confidence': proposal.get('confidence', 0.5),
            'reasoning': proposal.get('reasoning', 'Baserat på strategi'),
            'rl_enabled': self.rl_enabled
        }
        
        # Justera baserat på risk
        risk_level = risk_profile.get('risk_level', 'MEDIUM')
        risk_confidence = risk_profile.get('confidence', 0.5)
        
        if risk_level == 'HIGH':
            # Minska quantity vid hög risk (men minimum 1)
            original_qty = decision['quantity']
            decision['quantity'] = max(1, int(decision['quantity'] * 0.5))
            if decision['quantity'] != original_qty:
                decision['reasoning'] += f', reducerad position från {original_qty} till {decision["quantity"]} pga hög risk'
            decision['confidence'] *= 0.7
            
            # Vid mycket hög risk, överväg att avbryta
            if risk_confidence > 0.7 and decision['action'] in ['BUY', 'SELL']:
                decision['action'] = 'HOLD'
                decision['quantity'] = 0
                decision['reasoning'] = f'Avbrutet: {decision["reasoning"]} (för hög risk)'
                decision['confidence'] = 0.3
                
        elif risk_level == 'LOW':
            # Öka confidence vid låg risk
            decision['confidence'] = min(1.0, decision['confidence'] * 1.2)
            decision['reasoning'] += ' (låg risk, hög confidence)'
            
        elif risk_level == 'MEDIUM':
            # Balansera confidence
            decision['confidence'] = (proposal.get('confidence', 0.5) + risk_confidence) / 2
        
        # Kontrollera insufficient funds för BUY-beslut
        if decision['action'] == 'BUY' and current_price and self.portfolio_status:
            available_cash = self.portfolio_status.get('cash', 0)
            estimated_cost = current_price * decision['quantity'] * 1.0025  # Inkludera 0.25% fee
            
            if estimated_cost > available_cash:
                # Justera quantity så det passar i budget
                max_affordable_qty = int(available_cash / (current_price * 1.0025))
                if max_affordable_qty >= 1:
                    decision['quantity'] = max_affordable_qty
                    decision['reasoning'] += f', justerad till {max_affordable_qty} aktier pga tillgängligt kapital'
                else:
                    # Kan inte köpa ens 1 aktie
                    decision['action'] = 'HOLD'
                    decision['quantity'] = 0
                    decision['reasoning'] = 'Avbrutet: Otillräckligt kapital'
                    decision['confidence'] = 0.0
        
        # Kontrollera insufficient holdings för SELL-beslut
        if decision['action'] == 'SELL' and self.portfolio_status:
            positions = self.portfolio_status.get('positions', {})
            if symbol not in positions or positions[symbol]['quantity'] < decision['quantity']:
                available_qty = positions.get(symbol, {}).get('quantity', 0)
                if available_qty > 0:
                    decision['quantity'] = available_qty
                    decision['reasoning'] += f', justerad till {available_qty} aktier (allt vi äger)'
                else:
                    decision['action'] = 'HOLD'
                    decision['quantity'] = 0
                    decision['reasoning'] = 'Avbrutet: Inga aktier att sälja'
                    decision['confidence'] = 0.0
        
        # RL-justering (om aktiverad)
        if self.rl_enabled and self.agent_performance > 0:
            # Förbättra confidence om RL-agent presterar bra
            decision['confidence'] = min(1.0, decision['confidence'] * 1.15)
            decision['reasoning'] += f' [RL-optimerat, perf: {self.agent_performance:.2f}]'
        elif self.rl_enabled and self.agent_performance < -0.5:
            # Minska confidence om RL-agent presterar dåligt
            decision['confidence'] *= 0.8
            decision['reasoning'] += ' [RL försiktig pga negativ trend]'
        
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

