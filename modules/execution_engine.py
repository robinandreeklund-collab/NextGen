"""
execution_engine.py - Exekveringsmotor för handel

Beskrivning:
    Exekverar eller simulerar trades och loggar resultat.
    Hanterar faktisk orderhantering och slippage-simulering.

Roll:
    - Tar emot final_decision från decision_engine
    - Simulerar eller exekverar trade mot marknad/broker
    - Beräknar slippage och execution cost
    - Loggar execution_result för portfolio_manager
    - Genererar feedback om trade outcome

Inputs:
    - final_decision: Dict - Slutgiltigt beslut från decision_engine/consensus_engine

Outputs:
    - execution_result: Dict - Resultat av trade (executed_price, quantity, cost, slippage)

Publicerar till message_bus:
    - trade_log: Loggdata för varje exekverad trade

Prenumererar på (från functions.yaml):
    - final_decision (från decision_engine eller consensus_engine)

Använder RL: Ja (från functions.yaml)
Tar emot feedback: Ja (från portfolio_manager)

Anslutningar (från flowchart.yaml - execution_flow):
    Från: consensus_engine (eller decision_engine i Sprint 1)
    Till:
    - portfolio_manager (execution_result)
    - feedback_router (trade outcome feedback)
    - strategic_memory_engine (för loggning)

RL-anslutning (från feedback_loop.yaml):
    - Tar emot agent_update från rl_controller
    - Genererar feedback_event med triggers: trade_result, slippage, latency
    - Belöning baserad på execution quality

Feedback-generering (från feedback_loop.yaml):
    Triggers:
    - trade_result: Om trade lyckades/misslyckades
    - slippage: Skillnad mellan expected och executed price
    - latency: Tidsfördröjning i execution

Indikatorer från indicator_map.yaml:
    Använder:
    - OHLC: entry/exit signals
    - VWAP: price fairness, execution quality

Används i Sprint: 1, 2, 3
"""

from typing import Dict, Any
import random


class ExecutionEngine:
    """Hanterar exekvering av trades och resultatloggning."""
    
    def __init__(self, message_bus, simulation_mode: bool = True):
        """
        Initialiserar exekveringsmotorn.
        
        Args:
            message_bus: Referens till central message_bus
            simulation_mode: Om True, simuleras trades istället för verklig exekvering
        """
        self.message_bus = message_bus
        self.simulation_mode = simulation_mode
        self.rl_agent = None  # Kommer från rl_controller i Sprint 2
        
        # Prenumerera på final_decision
        self.message_bus.subscribe('final_decision', self._on_final_decision)
    
    def _on_final_decision(self, decision: Dict[str, Any]) -> None:
        """
        Callback för slutgiltigt beslut från decision_engine.
        
        Args:
            decision: Handelsbeslut att exekvera
        """
        if decision.get('action') != 'HOLD':
            result = self.execute_trade(decision)
            self.publish_result(result)
    
    def execute_trade(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Exekverar en trade (simulerad eller verklig).
        
        Args:
            decision: Handelsbeslut med action, symbol, quantity
            
        Returns:
            Dict med execution_result (executed_price, quantity, cost, slippage, success)
        """
        symbol = decision['symbol']
        action = decision['action']
        quantity = decision['quantity']
        
        # Stub: Simulerad exekvering för Sprint 1
        # I produktion skulle detta anropa broker API
        
        # Simulera market price (i verkligheten från market data)
        market_price = 150.0
        
        # Simulera slippage (0-0.5%)
        slippage_pct = random.uniform(0, 0.005)
        if action == 'BUY':
            executed_price = market_price * (1 + slippage_pct)
        else:  # SELL
            executed_price = market_price * (1 - slippage_pct)
        
        # Beräkna total cost
        total_cost = executed_price * quantity
        
        result = {
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'market_price': market_price,
            'executed_price': executed_price,
            'slippage': slippage_pct,
            'total_cost': total_cost,
            'success': True,
            'timestamp': 'timestamp_placeholder'
        }
        
        # Generera feedback för feedback_loop
        self.generate_feedback(result)
        
        return result
    
    def publish_result(self, result: Dict[str, Any]) -> None:
        """
        Publicerar execution result till message_bus.
        
        Args:
            result: Execution result att publicera
        """
        # Publicera till portfolio_manager
        self.message_bus.publish('execution_result', result)
        
        # Logga trade
        self.message_bus.publish('trade_log', result)
    
    def generate_feedback(self, result: Dict[str, Any]) -> None:
        """
        Genererar feedback om execution quality (från feedback_loop.yaml).
        
        Args:
            result: Execution result att analysera
        """
        feedback = {
            'source': 'execution_engine',
            'triggers': [],
            'data': {}
        }
        
        # Trade result feedback
        if result['success']:
            feedback['triggers'].append('trade_result')
            feedback['data']['trade_result'] = 'success'
        else:
            feedback['triggers'].append('trade_result')
            feedback['data']['trade_result'] = 'failure'
        
        # Slippage feedback
        if result['slippage'] > 0.002:  # >0.2%
            feedback['triggers'].append('slippage')
            feedback['data']['slippage'] = result['slippage']
        
        # Latency feedback (stub)
        feedback['triggers'].append('latency')
        feedback['data']['latency_ms'] = 50  # Placeholder
        
        # Publicera feedback till feedback_router
        self.message_bus.publish('feedback_event', feedback)
    
    def receive_feedback(self, feedback: Dict[str, Any]) -> None:
        """
        Tar emot feedback från portfolio_manager om trade performance.
        
        Args:
            feedback: Feedback om execution quality
        """
        # Stub: Implementeras fullt ut i Sprint 3
        pass
    
    def update_from_rl(self, agent_update: Dict[str, Any]) -> None:
        """
        Uppdaterar execution strategi baserat på RL-träning.
        
        Args:
            agent_update: Uppdatering från rl_controller
        """
        # Stub: Implementeras i Sprint 2
        pass

