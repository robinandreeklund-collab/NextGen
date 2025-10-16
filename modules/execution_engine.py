"""
execution_engine.py - Exekveringsmotor för handel

Beskrivning:
    Exekverar eller simulerar trades och loggar resultat.
    Hanterar faktisk orderhantering och slippage-simulering.
    Sprint 4.3: Använder adaptiva parametrar (execution_delay, slippage_tolerance).

Roll:
    - Tar emot final_decision från decision_engine
    - Simulerar eller exekverar trade mot marknad/broker
    - Beräknar slippage och execution cost
    - Loggar execution_result för portfolio_manager
    - Genererar feedback om trade outcome
    - Använder adaptiva parametrar från rl_controller (Sprint 4.3)

Inputs:
    - final_decision: Dict - Slutgiltigt beslut från decision_engine/consensus_engine
    - parameter_adjustment: Dict - Adaptiva parametrar från rl_controller (Sprint 4.3)

Outputs:
    - execution_result: Dict - Resultat av trade (executed_price, quantity, cost, slippage)

Publicerar till message_bus:
    - trade_log: Loggdata för varje exekverad trade
    - execution_result: Resultat av exekverad trade (executed_price, quantity, cost, slippage)

Prenumererar på (från functions.yaml):
    - final_decision (från decision_engine eller consensus_engine)
    - parameter_adjustment (från rl_controller, Sprint 4.3)

Använder RL: Ja (från functions.yaml)
Tar emot feedback: Ja (från portfolio_manager)
Har adaptiva parametrar: Ja (från functions_2.yaml, Sprint 4.3)

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

Adaptiva parametrar (Sprint 4.3):
    - execution_delay (0-10, default 0): Fördröjning för optimal timing
    - slippage_tolerance (0.001-0.05, default 0.01): Tolerans för slippage

Feedback-generering (från feedback_loop.yaml):
    Triggers:
    - trade_result: Om trade lyckades/misslyckades
    - slippage: Skillnad mellan expected och executed price
    - latency: Tidsfördröjning i execution

Indikatorer från indicator_map.yaml:
    Använder:
    - OHLC: entry/exit signals
    - VWAP: price fairness, execution quality

Används i Sprint: 1, 2, 3, 4.3
"""

from typing import Dict, Any
import random
import time


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
        
        # Sprint 4.3: Adaptiva parametrar
        self.execution_delay = 0  # Default från adaptive_parameters.yaml
        self.slippage_tolerance = 0.01  # Default från adaptive_parameters.yaml
        
        # Prenumerera på final_decision och parameter_adjustment
        self.message_bus.subscribe('final_decision', self._on_final_decision)
        self.message_bus.subscribe('parameter_adjustment', self._on_parameter_adjustment)
    
    def _on_final_decision(self, decision: Dict[str, Any]) -> None:
        """
        Callback för slutgiltigt beslut från decision_engine.
        
        Args:
            decision: Handelsbeslut att exekvera
        """
        if decision.get('action') != 'HOLD':
            result = self.execute_trade(decision)
            self.publish_result(result)
    
    def _on_parameter_adjustment(self, adjustment: Dict[str, Any]) -> None:
        """
        Callback för parameter adjustments från rl_controller (Sprint 4.3).
        
        Args:
            adjustment: Justerade parametrar med värden
        """
        params = adjustment.get('parameters', {})
        
        # Uppdatera execution_delay om justerad
        if 'execution_delay' in params:
            self.execution_delay = params['execution_delay']
        
        # Uppdatera slippage_tolerance om justerad
        if 'slippage_tolerance' in params:
            self.slippage_tolerance = params['slippage_tolerance']
    
    def execute_trade(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Exekverar en trade (simulerad eller verklig).
        Sprint 4.3: Använder adaptiva parametrar (execution_delay, slippage_tolerance).
        
        Args:
            decision: Handelsbeslut med action, symbol, quantity, current_price
            
        Returns:
            Dict med execution_result (executed_price, quantity, cost, slippage, success)
        """
        symbol = decision['symbol']
        action = decision['action']
        quantity = decision['quantity']
        
        # Sprint 4.3: Använd adaptiv execution_delay för timing
        if self.execution_delay > 0:
            time.sleep(self.execution_delay)
        
        # Använd aktuellt pris från decision (krävs nu)
        market_price = decision.get('current_price')
        if market_price is None:
            return {
                'symbol': symbol,
                'action': action,
                'quantity': 0,
                'market_price': 0,
                'executed_price': 0,
                'slippage': 0,
                'total_cost': 0,
                'success': False,
                'error': 'No market price available',
                'timestamp': self._get_timestamp(),
                'execution_delay': self.execution_delay,  # Sprint 4.3
                'slippage_tolerance': self.slippage_tolerance  # Sprint 4.3
            }
        
        # Sprint 4.3: Simulera slippage med adaptiv slippage_tolerance som max
        # Slippage mellan 0 och slippage_tolerance
        slippage_pct = random.uniform(0, self.slippage_tolerance)
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
            'timestamp': self._get_timestamp(),
            'execution_delay': self.execution_delay,  # Sprint 4.3
            'slippage_tolerance': self.slippage_tolerance  # Sprint 4.3
        }
        
        # Generera feedback för feedback_loop
        self.generate_feedback(result)
        
        return result
    
    def _get_timestamp(self) -> str:
        """Genererar aktuell tidsstämpel."""
        from datetime import datetime
        return datetime.now().isoformat()
    
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

