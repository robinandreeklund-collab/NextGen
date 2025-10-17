"""
decision_simulator.py - Beslutssimulator

Beskrivning:
    Testar alternativa beslut i sandbox och jämför utfall.

Roll:
    - Tar emot trade_proposal från strategy_engine
    - Simulerar olika beslutsscenarier
    - Jämför utfall utan verklig exekvering
    - Publicerar simulation_result

Inputs:
    - trade_proposal: Dict - Förslag att simulera

Outputs:
    - simulation_result: Dict - Simulerade utfall

Publicerar till message_bus:
    - simulation_result

Prenumererar på (från functions.yaml):
    - trade_proposal (från strategy_engine)

Använder RL: Nej
Tar emot feedback: Nej

Används i Sprint: 5
"""

from typing import Dict, Any, List
from datetime import datetime


class DecisionSimulator:
    """Simulerar beslut i sandbox för testing."""
    
    def __init__(self, message_bus):
        """
        Initialiserar beslutssimulator.
        
        Args:
            message_bus: Referens till central message_bus
        """
        self.message_bus = message_bus
        self.message_bus.subscribe('decision_proposal', self._on_proposal)
        self.simulation_history: List[Dict[str, Any]] = []
        
    def _on_proposal(self, proposal: Dict[str, Any]) -> None:
        """
        Callback för trade proposals.
        
        Args:
            proposal: Handelsförslag att simulera
        """
        # Simulera olika scenarier för förslaget
        simulation_result = self.simulate_decision(proposal)
        
        # Publicera resultat
        self.message_bus.publish('simulation_result', simulation_result)
        
        # Logga simulation
        self.simulation_history.append({
            'proposal': proposal,
            'result': simulation_result,
            'timestamp': datetime.now().isoformat()
        })
        
        # Limit history to prevent memory leak (keep last 1000)
        if len(self.simulation_history) > 1000:
            self.simulation_history = self.simulation_history[-1000:]
    
    def simulate_decision(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulerar olika utfall för ett beslut.
        
        Args:
            proposal: Handelsförslag att simulera
            
        Returns:
            Dict med simulationsresultat
        """
        symbol = proposal.get('symbol', 'UNKNOWN')
        action = proposal.get('action', 'HOLD')
        confidence = proposal.get('confidence', 0.5)
        quantity = proposal.get('quantity', 10)
        price = proposal.get('price', 100.0)
        
        # Simulera olika scenarier
        scenarios = []
        
        # Scenario 1: Best case (prisrörelse i rätt riktning)
        best_case_pnl = self._calculate_best_case(action, quantity, price)
        scenarios.append({
            'scenario': 'best_case',
            'pnl': best_case_pnl,
            'probability': confidence * 0.3
        })
        
        # Scenario 2: Expected case (förväntad prisrörelse)
        expected_pnl = self._calculate_expected_case(action, quantity, price, confidence)
        scenarios.append({
            'scenario': 'expected_case',
            'pnl': expected_pnl,
            'probability': 0.5
        })
        
        # Scenario 3: Worst case (prisrörelse mot oss)
        worst_case_pnl = self._calculate_worst_case(action, quantity, price)
        scenarios.append({
            'scenario': 'worst_case',
            'pnl': worst_case_pnl,
            'probability': (1 - confidence) * 0.3
        })
        
        # Scenario 4: No action (HOLD)
        scenarios.append({
            'scenario': 'no_action',
            'pnl': 0.0,
            'probability': 0.2
        })
        
        # Beräkna expected value
        expected_value = sum(s['pnl'] * s['probability'] for s in scenarios)
        
        # Rekommendation baserat på expected value
        recommendation = self._make_recommendation(expected_value, confidence, action)
        
        return {
            'symbol': symbol,
            'original_action': action,
            'scenarios': scenarios,
            'expected_value': expected_value,
            'recommendation': recommendation,
            'simulation_confidence': self._calculate_simulation_confidence(scenarios),
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_best_case(self, action: str, quantity: int, price: float) -> float:
        """Beräknar best case P&L."""
        if action == 'BUY':
            # Bästa fall: Priset går upp 5%
            return quantity * price * 0.05
        elif action == 'SELL':
            # Bästa fall: Priset går ner 5% (vi sålde i tid)
            return quantity * price * 0.05
        return 0.0
    
    def _calculate_expected_case(self, action: str, quantity: int, price: float, confidence: float) -> float:
        """Beräknar expected case P&L baserat på confidence."""
        if action == 'BUY':
            # Förväntad prisrörelse baserat på confidence
            expected_change = (confidence - 0.5) * 0.04  # -2% till +2%
            return quantity * price * expected_change
        elif action == 'SELL':
            # Förväntad prisrörelse (negativ för SELL)
            expected_change = (confidence - 0.5) * 0.04
            return quantity * price * expected_change
        return 0.0
    
    def _calculate_worst_case(self, action: str, quantity: int, price: float) -> float:
        """Beräknar worst case P&L."""
        if action == 'BUY':
            # Värsta fall: Priset går ner 3%
            return quantity * price * -0.03
        elif action == 'SELL':
            # Värsta fall: Priset går upp 3% (vi sålde för tidigt)
            return quantity * price * -0.03
        return 0.0
    
    def _make_recommendation(self, expected_value: float, confidence: float, original_action: str) -> str:
        """
        Ger rekommendation baserat på simulering.
        
        Args:
            expected_value: Förväntad P&L
            confidence: Original confidence
            original_action: Original åtgärd
            
        Returns:
            Rekommendation: 'proceed', 'caution', 'reject'
        """
        if expected_value > 0 and confidence > 0.7:
            return 'proceed'
        elif expected_value > 0 and confidence > 0.5:
            return 'caution'
        else:
            return 'reject'
    
    def _calculate_simulation_confidence(self, scenarios: List[Dict[str, Any]]) -> float:
        """
        Beräknar confidence i simuleringen baserat på scenario-spridning.
        
        Args:
            scenarios: Lista med scenarier
            
        Returns:
            Confidence-värde mellan 0 och 1
        """
        pnl_values = [s['pnl'] for s in scenarios]
        if not pnl_values:
            return 0.5
        
        # Låg spridning = hög confidence
        pnl_range = max(pnl_values) - min(pnl_values)
        avg_pnl = sum(pnl_values) / len(pnl_values)
        
        if avg_pnl == 0:
            return 0.5
        
        # Normalisera confidence baserat på spridning relativt genomsnitt
        volatility = abs(pnl_range / avg_pnl) if avg_pnl != 0 else 1.0
        confidence = max(0.3, min(0.95, 1.0 - (volatility * 0.2)))
        
        return confidence
    
    def get_simulation_history(self) -> List[Dict[str, Any]]:
        """
        Hämtar simulationshistorik.
        
        Returns:
            Lista med tidigare simuleringar
        """
        return self.simulation_history
    
    def get_simulation_statistics(self) -> Dict[str, Any]:
        """
        Beräknar statistik över simuleringar.
        
        Returns:
            Dict med statistik
        """
        if not self.simulation_history:
            return {
                'total_simulations': 0,
                'proceed_recommendations': 0,
                'caution_recommendations': 0,
                'reject_recommendations': 0,
                'average_expected_value': 0.0
            }
        
        proceed_count = sum(1 for s in self.simulation_history 
                          if s['result']['recommendation'] == 'proceed')
        caution_count = sum(1 for s in self.simulation_history 
                          if s['result']['recommendation'] == 'caution')
        reject_count = sum(1 for s in self.simulation_history 
                         if s['result']['recommendation'] == 'reject')
        
        avg_ev = sum(s['result']['expected_value'] for s in self.simulation_history) / len(self.simulation_history)
        
        return {
            'total_simulations': len(self.simulation_history),
            'proceed_recommendations': proceed_count,
            'caution_recommendations': caution_count,
            'reject_recommendations': reject_count,
            'average_expected_value': avg_ev
        }

