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

from typing import Dict, Any


class DecisionSimulator:
    """Simulerar beslut i sandbox för testing."""
    
    def __init__(self, message_bus):
        self.message_bus = message_bus
        self.message_bus.subscribe('decision_proposal', self._on_proposal)
    
    def _on_proposal(self, proposal: Dict[str, Any]) -> None:
        """Callback för trade proposals."""
        pass

