"""
strategic_memory_engine.py - Strategiskt minne

Beskrivning:
    Loggar beslut, indikatorer, utfall och reward för historisk analys.
    Identifierar mönster och ger insights till decision_engine.

Roll:
    - Loggar alla beslut och deras utfall
    - Lagrar indikatordata kopplat till beslut
    - Analyserar historisk performance
    - Genererar memory_insights för decision_engine
    - Genererar feedback om decision outcomes och indicator correlations

Inputs:
    - final_decision: Dict - Beslut från decision_engine
    - indicator_data: Dict - Indikatorer från indicator_registry
    - execution_result: Dict - Resultat från execution_engine

Outputs:
    - memory_insights: Dict - Historiska lärdomar och mönster

Publicerar till message_bus:
    - memory_insights: Insights för decision_engine
    - feedback_event: Feedback om decision outcomes

Prenumererar på (från functions.yaml):
    - final_decision (från decision_engine)
    - indicator_data (från indicator_registry)
    - execution_result (från execution_engine)

Använder RL: Nej (från functions.yaml)
Tar emot feedback: Ja (från feedback_analyzer, meta_agent_evolution_engine)

Anslutningar (från flowchart.yaml - memory_flow):
    Från:
    - decision_engine (final_decision)
    - indicator_registry (indicator_data)
    - execution_engine (execution_result)
    - risk_manager (risk_profile)
    Till:
    - decision_engine (memory_insights)
    - feedback_analyzer (historical data)
    - introspection_panel (för visualisering)

Feedback-generering (från feedback_loop.yaml):
    Triggers:
    - decision_outcome: Om beslut ledde till vinst/förlust
    - indicator_correlation: Vilka indikatorer korrelerade med utfall
    
    Stores:
    - feedback_event: Från andra moduler
    - feedback_insight: Från feedback_analyzer
    - agent_response: Från RL-agenter

Används i Sprint: 4, 6
"""

from typing import Dict, Any, List


class StrategicMemoryEngine:
    """Loggar och analyserar historiska beslut och utfall."""
    
    def __init__(self, message_bus):
        """
        Initialiserar strategiskt minne.
        
        Args:
            message_bus: Referens till central message_bus
        """
        self.message_bus = message_bus
        self.decision_history: List[Dict[str, Any]] = []
        self.indicator_history: Dict[str, List[Dict[str, Any]]] = {}
        self.execution_history: List[Dict[str, Any]] = []
        
        # Prenumerera på relevanta events
        self.message_bus.subscribe('final_decision', self._on_decision)
        self.message_bus.subscribe('indicator_data', self._on_indicators)
        self.message_bus.subscribe('execution_result', self._on_execution)
    
    def _on_decision(self, decision: Dict[str, Any]) -> None:
        """
        Callback för beslut från decision_engine.
        
        Args:
            decision: Handelsbeslut att logga
        """
        self.decision_history.append(decision)
    
    def _on_indicators(self, indicators: Dict[str, Any]) -> None:
        """
        Callback för indikatordata.
        
        Args:
            indicators: Indikatordata att lagra
        """
        symbol = indicators.get('symbol')
        if symbol not in self.indicator_history:
            self.indicator_history[symbol] = []
        self.indicator_history[symbol].append(indicators)
    
    def _on_execution(self, result: Dict[str, Any]) -> None:
        """
        Callback för execution results.
        
        Args:
            result: Execution result att analysera
        """
        self.execution_history.append(result)
        
        # Generera feedback om decision outcome
        self.generate_decision_feedback(result)
    
    def generate_insights(self) -> Dict[str, Any]:
        """
        Genererar insights baserat på historisk data.
        
        Returns:
            Dict med memory_insights
        """
        # Stub: I Sprint 4 kommer avancerad mönsteranalys implementeras
        insights = {
            'total_decisions': len(self.decision_history),
            'total_executions': len(self.execution_history),
            'success_rate': 0.0,
            'patterns': [],
            'recommendations': []
        }
        
        # Beräkna success rate
        if self.execution_history:
            successful = sum(1 for e in self.execution_history if e.get('success', False))
            insights['success_rate'] = successful / len(self.execution_history)
        
        return insights
    
    def publish_insights(self) -> None:
        """Publicerar memory insights till message_bus."""
        insights = self.generate_insights()
        self.message_bus.publish('memory_insights', insights)
    
    def generate_decision_feedback(self, execution_result: Dict[str, Any]) -> None:
        """
        Genererar feedback om decision outcome (från feedback_loop.yaml).
        
        Args:
            execution_result: Execution result att analysera
        """
        feedback = {
            'source': 'strategic_memory_engine',
            'triggers': ['decision_outcome'],
            'data': {
                'outcome': 'success' if execution_result.get('success') else 'failure',
                'symbol': execution_result.get('symbol'),
                'action': execution_result.get('action')
            }
        }
        
        # Analysera indicator correlation (stub)
        # I Sprint 4 kommer detaljerad korrelationsanalys
        feedback['triggers'].append('indicator_correlation')
        feedback['data']['indicator_correlation'] = 'pending_analysis'
        
        # Publicera feedback
        self.message_bus.publish('feedback_event', feedback)
    
    def receive_feedback(self, feedback: Dict[str, Any]) -> None:
        """
        Tar emot feedback från feedback_analyzer.
        
        Args:
            feedback: Feedback om memory insights quality
        """
        # Stub: Implementeras i Sprint 4
        pass
    
    def get_decision_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Hämtar beslutshistorik.
        
        Args:
            limit: Max antal beslut att returnera
            
        Returns:
            Lista med senaste besluten
        """
        return self.decision_history[-limit:]

