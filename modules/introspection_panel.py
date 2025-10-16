"""
introspection_panel.py - Introspektionspanel

Beskrivning:
    Visualiserar modulstatus, RL-performance och kommunikation.
    Dashboard för transparens och debugging av systemet.

Roll:
    - Prenumererar på agent_status, feedback_event, indicator_data
    - Visualiserar feedback flow mellan moduler
    - Visar agent adaptation och RL-performance
    - Visar reward trends över tid
    - Renderar dashboard för Dash-applikation

Inputs:
    - agent_status: Dict - Status från rl_controller
    - feedback_event: Dict - Feedback från feedback_router
    - indicator_data: Dict - Indikatorer från indicator_registry

Outputs:
    - dashboard_render: Dict - Data för Dash-visualisering

Publicerar till message_bus:
    - Ingen (konsumerar endast data)

Prenumererar på (från functions.yaml):
    - agent_status (från rl_controller)
    - feedback_event (från feedback_router)
    - indicator_data (från indicator_registry)

Använder RL: Nej (från functions.yaml)
Tar emot feedback: Nej (från functions.yaml)

Anslutningar (från flowchart.yaml - visualization):
    Från:
    - rl_controller (agent_status)
    - feedback_router (feedback_event)
    - indicator_registry (indicator_data)
    - portfolio_manager (portfolio_status)
    - strategic_memory_engine (memory_insights)
    - decision_simulator (simulation_result)
    Till: dashboard_render

Visualisering (från feedback_loop.yaml):
    Displays:
    - feedback flow: Visuell representation av feedback mellan moduler
    - agent adaptation: Hur agenter anpassar sig över tid
    - reward trends: Utveckling av reward över episodes

Används i Sprint: 3, 7
"""

from typing import Dict, Any, List


class IntrospectionPanel:
    """Visualiserar system status och modul-kommunikation."""
    
    def __init__(self, message_bus):
        """
        Initialiserar introspektionspanelen.
        
        Args:
            message_bus: Referens till central message_bus
        """
        self.message_bus = message_bus
        self.agent_status_history: List[Dict[str, Any]] = []
        self.feedback_events: List[Dict[str, Any]] = []
        self.indicator_snapshots: List[Dict[str, Any]] = []
        
        # Prenumerera på relevanta topics
        self.message_bus.subscribe('agent_status', self._on_agent_status)
        self.message_bus.subscribe('feedback_event', self._on_feedback)
        self.message_bus.subscribe('indicator_data', self._on_indicators)
        self.message_bus.subscribe('portfolio_status', self._on_portfolio)
    
    def _on_agent_status(self, status: Dict[str, Any]) -> None:
        """Callback för agent status från rl_controller."""
        self.agent_status_history.append(status)
        # Behåll senaste 100
        if len(self.agent_status_history) > 100:
            self.agent_status_history = self.agent_status_history[-100:]
    
    def _on_feedback(self, feedback: Dict[str, Any]) -> None:
        """Callback för feedback events."""
        self.feedback_events.append(feedback)
        # Behåll senaste 100
        if len(self.feedback_events) > 100:
            self.feedback_events = self.feedback_events[-100:]
    
    def _on_indicators(self, indicators: Dict[str, Any]) -> None:
        """Callback för indikatordata."""
        self.indicator_snapshots.append(indicators)
        # Behåll senaste 50
        if len(self.indicator_snapshots) > 50:
            self.indicator_snapshots = self.indicator_snapshots[-50:]
    
    def _on_portfolio(self, status: Dict[str, Any]) -> None:
        """Callback för portföljstatus."""
        # Stub: Lagras för visualisering
        pass
    
    def render_dashboard(self) -> Dict[str, Any]:
        """
        Genererar dashboard-data för visualisering.
        
        Returns:
            Dict med alla data för dashboard
        """
        # Stub: I Sprint 3 kommer faktisk Dash-integration
        dashboard_data = {
            'agent_status': self.agent_status_history[-10:] if self.agent_status_history else [],
            'feedback_flow': self.feedback_events[-20:] if self.feedback_events else [],
            'indicators': self.indicator_snapshots[-10:] if self.indicator_snapshots else [],
            'reward_trends': [],  # Extraheras från agent_status
            'module_connections': self.get_module_connections()
        }
        return dashboard_data
    
    def get_module_connections(self) -> List[Dict[str, Any]]:
        """
        Analyserar feedback för att visa modul-kopplingar.
        
        Returns:
            Lista med modul-till-modul kopplingar
        """
        # Stub: I Sprint 3 kommer nätverksanalys av kommunikation
        return []

