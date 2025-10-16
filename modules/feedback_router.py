"""
feedback_router.py - Feedback-router för modulkommunikation

Beskrivning:
    Distribuerar feedback mellan moduler enligt feedback_loop.yaml.
    Central för feedbackhantering och routing av feedback events.

Roll:
    - Tar emot feedback_event från execution_engine, portfolio_manager, strategic_memory_engine
    - Routar feedback till rätt mottagare (rl_controller, feedback_analyzer, etc.)
    - Loggar alla feedback för analys och introspektion

Inputs:
    - feedback: Dict - Feedback event från olika källor

Outputs:
    - feedback_event: Dict - Routad feedback till prenumeranter

Publicerar till message_bus:
    - feedback_event: Feedback till alla intresserade moduler

Prenumererar på:
    - Ingen direkt (från functions.yaml)
    - Tar emot via publish från andra moduler

Använder RL: Nej (från functions.yaml)
Tar emot feedback: Nej (från functions.yaml)

Anslutningar (från flowchart.yaml - feedback_flow):
    Från: execution_engine, portfolio_manager, strategic_memory_engine
    Till:
    - rl_controller (för agentträning)
    - feedback_analyzer (för mönsteranalys)
    - strategic_memory_engine (för loggning)

Feedback Routing (från feedback_loop.yaml):
    Distributes to:
    - rl_controller: För RL-träning och belöning
    - feedback_analyzer: För mönsteridentifiering
    - strategic_memory_engine: För historisk loggning
    - meta_agent_evolution_engine: För agentförbättring

Används i Sprint: 3, 4
"""

from typing import Dict, Any, List


class FeedbackRouter:
    """Routar feedback mellan moduler enligt feedback_loop.yaml."""
    
    def __init__(self, message_bus):
        """
        Initialiserar feedback-routern.
        
        Args:
            message_bus: Referens till central message_bus
        """
        self.message_bus = message_bus
        self.feedback_log: List[Dict[str, Any]] = []
        
        # Prenumerera på feedback events
        self.message_bus.subscribe('feedback_event', self._on_feedback_event)
    
    def _on_feedback_event(self, feedback: Dict[str, Any]) -> None:
        """
        Callback för feedback events från olika moduler.
        
        Args:
            feedback: Feedback event att routa
        """
        # Logga feedback
        self.feedback_log.append(feedback)
        
        # Routa feedback till relevanta moduler
        self.route_feedback(feedback)
    
    def route_feedback(self, feedback: Dict[str, Any]) -> None:
        """
        Routar feedback till rätt mottagare baserat på källa och triggers.
        
        Args:
            feedback: Feedback event att distribuera
        """
        # Sprint 2: Implementera grundläggande routing
        source = feedback.get('source', 'unknown')
        triggers = feedback.get('triggers', [])
        
        # Routing enligt feedback_loop.yaml:
        # - Alla feedback till rl_controller (hanteras via prenumeration)
        # - Feedback till feedback_analyzer (hanteras via prenumeration)
        # - Memory-relevanta triggers till strategic_memory_engine
        
        # Notera: Routing sker automatiskt via message_bus prenumerationer
        # Router agerar som logger och kan filtrera i framtida sprintar
        
        # I Sprint 3: Intelligent filtrering och prioritering av feedback
        pass
    
    def get_feedback_log(self) -> List[Dict[str, Any]]:
        """
        Hämtar alla loggade feedback events.
        
        Returns:
            Lista med alla feedback events
        """
        return self.feedback_log
    
    def clear_log(self) -> None:
        """Rensar feedback-loggen."""
        self.feedback_log.clear()

