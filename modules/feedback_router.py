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
        # VIKTIGT: Anropa inte route_feedback här för att undvika infinite loop
        # route_feedback används för extern routing
        pass
    
    def route_feedback(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Routar feedback till rätt mottagare baserat på källa och triggers.
        Denna metod används för manuell routing, inte för automatisk callback.
        
        Args:
            feedback: Feedback event att distribuera
        
        Returns:
            Dict[str, Any]: Enriched och routad feedback-event med metadata.
        """
        source = feedback.get('source', 'unknown')
        triggers = feedback.get('triggers', [])
        
        # Sprint 3: Intelligent filtrering och prioritering av feedback
        
        # Prioritera feedback baserat på triggers
        priority = self._calculate_priority(feedback)
        
        # Lägg till metadata
        enriched_feedback = {
            **feedback,
            'priority': priority,
            'routed_by': 'feedback_router',
            'route_timestamp': self._get_timestamp()
        }
        
        # Logga enriched feedback
        self.feedback_log.append(enriched_feedback)
        
        # Специфик routing baserat på trigger-typ till specifika topics
        # INTE tillbaka till 'feedback_event' för att undvika loop
        if 'slippage' in triggers and priority == 'high':
            # Hög slippage → direkt till rl_controller för omedelbar justering
            self.message_bus.publish('high_priority_feedback', enriched_feedback)
        
        if 'capital_change' in triggers:
            # Kapitalförändringar → strategic_memory för långsiktig analys
            self.message_bus.publish('memory_update', enriched_feedback)
        
        if 'indicator_correlation' in triggers:
            # Indikatorkorrelation → meta_agent_evolution för strategi-förbättring
            self.message_bus.publish('evolution_insight', enriched_feedback)
        
        return enriched_feedback
    
    def _calculate_priority(self, feedback: Dict[str, Any]) -> str:
        """
        Beräknar prioritet för feedback baserat på triggers och data.
        
        Args:
            feedback: Feedback event
            
        Returns:
            Priority level: 'low', 'medium', 'high', 'critical'
        """
        triggers = feedback.get('triggers', [])
        data = feedback.get('data', {})
        
        # Critical: Stora kapitalförluster eller systemfel
        if 'capital_change' in triggers:
            change = data.get('change', 0)
            if change < -100:  # Förlust > $100
                return 'critical'
        
        # High: Hög slippage eller låg success rate
        if 'slippage' in triggers:
            slippage = data.get('slippage', 0)
            if slippage > 0.005:  # > 0.5%
                return 'high'
        
        if 'trade_result' in triggers:
            success = data.get('success', True)
            if not success:
                return 'high'
        
        # Medium: Standard operational feedback
        if triggers:
            return 'medium'
        
        return 'low'
    
    def _get_timestamp(self) -> float:
        """Returnerar nuvarande timestamp."""
        import time
        return time.time()
    
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

