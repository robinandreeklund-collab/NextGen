"""
message_bus.py - Central meddelandebuss för modulkommunikation

Beskrivning:
    Central hub för publicering och prenumeration av meddelanden mellan moduler.
    Hanterar event-driven kommunikation och säkerställer lös koppling mellan komponenter.

Roll:
    - Möjliggör pub/sub-mönster mellan moduler
    - Distribuerar indicator_data, portfolio_status, feedback_event och andra meddelanden
    - Loggar meddelandeflöde för debugging och introspektion

Inputs:
    - topic: str - Meddelandetopic (ex: 'indicator_data', 'trade_proposal')
    - message: dict - Meddelandedata att publicera

Outputs:
    - Distribuerar meddelanden till prenumeranter

Anslutningar (från flowchart.yaml och feedback_loop.yaml):
    Används av: Alla moduler för kommunikation
    - Alla moduler kan publicera och prenumerera på topics
    - Central för hela systemets dataflöde

Används av Sprint: 1, 2, 3, 4, 5, 6, 7
"""

from typing import Dict, List, Callable, Any
from collections import defaultdict


class MessageBus:
    """Central meddelandebuss för pub/sub-kommunikation mellan moduler."""
    
    def __init__(self):
        """Initialiserar meddelandebussen med tomma prenumerationslistor."""
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.message_log: List[Dict[str, Any]] = []
    
    def subscribe(self, topic: str, callback: Callable) -> None:
        """
        Prenumerera på ett topic.
        
        Args:
            topic: Namnet på topic att prenumerera på
            callback: Funktion som anropas när meddelande publiceras
        """
        self.subscribers[topic].append(callback)
    
    def publish(self, topic: str, message: Dict[str, Any]) -> None:
        """
        Publicera ett meddelande till ett topic.
        
        Args:
            topic: Namnet på topic att publicera till
            message: Meddelandedata att skicka till prenumeranter
        """
        # Logga meddelandet
        self.message_log.append({
            'topic': topic,
            'message': message
        })
        
        # Limit message log to prevent memory leak (keep last 10000)
        # With websocket: 100+ messages/sec → log grows to 360k/hour without limit
        if len(self.message_log) > 10000:
            self.message_log = self.message_log[-10000:]
        
        # Distribuera till alla prenumeranter
        for callback in self.subscribers.get(topic, []):
            callback(message)
    
    def unsubscribe(self, topic: str, callback: Callable) -> None:
        """
        Avsluta prenumeration på ett topic.
        
        Args:
            topic: Namnet på topic att avsluta prenumeration på
            callback: Callback-funktionen att ta bort
        """
        if topic in self.subscribers:
            self.subscribers[topic] = [cb for cb in self.subscribers[topic] if cb != callback]
    
    def get_message_log(self) -> List[Dict[str, Any]]:
        """
        Hämta alla loggade meddelanden.
        
        Returns:
            Lista med alla publicerade meddelanden
        """
        return self.message_log
    
    def clear_log(self) -> None:
        """Rensa meddelandeloggen."""
        self.message_log.clear()


# Global instans av meddelandebussen
message_bus = MessageBus()
