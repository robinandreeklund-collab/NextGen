"""
vote_engine.py - Röstningsmotor

Beskrivning:
    Genomför röstning mellan agenter och returnerar röstmatris.
    Används för konsensusbaserat beslutsfattande.
    Sprint 4.3: Använder adaptiv agent_vote_weight för meritbaserad röstning.

Roll:
    - Tar emot decision_vote från decision_engine (och andra agenter i framtiden)
    - Samlar röster från flera agenter
    - Skapar röstmatris med viktning och konfidenser
    - Skickar vote_matrix till consensus_engine
    - Använder adaptiva parametrar från rl_controller (Sprint 4.3)

Inputs:
    - decision_vote: Dict - Röst från en agent med beslut och confidence
    - parameter_adjustment: Dict - Adaptiva parametrar från rl_controller (Sprint 4.3)

Outputs:
    - vote_matrix: Dict - Matris med alla röster och viktning

Publicerar till message_bus:
    - vote_matrix: Röstmatris för consensus_engine

Prenumererar på (från functions.yaml):
    - decision_vote (från decision_engine)
    - parameter_adjustment (från rl_controller, Sprint 4.3)

Använder RL: Nej (från functions.yaml)
Tar emot feedback: Ja (från consensus_engine)
Har adaptiva parametrar: Ja (från functions_2.yaml, Sprint 4.3)

Adaptiva parametrar (Sprint 4.3):
    - agent_vote_weight (0.1-2.0, default 1.0): Röstvikt baserad på agentperformance

Anslutningar (från flowchart.yaml - voting_flow):
    Från: decision_engine (decision_vote)
    Till: consensus_engine (vote_matrix)

Används i Sprint: 4.3, 5
"""

from typing import Dict, Any, List
from datetime import datetime


class VoteEngine:
    """Hanterar röstning mellan agenter för konsensus."""
    
    def __init__(self, message_bus):
        """
        Initialiserar röstmotorn.
        
        Args:
            message_bus: Referens till central message_bus
        """
        self.message_bus = message_bus
        self.votes: List[Dict[str, Any]] = []
        
        # Sprint 4.3: Adaptiv parameter
        self.agent_vote_weight = 1.0  # Default från adaptive_parameters.yaml
        
        # Prenumerera på decision_vote och parameter_adjustment
        self.message_bus.subscribe('decision_vote', self._on_decision_vote)
        self.message_bus.subscribe('parameter_adjustment', self._on_parameter_adjustment)
    
    def _on_decision_vote(self, vote: Dict[str, Any]) -> None:
        """
        Callback för besluts-röster från agenter.
        
        Args:
            vote: Röst från en agent
        """
        self.votes.append(vote)
        
        # I Sprint 5 kommer logik för när alla röster samlats
    
    def _on_parameter_adjustment(self, adjustment: Dict[str, Any]) -> None:
        """
        Callback för parameter adjustments från rl_controller (Sprint 4.3).
        
        Args:
            adjustment: Justerade parametrar med värden
        """
        params = adjustment.get('parameters', {})
        
        # Uppdatera agent_vote_weight om justerad
        if 'agent_vote_weight' in params:
            self.agent_vote_weight = params['agent_vote_weight']
    
    def create_vote_matrix(self) -> Dict[str, Any]:
        """
        Skapar röstmatris från alla insamlade röster.
        Sprint 4.3: Använder adaptiv agent_vote_weight för meritbaserad viktning.
        
        Returns:
            Dict med vote_matrix
        """
        # Sprint 4.3: Applicera agent_vote_weight på röster
        weighted_votes = []
        for vote in self.votes:
            weighted_vote = vote.copy()
            # Applicera global agent_vote_weight multiplicerat med agent-specifik performance
            agent_performance = vote.get('agent_performance', 1.0)
            weighted_vote['weight'] = self.agent_vote_weight * agent_performance
            weighted_votes.append(weighted_vote)
        
        # Stub: I Sprint 5 kommer faktisk matris-konstruktion
        matrix = {
            'votes': weighted_votes,
            'num_voters': len(self.votes),
            'timestamp': datetime.now().isoformat(),
            'agent_vote_weight': self.agent_vote_weight  # Sprint 4.3
        }
        return matrix
    
    def publish_vote_matrix(self, matrix: Dict[str, Any]) -> None:
        """
        Publicerar röstmatris till consensus_engine.
        
        Args:
            matrix: Röstmatris att publicera
        """
        self.message_bus.publish('vote_matrix', matrix)
    
    def receive_feedback(self, feedback: Dict[str, Any]) -> None:
        """
        Tar emot feedback från consensus_engine.
        
        Args:
            feedback: Feedback om voting quality
        """
        # Stub: Implementeras i Sprint 5
        pass

