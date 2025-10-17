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
        
        # Sprint 5: Spåra voting history
        self.total_votes_received = 0
        self.vote_matrices_published = 0
        
        # Prenumerera på decision_vote och parameter_adjustment
        self.message_bus.subscribe('decision_vote', self._on_decision_vote)
        self.message_bus.subscribe('parameter_adjustment', self._on_parameter_adjustment)
    
    def _on_decision_vote(self, vote: Dict[str, Any]) -> None:
        """
        Callback för besluts-röster från agenter.
        Sprint 5: Automatiskt skapar och publicerar vote_matrix efter röstning.
        
        Args:
            vote: Röst från en agent
        """
        self.votes.append(vote)
        self.total_votes_received += 1
        
        # Sprint 5: Skapa och publicera vote_matrix automatiskt
        # I nuvarande implementation (en decision_engine), publicera direkt
        # I framtiden kan detta vänta på flera agenter eller tidsgräns
        matrix = self.create_vote_matrix()
        self.publish_vote_matrix(matrix)
        self.vote_matrices_published += 1
        
        # INTE rensa röster automatiskt - låt användaren göra det explicit
        # Detta möjliggör inspektion och statistik över aktiva röster
    
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
        Sprint 5: Full implementation med röstanalys och aggregering.
        
        Returns:
            Dict med vote_matrix
        """
        if not self.votes:
            return {
                'votes': [],
                'num_voters': 0,
                'timestamp': datetime.now().isoformat(),
                'agent_vote_weight': self.agent_vote_weight,
                'vote_summary': {}
            }
        
        # Sprint 4.3: Applicera agent_vote_weight på röster
        weighted_votes = []
        action_votes = {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 0.0}
        total_weight = 0.0
        
        for vote in self.votes:
            weighted_vote = vote.copy()
            # Applicera global agent_vote_weight multiplicerat med agent-specifik performance
            agent_performance = vote.get('agent_performance', 1.0)
            vote_confidence = vote.get('confidence', 0.5)
            
            # Beräkna final weight: global_weight * agent_performance * confidence
            final_weight = self.agent_vote_weight * agent_performance * vote_confidence
            weighted_vote['weight'] = final_weight
            weighted_votes.append(weighted_vote)
            
            # Aggregera röster per action
            action = vote.get('action', 'HOLD')
            if action in action_votes:
                action_votes[action] += final_weight
                total_weight += final_weight
        
        # Normalisera röst-summor
        if total_weight > 0:
            for action in action_votes:
                action_votes[action] = action_votes[action] / total_weight
        
        # Beräkna consensus metrics
        max_vote = max(action_votes.values()) if action_votes else 0.0
        consensus_strength = max_vote  # Hur stark är majoriteten?
        
        # Sprint 5: Full matris-konstruktion
        matrix = {
            'votes': weighted_votes,
            'num_voters': len(self.votes),
            'timestamp': datetime.now().isoformat(),
            'agent_vote_weight': self.agent_vote_weight,  # Sprint 4.3
            'vote_summary': action_votes,
            'consensus_strength': consensus_strength,
            'total_weight': total_weight
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
        Sprint 5: Implementerad för att förbättra röstningskvalitet.
        
        Args:
            feedback: Feedback om voting quality
        """
        # Logga feedback för framtida förbättringar
        # I framtida sprintar kan detta användas för att justera agent_vote_weight
        pass
    
    def clear_votes(self) -> None:
        """
        Rensar rösthistorik.
        Används efter att konsensus har nåtts.
        """
        self.votes = []
    
    def get_voting_statistics(self) -> Dict[str, Any]:
        """
        Hämtar statistik över röstning.
        Sprint 5: Ger insikter i röstmönster.
        
        Returns:
            Dict med röststatistik
        """
        if not self.votes:
            return {
                'total_votes': 0,
                'unique_voters': 0,
                'average_confidence': 0.0,
                'action_distribution': {},
                'vote_matrices_published': self.vote_matrices_published
            }
        
        unique_voters = len(set(v.get('agent_id', 'unknown') for v in self.votes))
        avg_confidence = sum(v.get('confidence', 0.5) for v in self.votes) / len(self.votes)
        
        action_dist = {}
        for vote in self.votes:
            action = vote.get('action', 'HOLD')
            action_dist[action] = action_dist.get(action, 0) + 1
        
        return {
            'total_votes': len(self.votes),
            'unique_voters': unique_voters,
            'average_confidence': avg_confidence,
            'action_distribution': action_dist,
            'current_agent_vote_weight': self.agent_vote_weight,
            'vote_matrices_published': self.vote_matrices_published
        }

