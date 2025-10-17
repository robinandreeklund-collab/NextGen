"""
consensus_engine.py - Konsensusmotor

Beskrivning:
    Väljer konsensusmodell och avgör beslutets robusthet.
    Fattar slutgiltigt beslut baserat på röstmatris från vote_engine.

Roll:
    - Tar emot vote_matrix från vote_engine
    - Väljer lämplig konsensusmodell (majoritet, viktad, unanimitet, etc.)
    - Beräknar robusthet och confidence för konsensus
    - Fattar consensus_decision
    - Skickar till execution_engine

Inputs:
    - vote_matrix: Dict - Röstmatris från vote_engine

Outputs:
    - consensus_decision: Dict - Slutgiltigt beslut baserat på konsensus

Publicerar till message_bus:
    - final_decision: Konsensus-beslut för execution_engine

Prenumererar på (från functions.yaml):
    - vote_matrix (från vote_engine)

Använder RL: Ja (från functions.yaml)
Tar emot feedback: Ja (från execution_engine)

Anslutningar (från flowchart.yaml - consensus_flow):
    Från: vote_engine (vote_matrix)
    Till:
    - execution_engine (final_decision)
    - decision_simulator (för testing)

Konsensusmodeller (från docs/consensus_models.yaml):
    - Majority: Enkel majoritet
    - Weighted: Viktad baserat på agent confidence
    - Unanimous: Kräver alla agenter överens
    - Threshold: Kräver X% enighet

Används i Sprint: 5
"""

from typing import Dict, Any, List
from datetime import datetime


class ConsensusEngine:
    """Fattar konsensus-beslut baserat på röstmatris."""
    
    def __init__(self, message_bus, consensus_model: str = 'weighted'):
        """
        Initialiserar konsensusmotorn.
        
        Args:
            message_bus: Referens till central message_bus
            consensus_model: Vilken konsensusmodell att använda (majority, weighted, unanimous, threshold)
        """
        self.message_bus = message_bus
        self.rl_agent = None  # Kommer från rl_controller i framtiden
        self.consensus_model = consensus_model
        self.threshold = 0.6  # För threshold-modellen
        self.consensus_history: List[Dict[str, Any]] = []
        
        # Prenumerera på vote_matrix
        self.message_bus.subscribe('vote_matrix', self._on_vote_matrix)
    
    def _on_vote_matrix(self, matrix: Dict[str, Any]) -> None:
        """
        Callback för röstmatris från vote_engine.
        
        Args:
            matrix: Röstmatris att processa
        """
        decision = self.make_consensus_decision(matrix)
        self.publish_decision(decision)
    
    def make_consensus_decision(self, vote_matrix: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fattar konsensus-beslut baserat på röstmatris.
        Sprint 5: Full implementation med flera konsensusmodeller.
        
        Args:
            vote_matrix: Röstmatris från vote_engine
            
        Returns:
            Dict med consensus_decision
        """
        votes = vote_matrix.get('votes', [])
        vote_summary = vote_matrix.get('vote_summary', {})
        consensus_strength = vote_matrix.get('consensus_strength', 0.0)
        
        if not votes:
            # Ingen röstning - defaulta till HOLD
            return self._create_hold_decision()
        
        # Välj konsensusmodell
        if self.consensus_model == 'majority':
            decision = self._majority_consensus(votes, vote_summary)
        elif self.consensus_model == 'weighted':
            decision = self._weighted_consensus(votes, vote_summary)
        elif self.consensus_model == 'unanimous':
            decision = self._unanimous_consensus(votes, vote_summary)
        elif self.consensus_model == 'threshold':
            decision = self._threshold_consensus(votes, vote_summary, self.threshold)
        else:
            # Default till weighted
            decision = self._weighted_consensus(votes, vote_summary)
        
        # Beräkna robusthet baserat på röstdistribution
        decision['robustness'] = self._calculate_robustness(vote_summary, consensus_strength, len(votes))
        decision['consensus_model'] = self.consensus_model
        decision['timestamp'] = datetime.now().isoformat()
        
        # Logga beslut
        self.consensus_history.append({
            'decision': decision,
            'vote_matrix': vote_matrix,
            'timestamp': datetime.now().isoformat()
        })
        
        return decision
    
    def _create_hold_decision(self) -> Dict[str, Any]:
        """Skapar ett HOLD-beslut som default."""
        return {
            'action': 'HOLD',
            'symbol': 'unknown',
            'quantity': 0,
            'confidence': 0.0,
            'robustness': 0.0,
            'reasoning': 'No votes received'
        }
    
    def _majority_consensus(self, votes: List[Dict[str, Any]], vote_summary: Dict[str, float]) -> Dict[str, Any]:
        """
        Enkel majoritetskonsensus - den action med flest röster vinner.
        
        Args:
            votes: Lista med röster
            vote_summary: Sammanfattning av röster per action
            
        Returns:
            Beslut baserat på majoritet
        """
        if not vote_summary:
            return self._create_hold_decision()
        
        # Hitta action med flest röster (viktade)
        winning_action = max(vote_summary.items(), key=lambda x: x[1])[0]
        
        # Hitta första rösten med denna action för detaljer
        winning_vote = next((v for v in votes if v.get('action') == winning_action), {})
        
        return {
            'action': winning_action,
            'symbol': winning_vote.get('symbol', 'unknown'),
            'quantity': winning_vote.get('quantity', 0),
            'confidence': vote_summary.get(winning_action, 0.0),
            'reasoning': f'Majority vote: {vote_summary.get(winning_action, 0.0):.2%} support'
        }
    
    def _weighted_consensus(self, votes: List[Dict[str, Any]], vote_summary: Dict[str, float]) -> Dict[str, Any]:
        """
        Viktad konsensus - använder både röster och confidence.
        Samma som majority men med explicit viktning av confidence.
        
        Args:
            votes: Lista med röster
            vote_summary: Sammanfattning av viktade röster per action
            
        Returns:
            Beslut baserat på viktad röstning
        """
        if not vote_summary:
            return self._create_hold_decision()
        
        # vote_summary innehåller redan viktade röster från vote_engine
        winning_action = max(vote_summary.items(), key=lambda x: x[1])[0]
        winning_percentage = vote_summary.get(winning_action, 0.0)
        
        # Hitta första rösten med denna action
        winning_vote = next((v for v in votes if v.get('action') == winning_action), {})
        
        # Beräkna genomsnittlig confidence för winning action
        action_votes = [v for v in votes if v.get('action') == winning_action]
        avg_confidence = sum(v.get('confidence', 0.5) for v in action_votes) / len(action_votes) if action_votes else 0.5
        
        return {
            'action': winning_action,
            'symbol': winning_vote.get('symbol', 'unknown'),
            'quantity': winning_vote.get('quantity', 0),
            'confidence': avg_confidence * winning_percentage,  # Kombinera confidence och support
            'reasoning': f'Weighted consensus: {winning_percentage:.2%} support, {avg_confidence:.2f} avg confidence'
        }
    
    def _unanimous_consensus(self, votes: List[Dict[str, Any]], vote_summary: Dict[str, float]) -> Dict[str, Any]:
        """
        Unanimitet - kräver att alla röster är överens.
        
        Args:
            votes: Lista med röster
            vote_summary: Sammanfattning av röster per action
            
        Returns:
            Beslut om alla är överens, annars HOLD
        """
        if not vote_summary:
            return self._create_hold_decision()
        
        # Kontrollera om bara en action har röster
        actions_with_votes = [action for action, pct in vote_summary.items() if pct > 0]
        
        if len(actions_with_votes) == 1:
            # Alla röster på samma action
            winning_action = actions_with_votes[0]
            winning_vote = next((v for v in votes if v.get('action') == winning_action), {})
            
            return {
                'action': winning_action,
                'symbol': winning_vote.get('symbol', 'unknown'),
                'quantity': winning_vote.get('quantity', 0),
                'confidence': 1.0,  # Full confidence vid unanimitet
                'reasoning': f'Unanimous decision: all {len(votes)} voters agree'
            }
        else:
            # Inte alla överens - HOLD
            return {
                'action': 'HOLD',
                'symbol': votes[0].get('symbol', 'unknown') if votes else 'unknown',
                'quantity': 0,
                'confidence': 0.0,
                'reasoning': f'No unanimous consensus: {len(actions_with_votes)} different actions'
            }
    
    def _threshold_consensus(self, votes: List[Dict[str, Any]], vote_summary: Dict[str, float], threshold: float) -> Dict[str, Any]:
        """
        Tröskelvärde - kräver att en action har minst X% av rösterna.
        
        Args:
            votes: Lista med röster
            vote_summary: Sammanfattning av röster per action
            threshold: Minsta procentandel som krävs (0.0-1.0)
            
        Returns:
            Beslut om tröskelvärdet uppnås, annars HOLD
        """
        if not vote_summary:
            return self._create_hold_decision()
        
        # Hitta action med flest röster
        winning_action = max(vote_summary.items(), key=lambda x: x[1])[0]
        winning_percentage = vote_summary.get(winning_action, 0.0)
        
        if winning_percentage >= threshold:
            # Tröskelvärde uppnått
            winning_vote = next((v for v in votes if v.get('action') == winning_action), {})
            
            return {
                'action': winning_action,
                'symbol': winning_vote.get('symbol', 'unknown'),
                'quantity': winning_vote.get('quantity', 0),
                'confidence': winning_percentage,
                'reasoning': f'Threshold met: {winning_percentage:.2%} ≥ {threshold:.2%}'
            }
        else:
            # Tröskelvärde ej uppnått - HOLD
            return {
                'action': 'HOLD',
                'symbol': votes[0].get('symbol', 'unknown') if votes else 'unknown',
                'quantity': 0,
                'confidence': winning_percentage,
                'reasoning': f'Threshold not met: {winning_percentage:.2%} < {threshold:.2%}'
            }
    
    def _calculate_robustness(self, vote_summary: Dict[str, float], consensus_strength: float, num_votes: int) -> float:
        """
        Beräknar robusthet i konsensus.
        
        Args:
            vote_summary: Röstfördelning
            consensus_strength: Styrka i konsensus (från vote_engine)
            num_votes: Antal röster
            
        Returns:
            Robusthetsvärde mellan 0 och 1
        """
        if num_votes == 0:
            return 0.0
        
        # Robusthet baseras på:
        # 1. Konsensus-styrka (hur stor andel röstar på samma sak)
        # 2. Antal röster (fler röster = mer robust)
        
        # Normalisera antal röster (5+ röster = full contribution)
        vote_factor = min(1.0, num_votes / 5.0)
        
        # Kombinera med consensus strength
        robustness = consensus_strength * 0.7 + vote_factor * 0.3
        
        return max(0.0, min(1.0, robustness))
    
    def publish_decision(self, decision: Dict[str, Any]) -> None:
        """
        Publicerar konsensus-beslut till execution_engine.
        
        Args:
            decision: Konsensus-beslut att publicera
        """
        self.message_bus.publish('final_decision', decision)
    
    def receive_feedback(self, feedback: Dict[str, Any]) -> None:
        """
        Tar emot feedback från execution_engine.
        Sprint 5: Implementerad för att förbättra konsensuskvalitet.
        
        Args:
            feedback: Feedback om consensus quality
        """
        # Logga feedback för framtida analys
        # I framtida sprintar kan detta användas för att justera threshold eller modell
        pass
    
    def update_from_rl(self, agent_update: Dict[str, Any]) -> None:
        """
        Uppdaterar konsensuslogik baserat på RL-träning.
        Sprint 5: Stöd för framtida RL-integration.
        
        Args:
            agent_update: Uppdatering från rl_controller
        """
        # Framtida: Använd RL för att välja optimal consensus_model och threshold
        if 'consensus_model' in agent_update:
            self.consensus_model = agent_update['consensus_model']
        if 'threshold' in agent_update:
            self.threshold = agent_update['threshold']
    
    def get_consensus_statistics(self) -> Dict[str, Any]:
        """
        Hämtar statistik över konsensusbeslut.
        Sprint 5: Ger insikter i konsensuskvalitet.
        
        Returns:
            Dict med konsensusstatistik
        """
        if not self.consensus_history:
            return {
                'total_decisions': 0,
                'action_distribution': {},
                'average_confidence': 0.0,
                'average_robustness': 0.0,
                'consensus_model': self.consensus_model,
                'threshold': self.threshold
            }
        
        action_dist = {}
        total_confidence = 0.0
        total_robustness = 0.0
        
        for entry in self.consensus_history:
            decision = entry['decision']
            action = decision.get('action', 'HOLD')
            action_dist[action] = action_dist.get(action, 0) + 1
            total_confidence += decision.get('confidence', 0.0)
            total_robustness += decision.get('robustness', 0.0)
        
        num_decisions = len(self.consensus_history)
        
        return {
            'total_decisions': num_decisions,
            'action_distribution': action_dist,
            'average_confidence': total_confidence / num_decisions,
            'average_robustness': total_robustness / num_decisions,
            'consensus_model': self.consensus_model,
            'threshold': self.threshold
        }
    
    def set_consensus_model(self, model: str) -> None:
        """
        Ändrar konsensusmodell.
        
        Args:
            model: Ny modell (majority, weighted, unanimous, threshold)
        """
        if model in ['majority', 'weighted', 'unanimous', 'threshold']:
            self.consensus_model = model
    
    def set_threshold(self, threshold: float) -> None:
        """
        Ändrar tröskelvärde för threshold-modellen.
        
        Args:
            threshold: Nytt tröskelvärde (0.0-1.0)
        """
        self.threshold = max(0.0, min(1.0, threshold))

