"""
test_vote_engine.py - Tester för röstningsmotor (Sprint 5)

Beskrivning:
    Testar VoteEngine för röstmatris och konsensus.
    Validerar viktning, aggregering och adaptiva parametrar.
"""

import pytest
from modules.vote_engine import VoteEngine
from modules.message_bus import MessageBus


class TestVoteEngine:
    """Tester för VoteEngine (Sprint 5)."""
    
    def setup_method(self):
        """Körs före varje test."""
        self.message_bus = MessageBus()
        self.vote_engine = VoteEngine(self.message_bus)
    
    def test_vote_engine_initialization(self):
        """VE-001: Testa att vote engine initialiseras korrekt."""
        assert self.vote_engine is not None
        assert self.vote_engine.message_bus == self.message_bus
        assert len(self.vote_engine.votes) == 0
        assert self.vote_engine.agent_vote_weight == 1.0  # Default
    
    def test_receive_single_vote(self):
        """VE-002: Testa mottagning av en röst."""
        vote = {
            'agent_id': 'strategy_agent',
            'action': 'BUY',
            'symbol': 'AAPL',
            'confidence': 0.8,
            'agent_performance': 1.0
        }
        
        self.message_bus.publish('decision_vote', vote)
        
        assert len(self.vote_engine.votes) == 1
        assert self.vote_engine.votes[0]['action'] == 'BUY'
    
    def test_receive_multiple_votes(self):
        """VE-003: Testa mottagning av flera röster."""
        votes = [
            {'agent_id': 'strategy_agent', 'action': 'BUY', 'symbol': 'AAPL', 'confidence': 0.8, 'agent_performance': 1.0},
            {'agent_id': 'risk_agent', 'action': 'HOLD', 'symbol': 'AAPL', 'confidence': 0.6, 'agent_performance': 0.9},
            {'agent_id': 'decision_agent', 'action': 'BUY', 'symbol': 'AAPL', 'confidence': 0.7, 'agent_performance': 1.1}
        ]
        
        for vote in votes:
            self.message_bus.publish('decision_vote', vote)
        
        assert len(self.vote_engine.votes) == 3
    
    def test_create_vote_matrix_structure(self):
        """VE-004: Testa struktur på röstmatris."""
        votes = [
            {'agent_id': 'agent1', 'action': 'BUY', 'symbol': 'MSFT', 'confidence': 0.8, 'agent_performance': 1.0},
            {'agent_id': 'agent2', 'action': 'BUY', 'symbol': 'MSFT', 'confidence': 0.7, 'agent_performance': 0.95}
        ]
        
        for vote in votes:
            self.message_bus.publish('decision_vote', vote)
        
        matrix = self.vote_engine.create_vote_matrix()
        
        assert 'votes' in matrix
        assert 'num_voters' in matrix
        assert 'vote_summary' in matrix
        assert 'consensus_strength' in matrix
        assert 'agent_vote_weight' in matrix
        assert matrix['num_voters'] == 2
    
    def test_vote_weighting(self):
        """VE-005: Testa att röster viktas korrekt."""
        self.vote_engine.agent_vote_weight = 1.5  # Sätt adaptiv vikt
        
        vote = {
            'agent_id': 'agent1',
            'action': 'BUY',
            'symbol': 'GOOGL',
            'confidence': 0.8,
            'agent_performance': 1.2
        }
        
        self.message_bus.publish('decision_vote', vote)
        matrix = self.vote_engine.create_vote_matrix()
        
        # Weight ska vara agent_vote_weight * agent_performance * confidence
        expected_weight = 1.5 * 1.2 * 0.8
        assert abs(matrix['votes'][0]['weight'] - expected_weight) < 0.01
    
    def test_vote_aggregation(self):
        """VE-006: Testa aggregering av röster per action."""
        votes = [
            {'agent_id': 'agent1', 'action': 'BUY', 'symbol': 'AMZN', 'confidence': 0.8, 'agent_performance': 1.0},
            {'agent_id': 'agent2', 'action': 'BUY', 'symbol': 'AMZN', 'confidence': 0.7, 'agent_performance': 1.0},
            {'agent_id': 'agent3', 'action': 'SELL', 'symbol': 'AMZN', 'confidence': 0.6, 'agent_performance': 1.0}
        ]
        
        for vote in votes:
            self.message_bus.publish('decision_vote', vote)
        
        matrix = self.vote_engine.create_vote_matrix()
        vote_summary = matrix['vote_summary']
        
        assert 'BUY' in vote_summary
        assert 'SELL' in vote_summary
        # BUY ska ha högre vikt än SELL (2 röster vs 1)
        assert vote_summary['BUY'] > vote_summary['SELL']
    
    def test_consensus_strength_calculation(self):
        """VE-007: Testa beräkning av consensus strength."""
        # Stark konsensus - alla röstar samma
        votes = [
            {'agent_id': 'agent1', 'action': 'BUY', 'symbol': 'TSLA', 'confidence': 0.8, 'agent_performance': 1.0},
            {'agent_id': 'agent2', 'action': 'BUY', 'symbol': 'TSLA', 'confidence': 0.9, 'agent_performance': 1.0}
        ]
        
        for vote in votes:
            self.message_bus.publish('decision_vote', vote)
        
        matrix = self.vote_engine.create_vote_matrix()
        
        # Consensus strength ska vara hög när alla röstar samma
        assert matrix['consensus_strength'] > 0.8
    
    def test_parameter_adjustment(self):
        """VE-008: Testa mottagning av parameter adjustments (Sprint 4.3)."""
        adjustment = {
            'parameters': {
                'agent_vote_weight': 1.8
            }
        }
        
        self.message_bus.publish('parameter_adjustment', adjustment)
        
        assert self.vote_engine.agent_vote_weight == 1.8
    
    def test_clear_votes(self):
        """VE-009: Testa att röster kan rensas."""
        votes = [
            {'agent_id': 'agent1', 'action': 'BUY', 'symbol': 'NVDA', 'confidence': 0.8, 'agent_performance': 1.0}
        ]
        
        for vote in votes:
            self.message_bus.publish('decision_vote', vote)
        
        assert len(self.vote_engine.votes) == 1
        
        self.vote_engine.clear_votes()
        
        assert len(self.vote_engine.votes) == 0
    
    def test_voting_statistics(self):
        """VE-010: Testa beräkning av röststatistik."""
        votes = [
            {'agent_id': 'agent1', 'action': 'BUY', 'symbol': 'META', 'confidence': 0.8, 'agent_performance': 1.0},
            {'agent_id': 'agent2', 'action': 'BUY', 'symbol': 'META', 'confidence': 0.7, 'agent_performance': 1.0},
            {'agent_id': 'agent3', 'action': 'SELL', 'symbol': 'META', 'confidence': 0.6, 'agent_performance': 1.0}
        ]
        
        for vote in votes:
            self.message_bus.publish('decision_vote', vote)
        
        stats = self.vote_engine.get_voting_statistics()
        
        assert stats['total_votes'] == 3
        assert stats['unique_voters'] == 3
        assert 'average_confidence' in stats
        assert 'action_distribution' in stats
        assert stats['action_distribution']['BUY'] == 2
        assert stats['action_distribution']['SELL'] == 1
    
    def test_empty_vote_matrix(self):
        """VE-011: Testa skapande av tom röstmatris."""
        matrix = self.vote_engine.create_vote_matrix()
        
        assert matrix['num_voters'] == 0
        assert len(matrix['votes']) == 0
        assert 'vote_summary' in matrix
    
    def test_publish_vote_matrix(self):
        """VE-012: Testa publicering av röstmatris."""
        received_matrix = None
        
        def capture_matrix(matrix):
            nonlocal received_matrix
            received_matrix = matrix
        
        self.message_bus.subscribe('vote_matrix', capture_matrix)
        
        vote = {'agent_id': 'agent1', 'action': 'BUY', 'symbol': 'INTC', 'confidence': 0.8, 'agent_performance': 1.0}
        self.message_bus.publish('decision_vote', vote)
        
        matrix = self.vote_engine.create_vote_matrix()
        self.vote_engine.publish_vote_matrix(matrix)
        
        assert received_matrix is not None
        assert received_matrix['num_voters'] == 1
