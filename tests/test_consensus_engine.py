"""
test_consensus_engine.py - Tester för konsensusmotor (Sprint 5)

Beskrivning:
    Testar ConsensusEngine för olika konsensusmodeller.
    Validerar majoritet, viktad, unanimitet och threshold-konsensus.
"""

import pytest
from modules.consensus_engine import ConsensusEngine
from modules.message_bus import MessageBus


class TestConsensusEngine:
    """Tester för ConsensusEngine (Sprint 5)."""
    
    def setup_method(self):
        """Körs före varje test."""
        self.message_bus = MessageBus()
    
    def test_consensus_engine_initialization(self):
        """CE-001: Testa att consensus engine initialiseras korrekt."""
        engine = ConsensusEngine(self.message_bus, consensus_model='weighted')
        
        assert engine is not None
        assert engine.message_bus == self.message_bus
        assert engine.consensus_model == 'weighted'
        assert engine.threshold == 0.6
    
    def test_majority_consensus(self):
        """CE-002: Testa majoritetskonsensus."""
        engine = ConsensusEngine(self.message_bus, consensus_model='majority')
        
        vote_matrix = {
            'votes': [
                {'agent_id': 'agent1', 'action': 'BUY', 'symbol': 'AAPL', 'confidence': 0.8, 'weight': 0.8},
                {'agent_id': 'agent2', 'action': 'BUY', 'symbol': 'AAPL', 'confidence': 0.7, 'weight': 0.7},
                {'agent_id': 'agent3', 'action': 'SELL', 'symbol': 'AAPL', 'confidence': 0.6, 'weight': 0.6}
            ],
            'vote_summary': {'BUY': 0.7, 'SELL': 0.3},
            'consensus_strength': 0.7,
            'num_voters': 3
        }
        
        decision = engine.make_consensus_decision(vote_matrix)
        
        assert decision['action'] == 'BUY'  # Majoriteten röstar BUY
        assert decision['symbol'] == 'AAPL'
        assert decision['consensus_model'] == 'majority'
    
    def test_weighted_consensus(self):
        """CE-003: Testa viktad konsensus."""
        engine = ConsensusEngine(self.message_bus, consensus_model='weighted')
        
        vote_matrix = {
            'votes': [
                {'agent_id': 'agent1', 'action': 'BUY', 'symbol': 'MSFT', 'confidence': 0.9, 'weight': 0.9},
                {'agent_id': 'agent2', 'action': 'SELL', 'symbol': 'MSFT', 'confidence': 0.5, 'weight': 0.5}
            ],
            'vote_summary': {'BUY': 0.65, 'SELL': 0.35},
            'consensus_strength': 0.65,
            'num_voters': 2
        }
        
        decision = engine.make_consensus_decision(vote_matrix)
        
        assert decision['action'] == 'BUY'  # Högre viktad röst
        assert 'confidence' in decision
        assert decision['consensus_model'] == 'weighted'
    
    def test_unanimous_consensus_all_agree(self):
        """CE-004: Testa unanimitet när alla är överens."""
        engine = ConsensusEngine(self.message_bus, consensus_model='unanimous')
        
        vote_matrix = {
            'votes': [
                {'agent_id': 'agent1', 'action': 'BUY', 'symbol': 'GOOGL', 'confidence': 0.8, 'weight': 0.8},
                {'agent_id': 'agent2', 'action': 'BUY', 'symbol': 'GOOGL', 'confidence': 0.9, 'weight': 0.9}
            ],
            'vote_summary': {'BUY': 1.0},
            'consensus_strength': 1.0,
            'num_voters': 2
        }
        
        decision = engine.make_consensus_decision(vote_matrix)
        
        assert decision['action'] == 'BUY'
        assert decision['confidence'] == 1.0  # Full confidence vid unanimitet
    
    def test_unanimous_consensus_no_agreement(self):
        """CE-005: Testa unanimitet när inte alla är överens."""
        engine = ConsensusEngine(self.message_bus, consensus_model='unanimous')
        
        vote_matrix = {
            'votes': [
                {'agent_id': 'agent1', 'action': 'BUY', 'symbol': 'AMZN', 'confidence': 0.8, 'weight': 0.8},
                {'agent_id': 'agent2', 'action': 'SELL', 'symbol': 'AMZN', 'confidence': 0.7, 'weight': 0.7}
            ],
            'vote_summary': {'BUY': 0.53, 'SELL': 0.47},
            'consensus_strength': 0.53,
            'num_voters': 2
        }
        
        decision = engine.make_consensus_decision(vote_matrix)
        
        assert decision['action'] == 'HOLD'  # Inget unanimt beslut
        assert decision['confidence'] == 0.0
    
    def test_threshold_consensus_met(self):
        """CE-006: Testa threshold-konsensus när tröskelvärde uppnås."""
        engine = ConsensusEngine(self.message_bus, consensus_model='threshold')
        engine.set_threshold(0.6)
        
        vote_matrix = {
            'votes': [
                {'agent_id': 'agent1', 'action': 'BUY', 'symbol': 'TSLA', 'confidence': 0.8, 'weight': 0.8},
                {'agent_id': 'agent2', 'action': 'BUY', 'symbol': 'TSLA', 'confidence': 0.7, 'weight': 0.7},
                {'agent_id': 'agent3', 'action': 'SELL', 'symbol': 'TSLA', 'confidence': 0.5, 'weight': 0.5}
            ],
            'vote_summary': {'BUY': 0.75, 'SELL': 0.25},
            'consensus_strength': 0.75,
            'num_voters': 3
        }
        
        decision = engine.make_consensus_decision(vote_matrix)
        
        assert decision['action'] == 'BUY'  # 0.75 > 0.6 threshold
    
    def test_threshold_consensus_not_met(self):
        """CE-007: Testa threshold-konsensus när tröskelvärde ej uppnås."""
        engine = ConsensusEngine(self.message_bus, consensus_model='threshold')
        engine.set_threshold(0.8)
        
        vote_matrix = {
            'votes': [
                {'agent_id': 'agent1', 'action': 'BUY', 'symbol': 'NVDA', 'confidence': 0.7, 'weight': 0.7},
                {'agent_id': 'agent2', 'action': 'SELL', 'symbol': 'NVDA', 'confidence': 0.6, 'weight': 0.6}
            ],
            'vote_summary': {'BUY': 0.54, 'SELL': 0.46},
            'consensus_strength': 0.54,
            'num_voters': 2
        }
        
        decision = engine.make_consensus_decision(vote_matrix)
        
        assert decision['action'] == 'HOLD'  # 0.54 < 0.8 threshold
    
    def test_robustness_calculation(self):
        """CE-008: Testa beräkning av robusthet."""
        engine = ConsensusEngine(self.message_bus, consensus_model='weighted')
        
        vote_matrix = {
            'votes': [
                {'agent_id': 'agent1', 'action': 'BUY', 'symbol': 'META', 'confidence': 0.9, 'weight': 0.9},
                {'agent_id': 'agent2', 'action': 'BUY', 'symbol': 'META', 'confidence': 0.8, 'weight': 0.8},
                {'agent_id': 'agent3', 'action': 'BUY', 'symbol': 'META', 'confidence': 0.85, 'weight': 0.85}
            ],
            'vote_summary': {'BUY': 1.0},
            'consensus_strength': 1.0,
            'num_voters': 3
        }
        
        decision = engine.make_consensus_decision(vote_matrix)
        
        assert 'robustness' in decision
        assert 0.0 <= decision['robustness'] <= 1.0
        # Hög consensus strength + flera röster = hög robusthet
        assert decision['robustness'] > 0.7
    
    def test_empty_vote_matrix(self):
        """CE-009: Testa beslut med tom röstmatris."""
        engine = ConsensusEngine(self.message_bus, consensus_model='weighted')
        
        vote_matrix = {
            'votes': [],
            'vote_summary': {},
            'consensus_strength': 0.0,
            'num_voters': 0
        }
        
        decision = engine.make_consensus_decision(vote_matrix)
        
        assert decision['action'] == 'HOLD'
        assert decision['confidence'] == 0.0
    
    def test_consensus_history_logging(self):
        """CE-010: Testa att konsensusbeslut loggas."""
        engine = ConsensusEngine(self.message_bus, consensus_model='weighted')
        
        vote_matrix = {
            'votes': [{'agent_id': 'agent1', 'action': 'BUY', 'symbol': 'INTC', 'confidence': 0.8, 'weight': 0.8}],
            'vote_summary': {'BUY': 1.0},
            'consensus_strength': 1.0,
            'num_voters': 1
        }
        
        initial_count = len(engine.consensus_history)
        engine.make_consensus_decision(vote_matrix)
        
        assert len(engine.consensus_history) == initial_count + 1
    
    def test_consensus_statistics(self):
        """CE-011: Testa beräkning av konsensusstatistik."""
        engine = ConsensusEngine(self.message_bus, consensus_model='weighted')
        
        # Kör några beslut
        vote_matrices = [
            {
                'votes': [{'agent_id': 'agent1', 'action': 'BUY', 'symbol': 'AMD', 'confidence': 0.8, 'weight': 0.8}],
                'vote_summary': {'BUY': 1.0},
                'consensus_strength': 1.0,
                'num_voters': 1
            },
            {
                'votes': [{'agent_id': 'agent1', 'action': 'SELL', 'symbol': 'AMD', 'confidence': 0.7, 'weight': 0.7}],
                'vote_summary': {'SELL': 1.0},
                'consensus_strength': 1.0,
                'num_voters': 1
            }
        ]
        
        for matrix in vote_matrices:
            engine.make_consensus_decision(matrix)
        
        stats = engine.get_consensus_statistics()
        
        assert stats['total_decisions'] == 2
        assert 'action_distribution' in stats
        assert 'average_confidence' in stats
        assert 'average_robustness' in stats
    
    def test_set_consensus_model(self):
        """CE-012: Testa att ändra konsensusmodell."""
        engine = ConsensusEngine(self.message_bus, consensus_model='majority')
        
        assert engine.consensus_model == 'majority'
        
        engine.set_consensus_model('unanimous')
        
        assert engine.consensus_model == 'unanimous'
    
    def test_set_threshold(self):
        """CE-013: Testa att ändra tröskelvärde."""
        engine = ConsensusEngine(self.message_bus, consensus_model='threshold')
        
        assert engine.threshold == 0.6
        
        engine.set_threshold(0.75)
        
        assert engine.threshold == 0.75
    
    def test_publish_decision(self):
        """CE-014: Testa publicering av konsensusbeslut."""
        engine = ConsensusEngine(self.message_bus, consensus_model='weighted')
        
        received_decision = None
        
        def capture_decision(decision):
            nonlocal received_decision
            received_decision = decision
        
        self.message_bus.subscribe('final_decision', capture_decision)
        
        decision = {
            'action': 'BUY',
            'symbol': 'NFLX',
            'confidence': 0.8,
            'robustness': 0.9
        }
        
        engine.publish_decision(decision)
        
        assert received_decision is not None
        assert received_decision['action'] == 'BUY'
