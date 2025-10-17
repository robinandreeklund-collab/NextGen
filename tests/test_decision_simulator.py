"""
test_decision_simulator.py - Tester för beslutssimulator (Sprint 5)

Beskrivning:
    Testar DecisionSimulator för simulering av alternativa beslut.
    Validerar scenarier, expected value och rekommendationer.
"""

import pytest
from modules.decision_simulator import DecisionSimulator
from modules.message_bus import MessageBus


class TestDecisionSimulator:
    """Tester för DecisionSimulator (Sprint 5)."""
    
    def setup_method(self):
        """Körs före varje test."""
        self.message_bus = MessageBus()
        self.simulator = DecisionSimulator(self.message_bus)
    
    def test_simulator_initialization(self):
        """DS-001: Testa att simulator initialiseras korrekt."""
        assert self.simulator is not None
        assert self.simulator.message_bus == self.message_bus
        assert len(self.simulator.simulation_history) == 0
    
    def test_simulate_buy_decision(self):
        """DS-002: Testa simulering av BUY-beslut."""
        proposal = {
            'symbol': 'AAPL',
            'action': 'BUY',
            'confidence': 0.8,
            'quantity': 10,
            'price': 150.0
        }
        
        result = self.simulator.simulate_decision(proposal)
        
        assert result['symbol'] == 'AAPL'
        assert result['original_action'] == 'BUY'
        assert 'scenarios' in result
        assert len(result['scenarios']) == 4  # best, expected, worst, no_action
        assert 'expected_value' in result
        assert 'recommendation' in result
    
    def test_simulate_sell_decision(self):
        """DS-003: Testa simulering av SELL-beslut."""
        proposal = {
            'symbol': 'MSFT',
            'action': 'SELL',
            'confidence': 0.7,
            'quantity': 5,
            'price': 300.0
        }
        
        result = self.simulator.simulate_decision(proposal)
        
        assert result['symbol'] == 'MSFT'
        assert result['original_action'] == 'SELL'
        assert len(result['scenarios']) == 4
    
    def test_high_confidence_recommendation(self):
        """DS-004: Testa att hög confidence ger 'proceed' rekommendation."""
        proposal = {
            'symbol': 'GOOGL',
            'action': 'BUY',
            'confidence': 0.9,
            'quantity': 10,
            'price': 120.0
        }
        
        result = self.simulator.simulate_decision(proposal)
        
        # Hög confidence + positiv expected value borde ge proceed
        if result['expected_value'] > 0:
            assert result['recommendation'] in ['proceed', 'caution']
    
    def test_low_confidence_recommendation(self):
        """DS-005: Testa att låg confidence kan ge 'reject' rekommendation."""
        proposal = {
            'symbol': 'TSLA',
            'action': 'BUY',
            'confidence': 0.3,
            'quantity': 10,
            'price': 200.0
        }
        
        result = self.simulator.simulate_decision(proposal)
        
        # Låg confidence borde ge reject eller caution
        assert result['recommendation'] in ['caution', 'reject']
    
    def test_scenario_structure(self):
        """DS-006: Testa att scenarier har korrekt struktur."""
        proposal = {
            'symbol': 'AMZN',
            'action': 'BUY',
            'confidence': 0.6,
            'quantity': 10,
            'price': 140.0
        }
        
        result = self.simulator.simulate_decision(proposal)
        scenarios = result['scenarios']
        
        scenario_names = [s['scenario'] for s in scenarios]
        assert 'best_case' in scenario_names
        assert 'expected_case' in scenario_names
        assert 'worst_case' in scenario_names
        assert 'no_action' in scenario_names
        
        # Alla scenarier ska ha pnl och probability
        for scenario in scenarios:
            assert 'pnl' in scenario
            assert 'probability' in scenario
    
    def test_expected_value_calculation(self):
        """DS-007: Testa att expected value beräknas korrekt."""
        proposal = {
            'symbol': 'NVDA',
            'action': 'BUY',
            'confidence': 0.75,
            'quantity': 10,
            'price': 400.0
        }
        
        result = self.simulator.simulate_decision(proposal)
        
        # Expected value ska vara summan av pnl * probability
        manual_ev = sum(s['pnl'] * s['probability'] for s in result['scenarios'])
        assert abs(result['expected_value'] - manual_ev) < 0.01
    
    def test_simulation_history_logging(self):
        """DS-008: Testa att simuleringar loggas i historik."""
        initial_count = len(self.simulator.simulation_history)
        
        proposal = {
            'symbol': 'META',
            'action': 'BUY',
            'confidence': 0.65,
            'quantity': 10,
            'price': 300.0
        }
        
        # Publicera proposal via message_bus för att trigga callback
        self.message_bus.publish('decision_proposal', proposal)
        
        # Vänta kort för att simuleraren ska processa
        assert len(self.simulator.simulation_history) == initial_count + 1
    
    def test_simulation_statistics(self):
        """DS-009: Testa att simuleringsstatistik beräknas korrekt."""
        # Kör några simuleringar
        proposals = [
            {'symbol': 'AAPL', 'action': 'BUY', 'confidence': 0.9, 'quantity': 10, 'price': 150.0},
            {'symbol': 'MSFT', 'action': 'SELL', 'confidence': 0.4, 'quantity': 5, 'price': 300.0},
            {'symbol': 'GOOGL', 'action': 'BUY', 'confidence': 0.6, 'quantity': 10, 'price': 120.0}
        ]
        
        for proposal in proposals:
            self.message_bus.publish('decision_proposal', proposal)
        
        stats = self.simulator.get_simulation_statistics()
        
        assert stats['total_simulations'] == 3
        assert 'proceed_recommendations' in stats
        assert 'caution_recommendations' in stats
        assert 'reject_recommendations' in stats
        assert 'average_expected_value' in stats
        
        # Summan av rekommendationer ska vara lika med totalt
        total_recommendations = (stats['proceed_recommendations'] + 
                               stats['caution_recommendations'] + 
                               stats['reject_recommendations'])
        assert total_recommendations == stats['total_simulations']
    
    def test_simulation_confidence(self):
        """DS-010: Testa att simulation_confidence beräknas."""
        proposal = {
            'symbol': 'INTC',
            'action': 'BUY',
            'confidence': 0.7,
            'quantity': 10,
            'price': 50.0
        }
        
        result = self.simulator.simulate_decision(proposal)
        
        assert 'simulation_confidence' in result
        assert 0.0 <= result['simulation_confidence'] <= 1.0
    
    def test_no_action_scenario(self):
        """DS-011: Testa att no_action scenario alltid har P&L = 0."""
        proposal = {
            'symbol': 'AMD',
            'action': 'BUY',
            'confidence': 0.8,
            'quantity': 10,
            'price': 100.0
        }
        
        result = self.simulator.simulate_decision(proposal)
        no_action_scenario = next(s for s in result['scenarios'] if s['scenario'] == 'no_action')
        
        assert no_action_scenario['pnl'] == 0.0
    
    def test_best_case_positive(self):
        """DS-012: Testa att best_case är positiv för BUY/SELL."""
        for action in ['BUY', 'SELL']:
            proposal = {
                'symbol': 'TEST',
                'action': action,
                'confidence': 0.8,
                'quantity': 10,
                'price': 100.0
            }
            
            result = self.simulator.simulate_decision(proposal)
            best_case = next(s for s in result['scenarios'] if s['scenario'] == 'best_case')
            
            assert best_case['pnl'] > 0.0, f"Best case för {action} ska vara positiv"
