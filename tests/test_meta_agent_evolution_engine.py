"""
Tester för meta_agent_evolution_engine.py

Testar:
- Initialization
- Agent status tracking
- Feedback insight processing
- Evolution need detection
- Performance trend analysis
- Evolution suggestions
- Evolution tree generation
"""

import pytest
from modules.meta_agent_evolution_engine import MetaAgentEvolutionEngine
from modules.message_bus import MessageBus


class TestMetaAgentEvolutionEngine:
    """Tester för MetaAgentEvolutionEngine."""
    
    def setup_method(self):
        """Initiera test environment."""
        self.message_bus = MessageBus()
        self.evolution_engine = MetaAgentEvolutionEngine(self.message_bus)
    
    def test_initialization(self):
        """Testar att evolution engine initialiseras korrekt."""
        assert self.evolution_engine is not None
        assert self.evolution_engine.agent_performance_history == {}
        assert self.evolution_engine.feedback_insights == []
        assert self.evolution_engine.evolution_history == []
        assert self.evolution_engine.performance_threshold == 0.15
    
    def test_agent_status_tracking(self):
        """Testar att agent status spåras."""
        status = {
            'agent_id': 'strategy_agent',
            'reward': 0.8,
            'performance': 0.75
        }
        
        self.message_bus.publish('agent_status', status)
        
        assert 'strategy_agent' in self.evolution_engine.agent_performance_history
        assert len(self.evolution_engine.agent_performance_history['strategy_agent']) == 1
        assert 'timestamp' in self.evolution_engine.agent_performance_history['strategy_agent'][0]
    
    def test_feedback_insight_processing(self):
        """Testar att feedback insights processas."""
        insight = {
            'patterns': [{'type': 'agent_drift', 'agent_id': 'strategy_agent'}],
            'recommendations': ['test']
        }
        
        self.message_bus.publish('feedback_insight', insight)
        
        assert len(self.evolution_engine.feedback_insights) == 1
        assert 'timestamp' in self.evolution_engine.feedback_insights[0]
    
    def test_evolution_triggered_by_performance_degradation(self):
        """Testar att evolution triggas vid performance degradation."""
        agent_id = 'strategy_agent'
        
        # Publicera 10 status med hög performance
        for i in range(10):
            status = {
                'agent_id': agent_id,
                'reward': 0.9,
                'performance': 0.85
            }
            self.message_bus.publish('agent_status', status)
        
        # Publicera 10 status med låg performance
        for i in range(10):
            status = {
                'agent_id': agent_id,
                'reward': 0.3,
                'performance': 0.25
            }
            self.message_bus.publish('agent_status', status)
        
        # Evolution borde ha triggats
        assert len(self.evolution_engine.evolution_history) > 0
        assert self.evolution_engine.evolution_history[0]['agent_id'] == agent_id
        assert self.evolution_engine.evolution_history[0]['analysis']['reason'] == 'performance_degradation'
    
    def test_get_agent_performance_trend_improving(self):
        """Testar detektering av improving trend."""
        agent_id = 'strategy_agent'
        
        # Lägg till 10 status med ökande performance
        for i in range(10):
            status = {
                'agent_id': agent_id,
                'reward': 0.5 + (i * 0.05),
                'performance': 0.5 + (i * 0.05)
            }
            self.message_bus.publish('agent_status', status)
        
        trend = self.evolution_engine.get_agent_performance_trend(agent_id)
        
        assert trend['trend'] == 'improving'
        assert trend['data_points'] == 10
        assert trend['change_percentage'] > 0
    
    def test_generate_evolution_tree(self):
        """Testar generering av evolution tree."""
        # Lägg till evolution events
        self.evolution_engine.evolution_history = [
            {'agent_id': 'agent_1', 'analysis': {}},
            {'agent_id': 'agent_1', 'analysis': {}},
            {'agent_id': 'agent_2', 'analysis': {}},
            {'scope': 'system_wide', 'analysis': {}}
        ]
        
        tree = self.evolution_engine.generate_evolution_tree()
        
        assert tree['total_evolution_events'] == 4
        assert 'agent_1' in tree['agents']
        assert 'agent_2' in tree['agents']
        assert tree['agents']['agent_1']['evolution_count'] == 2
        assert tree['agents']['agent_2']['evolution_count'] == 1
        assert len(tree['system_wide_events']) == 1

