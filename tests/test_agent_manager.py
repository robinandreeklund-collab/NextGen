"""
Tester för agent_manager.py

Testar:
- Initialization med default profiles
- Agent update handling
- Version management
- Profile publishing
- Evolution suggestions
- Rollback functionality
- Evolution tree generation
"""

import pytest
from modules.agent_manager import AgentManager
from modules.message_bus import MessageBus


class TestAgentManager:
    """Tester för AgentManager."""
    
    def setup_method(self):
        """Initiera test environment."""
        self.message_bus = MessageBus()
        self.agent_manager = AgentManager(self.message_bus)
    
    def test_initialization(self):
        """Testar att agent manager initialiseras med default profiles."""
        assert self.agent_manager is not None
        assert len(self.agent_manager.agent_profiles) > 0
        assert 'strategy_agent' in self.agent_manager.agent_profiles
        assert 'risk_agent' in self.agent_manager.agent_profiles
        assert 'decision_agent' in self.agent_manager.agent_profiles
        assert 'execution_agent' in self.agent_manager.agent_profiles
    
    def test_default_agent_profiles(self):
        """Testar att default profiles är korrekt konfigurerade."""
        strategy_profile = self.agent_manager.agent_profiles['strategy_agent']
        
        assert strategy_profile['name'] == 'Strategy Agent'
        assert strategy_profile['module'] == 'strategy_engine'
        assert strategy_profile['uses_rl'] is True
        assert strategy_profile['version'] == '1.0.0'
        assert 'created_at' in strategy_profile
    
    def test_evolution_suggestion_handling(self):
        """Testar hantering av evolution suggestions."""
        update = {
            'update_type': 'evolution_suggestion',
            'agent_id': 'strategy_agent',
            'evolution_suggestion': {
                'analysis': {
                    'reason': 'performance_degradation'
                },
                'suggestions': ['Justera learning rate']
            }
        }
        
        # Prenumerera på agent_profile
        profiles = []
        def capture_profile(profile):
            profiles.append(profile)
        
        self.message_bus.subscribe('agent_profile', capture_profile)
        self.message_bus.publish('agent_update', update)
        
        # Verifiera att version uppdaterades
        profile = self.agent_manager.agent_profiles['strategy_agent']
        assert profile['version'] == '1.0.1'  # Incremented patch version
        assert 'last_evolution' in profile
        
        # Verifiera att profile publicerades
        assert len(profiles) > 0
    
    def test_version_increment_patch(self):
        """Testar patch version increment."""
        new_version = self.agent_manager._increment_version('1.0.0', minor=False)
        assert new_version == '1.0.1'
        
        new_version = self.agent_manager._increment_version('1.0.5', minor=False)
        assert new_version == '1.0.6'
    
    def test_get_agent_profile(self):
        """Testar att hämta specifik agent profile."""
        profile = self.agent_manager.get_agent_profile('strategy_agent')
        
        assert profile is not None
        assert profile['name'] == 'Strategy Agent'
        assert profile['version'] == '1.0.0'
    
    def test_get_all_profiles(self):
        """Testar att hämta alla profiles."""
        all_profiles = self.agent_manager.get_all_profiles()
        
        assert len(all_profiles) == 4  # Default 4 agents
        assert 'strategy_agent' in all_profiles
        assert 'risk_agent' in all_profiles
    
    def test_get_evolution_tree(self):
        """Testar generering av evolution tree."""
        # Skapa flera versioner
        for i in range(3):
            update = {
                'update_type': 'evolution_suggestion',
                'agent_id': 'strategy_agent',
                'evolution_suggestion': {
                    'analysis': {'reason': f'test_{i}'},
                    'suggestions': []
                }
            }
            self.message_bus.publish('agent_update', update)
        
        tree = self.agent_manager.get_evolution_tree()
        
        assert tree['total_agents'] == 4
        assert 'strategy_agent' in tree['agents']
        assert tree['agents']['strategy_agent']['total_versions'] > 1

