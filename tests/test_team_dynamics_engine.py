"""
Tests for Team Dynamics Engine - Sprint 7
"""

import pytest
import time
from modules.message_bus import MessageBus
from modules.team_dynamics_engine import TeamDynamicsEngine


class TestTeamDynamicsEngine:
    """Test Team Dynamics Engine functionality"""
    
    def setup_method(self):
        """Setup for each test"""
        self.message_bus = MessageBus()
        self.team_dynamics = TeamDynamicsEngine(self.message_bus)
    
    def test_initialization(self):
        """Test team dynamics initialization"""
        assert len(self.team_dynamics.team_patterns) == 4
        assert 'aggressive_trading' in self.team_dynamics.team_patterns
        assert 'conservative_trading' in self.team_dynamics.team_patterns
        assert 'balanced_trading' in self.team_dynamics.team_patterns
    
    def test_form_team(self):
        """Test team formation"""
        # Form a team
        team_id = self.team_dynamics.form_team({
            'team_id': 'test_team',
            'pattern': 'balanced_trading'
        })
        
        assert team_id == 'test_team'
        assert team_id in self.team_dynamics.active_teams
        
        # Check team info
        team_info = self.team_dynamics.get_team_info(team_id)
        assert team_info['pattern'] == 'balanced_trading'
        assert len(team_info['members']) > 0
    
    def test_dissolve_team(self):
        """Test team dissolution"""
        # Form and then dissolve
        team_id = self.team_dynamics.form_team({
            'team_id': 'temp_team',
            'pattern': 'aggressive_trading'
        })
        
        assert team_id in self.team_dynamics.active_teams
        
        self.team_dynamics.dissolve_team({'team_id': team_id})
        
        assert team_id not in self.team_dynamics.active_teams
    
    def test_track_interaction(self):
        """Test agent interaction tracking"""
        # Simulate voting interaction
        self.message_bus.publish('decision_vote', {
            'agent_id': 'strategy_agent',
            'action': 'BUY',
            'confidence': 0.8
        })
        
        time.sleep(0.1)
        
        # Check communication flows
        assert len(self.team_dynamics.communication_flows) > 0
    
    def test_agent_synergies(self):
        """Test agent synergy definitions"""
        synergies = self.team_dynamics.get_synergy_recommendations('strategy_agent')
        
        assert 'risk_agent' in synergies or 'decision_agent' in synergies
    
    def test_team_synergy_calculation(self):
        """Test team synergy score calculation"""
        # Form team with synergistic agents
        team_id = self.team_dynamics.form_team({
            'team_id': 'synergy_team',
            'members': ['strategy_agent', 'risk_agent', 'decision_agent']
        })
        
        team = self.team_dynamics.teams[team_id]
        synergy = self.team_dynamics._calculate_team_synergy(team)
        
        assert 0.0 <= synergy <= 1.0
    
    def test_coordination_score(self):
        """Test coordination score calculation"""
        # Create team
        team_id = self.team_dynamics.form_team({
            'team_id': 'coord_team',
            'members': ['strategy_agent', 'risk_agent']
        })
        
        # Simulate interactions
        self.team_dynamics.agent_interactions['strategy_agent']['risk_agent'] = 5
        self.team_dynamics.agent_interactions['risk_agent']['strategy_agent'] = 5
        
        team = self.team_dynamics.teams[team_id]
        coord_score = self.team_dynamics._calculate_coordination_score(team)
        
        assert coord_score > 0.0
    
    def test_evaluate_team_performance(self):
        """Test team performance evaluation"""
        # Form team
        team_id = self.team_dynamics.form_team({
            'team_id': 'perf_team',
            'pattern': 'balanced_trading'
        })
        
        # Simulate decision
        self.message_bus.publish('final_decision', {
            'decision_id': 'decision_1',
            'action': 'BUY',
            'confidence': 0.8
        })
        
        time.sleep(0.1)
        
        # Check performance tracked
        perf = self.team_dynamics.team_performance[team_id]
        assert perf['decisions_made'] > 0
    
    def test_team_patterns(self):
        """Test different team patterns"""
        patterns = ['aggressive_trading', 'conservative_trading', 'balanced_trading']
        
        for pattern in patterns:
            team_id = self.team_dynamics.form_team({
                'team_id': f'{pattern}_team',
                'pattern': pattern
            })
            
            team_info = self.team_dynamics.get_team_info(team_id)
            assert team_info['pattern'] == pattern
            assert 'resource_boost' in team_info
    
    def test_get_all_teams(self):
        """Test getting all teams info"""
        # Form multiple teams
        self.team_dynamics.form_team({'team_id': 'team1', 'pattern': 'aggressive_trading'})
        self.team_dynamics.form_team({'team_id': 'team2', 'pattern': 'conservative_trading'})
        
        all_teams = self.team_dynamics.get_all_teams()
        
        assert len(all_teams) >= 2
    
    def test_high_performing_teams(self):
        """Test identification of high-performing teams"""
        # Create team
        team_id = self.team_dynamics.form_team({
            'team_id': 'high_perf_team',
            'pattern': 'balanced_trading'
        })
        
        # Set high performance
        perf = self.team_dynamics.team_performance[team_id]
        perf['synergy_score'] = 0.85
        perf['coordination_score'] = 0.80
        
        high_performers = self.team_dynamics.identify_high_performing_teams()
        
        assert team_id in high_performers
    
    def test_recommend_team_pattern(self):
        """Test team pattern recommendation"""
        # High risk context
        pattern = self.team_dynamics.recommend_team_pattern({
            'risk_tolerance': 'high',
            'phase': 'trading'
        })
        assert pattern == 'aggressive_trading'
        
        # Low risk context
        pattern = self.team_dynamics.recommend_team_pattern({
            'risk_tolerance': 'low',
            'phase': 'trading'
        })
        assert pattern == 'conservative_trading'
        
        # Exploration phase
        pattern = self.team_dynamics.recommend_team_pattern({
            'phase': 'exploration'
        })
        assert pattern == 'exploration_phase'
    
    def test_agent_interactions_tracking(self):
        """Test agent interaction counts"""
        # Simulate multiple interactions
        for i in range(5):
            self.message_bus.publish('decision_vote', {
                'agent_id': 'strategy_agent',
                'action': 'BUY',
                'confidence': 0.7
            })
        
        time.sleep(0.1)
        
        interactions = self.team_dynamics.get_agent_interactions()
        assert len(interactions) > 0
    
    def test_dashboard_data(self):
        """Test dashboard data generation"""
        # Form a team
        self.team_dynamics.form_team({
            'team_id': 'dashboard_team',
            'pattern': 'balanced_trading'
        })
        
        dashboard = self.team_dynamics.get_dashboard_data()
        
        assert dashboard['module'] == 'team_dynamics_engine'
        assert dashboard['status'] == 'active'
        assert 'metrics' in dashboard
        assert 'active_teams' in dashboard['metrics']
    
    def test_resource_boost_allocation(self):
        """Test resource boost values for different patterns"""
        # Aggressive pattern has higher boost
        aggressive_id = self.team_dynamics.form_team({
            'team_id': 'aggressive',
            'pattern': 'aggressive_trading'
        })
        
        # Conservative has normal boost
        conservative_id = self.team_dynamics.form_team({
            'team_id': 'conservative',
            'pattern': 'conservative_trading'
        })
        
        aggressive_info = self.team_dynamics.get_team_info(aggressive_id)
        conservative_info = self.team_dynamics.get_team_info(conservative_id)
        
        assert aggressive_info['resource_boost'] > conservative_info['resource_boost']
    
    def test_communication_flow_logging(self):
        """Test communication flow logging"""
        initial_count = len(self.team_dynamics.communication_flows)
        
        # Send multiple votes
        for i in range(3):
            self.message_bus.publish('decision_vote', {
                'agent_id': f'agent_{i}',
                'action': 'HOLD',
                'confidence': 0.5
            })
        
        time.sleep(0.1)
        
        assert len(self.team_dynamics.communication_flows) > initial_count
