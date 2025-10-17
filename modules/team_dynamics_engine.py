"""
Team Dynamics Engine Module - Sprint 7
Koordinerar agentteam och samarbete
"""

import time
from typing import Dict, List, Optional, Any, Set
from collections import defaultdict


class TeamDynamicsEngine:
    """Hanterar teamkoordinering och agentsamarbete"""
    
    def __init__(self, message_bus):
        self.message_bus = message_bus
        
        # Team definitions
        self.teams = {}
        
        # Agent interactions
        self.agent_interactions = defaultdict(lambda: defaultdict(int))
        
        # Team performance
        self.team_performance = defaultdict(lambda: {
            'synergy_score': 0.0,
            'coordination_score': 0.0,
            'efficiency': 0.0,
            'decisions_made': 0
        })
        
        # Agent synergies (from evolution_matrix)
        self.agent_synergies = {
            'strategy_agent': ['risk_agent', 'decision_agent'],
            'risk_agent': ['strategy_agent', 'portfolio_agent'],
            'decision_agent': ['strategy_agent', 'execution_agent'],
            'execution_agent': ['decision_agent', 'portfolio_agent'],
            'meta_parameter_agent': ['all_agents'],
            'reward_tuner_agent': ['rl_controller', 'meta_parameter_agent']
        }
        
        # Team composition patterns
        self.team_patterns = {
            'aggressive_trading': {
                'agents': ['strategy_agent', 'execution_agent'],
                'resource_boost': 1.3,
                'risk_tolerance': 'high'
            },
            'conservative_trading': {
                'agents': ['risk_agent', 'decision_agent'],
                'resource_boost': 1.0,
                'risk_tolerance': 'low'
            },
            'balanced_trading': {
                'agents': ['strategy_agent', 'risk_agent', 'decision_agent'],
                'resource_boost': 1.1,
                'risk_tolerance': 'medium'
            },
            'exploration_phase': {
                'agents': ['strategy_agent', 'meta_parameter_agent'],
                'resource_boost': 0.9,
                'risk_tolerance': 'medium'
            }
        }
        
        # Communication flows
        self.communication_flows = []
        
        # Active teams tracking
        self.active_teams = set()
        
        # Subscribe to relevant topics
        self.message_bus.subscribe('decision_vote', self.track_interaction)
        self.message_bus.subscribe('agent_status', self.track_agent_activity)
        self.message_bus.subscribe('final_decision', self.evaluate_team_performance)
        self.message_bus.subscribe('form_team', self.form_team)
        self.message_bus.subscribe('dissolve_team', self.dissolve_team)
        
        print("[TeamDynamicsEngine] Initialized with 4 team patterns")
    
    def track_interaction(self, data: Dict[str, Any]):
        """Track agent interactions during voting"""
        agent_id = data.get('agent_id', 'unknown')
        action = data.get('action', 'HOLD')
        confidence = data.get('confidence', 0.0)
        
        # Track interaction timestamp
        interaction = {
            'timestamp': time.time(),
            'agent_id': agent_id,
            'action': action,
            'confidence': confidence
        }
        self.communication_flows.append(interaction)
        
        # Keep last 500 interactions
        if len(self.communication_flows) > 500:
            self.communication_flows = self.communication_flows[-500:]
        
        # Update interaction counts between agents
        for other_agent in self.agent_synergies.get(agent_id, []):
            if other_agent != 'all_agents':
                self.agent_interactions[agent_id][other_agent] += 1
    
    def track_agent_activity(self, data: Dict[str, Any]):
        """Track agent status updates"""
        agent_id = data.get('agent_id', 'unknown')
        status = data.get('status', 'active')
        
        # Check if agent is part of an active team
        for team_id in self.active_teams:
            team = self.teams.get(team_id)
            if team and agent_id in team['members']:
                team['last_activity'] = time.time()
    
    def evaluate_team_performance(self, data: Dict[str, Any]):
        """Evaluate team performance after a decision"""
        decision_id = data.get('decision_id', 'unknown')
        action = data.get('action', 'HOLD')
        confidence = data.get('confidence', 0.0)
        
        # Find which team made this decision
        for team_id in self.active_teams:
            team = self.teams.get(team_id)
            if team:
                # Update team performance
                perf = self.team_performance[team_id]
                perf['decisions_made'] += 1
                
                # Calculate synergy based on confidence and participation
                synergy = self._calculate_team_synergy(team)
                perf['synergy_score'] = (perf['synergy_score'] * 0.9 + synergy * 0.1)
                
                # Calculate coordination based on interaction frequency
                coordination = self._calculate_coordination_score(team)
                perf['coordination_score'] = coordination
    
    def _calculate_team_synergy(self, team: Dict[str, Any]) -> float:
        """Calculate synergy score for a team"""
        members = team['members']
        synergy_count = 0
        total_pairs = 0
        
        for i, agent1 in enumerate(members):
            for agent2 in members[i+1:]:
                total_pairs += 1
                # Check if agents have natural synergy
                if agent2 in self.agent_synergies.get(agent1, []):
                    synergy_count += 1
                elif agent1 in self.agent_synergies.get(agent2, []):
                    synergy_count += 1
        
        if total_pairs == 0:
            return 0.0
        
        return synergy_count / total_pairs
    
    def _calculate_coordination_score(self, team: Dict[str, Any]) -> float:
        """Calculate coordination score based on interaction frequency"""
        members = team['members']
        interaction_count = 0
        total_pairs = 0
        
        for i, agent1 in enumerate(members):
            for agent2 in members[i+1:]:
                total_pairs += 1
                # Count interactions between these agents
                count = self.agent_interactions[agent1].get(agent2, 0)
                count += self.agent_interactions[agent2].get(agent1, 0)
                interaction_count += count
        
        if total_pairs == 0:
            return 0.0
        
        # Normalize to 0-1 range (assume 10 interactions = max coordination)
        avg_interactions = interaction_count / total_pairs
        return min(1.0, avg_interactions / 10.0)
    
    def form_team(self, data: Dict[str, Any]):
        """Form a new team"""
        team_id = data.get('team_id', f'team_{len(self.teams)}')
        pattern = data.get('pattern', 'balanced_trading')
        members = data.get('members', [])
        
        # Use pattern if no members specified
        if not members and pattern in self.team_patterns:
            members = self.team_patterns[pattern]['agents']
        
        # Create team
        team = {
            'team_id': team_id,
            'pattern': pattern,
            'members': members,
            'formed_at': time.time(),
            'last_activity': time.time(),
            'resource_boost': self.team_patterns.get(pattern, {}).get('resource_boost', 1.0),
            'risk_tolerance': self.team_patterns.get(pattern, {}).get('risk_tolerance', 'medium')
        }
        
        self.teams[team_id] = team
        self.active_teams.add(team_id)
        
        # Publish team formation
        self.message_bus.publish('team_formed', {
            'team_id': team_id,
            'pattern': pattern,
            'members': members,
            'resource_boost': team['resource_boost'],
            'timestamp': time.time()
        })
        
        print(f"[TeamDynamicsEngine] Formed team '{team_id}' with pattern '{pattern}'")
        
        return team_id
    
    def dissolve_team(self, data: Dict[str, Any]):
        """Dissolve an existing team"""
        team_id = data.get('team_id')
        
        if team_id in self.active_teams:
            self.active_teams.remove(team_id)
            
            # Publish team dissolution
            self.message_bus.publish('team_dissolved', {
                'team_id': team_id,
                'timestamp': time.time()
            })
            
            print(f"[TeamDynamicsEngine] Dissolved team '{team_id}'")
    
    def get_team_info(self, team_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a team"""
        team = self.teams.get(team_id)
        if not team:
            return None
        
        perf = self.team_performance[team_id]
        
        return {
            'team_id': team_id,
            'pattern': team['pattern'],
            'members': team['members'],
            'formed_at': team['formed_at'],
            'last_activity': team['last_activity'],
            'resource_boost': team['resource_boost'],
            'risk_tolerance': team['risk_tolerance'],
            'synergy_score': perf['synergy_score'],
            'coordination_score': perf['coordination_score'],
            'decisions_made': perf['decisions_made']
        }
    
    def get_all_teams(self) -> List[Dict[str, Any]]:
        """Get information about all teams"""
        return [self.get_team_info(team_id) for team_id in self.teams.keys()]
    
    def get_agent_interactions(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get interaction counts for an agent or all agents"""
        if agent_id:
            return dict(self.agent_interactions.get(agent_id, {}))
        return {k: dict(v) for k, v in self.agent_interactions.items()}
    
    def get_synergy_recommendations(self, agent_id: str) -> List[str]:
        """Get recommended team members for an agent based on synergies"""
        return self.agent_synergies.get(agent_id, [])
    
    def identify_high_performing_teams(self, threshold: float = 0.75) -> List[str]:
        """Identify teams with high synergy and coordination"""
        high_performers = []
        
        for team_id in self.active_teams:
            perf = self.team_performance[team_id]
            avg_score = (perf['synergy_score'] + perf['coordination_score']) / 2
            
            if avg_score >= threshold:
                high_performers.append(team_id)
        
        return high_performers
    
    def recommend_team_pattern(self, context: Dict[str, Any]) -> str:
        """Recommend a team pattern based on context"""
        risk_tolerance = context.get('risk_tolerance', 'medium')
        phase = context.get('phase', 'trading')
        
        if phase == 'exploration':
            return 'exploration_phase'
        elif risk_tolerance == 'high':
            return 'aggressive_trading'
        elif risk_tolerance == 'low':
            return 'conservative_trading'
        else:
            return 'balanced_trading'
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get dashboard data for introspection panel"""
        # Calculate average synergy and coordination
        avg_synergy = 0.0
        avg_coordination = 0.0
        
        if self.team_performance:
            avg_synergy = sum(p['synergy_score'] for p in self.team_performance.values()) / len(self.team_performance)
            avg_coordination = sum(p['coordination_score'] for p in self.team_performance.values()) / len(self.team_performance)
        
        # Get active teams count
        active_count = len(self.active_teams)
        
        # Get total interactions
        total_interactions = sum(
            sum(interactions.values())
            for interactions in self.agent_interactions.values()
        )
        
        return {
            'module': 'team_dynamics_engine',
            'status': 'active',
            'metrics': {
                'active_teams': active_count,
                'total_teams': len(self.teams),
                'avg_synergy_score': avg_synergy,
                'avg_coordination_score': avg_coordination,
                'total_interactions': total_interactions,
                'communication_flows': len(self.communication_flows)
            },
            'high_performing_teams': self.identify_high_performing_teams(),
            'timestamp': time.time()
        }
