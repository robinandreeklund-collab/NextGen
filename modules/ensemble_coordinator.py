"""
ensemble_coordinator.py - Ensemble Coordinator for Multi-Agent System

Beskrivning:
    Koordinerar beslut från PPO, DQN, DT, GAN och GNN agenter.
    Hanterar konfliktlösning och ensemble voting för final decisions.
    
Roll:
    - Samlar beslut från alla RL-agenter (PPO, DQN, DT)
    - Tar in förslag från GAN evolution och GNN analys
    - Weighted ensemble för consensus
    - Konfliktdetektion och resolution
    - Publicerar ensemble decisions
    
Inputs:
    - ppo_action: från rl_controller
    - dqn_action: från dqn_controller
    - dt_action: från decision_transformer_agent
    - gan_candidate: från gan_evolution_engine
    - gnn_insight: från gnn_timespan_analyzer
    
Outputs:
    - ensemble_decision: Weighted ensemble decision
    - ensemble_metrics: Performance comparison
    
Publicerar till message_bus:
    - ensemble_decision: Final ensemble action
    - ensemble_metrics: Agent comparison metrics
    - ensemble_conflict: Conflict notifications
    
Prenumererar på:
    - ppo_action
    - dqn_action  
    - dt_action
    - gan_candidate
    - gnn_insight
    - parameter_adjustment
    
Används i Sprint: 10 (Decision Transformer Integration)
"""

from typing import Dict, Any, List, Tuple
import numpy as np
from collections import deque
import time


class EnsembleCoordinator:
    """
    Koordinerar ensemble av PPO, DQN, DT, GAN och GNN agenter.
    
    Använder weighted voting för final decisions och hanterar konflikter.
    """
    
    def __init__(self, message_bus, 
                 ppo_weight: float = 0.3,
                 dqn_weight: float = 0.3,
                 dt_weight: float = 0.2,
                 gan_weight: float = 0.1,
                 gnn_weight: float = 0.1,
                 conflict_threshold: float = 0.3):
        """
        Initialize ensemble coordinator
        
        Args:
            message_bus: Central message bus
            ppo_weight: Weight for PPO in ensemble (default 0.3)
            dqn_weight: Weight for DQN in ensemble (default 0.3)
            dt_weight: Weight for DT in ensemble (default 0.2)
            gan_weight: Weight for GAN in ensemble (default 0.1)
            gnn_weight: Weight for GNN in ensemble (default 0.1)
            conflict_threshold: Threshold for detecting conflicts
        """
        self.message_bus = message_bus
        
        # Agent weights (must sum to 1.0)
        self.weights = {
            'ppo': ppo_weight,
            'dqn': dqn_weight,
            'dt': dt_weight,
            'gan': gan_weight,
            'gnn': gnn_weight
        }
        self._normalize_weights()
        
        self.conflict_threshold = conflict_threshold
        
        # Latest actions from each agent
        self.agent_actions = {
            'ppo': None,
            'dqn': None,
            'dt': None,
            'gan': None,
            'gnn': None
        }
        
        # Performance tracking
        self.agent_performance = {
            'ppo': deque(maxlen=100),
            'dqn': deque(maxlen=100),
            'dt': deque(maxlen=100),
            'gan': deque(maxlen=100),
            'gnn': deque(maxlen=100)
        }
        
        # Conflict history
        self.conflicts = deque(maxlen=100)
        self.ensemble_history = deque(maxlen=100)
        
        # Subscribe to agent actions
        self.message_bus.subscribe('ppo_action', self._on_ppo_action)
        self.message_bus.subscribe('dqn_action', self._on_dqn_action)
        self.message_bus.subscribe('dt_action', self._on_dt_action)
        self.message_bus.subscribe('gan_candidate', self._on_gan_candidate)
        self.message_bus.subscribe('gnn_insights', self._on_gnn_insight)
        self.message_bus.subscribe('parameter_adjustment', self._on_parameter_adjustment)
        self.message_bus.subscribe('reward', self._on_reward)
        
    def _normalize_weights(self) -> None:
        """Normalize weights to sum to 1.0"""
        total = sum(self.weights.values())
        if total > 0:
            for key in self.weights:
                self.weights[key] /= total
                
    def _on_ppo_action(self, action_data: Dict[str, Any]) -> None:
        """Handle PPO action"""
        self.agent_actions['ppo'] = action_data
        self._try_create_ensemble_decision()
        
    def _on_dqn_action(self, action_data: Dict[str, Any]) -> None:
        """Handle DQN action"""
        self.agent_actions['dqn'] = action_data
        self._try_create_ensemble_decision()
        
    def _on_dt_action(self, action_data: Dict[str, Any]) -> None:
        """Handle Decision Transformer action"""
        self.agent_actions['dt'] = action_data
        self._try_create_ensemble_decision()
        
    def _on_gan_candidate(self, candidate_data: Dict[str, Any]) -> None:
        """Handle GAN candidate (converted to action suggestion)"""
        # GAN candidates are parameter suggestions, not direct actions
        # Convert to action influence if applicable
        self.agent_actions['gan'] = candidate_data
        
    def _on_gnn_insight(self, insight_data: Dict[str, Any]) -> None:
        """Handle GNN temporal insight (converted to action suggestion)"""
        # GNN provides pattern insights that influence actions
        self.agent_actions['gnn'] = insight_data
        
    def _on_parameter_adjustment(self, params: Dict[str, Any]) -> None:
        """Handle adaptive parameter adjustments"""
        if 'ppo_weight' in params:
            self.weights['ppo'] = params['ppo_weight']
        if 'dqn_weight' in params:
            self.weights['dqn'] = params['dqn_weight']
        if 'dt_weight' in params:
            self.weights['dt'] = params['dt_weight']
        if 'gan_weight' in params:
            self.weights['gan'] = params['gan_weight']
        if 'gnn_weight' in params:
            self.weights['gnn'] = params['gnn_weight']
            
        self._normalize_weights()
        
    def _on_reward(self, reward_data: Dict[str, Any]) -> None:
        """Update performance tracking with rewards"""
        reward = reward_data.get('reward', 0.0)
        
        # Attribute reward to agents that participated in last decision
        if len(self.ensemble_history) > 0:
            last_decision = self.ensemble_history[-1]
            participating_agents = last_decision.get('participating_agents', [])
            
            for agent in participating_agents:
                if agent in self.agent_performance:
                    self.agent_performance[agent].append(reward)
                    
    def _try_create_ensemble_decision(self) -> None:
        """
        Try to create ensemble decision if we have enough agent inputs.
        
        Requires at least PPO, DQN, or DT to make a decision.
        """
        # Check if we have at least one RL agent action
        rl_agents_available = [
            self.agent_actions['ppo'] is not None,
            self.agent_actions['dqn'] is not None,
            self.agent_actions['dt'] is not None
        ]
        
        if not any(rl_agents_available):
            return  # Not enough data yet
            
        # Create ensemble decision
        decision = self.create_ensemble_decision()
        
        # Detect conflicts
        conflicts = self.detect_conflicts()
        if conflicts:
            self.conflicts.append({
                'timestamp': time.time(),
                'conflicts': conflicts,
                'resolution': decision['resolution_method']
            })
            self.message_bus.publish('ensemble_conflict', {
                'conflicts': conflicts,
                'resolution': decision
            })
        
        # Publish decision
        self.message_bus.publish('ensemble_decision', decision)
        
        # Publish metrics
        metrics = self.get_ensemble_metrics()
        self.message_bus.publish('ensemble_metrics', metrics)
        
        # Store in history
        self.ensemble_history.append(decision)
        
    def create_ensemble_decision(self) -> Dict[str, Any]:
        """
        Create weighted ensemble decision from agent actions.
        
        Returns:
            Dictionary with ensemble decision and metadata
        """
        # Extract actions from agents
        actions = {}
        confidences = {}
        participating_agents = []
        
        for agent, data in self.agent_actions.items():
            if data is not None:
                # Extract action (different agents may have different formats)
                action = self._extract_action(agent, data)
                confidence = data.get('confidence', 0.5)
                
                if action is not None:
                    actions[agent] = action
                    confidences[agent] = confidence
                    participating_agents.append(agent)
        
        if not actions:
            # No actions available - default to HOLD
            return {
                'action': 'HOLD',
                'action_vector': np.array([1.0, 0.0, 0.0]),
                'confidence': 0.0,
                'participating_agents': [],
                'resolution_method': 'default',
                'timestamp': time.time()
            }
        
        # Weighted voting
        action_vectors = {}
        for agent, action in actions.items():
            action_vectors[agent] = self._action_to_vector(action)
        
        # Compute weighted average
        weighted_sum = np.zeros(3)  # HOLD, BUY, SELL
        total_weight = 0.0
        
        for agent, vector in action_vectors.items():
            weight = self.weights.get(agent, 0.0) * confidences.get(agent, 0.5)
            weighted_sum += weight * vector
            total_weight += weight
        
        if total_weight > 0:
            weighted_avg = weighted_sum / total_weight
        else:
            weighted_avg = np.array([1.0, 0.0, 0.0])  # Default to HOLD
        
        # Convert to action
        final_action = self._vector_to_action(weighted_avg)
        
        # Calculate ensemble confidence
        ensemble_confidence = self._calculate_ensemble_confidence(
            actions, confidences, weighted_avg
        )
        
        return {
            'action': final_action,
            'action_vector': weighted_avg.tolist(),
            'confidence': ensemble_confidence,
            'participating_agents': participating_agents,
            'agent_actions': actions,
            'agent_confidences': confidences,
            'weights': {k: v for k, v in self.weights.items() if k in participating_agents},
            'resolution_method': 'weighted_average',
            'timestamp': time.time()
        }
    
    def _extract_action(self, agent: str, data: Dict[str, Any]) -> Any:
        """Extract action from agent-specific data format"""
        if agent in ['ppo', 'dqn', 'dt']:
            # RL agents return action or action_type
            return data.get('action_type', data.get('action'))
        elif agent == 'gan':
            # GAN provides candidate parameters, not direct actions
            # Return None to exclude from voting
            return None
        elif agent == 'gnn':
            # GNN provides insights, not direct actions
            # Return None to exclude from voting
            return None
        return None
    
    def _action_to_vector(self, action: Any) -> np.ndarray:
        """Convert action to vector [HOLD, BUY, SELL]"""
        if isinstance(action, str):
            action_map = {'HOLD': 0, 'BUY': 1, 'SELL': 2}
            idx = action_map.get(action.upper(), 0)
            vector = np.zeros(3)
            vector[idx] = 1.0
            return vector
        elif isinstance(action, (list, np.ndarray)):
            return np.array(action)
        else:
            # Unknown format - default to HOLD
            return np.array([1.0, 0.0, 0.0])
    
    def _vector_to_action(self, vector: np.ndarray) -> str:
        """Convert action vector to string"""
        idx = np.argmax(vector)
        action_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
        return action_map.get(idx, 'HOLD')
    
    def _calculate_ensemble_confidence(self, actions: Dict[str, Any],
                                       confidences: Dict[str, float],
                                       weighted_avg: np.ndarray) -> float:
        """
        Calculate ensemble confidence based on agreement and individual confidences.
        
        High confidence when:
        - Agents agree on action
        - Individual confidences are high
        - Final weighted average is clear (not ambiguous)
        """
        if not actions:
            return 0.0
        
        # Agreement score: how many agents agree with final action
        final_action_idx = np.argmax(weighted_avg)
        agreement_count = 0
        
        for agent, action in actions.items():
            action_vec = self._action_to_vector(action)
            if np.argmax(action_vec) == final_action_idx:
                agreement_count += 1
        
        agreement_score = agreement_count / len(actions)
        
        # Average individual confidence
        avg_confidence = np.mean(list(confidences.values()))
        
        # Clarity of final decision (entropy)
        # Lower entropy = more clear decision
        probs = np.abs(weighted_avg)
        probs = probs / (probs.sum() + 1e-8)
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        max_entropy = np.log(3)  # log(num_actions)
        clarity = 1.0 - (entropy / max_entropy)
        
        # Combined confidence
        ensemble_confidence = 0.4 * agreement_score + 0.3 * avg_confidence + 0.3 * clarity
        
        return float(ensemble_confidence)
    
    def detect_conflicts(self) -> List[Dict[str, Any]]:
        """
        Detect conflicts between agent actions.
        
        Returns:
            List of detected conflicts
        """
        conflicts = []
        
        # Get actions from RL agents
        rl_actions = {}
        for agent in ['ppo', 'dqn', 'dt']:
            if self.agent_actions[agent] is not None:
                action = self._extract_action(agent, self.agent_actions[agent])
                if action is not None:
                    rl_actions[agent] = action
        
        if len(rl_actions) < 2:
            return []  # No conflict possible with < 2 agents
        
        # Check for disagreements
        action_types = list(rl_actions.values())
        unique_actions = set(str(a) for a in action_types)
        
        if len(unique_actions) > 1:
            # Agents disagree
            conflicts.append({
                'type': 'action_disagreement',
                'agents': list(rl_actions.keys()),
                'actions': rl_actions,
                'severity': 'high' if 'BUY' in unique_actions and 'SELL' in unique_actions else 'medium'
            })
        
        return conflicts
    
    def get_ensemble_metrics(self) -> Dict[str, Any]:
        """
        Get current ensemble performance metrics.
        
        Returns:
            Dictionary of ensemble metrics
        """
        metrics = {
            'weights': self.weights.copy(),
            'agent_performance': {},
            'agent_accuracy': {},
            'agent_rewards': {},
            'conflict_rate': 0.0,
            'ensemble_diversity': 0.0,
            'timestamp': time.time()
        }
        
        # Calculate performance metrics for each agent
        for agent, perf_history in self.agent_performance.items():
            if len(perf_history) > 0:
                metrics['agent_performance'][agent] = {
                    'avg_reward': float(np.mean(perf_history)),
                    'total_reward': float(np.sum(perf_history)),
                    'num_episodes': len(perf_history)
                }
                metrics['agent_rewards'][f'{agent}_rewards'] = float(np.sum(perf_history))
                
                # Simple accuracy estimate (positive rewards = correct decisions)
                correct = sum(1 for r in perf_history if r > 0)
                metrics['agent_accuracy'][f'{agent}_accuracy'] = correct / len(perf_history)
        
        # Calculate conflict rate
        if len(self.ensemble_history) > 0:
            recent_decisions = list(self.ensemble_history)[-20:]
            conflicts_count = len([d for d in recent_decisions 
                                  if d.get('resolution_method') == 'conflict_resolution'])
            metrics['conflict_rate'] = conflicts_count / len(recent_decisions)
        
        # Calculate ensemble diversity (variance in agent actions)
        if len(self.ensemble_history) > 0:
            last_decision = self.ensemble_history[-1]
            agent_actions = last_decision.get('agent_actions', {})
            
            if len(agent_actions) > 1:
                action_vectors = [self._action_to_vector(a) for a in agent_actions.values()]
                action_matrix = np.array(action_vectors)
                diversity = np.std(action_matrix)
                metrics['ensemble_diversity'] = float(diversity)
        
        return metrics
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about ensemble performance"""
        return {
            'total_decisions': len(self.ensemble_history),
            'total_conflicts': len(self.conflicts),
            'agent_weights': self.weights.copy(),
            'metrics': self.get_ensemble_metrics()
        }
