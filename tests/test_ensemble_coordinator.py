"""
Tests for Ensemble Coordinator
"""

import pytest
import numpy as np
from modules.ensemble_coordinator import EnsembleCoordinator
from modules.message_bus import MessageBus


class TestEnsembleCoordinator:
    """Tests for EnsembleCoordinator"""
    
    def test_initialization(self):
        """Test ensemble coordinator initialization"""
        message_bus = MessageBus()
        coordinator = EnsembleCoordinator(message_bus)
        
        assert coordinator is not None
        assert coordinator.weights['ppo'] == 0.3
        assert coordinator.weights['dqn'] == 0.3
        assert coordinator.weights['dt'] == 0.2
        assert coordinator.weights['gan'] == 0.1
        assert coordinator.weights['gnn'] == 0.1
        assert sum(coordinator.weights.values()) == pytest.approx(1.0)
        
    def test_custom_weights(self):
        """Test initialization with custom weights"""
        message_bus = MessageBus()
        coordinator = EnsembleCoordinator(
            message_bus,
            ppo_weight=0.4,
            dqn_weight=0.3,
            dt_weight=0.2,
            gan_weight=0.05,
            gnn_weight=0.05
        )
        
        assert coordinator.weights['ppo'] == 0.4
        assert sum(coordinator.weights.values()) == pytest.approx(1.0)
        
    def test_weight_normalization(self):
        """Test that weights are normalized to sum to 1.0"""
        message_bus = MessageBus()
        coordinator = EnsembleCoordinator(
            message_bus,
            ppo_weight=1.0,
            dqn_weight=1.0,
            dt_weight=1.0,
            gan_weight=0.0,
            gnn_weight=0.0
        )
        
        # Should be normalized
        assert coordinator.weights['ppo'] == pytest.approx(1/3)
        assert coordinator.weights['dqn'] == pytest.approx(1/3)
        assert coordinator.weights['dt'] == pytest.approx(1/3)
        assert sum(coordinator.weights.values()) == pytest.approx(1.0)
        
    def test_ppo_action_handling(self):
        """Test handling of PPO action"""
        message_bus = MessageBus()
        coordinator = EnsembleCoordinator(message_bus)
        
        ppo_action = {
            'action': 'BUY',
            'action_type': 'BUY',
            'confidence': 0.8
        }
        
        message_bus.publish('ppo_action', ppo_action)
        
        assert coordinator.agent_actions['ppo'] is not None
        assert coordinator.agent_actions['ppo']['action_type'] == 'BUY'
        
    def test_dqn_action_handling(self):
        """Test handling of DQN action"""
        message_bus = MessageBus()
        coordinator = EnsembleCoordinator(message_bus)
        
        dqn_action = {
            'action': 'SELL',
            'action_type': 'SELL',
            'confidence': 0.7
        }
        
        message_bus.publish('dqn_action', dqn_action)
        
        assert coordinator.agent_actions['dqn'] is not None
        
    def test_dt_action_handling(self):
        """Test handling of DT action"""
        message_bus = MessageBus()
        coordinator = EnsembleCoordinator(message_bus)
        
        dt_action = {
            'action': [0.1, 0.7, 0.2],
            'action_type': 'BUY',
            'confidence': 0.75
        }
        
        message_bus.publish('dt_action', dt_action)
        
        assert coordinator.agent_actions['dt'] is not None
        
    def test_action_to_vector(self):
        """Test action to vector conversion"""
        message_bus = MessageBus()
        coordinator = EnsembleCoordinator(message_bus)
        
        # String action
        hold_vec = coordinator._action_to_vector('HOLD')
        assert np.allclose(hold_vec, [1.0, 0.0, 0.0])
        
        buy_vec = coordinator._action_to_vector('BUY')
        assert np.allclose(buy_vec, [0.0, 1.0, 0.0])
        
        sell_vec = coordinator._action_to_vector('SELL')
        assert np.allclose(sell_vec, [0.0, 0.0, 1.0])
        
        # Array action
        array_action = [0.2, 0.7, 0.1]
        array_vec = coordinator._action_to_vector(array_action)
        assert np.allclose(array_vec, array_action)
        
    def test_vector_to_action(self):
        """Test vector to action conversion"""
        message_bus = MessageBus()
        coordinator = EnsembleCoordinator(message_bus)
        
        hold_vec = np.array([1.0, 0.0, 0.0])
        assert coordinator._vector_to_action(hold_vec) == 'HOLD'
        
        buy_vec = np.array([0.0, 1.0, 0.0])
        assert coordinator._vector_to_action(buy_vec) == 'BUY'
        
        sell_vec = np.array([0.0, 0.0, 1.0])
        assert coordinator._vector_to_action(sell_vec) == 'SELL'
        
    def test_ensemble_decision_single_agent(self):
        """Test ensemble decision with single agent"""
        message_bus = MessageBus()
        coordinator = EnsembleCoordinator(message_bus)
        
        coordinator.agent_actions['ppo'] = {
            'action_type': 'BUY',
            'confidence': 0.8
        }
        
        decision = coordinator.create_ensemble_decision()
        
        assert decision['action'] == 'BUY'
        assert 'ppo' in decision['participating_agents']
        assert decision['confidence'] > 0
        
    def test_ensemble_decision_multiple_agents_agree(self):
        """Test ensemble decision when agents agree"""
        message_bus = MessageBus()
        coordinator = EnsembleCoordinator(message_bus)
        
        coordinator.agent_actions['ppo'] = {
            'action_type': 'BUY',
            'confidence': 0.8
        }
        coordinator.agent_actions['dqn'] = {
            'action_type': 'BUY',
            'confidence': 0.7
        }
        coordinator.agent_actions['dt'] = {
            'action_type': 'BUY',
            'confidence': 0.75
        }
        
        decision = coordinator.create_ensemble_decision()
        
        assert decision['action'] == 'BUY'
        assert len(decision['participating_agents']) == 3
        assert decision['confidence'] > 0.5  # High confidence when agents agree
        
    def test_ensemble_decision_multiple_agents_disagree(self):
        """Test ensemble decision when agents disagree"""
        message_bus = MessageBus()
        coordinator = EnsembleCoordinator(message_bus)
        
        coordinator.agent_actions['ppo'] = {
            'action_type': 'BUY',
            'confidence': 0.8
        }
        coordinator.agent_actions['dqn'] = {
            'action_type': 'SELL',
            'confidence': 0.7
        }
        coordinator.agent_actions['dt'] = {
            'action_type': 'HOLD',
            'confidence': 0.6
        }
        
        decision = coordinator.create_ensemble_decision()
        
        # Should use weighted voting to resolve
        assert decision['action'] in ['BUY', 'SELL', 'HOLD']
        assert len(decision['participating_agents']) == 3
        
    def test_conflict_detection_agreement(self):
        """Test conflict detection when agents agree"""
        message_bus = MessageBus()
        coordinator = EnsembleCoordinator(message_bus)
        
        coordinator.agent_actions['ppo'] = {'action_type': 'BUY', 'confidence': 0.8}
        coordinator.agent_actions['dqn'] = {'action_type': 'BUY', 'confidence': 0.7}
        coordinator.agent_actions['dt'] = {'action_type': 'BUY', 'confidence': 0.75}
        
        conflicts = coordinator.detect_conflicts()
        
        assert len(conflicts) == 0  # No conflicts when all agree
        
    def test_conflict_detection_disagreement(self):
        """Test conflict detection when agents disagree"""
        message_bus = MessageBus()
        coordinator = EnsembleCoordinator(message_bus)
        
        coordinator.agent_actions['ppo'] = {'action_type': 'BUY', 'confidence': 0.8}
        coordinator.agent_actions['dqn'] = {'action_type': 'SELL', 'confidence': 0.7}
        
        conflicts = coordinator.detect_conflicts()
        
        assert len(conflicts) > 0
        assert conflicts[0]['type'] == 'action_disagreement'
        assert conflicts[0]['severity'] == 'high'  # BUY vs SELL is high severity
        
    def test_parameter_adjustment(self):
        """Test parameter adjustment handling"""
        message_bus = MessageBus()
        coordinator = EnsembleCoordinator(message_bus)
        
        params = {
            'ppo_weight': 0.4,
            'dqn_weight': 0.3,
            'dt_weight': 0.3
        }
        
        message_bus.publish('parameter_adjustment', params)
        
        # Weights should be normalized (0.4 + 0.3 + 0.3 + 0.1 + 0.1 = 1.2 -> normalized)
        # ppo: 0.4/1.2 = 0.333...
        assert coordinator.weights['ppo'] == pytest.approx(0.4/1.2)
        assert sum(coordinator.weights.values()) == pytest.approx(1.0)
        
    def test_reward_tracking(self):
        """Test reward tracking for performance"""
        message_bus = MessageBus()
        coordinator = EnsembleCoordinator(message_bus)
        
        # Create a decision
        coordinator.agent_actions['ppo'] = {'action_type': 'BUY', 'confidence': 0.8}
        coordinator.agent_actions['dqn'] = {'action_type': 'BUY', 'confidence': 0.7}
        
        decision = coordinator.create_ensemble_decision()
        coordinator.ensemble_history.append(decision)
        
        # Send reward
        message_bus.publish('reward', {'reward': 5.0})
        
        # Check that reward was tracked
        assert len(coordinator.agent_performance['ppo']) == 1
        assert len(coordinator.agent_performance['dqn']) == 1
        assert coordinator.agent_performance['ppo'][0] == 5.0
        
    def test_ensemble_metrics(self):
        """Test ensemble metrics retrieval"""
        message_bus = MessageBus()
        coordinator = EnsembleCoordinator(message_bus)
        
        # Add some performance data
        coordinator.agent_performance['ppo'].extend([1.0, 2.0, 3.0])
        coordinator.agent_performance['dqn'].extend([0.5, 1.5, 2.5])
        coordinator.agent_performance['dt'].extend([1.5, 2.5, 3.5])
        
        metrics = coordinator.get_ensemble_metrics()
        
        assert 'weights' in metrics
        assert 'agent_performance' in metrics
        assert 'ppo' in metrics['agent_performance']
        assert metrics['agent_performance']['ppo']['avg_reward'] == pytest.approx(2.0)
        
    def test_statistics(self):
        """Test statistics retrieval"""
        message_bus = MessageBus()
        coordinator = EnsembleCoordinator(message_bus)
        
        stats = coordinator.get_statistics()
        
        assert 'total_decisions' in stats
        assert 'total_conflicts' in stats
        assert 'agent_weights' in stats
        assert 'metrics' in stats
        
    def test_full_ensemble_flow(self):
        """Test full ensemble decision flow"""
        message_bus = MessageBus()
        coordinator = EnsembleCoordinator(message_bus)
        
        # Track published decisions
        decisions = []
        metrics_list = []
        
        def capture_decision(d):
            decisions.append(d)
            
        def capture_metrics(m):
            metrics_list.append(m)
        
        message_bus.subscribe('ensemble_decision', capture_decision)
        message_bus.subscribe('ensemble_metrics', capture_metrics)
        
        # Simulate agent actions
        message_bus.publish('ppo_action', {'action_type': 'BUY', 'confidence': 0.8})
        message_bus.publish('dqn_action', {'action_type': 'BUY', 'confidence': 0.75})
        message_bus.publish('dt_action', {'action_type': 'BUY', 'confidence': 0.7})
        
        # Should have triggered ensemble decision
        assert len(decisions) >= 1
        assert len(metrics_list) >= 1
        
        # Check decision quality
        assert decisions[-1]['action'] == 'BUY'
        assert decisions[-1]['confidence'] > 0.5
        
    def test_ensemble_confidence_calculation(self):
        """Test ensemble confidence calculation"""
        message_bus = MessageBus()
        coordinator = EnsembleCoordinator(message_bus)
        
        # High agreement scenario
        actions = {'ppo': 'BUY', 'dqn': 'BUY', 'dt': 'BUY'}
        confidences = {'ppo': 0.9, 'dqn': 0.85, 'dt': 0.8}
        weighted_avg = np.array([0.0, 1.0, 0.0])  # Clear BUY
        
        confidence = coordinator._calculate_ensemble_confidence(
            actions, confidences, weighted_avg
        )
        
        assert confidence > 0.7  # Should be high
        
        # Low agreement scenario
        actions = {'ppo': 'BUY', 'dqn': 'SELL', 'dt': 'HOLD'}
        confidences = {'ppo': 0.5, 'dqn': 0.5, 'dt': 0.5}
        weighted_avg = np.array([0.3, 0.4, 0.3])  # Ambiguous
        
        confidence = coordinator._calculate_ensemble_confidence(
            actions, confidences, weighted_avg
        )
        
        assert confidence < 0.5  # Should be low
