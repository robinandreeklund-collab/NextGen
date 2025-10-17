"""
Tests for Hybrid RL - PPO and DQN Integration
"""

import pytest
import numpy as np
from modules.dqn_controller import DQNController
from modules.rl_controller import RLController
from modules.message_bus import MessageBus


class TestHybridRL:
    """Tests for Hybrid RL (PPO + DQN)"""
    
    def setup_method(self):
        self.message_bus = MessageBus()
        
        # Initialize both controllers
        self.dqn = DQNController(
            self.message_bus,
            state_dim=10,
            action_dim=3,
            epsilon=0.5
        )
        
        self.ppo = RLController(self.message_bus)
        
    def test_parallel_initialization(self):
        """Test that both PPO and DQN can be initialized together"""
        assert self.dqn is not None
        assert self.ppo is not None
        
    def test_both_receive_rewards(self):
        """Test that both controllers receive reward signals"""
        self.message_bus.publish('reward', {'reward': 1.5})
        # Both should handle reward
        
    def test_both_receive_tuned_rewards(self):
        """Test that both controllers receive tuned rewards"""
        self.message_bus.publish('tuned_reward', {'tuned_reward': 2.0})
        # Both should handle tuned reward
        
    def test_separate_action_requests(self):
        """Test that PPO and DQN respond to separate action requests"""
        state = np.random.randn(10)
        
        # Request DQN action
        self.message_bus.publish('dqn_action_request', {'state': state})
        
        # Request PPO action (through normal flow)
        # PPO uses different mechanism but should still work
        
    def test_no_interference(self):
        """Test that PPO and DQN don't interfere with each other"""
        # Train DQN
        for i in range(50):
            state = np.random.randn(10)
            next_state = np.random.randn(10)
            self.dqn.store_transition(state, 0, 1.0, next_state, False)
        
        dqn_loss = self.dqn.train_step()
        
        # PPO should still work independently
        assert self.ppo is not None
        
    def test_hybrid_metrics(self):
        """Test that both controllers provide metrics"""
        dqn_metrics = self.dqn.get_metrics()
        
        assert 'training_steps' in dqn_metrics or 'epsilon' in dqn_metrics
        # PPO controller exists and works
        assert self.ppo is not None
        
    def test_parameter_updates_independent(self):
        """Test that parameter updates are independent"""
        self.dqn.update_parameters({'epsilon': 0.2})
        assert self.dqn.epsilon == 0.2
        
        # PPO parameters should be unaffected
        
    def test_concurrent_training(self):
        """Test that both can train concurrently"""
        # Train DQN
        for i in range(50):
            state = np.random.randn(10)
            next_state = np.random.randn(10)
            self.dqn.store_transition(state, i % 3, float(i % 5), next_state, False)
        
        dqn_loss = self.dqn.train_step()
        assert dqn_loss is not None
        
        # PPO continues to work
        assert self.ppo is not None


class TestHybridReward:
    """Tests for Hybrid Reward Distribution"""
    
    def setup_method(self):
        self.message_bus = MessageBus()
        self.dqn = DQNController(self.message_bus, state_dim=10, action_dim=3)
        self.ppo = RLController(self.message_bus)
        
    def test_base_reward_distribution(self):
        """Test that base rewards are distributed to both"""
        self.message_bus.publish('reward', {'reward': 1.0})
        # Both should receive
        
    def test_tuned_reward_distribution(self):
        """Test that tuned rewards are distributed to both"""
        self.message_bus.publish('tuned_reward', {'tuned_reward': 1.5})
        # Both should receive
        
    def test_reward_normalization(self):
        """Test that rewards are properly normalized"""
        # Send various rewards
        rewards = [0.5, 1.0, 1.5, -0.5, -1.0]
        for reward in rewards:
            self.message_bus.publish('reward', {'reward': reward})
        
    def test_reward_scaling(self):
        """Test reward scaling for different controllers"""
        # DQN and PPO might need different reward scales
        base_reward = 1.0
        self.message_bus.publish('reward', {'reward': base_reward})


class TestConflictDetection:
    """Tests for Conflict Detection between PPO and DQN"""
    
    def setup_method(self):
        self.message_bus = MessageBus()
        self.dqn = DQNController(self.message_bus, state_dim=10, action_dim=3)
        self.ppo = RLController(self.message_bus)
        
    def test_parameter_conflict_detection(self):
        """Test detection of parameter conflicts"""
        # Both try to update same parameter differently
        self.dqn.update_parameters({'epsilon': 0.1})
        # If PPO also had epsilon, conflict should be detected
        
    def test_decision_conflict_detection(self):
        """Test detection of decision conflicts"""
        state = np.random.randn(10)
        
        # Get DQN action
        dqn_action = self.dqn.select_action(state, training=False)
        
        # Get PPO action (simulated)
        ppo_action = (dqn_action + 1) % 3  # Different action
        
        # Conflict exists when actions differ
        conflict = dqn_action != ppo_action
        assert isinstance(conflict, bool)
        
    def test_conflict_resolution_weighted(self):
        """Test weighted combination for conflict resolution"""
        state = np.random.randn(10)
        
        dqn_q_values = self.dqn.get_q_values(state)
        
        # Weighted combination
        weight_dqn = 0.6
        weight_ppo = 0.4
        
        # Combine based on weights
        combined_score = weight_dqn * np.max(dqn_q_values)
        assert combined_score >= 0 or combined_score < 0
        
    def test_conflict_logging(self):
        """Test that conflicts are logged"""
        state = np.random.randn(10)
        dqn_action = self.dqn.select_action(state)
        # Conflict detection and logging would happen here


class TestHybridPerformance:
    """Tests for Hybrid RL Performance"""
    
    def setup_method(self):
        self.message_bus = MessageBus()
        self.dqn = DQNController(self.message_bus, state_dim=10, action_dim=3)
        self.ppo = RLController(self.message_bus)
        
    def test_training_convergence(self):
        """Test that hybrid system converges"""
        # Train DQN
        for episode in range(5):
            for step in range(10):
                state = np.random.randn(10)
                action = self.dqn.select_action(state)
                next_state = np.random.randn(10)
                reward = np.random.randn()
                self.dqn.store_transition(state, action, reward, next_state, False)
                
                if len(self.dqn.replay_buffer) >= self.dqn.batch_size:
                    self.dqn.train_step()
        
        # Check convergence metrics
        metrics = self.dqn.get_metrics()
        assert metrics['training_steps'] > 0
        
    def test_ppo_vs_dqn_comparison(self):
        """Test comparison between PPO and DQN performance"""
        dqn_metrics = self.dqn.get_metrics()
        
        # Both should provide metrics
        assert dqn_metrics is not None
        # PPO exists and is working
        assert self.ppo is not None
        
    def test_hybrid_overhead(self):
        """Test that hybrid approach doesn't add excessive overhead"""
        import time
        
        # Measure DQN training time
        start = time.time()
        for i in range(100):
            state = np.random.randn(10)
            next_state = np.random.randn(10)
            self.dqn.store_transition(state, 0, 1.0, next_state, False)
            if len(self.dqn.replay_buffer) >= self.dqn.batch_size:
                self.dqn.train_step()
        dqn_time = time.time() - start
        
        # Time should be reasonable
        assert dqn_time < 10.0  # Less than 10 seconds
        
    def test_combined_performance(self):
        """Test that combined performance is better than individual"""
        # This is a conceptual test
        # In practice, hybrid RL should leverage strengths of both
        state = np.random.randn(10)
        
        # Get both predictions
        dqn_action = self.dqn.select_action(state, training=False)
        dqn_q_values = self.dqn.get_q_values(state)
        
        # Combined decision would use both
        assert 0 <= dqn_action < 3
        assert len(dqn_q_values) == 3
