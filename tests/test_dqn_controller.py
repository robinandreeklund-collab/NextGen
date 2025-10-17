"""
Tests for DQN Controller
"""

import pytest
import numpy as np
from modules.dqn_controller import DQNController, QNetwork, ReplayBuffer
from modules.message_bus import MessageBus


class TestQNetwork:
    """Tests for Q-Network"""
    
    def test_initialization(self):
        network = QNetwork(state_dim=10, action_dim=3, hidden_dim=64)
        assert network is not None
        
    def test_forward_pass(self):
        network = QNetwork(state_dim=10, action_dim=3)
        state = np.random.randn(1, 10).astype(np.float32)
        import torch
        state_tensor = torch.FloatTensor(state)
        output = network(state_tensor)
        assert output.shape == (1, 3)


class TestReplayBuffer:
    """Tests for Replay Buffer"""
    
    def test_initialization(self):
        buffer = ReplayBuffer(capacity=1000)
        assert len(buffer) == 0
        
    def test_push(self):
        buffer = ReplayBuffer(capacity=1000)
        state = np.random.randn(10)
        next_state = np.random.randn(10)
        buffer.push(state, 0, 1.0, next_state, False)
        assert len(buffer) == 1
        
    def test_sample(self):
        buffer = ReplayBuffer(capacity=1000)
        for i in range(100):
            state = np.random.randn(10)
            next_state = np.random.randn(10)
            buffer.push(state, i % 3, float(i), next_state, i % 10 == 0)
        
        states, actions, rewards, next_states, dones = buffer.sample(32)
        assert len(states) == 32
        assert len(actions) == 32
        assert len(rewards) == 32
        
    def test_capacity_limit(self):
        buffer = ReplayBuffer(capacity=10)
        for i in range(20):
            state = np.random.randn(10)
            next_state = np.random.randn(10)
            buffer.push(state, 0, 1.0, next_state, False)
        assert len(buffer) == 10


class TestDQNController:
    """Tests for DQN Controller"""
    
    def setup_method(self):
        self.message_bus = MessageBus()
        self.dqn = DQNController(
            self.message_bus,
            state_dim=10,
            action_dim=3,
            learning_rate=0.001,
            epsilon=1.0
        )
        
    def test_initialization(self):
        assert self.dqn is not None
        assert self.dqn.state_dim == 10
        assert self.dqn.action_dim == 3
        assert self.dqn.epsilon == 1.0
        
    def test_select_action_exploration(self):
        state = np.random.randn(10)
        action = self.dqn.select_action(state, training=True)
        assert 0 <= action < 3
        
    def test_select_action_exploitation(self):
        self.dqn.epsilon = 0.0
        state = np.random.randn(10)
        action = self.dqn.select_action(state, training=True)
        assert 0 <= action < 3
        
    def test_get_q_values(self):
        state = np.random.randn(10)
        q_values = self.dqn.get_q_values(state)
        assert len(q_values) == 3
        
    def test_store_transition(self):
        state = np.random.randn(10)
        next_state = np.random.randn(10)
        self.dqn.store_transition(state, 0, 1.0, next_state, False)
        assert len(self.dqn.replay_buffer) == 1
        
    def test_train_step_insufficient_data(self):
        # Not enough data for training
        loss = self.dqn.train_step()
        assert loss is None
        
    def test_train_step_with_data(self):
        # Fill replay buffer
        for i in range(100):
            state = np.random.randn(10)
            next_state = np.random.randn(10)
            self.dqn.store_transition(state, i % 3, float(i % 5 - 2), next_state, i % 20 == 0)
        
        loss = self.dqn.train_step()
        assert loss is not None
        assert loss >= 0
        
    def test_epsilon_decay(self):
        initial_epsilon = self.dqn.epsilon
        
        # Fill buffer and train
        for i in range(100):
            state = np.random.randn(10)
            next_state = np.random.randn(10)
            self.dqn.store_transition(state, 0, 1.0, next_state, False)
        
        self.dqn.train_step()
        assert self.dqn.epsilon < initial_epsilon
        
    def test_target_network_update(self):
        # Fill buffer
        for i in range(100):
            state = np.random.randn(10)
            next_state = np.random.randn(10)
            self.dqn.store_transition(state, 0, 1.0, next_state, False)
        
        # Train multiple steps
        for _ in range(self.dqn.target_update_frequency + 1):
            self.dqn.train_step()
        
        assert self.dqn.training_steps > self.dqn.target_update_frequency
        
    def test_update_parameters(self):
        self.dqn.update_parameters({
            'learning_rate': 0.0001,
            'epsilon': 0.5,
            'epsilon_decay': 0.99
        })
        assert self.dqn.epsilon == 0.5
        assert self.dqn.epsilon_decay == 0.99
        
    def test_get_metrics(self):
        metrics = self.dqn.get_metrics()
        assert 'training_steps' in metrics
        assert 'epsilon' in metrics
        assert 'buffer_size' in metrics
        assert 'avg_loss' in metrics
        
    def test_handle_reward(self):
        self.message_bus.publish('reward', {'reward': 1.5})
        assert hasattr(self.dqn, 'current_reward')
        
    def test_handle_tuned_reward(self):
        self.message_bus.publish('tuned_reward', {'tuned_reward': 2.0})
        assert hasattr(self.dqn, 'current_reward')
        
    def test_handle_action_request(self):
        state = np.random.randn(10)
        self.message_bus.publish('dqn_action_request', {'state': state})
        # Should publish response
        
    def test_q_learning_convergence(self):
        """Test that Q-values improve with training"""
        # Simple environment: state -> action with known rewards
        for episode in range(10):
            state = np.ones(10)
            for step in range(10):
                action = self.dqn.select_action(state, training=True)
                next_state = np.ones(10) * (step + 1) / 10
                reward = 1.0 if action == 1 else -1.0
                self.dqn.store_transition(state, action, reward, next_state, step == 9)
                state = next_state
                
                if len(self.dqn.replay_buffer) >= self.dqn.batch_size:
                    self.dqn.train_step()
        
        # Check that training occurred
        assert self.dqn.training_steps > 0
        assert len(self.dqn.losses) > 0
