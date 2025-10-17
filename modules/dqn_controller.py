"""
DQN Controller - Deep Q-Network for reinforcement learning

Implements DQN with experience replay and target network for stable training.
Works in parallel with PPO from rl_controller for hybrid RL approach.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from collections import deque
import random


class QNetwork(nn.Module):
    """Q-Network for approximating Q-values"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class ReplayBuffer:
    """Experience replay buffer for DQN"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
        
    def __len__(self) -> int:
        return len(self.buffer)


class DQNController:
    """
    DQN Controller for reinforcement learning
    
    Implements Deep Q-Network with:
    - Experience replay for stable training
    - Target network for stable Q-value estimation
    - Epsilon-greedy exploration
    - Hybrid RL integration with PPO
    """
    
    def __init__(self, message_bus, state_dim: int = 10, action_dim: int = 3,
                 learning_rate: float = 0.001, discount_factor: float = 0.99,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01, replay_buffer_size: int = 10000,
                 batch_size: int = 32, target_update_frequency: int = 100):
        self.message_bus = message_bus
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        
        # Networks
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        
        # Training metrics
        self.training_steps = 0
        self.episodes = 0
        self.losses = []
        self.q_values_history = []
        
        # Subscribe to topics
        self.message_bus.subscribe('reward', self._handle_reward)
        self.message_bus.subscribe('tuned_reward', self._handle_tuned_reward)
        self.message_bus.subscribe('dqn_action_request', self._handle_action_request)
        
    def _handle_reward(self, data: Dict[str, Any]):
        """Handle reward signal"""
        if 'reward' in data:
            self.current_reward = data['reward']
            
    def _handle_tuned_reward(self, data: Dict[str, Any]):
        """Handle tuned reward from reward_tuner"""
        if 'tuned_reward' in data:
            self.current_reward = data['tuned_reward']
            
    def _handle_action_request(self, data: Dict[str, Any]):
        """Handle action request from decision engine"""
        if 'state' in data:
            action = self.select_action(data['state'])
            self.message_bus.publish('dqn_action_response', {
                'action': action,
                'q_values': self.get_q_values(data['state']).tolist(),
                'epsilon': self.epsilon
            })
            
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state
            training: Whether in training mode (affects epsilon)
            
        Returns:
            Selected action index
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
            
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions in given state"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.numpy()[0]
            
    def store_transition(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """Store transition in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
        
    def train_step(self) -> Optional[float]:
        """
        Perform one training step using experience replay
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
            
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.discount_factor * next_q_values
            
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update metrics
        self.training_steps += 1
        self.losses.append(loss.item())
        
        # Update target network
        if self.training_steps % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Publish metrics
        self.message_bus.publish('dqn_metrics', {
            'loss': loss.item(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'buffer_size': len(self.replay_buffer)
        })
        
        return loss.item()
        
    def update_parameters(self, parameters: Dict[str, Any]):
        """Update DQN parameters (for adaptive control)"""
        if 'learning_rate' in parameters:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = parameters['learning_rate']
                
        if 'epsilon' in parameters:
            self.epsilon = max(self.epsilon_min, parameters['epsilon'])
            
        if 'epsilon_decay' in parameters:
            self.epsilon_decay = parameters['epsilon_decay']
            
        if 'discount_factor' in parameters:
            self.discount_factor = parameters['discount_factor']
            
    def get_metrics(self) -> Dict[str, Any]:
        """Get current training metrics"""
        avg_loss = np.mean(self.losses[-100:]) if self.losses else 0.0
        
        return {
            'training_steps': self.training_steps,
            'episodes': self.episodes,
            'epsilon': self.epsilon,
            'buffer_size': len(self.replay_buffer),
            'avg_loss': avg_loss,
            'recent_losses': self.losses[-10:] if self.losses else []
        }
        
    def save_model(self, path: str):
        """Save Q-network state"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_steps': self.training_steps,
            'epsilon': self.epsilon
        }, path)
        
    def load_model(self, path: str):
        """Load Q-network state"""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.training_steps = checkpoint['training_steps']
        self.epsilon = checkpoint['epsilon']
