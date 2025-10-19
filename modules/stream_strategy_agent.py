"""
stream_strategy_agent.py - Stream Strategy Agent

Description:
    RL agent that optimizes streaming strategies and resource allocation.
    Uses reinforcement learning to improve symbol selection and prioritization.

Features:
    - RL-based symbol scoring
    - Dynamic resource allocation
    - Strategy optimization
    - Feedback processing
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import numpy as np
import torch
import torch.nn as nn


class StrategyNetwork(nn.Module):
    """Simple neural network for strategy decisions."""
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)


class StreamStrategyAgent:
    """
    RL agent for optimizing streaming strategies.
    """
    
    def __init__(self, message_bus):
        """
        Initialize the stream strategy agent.
        
        Args:
            message_bus: Reference to message bus
        """
        self.message_bus = message_bus
        self.logger = logging.getLogger('StreamStrategyAgent')
        
        # RL model
        self.model = StrategyNetwork()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # Experience buffer
        self.experience_buffer = []
        self.max_buffer_size = 1000
        
        # Current strategy
        self.current_strategy = {
            'resource_allocation': {},
            'batch_size': 10,
            'priority_weights': {}
        }
    
    def get_symbol_scores(
        self,
        symbols: List[str],
        metrics: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Compute RL-based scores for symbols.
        
        Args:
            symbols: List of symbols to score
            metrics: Current metrics data
            
        Returns:
            Dictionary of symbol scores
        """
        scores = {}
        
        for symbol in symbols:
            # Extract features for the symbol
            features = self._extract_features(symbol, metrics)
            
            # Get score from RL model
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features).unsqueeze(0)
                score = self.model(features_tensor).item()
            
            scores[symbol] = score
        
        return scores
    
    def _extract_features(
        self,
        symbol: str,
        metrics: Dict[str, Dict[str, Any]]
    ) -> List[float]:
        """
        Extract features for RL model.
        
        Args:
            symbol: Symbol to extract features for
            metrics: Metrics data
            
        Returns:
            Feature vector
        """
        # Placeholder features - could be enhanced with real metrics
        features = [
            np.random.random(),  # Historical performance
            np.random.random(),  # Volatility
            np.random.random(),  # Volume
            np.random.random(),  # Momentum
            np.random.random(),  # Trend strength
            np.random.random(),  # Recent profit/loss
            np.random.random(),  # Correlation with portfolio
            np.random.random(),  # Market regime indicator
            np.random.random(),  # Liquidity
            np.random.random()   # Time since last rotation
        ]
        
        return features
    
    def update_strategy(
        self,
        metrics: Dict[str, Dict[str, Any]],
        priorities: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Update streaming strategy based on metrics.
        
        Args:
            metrics: Current metrics
            priorities: Symbol priorities
            
        Returns:
            Updated strategy
        """
        # Adaptive batch sizing
        avg_priority = np.mean(list(priorities.values())) if priorities else 0.5
        
        if avg_priority > 0.7:
            batch_size = 15  # Larger batches for high priority
        elif avg_priority < 0.3:
            batch_size = 5   # Smaller batches for low priority
        else:
            batch_size = 10
        
        # Update resource allocation
        resource_allocation = {}
        total_priority = sum(priorities.values())
        
        if total_priority > 0:
            for symbol, priority in priorities.items():
                resource_allocation[symbol] = priority / total_priority
        
        # Update strategy
        self.current_strategy = {
            'timestamp': datetime.now().isoformat(),
            'resource_allocation': resource_allocation,
            'batch_size': batch_size,
            'priority_weights': priorities.copy()
        }
        
        return self.current_strategy
    
    def process_feedback(self, feedback: Dict[str, Any]):
        """
        Process feedback to train RL model.
        
        Args:
            feedback: Feedback data
        """
        # Add to experience buffer
        self.experience_buffer.append(feedback)
        
        # Limit buffer size
        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer.pop(0)
        
        # Train model if enough experience
        if len(self.experience_buffer) >= 32:
            self._train_step()
    
    def _train_step(self):
        """Perform a training step on the RL model."""
        # Sample batch from experience buffer
        batch_size = min(32, len(self.experience_buffer))
        batch_indices = np.random.choice(
            len(self.experience_buffer),
            batch_size,
            replace=False
        )
        
        batch = [self.experience_buffer[i] for i in batch_indices]
        
        # Extract features and rewards
        features = []
        rewards = []
        
        for experience in batch:
            # Placeholder feature extraction
            feat = [np.random.random() for _ in range(10)]
            features.append(feat)
            rewards.append(experience.get('reward', 0.0))
        
        # Convert to tensors
        features_tensor = torch.FloatTensor(features)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1)
        
        # Normalize rewards
        rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
        
        # Forward pass
        predictions = self.model(features_tensor)
        
        # Compute loss (regression on normalized rewards)
        loss = nn.MSELoss()(predictions, (rewards_tensor + 1) / 2)  # Scale to [0, 1]
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.logger.debug(f"Training step: loss={loss.item():.4f}")
    
    def get_current_strategy(self) -> Dict[str, Any]:
        """Get current streaming strategy."""
        return self.current_strategy.copy()
    
    def reset(self):
        """Reset the agent."""
        self.experience_buffer.clear()
        self.model = StrategyNetwork()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.logger.info("Stream strategy agent reset")
