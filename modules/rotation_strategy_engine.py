"""
rotation_strategy_engine.py - Rotation Strategy Engine

Description:
    Determines rotation strategies based on historical performance,
    RL feedback, and market state. Uses RL to optimize rotation decisions.

Features:
    - Performance-based strategy selection
    - RL-driven strategy optimization
    - Market regime adaptation
    - Feedback processing
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import numpy as np


class RotationStrategyEngine:
    """
    Engine for determining optimal rotation strategies.
    """
    
    def __init__(self, message_bus):
        """
        Initialize the rotation strategy engine.
        
        Args:
            message_bus: Reference to message bus
        """
        self.message_bus = message_bus
        self.logger = logging.getLogger('RotationStrategyEngine')
        
        # Strategy performance tracking
        self.strategy_performance = {
            'top_priority': {'successes': 0, 'failures': 0, 'avg_reward': 0.0},
            'random': {'successes': 0, 'failures': 0, 'avg_reward': 0.0},
            'hybrid': {'successes': 0, 'failures': 0, 'avg_reward': 0.0}
        }
        
        # Current strategy
        self.current_strategy = {
            'type': 'top_priority',
            'rotation_rate': 0.3,
            'recommend_rotation': False
        }
        
        # RL feedback buffer
        self.feedback_buffer = []
    
    def compute_rotation_strategy(
        self,
        priorities: Dict[str, float],
        metrics: Dict[str, Dict[str, Any]],
        current_symbols: List[str]
    ) -> Dict[str, Any]:
        """
        Compute optimal rotation strategy.
        
        Args:
            priorities: Symbol priority scores
            metrics: Current stream metrics
            current_symbols: Currently active symbols
            
        Returns:
            Strategy configuration
        """
        # Analyze current performance
        avg_priority = np.mean(list(priorities.values())) if priorities else 0.5
        
        # Select strategy based on performance and exploration
        strategy_type = self._select_strategy_type(avg_priority)
        
        # Determine rotation rate based on metrics
        rotation_rate = self._compute_rotation_rate(metrics, avg_priority)
        
        # Update current strategy
        self.current_strategy = {
            'type': strategy_type,
            'rotation_rate': rotation_rate,
            'recommend_rotation': False,
            'timestamp': datetime.now().isoformat()
        }
        
        # Publish strategy update
        self.message_bus.publish('strategy_update', self.current_strategy)
        
        return self.current_strategy
    
    def _select_strategy_type(self, avg_priority: float) -> str:
        """
        Select strategy type based on performance.
        
        Args:
            avg_priority: Average symbol priority
            
        Returns:
            Strategy type name
        """
        # Get best performing strategy
        best_strategy = max(
            self.strategy_performance.items(),
            key=lambda x: x[1]['avg_reward']
        )[0]
        
        # Exploration: occasionally try different strategies
        if np.random.random() < 0.1:  # 10% exploration
            return np.random.choice(['top_priority', 'random', 'hybrid'])
        
        # If avg priority is low, try more exploration
        if avg_priority < 0.3:
            return 'random' if np.random.random() < 0.5 else 'hybrid'
        
        return best_strategy
    
    def _compute_rotation_rate(
        self,
        metrics: Dict[str, Dict[str, Any]],
        avg_priority: float
    ) -> float:
        """
        Compute optimal rotation rate.
        
        Args:
            metrics: Stream metrics
            avg_priority: Average priority
            
        Returns:
            Rotation rate (0.0 to 1.0)
        """
        # Base rotation rate
        base_rate = 0.3
        
        # Adjust based on priority
        if avg_priority < 0.3:
            # Low priority -> higher rotation
            rate = base_rate + 0.2
        elif avg_priority > 0.7:
            # High priority -> lower rotation
            rate = base_rate - 0.1
        else:
            rate = base_rate
        
        # Clamp to valid range
        return max(0.1, min(0.8, rate))
    
    def process_feedback(self, feedback: Dict[str, Any]):
        """
        Process RL feedback to improve strategy selection.
        
        Args:
            feedback: Feedback data from RL controllers
        """
        self.feedback_buffer.append(feedback)
        
        # Update strategy performance based on feedback
        if 'reward' in feedback and 'strategy' in feedback:
            strategy_type = feedback['strategy']
            reward = feedback['reward']
            
            if strategy_type in self.strategy_performance:
                perf = self.strategy_performance[strategy_type]
                
                # Update success/failure counts
                if reward > 0:
                    perf['successes'] += 1
                else:
                    perf['failures'] += 1
                
                # Update average reward (exponential moving average)
                alpha = 0.1
                perf['avg_reward'] = (alpha * reward + 
                                     (1 - alpha) * perf['avg_reward'])
        
        # Check if rotation should be recommended
        if len(self.feedback_buffer) >= 10:
            recent_rewards = [f.get('reward', 0) for f in self.feedback_buffer[-10:]]
            avg_recent_reward = np.mean(recent_rewards)
            
            # Recommend rotation if performance is declining
            if avg_recent_reward < -0.1:
                self.current_strategy['recommend_rotation'] = True
    
    def get_current_strategy(self) -> Dict[str, Any]:
        """Get current rotation strategy."""
        return self.current_strategy.copy()
    
    def get_strategy_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance metrics for all strategies."""
        return {
            strategy: {
                'success_rate': (
                    perf['successes'] / (perf['successes'] + perf['failures'])
                    if (perf['successes'] + perf['failures']) > 0 else 0.0
                ),
                'avg_reward': perf['avg_reward'],
                'total_trials': perf['successes'] + perf['failures']
            }
            for strategy, perf in self.strategy_performance.items()
        }
    
    def reset_strategy(self):
        """Reset strategy to default."""
        self.current_strategy = {
            'type': 'top_priority',
            'rotation_rate': 0.3,
            'recommend_rotation': False
        }
        self.logger.info("Strategy reset to default")
