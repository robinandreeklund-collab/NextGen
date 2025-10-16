"""
reward_tuner.py - RewardTunerAgent för meta-belöningsjustering

Beskrivning:
    Meta-agent som justerar och optimerar belöningssignaler från portfolio_manager
    innan de når rl_controller. Implementerar reward scaling, volatility penalties
    och overfitting detection för robust RL-träning.

Roll:
    - Tar emot base_reward från portfolio_manager
    - Analyserar reward stability och volatilitet
    - Justerar reward med scaling_factor baserat på marknadsförhållanden
    - Applicerar volatility_penalty vid hög volatilitet
    - Detekterar overfitting patterns i agent performance
    - Publicerar tuned_reward till rl_controller
    - Loggar reward flow i strategic_memory_engine

Inputs:
    - base_reward: float - Raw reward från portfolio_manager
    - portfolio_status: Dict - Portfolio state för kontext
    - agent_status: Dict - RL-agent performance från rl_controller

Outputs:
    - tuned_reward: float - Justerad reward för RL-träning
    - reward_metrics: Dict - Metrics om reward tuning

Publicerar till message_bus:
    - tuned_reward: Justerad reward för rl_controller
    - reward_metrics: Metrics för introspection_panel
    - reward_log: Reward history för strategic_memory_engine

Prenumererar på:
    - base_reward: Från portfolio_manager
    - portfolio_status: Från portfolio_manager
    - agent_status: Från rl_controller
    - parameter_adjustment: Från rl_controller

Används i Sprint: 4.4
"""

from typing import Dict, Any, List
import numpy as np
import time


class RewardTunerAgent:
    """Meta-agent för reward justering och optimering."""
    
    def __init__(
        self,
        message_bus,
        reward_scaling_factor: float = 1.0,
        volatility_penalty_weight: float = 0.3,
        overfitting_detector_threshold: float = 0.2,
        history_window: int = 50
    ):
        """
        Initialiserar RewardTunerAgent.
        
        Args:
            message_bus: Referens till central message_bus
            reward_scaling_factor: Multiplikativ skalning av base reward (0.5-2.0)
            volatility_penalty_weight: Viktning av volatility penalty (0.0-1.0)
            overfitting_detector_threshold: Tröskelvärde för overfitting (0.05-0.5)
            history_window: Antal rewards att spara i history för analys
        """
        self.message_bus = message_bus
        self.reward_scaling_factor = reward_scaling_factor
        self.volatility_penalty_weight = volatility_penalty_weight
        self.overfitting_detector_threshold = overfitting_detector_threshold
        self.history_window = history_window
        
        # History tracking
        self.base_reward_history: List[float] = []
        self.tuned_reward_history: List[float] = []
        self.volatility_history: List[Dict[str, Any]] = []
        self.overfitting_events: List[Dict[str, Any]] = []
        self.parameter_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.agent_performance_history: List[float] = []
        self.transformation_ratios: List[float] = []
        
        # Subscribe to relevant topics
        self.message_bus.subscribe('base_reward', self._on_base_reward)
        self.message_bus.subscribe('portfolio_status', self._on_portfolio_status)
        self.message_bus.subscribe('agent_status', self._on_agent_status)
        self.message_bus.subscribe('parameter_adjustment', self._on_parameter_adjustment)
        
        # Cache for current portfolio and agent state
        self.current_portfolio_status: Dict[str, Any] = {}
        self.current_agent_status: Dict[str, Any] = {}
    
    def _on_base_reward(self, reward_data: Dict[str, Any]) -> None:
        """
        Callback när base_reward publiceras från portfolio_manager.
        
        Args:
            reward_data: Dict med base_reward och kontext
        """
        base_reward = reward_data.get('reward', 0.0)
        
        # Store in history
        self.base_reward_history.append(base_reward)
        if len(self.base_reward_history) > self.history_window:
            self.base_reward_history = self.base_reward_history[-self.history_window:]
        
        # Calculate volatility
        volatility, volatility_ratio = self._calculate_reward_volatility()
        
        # Detect overfitting
        overfitting_detected, overfitting_score = self._detect_overfitting()
        
        # Apply reward transformation
        tuned_reward = self._apply_reward_transformation(
            base_reward, volatility_ratio, overfitting_detected
        )
        
        # Store tuned reward
        self.tuned_reward_history.append(tuned_reward)
        if len(self.tuned_reward_history) > self.history_window:
            self.tuned_reward_history = self.tuned_reward_history[-self.history_window:]
        
        # Calculate transformation ratio
        transformation_ratio = tuned_reward / base_reward if base_reward != 0 else 1.0
        self.transformation_ratios.append(transformation_ratio)
        if len(self.transformation_ratios) > self.history_window:
            self.transformation_ratios = self.transformation_ratios[-self.history_window:]
        
        # Publish tuned reward and metrics
        self._publish_tuned_reward(tuned_reward, {
            'base_reward': base_reward,
            'tuned_reward': tuned_reward,
            'transformation_ratio': transformation_ratio,
            'volatility': volatility,
            'volatility_ratio': volatility_ratio,
            'overfitting_detected': overfitting_detected,
            'overfitting_score': overfitting_score,
            'reward_scaling_factor': self.reward_scaling_factor,
            'volatility_penalty_weight': self.volatility_penalty_weight
        })
        
        # Log reward flow
        self._log_reward_flow(base_reward, tuned_reward, {
            'volatility': volatility,
            'volatility_ratio': volatility_ratio,
            'overfitting_detected': overfitting_detected,
            'overfitting_score': overfitting_score,
            'transformation_ratio': transformation_ratio
        })
    
    def _calculate_reward_volatility(self) -> tuple:
        """
        Beräknar volatilitet i reward history.
        
        Returns:
            (volatility, volatility_ratio): Tuple med std dev och ratio
        """
        if len(self.base_reward_history) < 2:
            return 0.0, 1.0
        
        # Calculate standard deviation
        volatility = float(np.std(self.base_reward_history))
        
        # Calculate mean of volatility over longer history
        if len(self.volatility_history) > 0:
            historical_mean_volatility = np.mean([v['volatility'] for v in self.volatility_history[-20:]])
            if historical_mean_volatility > 0:
                volatility_ratio = volatility / historical_mean_volatility
            else:
                volatility_ratio = 1.0
        else:
            volatility_ratio = 1.0
        
        # Store volatility metrics
        self.volatility_history.append({
            'timestamp': time.time(),
            'volatility': volatility,
            'volatility_ratio': volatility_ratio
        })
        if len(self.volatility_history) > self.history_window:
            self.volatility_history = self.volatility_history[-self.history_window:]
        
        return volatility, volatility_ratio
    
    def _detect_overfitting(self) -> tuple:
        """
        Detekterar overfitting patterns i agent performance.
        
        Returns:
            (overfitting_detected, overfitting_score): Tuple med bool och score
        """
        if len(self.agent_performance_history) < 10:
            return False, 0.0
        
        # Compare recent performance vs long-term average
        recent_window = min(5, len(self.agent_performance_history))
        long_term_window = min(20, len(self.agent_performance_history))
        
        recent_performance = np.mean(self.agent_performance_history[-recent_window:])
        long_term_performance = np.mean(self.agent_performance_history[-long_term_window:])
        
        # Calculate overfitting score (performance drop)
        if long_term_performance > 0:
            overfitting_score = (long_term_performance - recent_performance) / long_term_performance
            overfitting_score = max(0.0, overfitting_score)  # Clamp to [0, inf)
        else:
            overfitting_score = 0.0
        
        # Detect overfitting if score exceeds threshold
        overfitting_detected = overfitting_score > self.overfitting_detector_threshold
        
        # Log overfitting event
        if overfitting_detected:
            self.overfitting_events.append({
                'timestamp': time.time(),
                'overfitting_score': overfitting_score,
                'recent_performance': recent_performance,
                'long_term_performance': long_term_performance
            })
            if len(self.overfitting_events) > 100:
                self.overfitting_events = self.overfitting_events[-100:]
        
        return overfitting_detected, overfitting_score
    
    def _apply_reward_transformation(
        self,
        base_reward: float,
        volatility_ratio: float,
        overfitting_detected: bool
    ) -> float:
        """
        Applicerar reward scaling och penalties.
        
        Args:
            base_reward: Raw reward från portfolio
            volatility_ratio: Ratio av current vs historical volatility
            overfitting_detected: Om overfitting detekterades
        
        Returns:
            tuned_reward: Justerad reward
        """
        adjusted_reward = base_reward
        
        # Apply volatility penalty if high volatility
        if volatility_ratio > 1.5:
            penalty = self.volatility_penalty_weight * (volatility_ratio - 1.0)
            penalty = min(penalty, 0.9)  # Max 90% penalty
            adjusted_reward = adjusted_reward * (1.0 - penalty)
        
        # Apply overfitting penalty
        if overfitting_detected:
            adjusted_reward = adjusted_reward * 0.5
        
        # Apply reward scaling factor
        tuned_reward = adjusted_reward * self.reward_scaling_factor
        
        # Bound checking - clamp to reasonable range
        tuned_reward = np.clip(tuned_reward, -10.0, 10.0)
        
        return float(tuned_reward)
    
    def _publish_tuned_reward(self, tuned_reward: float, metrics: Dict[str, Any]) -> None:
        """
        Publicerar justerad reward och metrics till message_bus.
        
        Args:
            tuned_reward: Justerad reward
            metrics: Reward metrics för visualization
        """
        # Publish tuned_reward for rl_controller
        self.message_bus.publish('tuned_reward', {
            'reward': tuned_reward,
            'timestamp': time.time()
        })
        
        # Publish reward_metrics for introspection_panel
        self.message_bus.publish('reward_metrics', metrics)
    
    def _log_reward_flow(
        self,
        base_reward: float,
        tuned_reward: float,
        metrics: Dict[str, Any]
    ) -> None:
        """
        Loggar reward transformation history för strategic_memory_engine.
        
        Args:
            base_reward: Raw reward
            tuned_reward: Justerad reward
            metrics: Transformation metrics
        """
        reward_log = {
            'timestamp': time.time(),
            'base_reward': base_reward,
            'tuned_reward': tuned_reward,
            'transformation_ratio': tuned_reward / base_reward if base_reward != 0 else 1.0,
            'volatility': metrics.get('volatility', 0.0),
            'volatility_ratio': metrics.get('volatility_ratio', 1.0),
            'overfitting_detected': metrics.get('overfitting_detected', False),
            'overfitting_score': metrics.get('overfitting_score', 0.0),
            'reward_scaling_factor': self.reward_scaling_factor,
            'volatility_penalty_weight': self.volatility_penalty_weight
        }
        
        self.message_bus.publish('reward_log', reward_log)
    
    def _on_portfolio_status(self, status: Dict[str, Any]) -> None:
        """
        Callback för portfolio status från portfolio_manager.
        
        Args:
            status: Portfolio status data
        """
        self.current_portfolio_status = status
    
    def _on_agent_status(self, status: Dict[str, Any]) -> None:
        """
        Callback för agent status från rl_controller.
        
        Args:
            status: Agent status med performance metrics
        """
        self.current_agent_status = status
        
        # Extract performance metric for overfitting detection
        if 'cumulative_reward' in status:
            performance = status['cumulative_reward']
            self.agent_performance_history.append(performance)
            if len(self.agent_performance_history) > self.history_window * 2:
                self.agent_performance_history = self.agent_performance_history[-self.history_window * 2:]
    
    def _on_parameter_adjustment(self, adjustment: Dict[str, Any]) -> None:
        """
        Callback för parameter adjustment från rl_controller.
        
        Args:
            adjustment: Parameter adjustments för reward_tuner
        """
        # Check if adjustment is for reward_tuner
        if adjustment.get('module') == 'reward_tuner':
            params = adjustment.get('parameters', {})
            
            # Update parameters
            if 'reward_scaling_factor' in params:
                self.reward_scaling_factor = params['reward_scaling_factor']
            if 'volatility_penalty_weight' in params:
                self.volatility_penalty_weight = params['volatility_penalty_weight']
            if 'overfitting_detector_threshold' in params:
                self.overfitting_detector_threshold = params['overfitting_detector_threshold']
            
            # Log parameter change
            self.parameter_history.append({
                'timestamp': time.time(),
                'reward_scaling_factor': self.reward_scaling_factor,
                'volatility_penalty_weight': self.volatility_penalty_weight,
                'overfitting_detector_threshold': self.overfitting_detector_threshold
            })
            if len(self.parameter_history) > 100:
                self.parameter_history = self.parameter_history[-100:]
    
    def get_reward_metrics(self) -> Dict[str, Any]:
        """
        Returnerar metrics för visualization.
        
        Returns:
            metrics: Dict med reward history och metrics
        """
        return {
            'base_reward_history': self.base_reward_history.copy(),
            'tuned_reward_history': self.tuned_reward_history.copy(),
            'transformation_ratios': self.transformation_ratios.copy(),
            'volatility_history': [v['volatility'] for v in self.volatility_history],
            'overfitting_events': self.overfitting_events.copy(),
            'current_parameters': {
                'reward_scaling_factor': self.reward_scaling_factor,
                'volatility_penalty_weight': self.volatility_penalty_weight,
                'overfitting_detector_threshold': self.overfitting_detector_threshold
            },
            'parameter_history': self.parameter_history.copy()
        }
    
    def get_parameter_history(self) -> List[Dict[str, Any]]:
        """
        Returnerar parameter history för analys.
        
        Returns:
            parameter_history: Lista av parameter snapshots
        """
        return self.parameter_history.copy()
