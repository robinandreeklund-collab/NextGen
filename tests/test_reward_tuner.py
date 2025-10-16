"""
Test suite för RewardTunerAgent - Sprint 4.4

Test cases RT-001 till RT-006 enligt reward_test_suite.yaml
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.reward_tuner import RewardTunerAgent
from modules.message_bus import MessageBus
import time


class TestRewardTunerAgent:
    """Test suite för RewardTunerAgent enligt reward_test_suite.yaml"""
    
    def setup_method(self):
        """Setup för varje test."""
        self.message_bus = MessageBus()
        self.reward_tuner = RewardTunerAgent(
            message_bus=self.message_bus,
            reward_scaling_factor=1.0,
            volatility_penalty_weight=0.3,
            overfitting_detector_threshold=0.2,
            history_window=50
        )
    
    # RT-001: Reward Volatility Calculation
    def test_RT001_low_volatility_rewards(self):
        """RT-001: Test låg volatilitet i rewards."""
        # Fyll history med låg volatilitet
        rewards = [1.0, 1.1, 0.9, 1.0, 1.05]
        for reward in rewards:
            self.reward_tuner.base_reward_history.append(reward)
        
        volatility, volatility_ratio = self.reward_tuner._calculate_reward_volatility()
        
        assert volatility < 0.1, f"Volatilitet {volatility} ska vara < 0.1"
        assert volatility_ratio < 1.5, f"Volatility ratio {volatility_ratio} ska vara < 1.5"
    
    def test_RT001_high_volatility_rewards(self):
        """RT-001: Test hög volatilitet i rewards."""
        # Fyll history med hög volatilitet
        rewards = [1.0, 5.0, -3.0, 2.0, -1.0]
        for reward in rewards:
            self.reward_tuner.base_reward_history.append(reward)
        
        volatility, volatility_ratio = self.reward_tuner._calculate_reward_volatility()
        
        assert volatility > 2.0, f"Volatilitet {volatility} ska vara > 2.0"
    
    def test_RT001_empty_history(self):
        """RT-001: Test tom history (edge case)."""
        self.reward_tuner.base_reward_history = []
        
        volatility, volatility_ratio = self.reward_tuner._calculate_reward_volatility()
        
        assert volatility == 0.0, "Volatilitet ska vara 0.0 för tom history"
        assert volatility_ratio == 1.0, "Volatility ratio ska vara 1.0 för tom history"
    
    # RT-002: Overfitting Detection
    def test_RT002_no_overfitting_stable_performance(self):
        """RT-002: Test ingen overfitting med stabil performance."""
        # Simulera stabil performance
        for _ in range(20):
            self.reward_tuner.agent_performance_history.append(0.75)
        
        overfitting_detected, overfitting_score = self.reward_tuner._detect_overfitting()
        
        assert not overfitting_detected, "Overfitting ska inte detekteras vid stabil performance"
        assert overfitting_score < 0.2, f"Overfitting score {overfitting_score} ska vara < 0.2"
    
    def test_RT002_clear_overfitting_performance_drop(self):
        """RT-002: Test tydlig overfitting med performance drop."""
        # Simulera performance degradation
        for _ in range(15):
            self.reward_tuner.agent_performance_history.append(0.80)
        for _ in range(5):
            self.reward_tuner.agent_performance_history.append(0.50)
        
        overfitting_detected, overfitting_score = self.reward_tuner._detect_overfitting()
        
        assert overfitting_detected, "Overfitting ska detekteras vid performance drop"
        assert overfitting_score > 0.2, f"Overfitting score {overfitting_score} ska vara > 0.2"
    
    def test_RT002_borderline_overfitting(self):
        """RT-002: Test borderline overfitting."""
        # Simulera borderline case
        for _ in range(15):
            self.reward_tuner.agent_performance_history.append(0.80)
        for _ in range(5):
            self.reward_tuner.agent_performance_history.append(0.65)
        
        overfitting_detected, overfitting_score = self.reward_tuner._detect_overfitting()
        
        # Score ska vara nära threshold (adjust range based on actual calculation)
        assert 0.14 < overfitting_score < 0.25, f"Overfitting score {overfitting_score} ska vara borderline"
    
    # RT-003: Reward Transformation with Volatility Penalty
    def test_RT003_no_penalty_low_volatility(self):
        """RT-003: Test ingen penalty vid låg volatilitet."""
        base_reward = 1.0
        volatility_ratio = 1.0
        overfitting_detected = False
        
        tuned_reward = self.reward_tuner._apply_reward_transformation(
            base_reward, volatility_ratio, overfitting_detected
        )
        
        assert abs(tuned_reward - 1.0) < 0.01, f"Tuned reward {tuned_reward} ska vara ~1.0"
    
    def test_RT003_volatility_penalty_applied(self):
        """RT-003: Test volatility penalty appliceras."""
        base_reward = 1.0
        volatility_ratio = 2.0
        overfitting_detected = False
        
        tuned_reward = self.reward_tuner._apply_reward_transformation(
            base_reward, volatility_ratio, overfitting_detected
        )
        
        # Förväntat: 1.0 * (1 - 0.3 * (2.0 - 1.0)) = 0.7
        assert 0.65 < tuned_reward < 0.75, f"Tuned reward {tuned_reward} ska vara ~0.7"
    
    def test_RT003_both_penalties_applied(self):
        """RT-003: Test både volatility och overfitting penalty."""
        base_reward = 1.0
        volatility_ratio = 2.5
        overfitting_detected = True
        
        tuned_reward = self.reward_tuner._apply_reward_transformation(
            base_reward, volatility_ratio, overfitting_detected
        )
        
        # Förväntat: 1.0 * (1 - 0.3 * 1.5) * 0.5 = 0.275
        assert 0.25 < tuned_reward < 0.3, f"Tuned reward {tuned_reward} ska vara ~0.275"
    
    # RT-004: Reward Scaling Factor Application
    def test_RT004_neutral_scaling(self):
        """RT-004: Test neutral scaling factor."""
        base_reward = 1.0
        self.reward_tuner.reward_scaling_factor = 1.0
        
        tuned_reward = self.reward_tuner._apply_reward_transformation(
            base_reward, 1.0, False
        )
        
        assert abs(tuned_reward - 1.0) < 0.01, "Neutral scaling ska ge samma reward"
    
    def test_RT004_conservative_scaling(self):
        """RT-004: Test konservativ scaling (0.5)."""
        base_reward = 1.0
        self.reward_tuner.reward_scaling_factor = 0.5
        
        tuned_reward = self.reward_tuner._apply_reward_transformation(
            base_reward, 1.0, False
        )
        
        assert abs(tuned_reward - 0.5) < 0.01, "Conservative scaling ska ge 0.5"
    
    def test_RT004_aggressive_scaling(self):
        """RT-004: Test aggressiv scaling (2.0)."""
        base_reward = 1.0
        self.reward_tuner.reward_scaling_factor = 2.0
        
        tuned_reward = self.reward_tuner._apply_reward_transformation(
            base_reward, 1.0, False
        )
        
        assert abs(tuned_reward - 2.0) < 0.01, "Aggressive scaling ska ge 2.0"
    
    def test_RT004_negative_reward_scaled(self):
        """RT-004: Test negativ reward skalas korrekt."""
        base_reward = -1.0
        self.reward_tuner.reward_scaling_factor = 1.5
        
        tuned_reward = self.reward_tuner._apply_reward_transformation(
            base_reward, 1.0, False
        )
        
        assert abs(tuned_reward - (-1.5)) < 0.01, "Negativ reward ska skalas korrekt"
    
    # RT-005: Integration - Portfolio to RL Controller
    def test_RT005_full_reward_flow(self):
        """RT-005: Test full reward flow från portfolio till rl_controller."""
        received_base_reward = []
        received_tuned_reward = []
        received_metrics = []
        
        # Subscribe to events
        def on_base_reward(data):
            received_base_reward.append(data)
        
        def on_tuned_reward(data):
            received_tuned_reward.append(data)
        
        def on_reward_metrics(data):
            received_metrics.append(data)
        
        self.message_bus.subscribe('base_reward', on_base_reward)
        self.message_bus.subscribe('tuned_reward', on_tuned_reward)
        self.message_bus.subscribe('reward_metrics', on_reward_metrics)
        
        # Simulera portfolio publicerar base_reward
        self.message_bus.publish('base_reward', {
            'reward': 1.0,
            'source': 'portfolio_manager',
            'portfolio_value': 1010.0,
            'num_trades': 1
        })
        
        # Vänta lite för att callbacks ska köras
        time.sleep(0.01)
        
        # Verifiera att alla events publicerades
        assert len(received_base_reward) == 1, "Base reward ska ha publicerats"
        assert len(received_tuned_reward) == 1, "Tuned reward ska ha publicerats"
        assert len(received_metrics) == 1, "Reward metrics ska ha publicerats"
        
        # Verifiera innehåll
        assert 'reward' in received_tuned_reward[0], "Tuned reward ska ha reward field"
        assert 'transformation_ratio' in received_metrics[0], "Metrics ska ha transformation_ratio"
    
    # RT-006: Reward Logging and Visualization
    def test_RT006_reward_logging_in_memory(self):
        """RT-006: Test att reward flow loggas i strategic_memory_engine."""
        from modules.strategic_memory_engine import StrategicMemoryEngine
        
        memory = StrategicMemoryEngine(self.message_bus)
        
        # Publicera base_reward
        self.message_bus.publish('base_reward', {
            'reward': 1.0,
            'source': 'portfolio_manager',
            'portfolio_value': 1010.0
        })
        
        time.sleep(0.01)
        
        # Verifiera logging
        reward_history = memory.get_reward_history(limit=10)
        assert len(reward_history) > 0, "Reward history ska ha loggats"
        
        # Hitta base_reward entry
        base_entries = [r for r in reward_history if r.get('type') == 'base_reward']
        assert len(base_entries) > 0, "Base reward ska finnas i history"
        assert base_entries[0].get('reward') == 1.0, "Base reward värde ska vara korrekt"
    
    def test_RT006_reward_visualization_in_panel(self):
        """RT-006: Test att reward visualization data genereras."""
        from modules.introspection_panel import IntrospectionPanel
        
        panel = IntrospectionPanel(self.message_bus)
        
        # Publicera reward metrics
        self.message_bus.publish('reward_metrics', {
            'base_reward': 1.0,
            'tuned_reward': 0.8,
            'transformation_ratio': 0.8,
            'volatility': 0.1,
            'overfitting_detected': False
        })
        
        time.sleep(0.01)
        
        # Verifiera att panel har data
        assert len(panel.reward_metrics_history) > 0, "Panel ska ha reward metrics"
        
        # Verifiera dashboard data generation
        dashboard_data = panel.render_dashboard()
        assert 'reward_flow' in dashboard_data, "Dashboard ska ha reward_flow data"
    
    def test_parameter_updates(self):
        """Test att parametrar kan uppdateras via parameter_adjustment."""
        initial_scaling = self.reward_tuner.reward_scaling_factor
        
        # Publicera parameter adjustment
        self.message_bus.publish('parameter_adjustment', {
            'module': 'reward_tuner',
            'parameters': {
                'reward_scaling_factor': 1.5,
                'volatility_penalty_weight': 0.4
            }
        })
        
        time.sleep(0.01)
        
        # Verifiera att parametrar uppdaterades
        assert self.reward_tuner.reward_scaling_factor == 1.5, "Scaling factor ska uppdaterats"
        assert self.reward_tuner.volatility_penalty_weight == 0.4, "Penalty weight ska uppdaterats"
        assert len(self.reward_tuner.parameter_history) > 0, "Parameter history ska ha loggats"
    
    def test_get_reward_metrics(self):
        """Test att reward metrics kan hämtas."""
        # Lägg till lite data
        self.reward_tuner.base_reward_history = [1.0, 1.1, 0.9]
        self.reward_tuner.tuned_reward_history = [0.9, 1.0, 0.85]
        
        metrics = self.reward_tuner.get_reward_metrics()
        
        assert 'base_reward_history' in metrics
        assert 'tuned_reward_history' in metrics
        assert 'current_parameters' in metrics
        assert len(metrics['base_reward_history']) == 3
        assert len(metrics['tuned_reward_history']) == 3
    
    def test_bounds_enforcement(self):
        """Test att reward bounds enforcement fungerar."""
        # Test extremt högt reward
        base_reward = 100.0
        tuned_reward = self.reward_tuner._apply_reward_transformation(
            base_reward, 1.0, False
        )
        
        assert tuned_reward <= 10.0, "Reward ska clampas till max 10.0"
        
        # Test extremt lågt reward
        base_reward = -100.0
        tuned_reward = self.reward_tuner._apply_reward_transformation(
            base_reward, 1.0, False
        )
        
        assert tuned_reward >= -10.0, "Reward ska clampas till min -10.0"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
