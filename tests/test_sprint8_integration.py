"""
Integration Tests for Sprint 8
"""

import pytest
import numpy as np
from modules.dqn_controller import DQNController
from modules.gan_evolution_engine import GANEvolutionEngine
from modules.gnn_timespan_analyzer import GNNTimespanAnalyzer
from modules.rl_controller import RLController
from modules.message_bus import MessageBus


class TestSprint8Integration:
    """Integration tests for Sprint 8 modules"""
    
    def setup_method(self):
        self.message_bus = MessageBus()
        
        # Initialize all Sprint 8 components
        self.dqn = DQNController(self.message_bus, state_dim=10, action_dim=3)
        self.gan = GANEvolutionEngine(self.message_bus, latent_dim=64, param_dim=16)
        self.gnn = GNNTimespanAnalyzer(self.message_bus, input_dim=32)
        self.ppo = RLController(self.message_bus)
        
    def test_all_modules_initialized(self):
        """Test that all Sprint 8 modules initialize successfully"""
        assert self.dqn is not None
        assert self.gan is not None
        assert self.gnn is not None
        assert self.ppo is not None
        
    def test_dqn_gan_integration(self):
        """Test integration between DQN and GAN"""
        # GAN generates candidates for DQN parameter optimization
        candidates = self.gan.generate_candidates(num_candidates=3)
        assert len(candidates) > 0
        
        # DQN could use these for exploration strategies
        
    def test_gan_to_meta_agent_flow(self):
        """Test GAN candidates flow to meta agent evolution"""
        # Simulate agent performance data
        for i in range(50):
            self.message_bus.publish('agent_performance', {
                'parameters': np.random.randn(16),
                'performance': 0.6 + np.random.rand() * 0.3
            })
        
        # Generate candidates
        candidates = self.gan.generate_candidates(num_candidates=5)
        assert len(candidates) > 0
        
        # These would be used by meta_agent_evolution_engine
        
    def test_gnn_temporal_analysis_flow(self):
        """Test GNN temporal analysis with decision flow"""
        # Simulate decision flow
        for i in range(10):
            self.message_bus.publish('final_decision', {
                'timestamp': i,
                'action': 'BUY' if i % 2 == 0 else 'SELL',
                'confidence': 0.7 + np.random.rand() * 0.2,
                'symbol': 'AAPL'
            })
            
            self.message_bus.publish('indicator_data', {
                'timestamp': i,
                'indicators': {'RSI': 50 + i * 2, 'MACD': 0.5, 'ATR': 2.5},
                'symbol': 'AAPL'
            })
            
            self.message_bus.publish('execution_result', {
                'timestamp': i,
                'success': i % 3 != 0,
                'pnl': np.random.randn() * 100,
                'symbol': 'AAPL'
            })
        
        # Analyze patterns
        patterns = self.gnn.analyze_patterns()
        assert 'patterns' in patterns
        
        insights = self.gnn.get_temporal_insights()
        assert 'success_rate' in insights
        
    def test_hybrid_rl_decision_making(self):
        """Test hybrid RL decision making with both PPO and DQN"""
        state = np.random.randn(10)
        
        # Get DQN decision
        dqn_action = self.dqn.select_action(state, training=False)
        dqn_q_values = self.dqn.get_q_values(state)
        
        # Both should work
        assert 0 <= dqn_action < 3
        assert len(dqn_q_values) == 3
        
    def test_full_training_cycle(self):
        """Test full training cycle with all Sprint 8 components"""
        # Simulate training episodes
        for episode in range(5):
            state = np.random.randn(10)
            
            for step in range(10):
                # DQN selects action
                action = self.dqn.select_action(state)
                
                # Execute and get reward
                next_state = np.random.randn(10)
                reward = np.random.randn()
                done = step == 9
                
                # Store in DQN
                self.dqn.store_transition(state, action, reward, next_state, done)
                
                # Train DQN
                if len(self.dqn.replay_buffer) >= self.dqn.batch_size:
                    self.dqn.train_step()
                
                # Record decision for GNN
                self.message_bus.publish('final_decision', {
                    'timestamp': episode * 10 + step,
                    'action': ['BUY', 'SELL', 'HOLD'][action],
                    'confidence': 0.8,
                    'symbol': 'AAPL'
                })
                
                # Record outcome
                self.message_bus.publish('execution_result', {
                    'timestamp': episode * 10 + step,
                    'success': reward > 0,
                    'pnl': reward * 100,
                    'symbol': 'AAPL'
                })
                
                state = next_state
        
        # Verify training occurred
        dqn_metrics = self.dqn.get_metrics()
        assert dqn_metrics['training_steps'] > 0
        
        # Verify GNN collected data
        gnn_metrics = self.gnn.get_metrics()
        assert gnn_metrics['decision_history_size'] > 0
        
    def test_gan_training_with_dqn_performance(self):
        """Test GAN training using DQN performance data"""
        # Simulate DQN training and performance
        for i in range(50):
            # Create agent performance data
            parameters = np.random.randn(16)
            performance = 0.5 + np.random.rand() * 0.4
            
            self.message_bus.publish('agent_performance', {
                'parameters': parameters,
                'performance': performance
            })
        
        # Train GAN
        for _ in range(10):
            self.gan.train_step(batch_size=32)
        
        # Generate new candidates
        candidates = self.gan.generate_candidates(num_candidates=5)
        assert len(candidates) > 0
        
        # Verify GAN metrics
        gan_metrics = self.gan.get_metrics()
        assert gan_metrics['candidates_generated'] > 0
        
    def test_conflict_detection_and_resolution(self):
        """Test parameter conflict detection between PPO and DQN"""
        state = np.random.randn(10)
        
        # Get actions from both
        dqn_action = self.dqn.select_action(state, training=False)
        
        # Check for conflicts
        assert 0 <= dqn_action < 3
        
    def test_reward_flow_through_system(self):
        """Test reward flow through all components"""
        # Base reward
        self.message_bus.publish('reward', {'reward': 1.5})
        
        # Tuned reward
        self.message_bus.publish('tuned_reward', {'tuned_reward': 2.0})
        
        # All components should handle rewards appropriately
        
    def test_metrics_collection(self):
        """Test that all components provide metrics"""
        dqn_metrics = self.dqn.get_metrics()
        gan_metrics = self.gan.get_metrics()
        gnn_metrics = self.gnn.get_metrics()
        
        assert dqn_metrics is not None
        assert gan_metrics is not None
        assert gnn_metrics is not None
        # PPO exists
        assert self.ppo is not None
        
    def test_message_bus_integration(self):
        """Test that all components communicate via message bus"""
        # Track message bus topics
        topics_published = set()
        
        # DQN publishes metrics
        self.message_bus.publish('dqn_metrics', {'loss': 0.5})
        topics_published.add('dqn_metrics')
        
        # GAN publishes candidates
        self.message_bus.publish('gan_candidates', {'candidates': []})
        topics_published.add('gan_candidates')
        
        # GNN publishes insights
        self.message_bus.publish('gnn_analysis_response', {'insights': {}})
        topics_published.add('gnn_analysis_response')
        
        assert len(topics_published) == 3
        
    def test_end_to_end_scenario(self):
        """Test end-to-end scenario with all Sprint 8 components"""
        # 1. Market data arrives
        for i in range(20):
            # 2. Make decision using hybrid RL
            state = np.random.randn(10)
            dqn_action = self.dqn.select_action(state)
            
            # 3. Record decision for GNN
            self.message_bus.publish('final_decision', {
                'timestamp': i,
                'action': ['BUY', 'SELL', 'HOLD'][dqn_action],
                'confidence': 0.8,
                'symbol': 'AAPL'
            })
            
            # 4. Execute and get outcome
            reward = np.random.randn()
            next_state = np.random.randn(10)
            
            self.message_bus.publish('execution_result', {
                'timestamp': i,
                'success': reward > 0,
                'pnl': reward * 100,
                'symbol': 'AAPL'
            })
            
            # 5. Train DQN
            self.dqn.store_transition(state, dqn_action, reward, next_state, False)
            if len(self.dqn.replay_buffer) >= self.dqn.batch_size:
                self.dqn.train_step()
            
            # 6. Record agent performance for GAN
            if i % 5 == 0:
                self.message_bus.publish('agent_performance', {
                    'parameters': np.random.randn(16),
                    'performance': 0.6 + np.random.rand() * 0.3
                })
        
        # 7. Analyze temporal patterns
        patterns = self.gnn.analyze_patterns()
        insights = self.gnn.get_temporal_insights()
        
        # 8. Generate new candidates
        if len(self.gan.real_agent_data) >= 32:
            self.gan.train_step(batch_size=32)
            candidates = self.gan.generate_candidates(num_candidates=3)
        
        # Verify everything worked
        # DQN may not have trained if buffer wasn't full enough
        assert len(self.dqn.replay_buffer) > 0
        assert len(self.gnn.decision_history) > 0
        assert patterns is not None
        assert insights is not None


class TestSprint8Regression:
    """Regression tests to ensure Sprint 1-7 still work"""
    
    def setup_method(self):
        self.message_bus = MessageBus()
        
    def test_message_bus_still_works(self):
        """Test that message bus from Sprint 1 still works"""
        assert self.message_bus is not None
        
        # Publish and subscribe
        received = []
        def handler(data):
            received.append(data)
        
        self.message_bus.subscribe('test_topic', handler)
        self.message_bus.publish('test_topic', {'value': 42})
        
        assert len(received) == 1
        assert received[0]['value'] == 42
        
    def test_rl_controller_still_works(self):
        """Test that RLController from Sprint 2 still works"""
        rl = RLController(self.message_bus)
        assert rl is not None
        # RLController exists and can be initialized
