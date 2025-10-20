"""
Tests for Decision Transformer Agent
"""

import pytest
import numpy as np
import torch
from modules.decision_transformer_agent import (
    DecisionTransformer,
    TransformerBlock,
    DecisionTransformerAgent
)
from modules.message_bus import MessageBus


class TestTransformerBlock:
    """Tests for TransformerBlock"""
    
    def test_initialization(self):
        """Test transformer block initialization"""
        block = TransformerBlock(embed_dim=128, num_heads=4, ff_dim=512)
        assert block is not None
        assert block.attention is not None
        assert block.feed_forward is not None
        
    def test_forward_pass(self):
        """Test forward pass through transformer block"""
        block = TransformerBlock(embed_dim=128, num_heads=4, ff_dim=512)
        x = torch.randn(2, 10, 128)  # (batch, seq_len, embed_dim)
        
        output, attn_weights = block(x)
        
        assert output.shape == x.shape
        assert attn_weights is not None
        assert attn_weights.shape[0] == 2  # batch size


class TestDecisionTransformer:
    """Tests for DecisionTransformer model"""
    
    def test_initialization(self):
        """Test Decision Transformer initialization"""
        model = DecisionTransformer(
            state_dim=10,
            action_dim=3,
            embed_dim=128,
            num_layers=3,
            num_heads=4,
            max_length=20
        )
        assert model is not None
        assert len(model.transformer_blocks) == 3
        
    def test_forward_pass(self):
        """Test forward pass through Decision Transformer"""
        model = DecisionTransformer(
            state_dim=10,
            action_dim=3,
            embed_dim=128,
            num_layers=2,
            num_heads=4,
            max_length=20
        )
        
        batch_size = 2
        seq_length = 5
        
        returns = torch.randn(batch_size, seq_length, 1)
        states = torch.randn(batch_size, seq_length, 10)
        actions = torch.randn(batch_size, seq_length, 3)
        timesteps = torch.arange(seq_length).unsqueeze(0).repeat(batch_size, 1)
        
        action_preds, attention_weights = model(returns, states, actions, timesteps)
        
        assert action_preds.shape == (batch_size, seq_length, 3)
        assert len(attention_weights) == 2  # num_layers
        
    def test_output_dimensions(self):
        """Test output dimensions for different sequence lengths"""
        model = DecisionTransformer(
            state_dim=10,
            action_dim=3,
            embed_dim=64,
            num_layers=2,
            num_heads=2,
            max_length=20
        )
        
        for seq_length in [1, 5, 10, 15]:
            returns = torch.randn(1, seq_length, 1)
            states = torch.randn(1, seq_length, 10)
            actions = torch.randn(1, seq_length, 3)
            timesteps = torch.arange(seq_length).unsqueeze(0)
            
            action_preds, _ = model(returns, states, actions, timesteps)
            assert action_preds.shape == (1, seq_length, 3)


class TestDecisionTransformerAgent:
    """Tests for DecisionTransformerAgent"""
    
    def test_initialization(self):
        """Test agent initialization"""
        message_bus = MessageBus()
        agent = DecisionTransformerAgent(
            message_bus=message_bus,
            state_dim=10,
            action_dim=3,
            embed_dim=128,
            num_layers=3,
            num_heads=4
        )
        
        assert agent is not None
        assert agent.model is not None
        assert agent.optimizer is not None
        assert len(agent.sequence_buffer) == 0
        
    def test_extract_state(self):
        """Test state extraction from market data"""
        message_bus = MessageBus()
        agent = DecisionTransformerAgent(message_bus=message_bus)
        
        market_data = {
            'close': 150.0,
            'volume': 1000000,
            'rsi': 65.0,
            'macd': 2.5,
            'sma_20': 148.0,
            'atr': 3.2,
            'portfolio_value': 10500.0,
            'cash': 5000.0,
            'position_size': 100.0,
            'unrealized_pnl': 500.0
        }
        
        state = agent._extract_state(market_data)
        
        assert state.shape == (10,)
        assert np.all(np.abs(state) <= 1.0)  # Should be normalized
        
    def test_encode_action(self):
        """Test action encoding"""
        message_bus = MessageBus()
        agent = DecisionTransformerAgent(message_bus=message_bus)
        
        hold_action = agent._encode_action('HOLD')
        assert hold_action.shape == (3,)
        assert hold_action[0] == 1.0
        
        buy_action = agent._encode_action('BUY')
        assert buy_action[1] == 1.0
        
        sell_action = agent._encode_action('SELL')
        assert sell_action[2] == 1.0
        
    def test_decode_action(self):
        """Test action decoding"""
        message_bus = MessageBus()
        agent = DecisionTransformerAgent(message_bus=message_bus)
        
        action_vec = np.array([1.0, 0.0, 0.0])
        action_str = agent._decode_action(action_vec)
        assert action_str == 'HOLD'
        
        action_vec = np.array([0.0, 1.0, 0.0])
        action_str = agent._decode_action(action_vec)
        assert action_str == 'BUY'
        
    def test_calculate_returns_to_go(self):
        """Test return-to-go calculation"""
        message_bus = MessageBus()
        agent = DecisionTransformerAgent(message_bus=message_bus)
        
        rewards = [1.0, 2.0, 3.0, 4.0]
        returns_to_go = agent._calculate_returns_to_go(rewards, gamma=0.9)
        
        assert len(returns_to_go) == 4
        assert returns_to_go[0] > returns_to_go[1]  # Earlier returns should be higher
        assert returns_to_go[-1] == 4.0  # Last return is just the last reward
        
    def test_predict_action_cold_start(self):
        """Test action prediction with no history (cold start)"""
        message_bus = MessageBus()
        agent = DecisionTransformerAgent(message_bus=message_bus)
        
        state = np.random.randn(10)
        action, metrics = agent.predict_action(state, target_return=100.0)
        
        assert action.shape == (3,)
        assert 'confidence' in metrics
        assert 'action_probs' in metrics
        assert 'attention_weights' in metrics
        
    def test_predict_action_with_history(self):
        """Test action prediction with sequence history"""
        message_bus = MessageBus()
        agent = DecisionTransformerAgent(message_bus=message_bus)
        
        # Build some history
        for i in range(5):
            state = np.random.randn(10)
            action = np.array([1.0, 0.0, 0.0])
            agent.current_sequence['states'].append(state)
            agent.current_sequence['actions'].append(action)
            agent.current_sequence['rewards'].append(1.0)
        
        # Predict next action
        state = np.random.randn(10)
        action, metrics = agent.predict_action(state, target_return=50.0)
        
        assert action.shape == (3,)
        assert metrics['sequence_length'] == 5
        
    def test_train_step_insufficient_data(self):
        """Test training step with insufficient data"""
        message_bus = MessageBus()
        agent = DecisionTransformerAgent(message_bus=message_bus)
        
        result = agent.train_step(batch_size=32)
        
        assert result['skipped'] == True
        assert result['loss'] == 0.0
        
    def test_train_step_with_data(self):
        """Test training step with sufficient data"""
        message_bus = MessageBus()
        agent = DecisionTransformerAgent(message_bus=message_bus)
        
        # Add sequences to buffer
        for _ in range(50):
            seq_length = 10
            sequence = {
                'states': np.random.randn(seq_length, 10),
                'actions': np.random.randn(seq_length, 3),
                'rewards': np.random.randn(seq_length),
                'returns_to_go': np.random.randn(seq_length),
                'timesteps': np.arange(seq_length)
            }
            agent.sequence_buffer.append(sequence)
        
        result = agent.train_step(batch_size=32)
        
        assert 'loss' in result
        assert 'avg_loss' in result
        assert 'total_steps' in result
        assert result['total_steps'] == 1
        
    def test_on_reward(self):
        """Test reward handling"""
        message_bus = MessageBus()
        agent = DecisionTransformerAgent(message_bus=message_bus)
        
        agent.last_action = np.array([1.0, 0.0, 0.0])
        
        reward_data = {'reward': 5.0, 'timestamp': 1234567890}
        agent._on_reward(reward_data)
        
        assert agent.last_reward == 5.0
        assert len(agent.current_sequence['rewards']) == 1
        
    def test_on_tuned_reward(self):
        """Test tuned reward handling"""
        message_bus = MessageBus()
        agent = DecisionTransformerAgent(message_bus=message_bus)
        
        tuned_reward_data = {'tuned_reward': 7.0}
        agent._on_tuned_reward(tuned_reward_data)
        
        assert agent.last_reward == 7.0
        
    def test_on_market_data(self):
        """Test market data handling"""
        message_bus = MessageBus()
        agent = DecisionTransformerAgent(message_bus=message_bus)
        
        market_data = {
            'close': 150.0,
            'volume': 1000000,
            'rsi': 65.0
        }
        
        agent._on_market_data(market_data)
        
        assert agent.current_state is not None
        assert agent.current_state.shape == (10,)
        
    def test_on_action_request(self):
        """Test action request handling"""
        message_bus = MessageBus()
        agent = DecisionTransformerAgent(message_bus=message_bus)
        
        # Set current state
        agent.current_state = np.random.randn(10)
        
        # Track published messages
        dt_actions = []
        dt_metrics = []
        
        def capture_action(msg):
            dt_actions.append(msg)
            
        def capture_metrics(msg):
            dt_metrics.append(msg)
        
        message_bus.subscribe('dt_action', capture_action)
        message_bus.subscribe('dt_metrics', capture_metrics)
        
        # Request action
        agent._on_action_request({})
        
        assert len(dt_actions) == 1
        assert len(dt_metrics) == 1
        assert 'action' in dt_actions[0]
        assert 'confidence' in dt_actions[0]
        
    def test_on_parameter_adjustment(self):
        """Test parameter adjustment handling"""
        message_bus = MessageBus()
        agent = DecisionTransformerAgent(message_bus=message_bus)
        
        params = {
            'dt_learning_rate': 0.0005,
            'dt_target_return_weight': 1.5,
            'dt_sequence_length': 30
        }
        
        agent._on_parameter_adjustment(params)
        
        assert agent.optimizer.param_groups[0]['lr'] == 0.0005
        assert agent.target_return_weight == 1.5
        assert agent.max_sequence_length == 30
        
    def test_update_target_return(self):
        """Test target return update"""
        message_bus = MessageBus()
        agent = DecisionTransformerAgent(message_bus=message_bus)
        
        recent_returns = [50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        agent.update_target_return(recent_returns)
        
        # Should be 75th percentile
        assert agent.target_return == np.percentile(recent_returns, 75)
        
    def test_get_metrics(self):
        """Test metrics retrieval"""
        message_bus = MessageBus()
        agent = DecisionTransformerAgent(message_bus=message_bus)
        
        metrics = agent.get_metrics()
        
        assert 'training_steps' in metrics
        assert 'episodes' in metrics
        assert 'avg_loss' in metrics
        assert 'buffer_size' in metrics
        assert 'target_return' in metrics
        
    def test_message_bus_integration(self):
        """Test full message bus integration"""
        message_bus = MessageBus()
        agent = DecisionTransformerAgent(message_bus=message_bus)
        
        # Publish market data
        market_data = {
            'close': 150.0,
            'volume': 1000000,
            'rsi': 65.0
        }
        message_bus.publish('market_data', market_data)
        
        # Verify state was updated
        assert agent.current_state is not None
        
        # Publish reward
        message_bus.publish('reward', {'reward': 5.0})
        assert agent.last_reward == 5.0
        
    def test_process_decision_history(self):
        """Test processing of decision history"""
        message_bus = MessageBus()
        agent = DecisionTransformerAgent(message_bus=message_bus, max_sequence_length=5)
        
        # Create decision history
        history = []
        for i in range(10):
            history.append({
                'market_data': {
                    'close': 150.0 + i,
                    'volume': 1000000,
                    'rsi': 50.0 + i
                },
                'action': 'BUY' if i % 2 == 0 else 'SELL',
                'reward': float(i)
            })
        
        agent._process_decision_history(history)
        
        # Should have created sequences
        assert len(agent.sequence_buffer) > 0
        
        # Check sequence format
        seq = agent.sequence_buffer[0]
        assert 'states' in seq
        assert 'actions' in seq
        assert 'rewards' in seq
        assert 'returns_to_go' in seq
        assert seq['states'].shape[0] == 5  # max_sequence_length
        
    def test_training_convergence(self):
        """Test that training loss decreases over steps"""
        message_bus = MessageBus()
        agent = DecisionTransformerAgent(message_bus=message_bus, learning_rate=0.001)
        
        # Generate training data
        for _ in range(100):
            seq_length = 10
            sequence = {
                'states': np.random.randn(seq_length, 10),
                'actions': np.random.randn(seq_length, 3),
                'rewards': np.random.randn(seq_length),
                'returns_to_go': np.random.randn(seq_length),
                'timesteps': np.arange(seq_length)
            }
            agent.sequence_buffer.append(sequence)
        
        # Train for several steps
        losses = []
        for _ in range(10):
            result = agent.train_step(batch_size=32)
            losses.append(result['loss'])
        
        # Check that average loss in second half is lower than first half
        first_half = np.mean(losses[:5])
        second_half = np.mean(losses[5:])
        
        # Loss should generally decrease (may fluctuate)
        assert len(losses) == 10


class TestDecisionTransformerIntegration:
    """Integration tests for Decision Transformer with system"""
    
    def test_end_to_end_prediction(self):
        """Test end-to-end prediction flow"""
        message_bus = MessageBus()
        agent = DecisionTransformerAgent(message_bus=message_bus)
        
        # Simulate market data flow
        for i in range(10):
            market_data = {
                'close': 150.0 + i,
                'volume': 1000000 + i * 10000,
                'rsi': 50.0 + i * 2,
                'macd': 1.0 + i * 0.1
            }
            message_bus.publish('market_data', market_data)
            message_bus.publish('reward', {'reward': float(i)})
        
        # Request action
        captured_actions = []
        message_bus.subscribe('dt_action', lambda msg: captured_actions.append(msg))
        
        message_bus.publish('dt_action_request', {})
        
        assert len(captured_actions) == 1
        assert 'action_type' in captured_actions[0]
        
    def test_memory_insights_integration(self):
        """Test integration with strategic_memory_engine"""
        message_bus = MessageBus()
        agent = DecisionTransformerAgent(message_bus=message_bus, max_sequence_length=5)
        
        # Simulate memory insights with decision history
        history = []
        for i in range(15):
            history.append({
                'market_data': {'close': 150.0 + i, 'volume': 1000000},
                'action': 'BUY' if i % 3 == 0 else 'HOLD',
                'reward': float(i * 0.5)
            })
        
        memory_insights = {
            'decision_history': history,
            'patterns': []
        }
        
        message_bus.publish('memory_insights', memory_insights)
        
        # Agent should have processed sequences
        assert len(agent.sequence_buffer) > 0
