"""
Tests for GNN Timespan Analyzer
"""

import pytest
import numpy as np
from modules.gnn_timespan_analyzer import GNNTimespanAnalyzer, GNNTemporalNetwork, GraphAttentionLayer
from modules.message_bus import MessageBus


class TestGraphAttentionLayer:
    """Tests for Graph Attention Layer"""
    
    def test_initialization(self):
        layer = GraphAttentionLayer(in_features=32, out_features=64, num_heads=4)
        assert layer is not None
        
    def test_forward_pass(self):
        layer = GraphAttentionLayer(in_features=32, out_features=64, num_heads=4)
        import torch
        
        # Create sample data
        node_features = torch.randn(10, 32)
        adjacency_matrix = torch.ones(10, 10)
        
        output = layer(node_features, adjacency_matrix)
        assert output.shape[0] == 10


class TestGNNTemporalNetwork:
    """Tests for GNN Temporal Network"""
    
    def test_initialization(self):
        network = GNNTemporalNetwork(input_dim=32, hidden_dim=64, num_layers=3)
        assert network is not None
        
    def test_forward_pass(self):
        network = GNNTemporalNetwork(input_dim=32, hidden_dim=64, num_layers=3)
        import torch
        
        node_features = torch.randn(10, 32)
        adjacency_matrix = torch.ones(10, 10)
        
        embeddings, patterns = network(node_features, adjacency_matrix)
        assert embeddings.shape == (10, 64)
        assert patterns.shape == (10, 8)


class TestGNNTimespanAnalyzer:
    """Tests for GNN Timespan Analyzer"""
    
    def setup_method(self):
        self.message_bus = MessageBus()
        self.gnn = GNNTimespanAnalyzer(
            self.message_bus,
            input_dim=32,
            hidden_dim=64,
            temporal_window=20
        )
        
    def test_initialization(self):
        assert self.gnn is not None
        assert self.gnn.input_dim == 32
        assert self.gnn.temporal_window == 20
        
    def test_handle_decision(self):
        self.message_bus.publish('final_decision', {
            'timestamp': 0,
            'action': 'BUY',
            'confidence': 0.8,
            'symbol': 'AAPL'
        })
        assert len(self.gnn.decision_history) == 1
        
    def test_handle_indicator(self):
        self.message_bus.publish('indicator_data', {
            'timestamp': 0,
            'indicators': {'RSI': 65, 'MACD': 0.5, 'ATR': 2.5},
            'symbol': 'AAPL'
        })
        assert len(self.gnn.indicator_history) == 1
        
    def test_handle_outcome(self):
        self.message_bus.publish('execution_result', {
            'timestamp': 0,
            'success': True,
            'pnl': 100.0,
            'symbol': 'AAPL'
        })
        assert len(self.gnn.outcome_history) == 1
        
    def test_temporal_window_limit_decisions(self):
        """Test that decision history respects temporal window"""
        for i in range(30):
            self.message_bus.publish('final_decision', {
                'timestamp': i,
                'action': 'BUY',
                'confidence': 0.8,
                'symbol': 'AAPL'
            })
        assert len(self.gnn.decision_history) == self.gnn.temporal_window
        
    def test_temporal_window_limit_indicators(self):
        """Test that indicator history respects temporal window"""
        for i in range(30):
            self.message_bus.publish('indicator_data', {
                'timestamp': i,
                'indicators': {'RSI': 50},
                'symbol': 'AAPL'
            })
        assert len(self.gnn.indicator_history) == self.gnn.temporal_window
        
    def test_temporal_window_limit_outcomes(self):
        """Test that outcome history respects temporal window"""
        for i in range(30):
            self.message_bus.publish('execution_result', {
                'timestamp': i,
                'success': True,
                'pnl': 10.0,
                'symbol': 'AAPL'
            })
        assert len(self.gnn.outcome_history) == self.gnn.temporal_window
        
    def test_construct_graph_empty(self):
        """Test graph construction with empty history"""
        node_features, adjacency = self.gnn.construct_graph()
        assert node_features.shape[0] >= 1
        assert adjacency.shape[0] >= 1
        
    def test_construct_graph_with_data(self):
        """Test graph construction with sample data"""
        # Add sample data
        for i in range(5):
            self.message_bus.publish('final_decision', {
                'timestamp': i,
                'action': 'BUY',
                'confidence': 0.8,
                'symbol': 'AAPL'
            })
            self.message_bus.publish('indicator_data', {
                'timestamp': i,
                'indicators': {'RSI': 50 + i, 'MACD': 0.5, 'ATR': 2.5},
                'symbol': 'AAPL'
            })
            self.message_bus.publish('execution_result', {
                'timestamp': i,
                'success': True,
                'pnl': 10.0 * i,
                'symbol': 'AAPL'
            })
        
        node_features, adjacency = self.gnn.construct_graph()
        # 5 decisions + 5 indicators + 5 outcomes = 15 nodes
        assert node_features.shape[0] == 15
        assert adjacency.shape == (15, 15)
        
    def test_graph_adjacency_structure(self):
        """Test that adjacency matrix has correct structure"""
        # Add sample data
        for i in range(3):
            self.message_bus.publish('final_decision', {
                'timestamp': i,
                'action': 'BUY',
                'confidence': 0.8,
                'symbol': 'AAPL'
            })
        
        node_features, adjacency = self.gnn.construct_graph()
        
        # Adjacency should be symmetric for undirected graph
        import torch
        assert torch.allclose(adjacency, adjacency.t(), atol=1e-6)
        
    def test_analyze_patterns_empty(self):
        """Test pattern analysis with no data"""
        patterns = self.gnn.analyze_patterns()
        assert 'patterns' in patterns
        assert 'confidence' in patterns
        
    def test_analyze_patterns_with_data(self):
        """Test pattern analysis with sample data"""
        # Add sample data
        for i in range(10):
            self.message_bus.publish('final_decision', {
                'timestamp': i,
                'action': 'BUY' if i % 2 == 0 else 'SELL',
                'confidence': 0.8,
                'symbol': 'AAPL'
            })
            self.message_bus.publish('indicator_data', {
                'timestamp': i,
                'indicators': {'RSI': 50 + i * 2, 'MACD': 0.5, 'ATR': 2.5},
                'symbol': 'AAPL'
            })
        
        patterns = self.gnn.analyze_patterns()
        assert 'patterns' in patterns
        assert 'num_nodes' in patterns
        assert patterns['num_nodes'] > 0
        
    def test_get_temporal_insights_empty(self):
        """Test insights with no data"""
        insights = self.gnn.get_temporal_insights()
        assert 'decision_frequency' in insights
        assert 'success_rate' in insights
        assert 'recommendations' in insights
        
    def test_get_temporal_insights_with_data(self):
        """Test insights with sample data"""
        # Add decisions and outcomes
        for i in range(10):
            self.message_bus.publish('final_decision', {
                'timestamp': i,
                'action': 'BUY',
                'confidence': 0.8,
                'symbol': 'AAPL'
            })
            self.message_bus.publish('execution_result', {
                'timestamp': i,
                'success': i % 3 == 0,
                'pnl': 10.0,
                'symbol': 'AAPL'
            })
        
        insights = self.gnn.get_temporal_insights()
        assert insights['decision_frequency'] > 0
        assert 0.0 <= insights['success_rate'] <= 1.0
        
    def test_pattern_types(self):
        """Test that all pattern types can be detected"""
        # Add varied data to trigger different patterns
        actions = ['BUY', 'BUY', 'BUY', 'SELL', 'SELL', 'HOLD', 'BUY', 'SELL']
        for i, action in enumerate(actions):
            self.message_bus.publish('final_decision', {
                'timestamp': i,
                'action': action,
                'confidence': 0.7 + (i % 3) * 0.1,
                'symbol': 'AAPL'
            })
            self.message_bus.publish('indicator_data', {
                'timestamp': i,
                'indicators': {'RSI': 40 + i * 5, 'MACD': 0.5, 'ATR': 2.5},
                'symbol': 'AAPL'
            })
        
        patterns = self.gnn.analyze_patterns()
        assert 'patterns' in patterns
        
    def test_handle_analysis_request(self):
        """Test handling of analysis request"""
        # Add some data
        for i in range(5):
            self.message_bus.publish('final_decision', {
                'timestamp': i,
                'action': 'BUY',
                'confidence': 0.8,
                'symbol': 'AAPL'
            })
        
        self.message_bus.publish('gnn_analysis_request', {})
        # Should publish response
        
    def test_update_parameters(self):
        """Test parameter updates"""
        self.gnn.update_parameters({'temporal_window': 30})
        assert self.gnn.temporal_window == 30
        
    def test_get_metrics(self):
        """Test metrics retrieval"""
        metrics = self.gnn.get_metrics()
        assert 'decision_history_size' in metrics
        assert 'indicator_history_size' in metrics
        assert 'outcome_history_size' in metrics
        assert 'temporal_window' in metrics
        
    def test_success_rate_calculation(self):
        """Test success rate calculation"""
        # Add outcomes with known success rate
        for i in range(10):
            self.message_bus.publish('execution_result', {
                'timestamp': i,
                'success': i < 7,  # 70% success rate
                'pnl': 10.0,
                'symbol': 'AAPL'
            })
        
        insights = self.gnn.get_temporal_insights()
        assert insights['success_rate'] == 0.7
        
    def test_low_success_rate_recommendation(self):
        """Test that low success rate triggers recommendation"""
        # Add outcomes with low success rate
        for i in range(10):
            self.message_bus.publish('execution_result', {
                'timestamp': i,
                'success': i < 2,  # 20% success rate
                'pnl': 10.0,
                'symbol': 'AAPL'
            })
        
        insights = self.gnn.get_temporal_insights()
        assert len(insights['recommendations']) > 0
