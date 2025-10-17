"""
GNN Timespan Analyzer - Graph Neural Network for temporal pattern analysis

Uses GNN to analyze temporal relationships between decisions, indicators,
and outcomes. Provides deeper insights than traditional time-series analysis.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from collections import defaultdict


class GraphAttentionLayer(nn.Module):
    """Graph Attention Layer for temporal analysis"""
    
    def __init__(self, in_features: int, out_features: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        
        # Linear transformation
        self.W = nn.Linear(in_features, out_features)
        self.attention = nn.Linear(out_features * 2, 1)
        
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, node_features: torch.Tensor, 
                adjacency_matrix: torch.Tensor) -> torch.Tensor:
        # Linear transformation
        h = self.W(node_features)
        
        # Attention mechanism
        N = node_features.size(0)
        
        # Compute attention scores for each pair
        attention_scores = torch.zeros((N, N), device=h.device)
        for i in range(N):
            for j in range(N):
                # Concatenate features
                combined = torch.cat([h[i], h[j]], dim=0)
                # Compute attention score
                attention_scores[i, j] = self.leakyrelu(self.attention(combined))
        
        # Mask with adjacency matrix
        zero_vec = -9e15 * torch.ones_like(attention_scores)
        attention = torch.where(adjacency_matrix > 0, attention_scores, zero_vec)
        attention = self.softmax(attention)
        
        # Apply attention to features
        h_prime = torch.matmul(attention, h)
        
        return h_prime


class GNNTemporalNetwork(nn.Module):
    """GNN for temporal pattern analysis"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, 
                 num_layers: int = 3, num_heads: int = 4):
        super().__init__()
        
        self.num_layers = num_layers
        
        # GAT layers
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(input_dim if i == 0 else hidden_dim, 
                              hidden_dim, num_heads)
            for i in range(num_layers)
        ])
        
        # Output layer for pattern classification
        self.output_layer = nn.Linear(hidden_dim, 8)  # 8 pattern types
        
    def forward(self, node_features: torch.Tensor, 
                adjacency_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = node_features
        
        # Pass through GAT layers
        for gat_layer in self.gat_layers:
            h = torch.relu(gat_layer(h, adjacency_matrix))
            
        # Pattern classification
        patterns = self.output_layer(h)
        
        return h, patterns


class GNNTimespanAnalyzer:
    """
    GNN Timespan Analyzer for temporal pattern detection
    
    Uses Graph Neural Networks to:
    - Model temporal relationships as graphs
    - Detect complex patterns across time
    - Provide insights for decision-making
    - Integrate with timespan_tracker
    """
    
    def __init__(self, message_bus, input_dim: int = 32, hidden_dim: int = 64,
                 num_layers: int = 3, attention_heads: int = 4,
                 temporal_window: int = 20):
        self.message_bus = message_bus
        self.input_dim = input_dim
        self.temporal_window = temporal_window
        
        # GNN model
        self.model = GNNTemporalNetwork(input_dim, hidden_dim, num_layers, attention_heads)
        
        # Temporal data storage
        self.decision_history = []
        self.indicator_history = []
        self.outcome_history = []
        
        # Graph construction
        self.node_features = []
        self.adjacency_matrix = None
        
        # Pattern detection
        self.detected_patterns = defaultdict(list)
        
        # Subscribe to topics
        self.message_bus.subscribe('final_decision', self._handle_decision)
        self.message_bus.subscribe('indicator_data', self._handle_indicator)
        self.message_bus.subscribe('execution_result', self._handle_outcome)
        self.message_bus.subscribe('gnn_analysis_request', self._handle_analysis_request)
        
    def _handle_decision(self, data: Dict[str, Any]):
        """Handle decision events"""
        self.decision_history.append({
            'timestamp': data.get('timestamp', len(self.decision_history)),
            'action': data.get('action', 'HOLD'),
            'confidence': data.get('confidence', 0.5),
            'symbol': data.get('symbol', 'UNKNOWN')
        })
        
        # Keep only recent history
        if len(self.decision_history) > self.temporal_window:
            self.decision_history = self.decision_history[-self.temporal_window:]
            
    def _handle_indicator(self, data: Dict[str, Any]):
        """Handle indicator data"""
        self.indicator_history.append({
            'timestamp': data.get('timestamp', len(self.indicator_history)),
            'indicators': data.get('indicators', {}),
            'symbol': data.get('symbol', 'UNKNOWN')
        })
        
        if len(self.indicator_history) > self.temporal_window:
            self.indicator_history = self.indicator_history[-self.temporal_window:]
            
    def _handle_outcome(self, data: Dict[str, Any]):
        """Handle execution outcomes"""
        self.outcome_history.append({
            'timestamp': data.get('timestamp', len(self.outcome_history)),
            'success': data.get('success', False),
            'pnl': data.get('pnl', 0.0),
            'symbol': data.get('symbol', 'UNKNOWN')
        })
        
        if len(self.outcome_history) > self.temporal_window:
            self.outcome_history = self.outcome_history[-self.temporal_window:]
            
    def _handle_analysis_request(self, data: Dict[str, Any]):
        """Handle analysis request"""
        patterns = self.analyze_patterns()
        insights = self.get_temporal_insights()
        
        self.message_bus.publish('gnn_analysis_response', {
            'patterns': patterns,
            'insights': insights,
            'graph_size': len(self.node_features)
        })
        
    def construct_graph(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Construct temporal graph from history
        
        Returns:
            Tuple of (node_features, adjacency_matrix)
        """
        nodes = []
        
        # Create nodes from decisions, indicators, outcomes
        for i, decision in enumerate(self.decision_history):
            # Decision node features
            feature = np.zeros(self.input_dim)
            feature[0] = 1.0  # Node type: decision
            feature[1] = {'BUY': 1.0, 'SELL': -1.0, 'HOLD': 0.0}.get(decision['action'], 0.0)
            feature[2] = decision['confidence']
            feature[3] = i / len(self.decision_history)  # Temporal position
            nodes.append(feature)
            
        for i, indicator in enumerate(self.indicator_history):
            # Indicator node features
            feature = np.zeros(self.input_dim)
            feature[0] = 2.0  # Node type: indicator
            # Encode key indicators
            indicators = indicator['indicators']
            feature[4] = indicators.get('RSI', 50) / 100.0
            feature[5] = indicators.get('MACD', 0.0)
            feature[6] = indicators.get('ATR', 0.0)
            feature[3] = i / len(self.indicator_history)  # Temporal position
            nodes.append(feature)
            
        for i, outcome in enumerate(self.outcome_history):
            # Outcome node features
            feature = np.zeros(self.input_dim)
            feature[0] = 3.0  # Node type: outcome
            feature[7] = 1.0 if outcome['success'] else 0.0
            feature[8] = np.tanh(outcome['pnl'])  # Normalized P&L
            feature[3] = i / len(self.outcome_history)  # Temporal position
            nodes.append(feature)
            
        if not nodes:
            return torch.zeros((1, self.input_dim)), torch.zeros((1, 1))
            
        node_features = torch.FloatTensor(np.array(nodes))
        
        # Construct adjacency matrix (temporal connections)
        num_nodes = len(nodes)
        adjacency = torch.zeros((num_nodes, num_nodes))
        
        # Connect temporally adjacent nodes
        for i in range(num_nodes - 1):
            adjacency[i, i + 1] = 1.0
            adjacency[i + 1, i] = 1.0
            
        # Connect nodes within temporal window
        window = min(5, num_nodes)
        for i in range(num_nodes):
            for j in range(max(0, i - window), min(num_nodes, i + window + 1)):
                if i != j:
                    adjacency[i, j] = 1.0 / (abs(i - j) + 1)
                    
        return node_features, adjacency
        
    def analyze_patterns(self) -> Dict[str, Any]:
        """
        Analyze temporal patterns using GNN
        
        Returns:
            Dictionary of detected patterns
        """
        if not self.decision_history:
            return {'patterns': [], 'confidence': 0.0}
            
        # Construct graph
        node_features, adjacency = self.construct_graph()
        
        # Run GNN
        self.model.eval()
        with torch.no_grad():
            embeddings, pattern_logits = self.model(node_features, adjacency)
            
        # Identify patterns
        patterns = []
        pattern_types = [
            'uptrend', 'downtrend', 'reversal', 'consolidation',
            'breakout', 'breakdown', 'divergence', 'convergence'
        ]
        
        for i, logits in enumerate(pattern_logits):
            pattern_probs = torch.softmax(logits, dim=0)
            top_pattern_idx = torch.argmax(pattern_probs).item()
            top_pattern_prob = pattern_probs[top_pattern_idx].item()
            
            if top_pattern_prob > 0.5:
                patterns.append({
                    'type': pattern_types[top_pattern_idx],
                    'confidence': top_pattern_prob,
                    'node_index': i
                })
                
        return {
            'patterns': patterns,
            'num_nodes': len(node_features),
            'avg_confidence': np.mean([p['confidence'] for p in patterns]) if patterns else 0.0
        }
        
    def get_temporal_insights(self) -> Dict[str, Any]:
        """Get insights from temporal analysis"""
        insights = {
            'decision_frequency': len(self.decision_history) / max(1, self.temporal_window),
            'success_rate': 0.0,
            'pattern_summary': {},
            'recommendations': []
        }
        
        # Calculate success rate
        if self.outcome_history:
            successes = sum(1 for o in self.outcome_history if o['success'])
            insights['success_rate'] = successes / len(self.outcome_history)
            
        # Pattern summary
        patterns = self.analyze_patterns()
        if patterns['patterns']:
            pattern_counts = defaultdict(int)
            for p in patterns['patterns']:
                pattern_counts[p['type']] += 1
            insights['pattern_summary'] = dict(pattern_counts)
            
        # Generate recommendations
        if insights['success_rate'] < 0.3:
            insights['recommendations'].append('Low success rate - consider reducing trading frequency')
        if 'reversal' in insights['pattern_summary']:
            insights['recommendations'].append('Reversal pattern detected - monitor for trend change')
            
        return insights
        
    def update_parameters(self, parameters: Dict[str, Any]):
        """Update GNN parameters"""
        if 'temporal_window' in parameters:
            self.temporal_window = parameters['temporal_window']
            
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return {
            'decision_history_size': len(self.decision_history),
            'indicator_history_size': len(self.indicator_history),
            'outcome_history_size': len(self.outcome_history),
            'temporal_window': self.temporal_window,
            'patterns_detected': len(self.detected_patterns)
        }
