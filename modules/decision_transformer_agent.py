"""
decision_transformer_agent.py - Decision Transformer Agent

Beskrivning:
    Decision Transformer (DT) för sekvensbaserad RL.
    Använder transformer-arkitektur för att modellera beslutsekvenser
    baserat på (state, action, reward, return-to-go).
    
Roll:
    - Tar in sekvenser från strategic_memory_engine
    - Predikterar actions baserat på desired return-to-go
    - Använder attention mechanism för temporal dependencies
    - Tränas offline på historiska sekvenser
    - Integreras med PPO, DQN, GAN, GNN i ensemble
    
Inputs:
    - memory_sequences: Dict - Sekvenser från strategic_memory_engine
    - target_return: float - Önskad return-to-go
    - market_data: Dict - Marknadsdata för state representation
    
Outputs:
    - dt_action: Dict - Predikterad action från DT
    - dt_metrics: Dict - Training metrics och attention weights
    
Publicerar till message_bus:
    - dt_action: Action predictions
    - dt_metrics: Training progress och attention analysis
    
Prenumererar på:
    - memory_insights (från strategic_memory_engine)
    - reward (från portfolio_manager via reward_tuner)
    - tuned_reward (från reward_tuner)
    
Använder RL: Ja (Transformer-based sequence modeling)
Tar emot feedback: Ja (via reward signals)

Används i Sprint: 10 (Decision Transformer Integration)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from collections import deque
import time


class TransformerBlock(nn.Module):
    """Single transformer block with multi-head attention"""
    
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Multi-head attention with residual
        attn_output, attn_weights = self.attention(x, x, x, attn_mask=mask, need_weights=True)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attn_weights


class DecisionTransformer(nn.Module):
    """
    Decision Transformer architecture for sequence-based RL
    
    Processes sequences of (return-to-go, state, action) tuples
    and predicts next action given desired return.
    """
    
    def __init__(self, state_dim: int, action_dim: int, embed_dim: int = 128,
                 num_layers: int = 3, num_heads: int = 4, max_length: int = 20,
                 dropout: float = 0.1):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.max_length = max_length
        
        # Embeddings for return-to-go, state, action
        self.return_embedding = nn.Linear(1, embed_dim)
        self.state_embedding = nn.Linear(state_dim, embed_dim)
        self.action_embedding = nn.Linear(action_dim, embed_dim)
        
        # Positional encoding
        self.position_embedding = nn.Parameter(torch.zeros(1, max_length * 3, embed_dim))
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, embed_dim * 4, dropout)
            for _ in range(num_layers)
        ])
        
        # Action prediction head
        self.action_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, action_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, returns: torch.Tensor, states: torch.Tensor, 
                actions: torch.Tensor, timesteps: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through Decision Transformer
        
        Args:
            returns: (batch_size, seq_length, 1) - return-to-go values
            states: (batch_size, seq_length, state_dim) - states
            actions: (batch_size, seq_length, action_dim) - actions
            timesteps: (batch_size, seq_length) - timestep indices
            
        Returns:
            action_preds: (batch_size, seq_length, action_dim) - predicted actions
            attention_weights: List of attention weight tensors from each layer
        """
        batch_size, seq_length = states.shape[0], states.shape[1]
        
        # Embed return-to-go, states, actions
        return_embeds = self.return_embedding(returns)  # (B, T, D)
        state_embeds = self.state_embedding(states)      # (B, T, D)
        action_embeds = self.action_embedding(actions)   # (B, T, D)
        
        # Interleave return, state, action embeddings
        # Shape: (B, 3*T, D)
        sequence = torch.stack([return_embeds, state_embeds, action_embeds], dim=2)
        sequence = sequence.reshape(batch_size, 3 * seq_length, self.embed_dim)
        
        # Add positional encoding
        sequence = sequence + self.position_embedding[:, :3 * seq_length, :]
        sequence = self.dropout(sequence)
        
        # Create causal mask to prevent looking into the future
        mask = torch.triu(torch.ones(3 * seq_length, 3 * seq_length), diagonal=1).bool()
        mask = mask.to(sequence.device)
        
        # Pass through transformer blocks
        attention_weights_list = []
        x = sequence
        for block in self.transformer_blocks:
            x, attn_weights = block(x, mask=mask)
            attention_weights_list.append(attn_weights)
        
        # Extract state positions (every 3rd position starting from 1)
        # We predict action from state embedding
        state_positions = torch.arange(1, 3 * seq_length, 3, device=x.device)
        x = x[:, state_positions, :]  # (B, T, D)
        
        # Predict actions
        action_preds = self.action_predictor(x)  # (B, T, action_dim)
        
        return action_preds, attention_weights_list


class DecisionTransformerAgent:
    """
    Decision Transformer Agent for NextGen trading system
    
    Integrates transformer-based sequence modeling with the existing
    PPO, DQN, GAN, GNN ensemble.
    """
    
    def __init__(self, message_bus, state_dim: int = 10, action_dim: int = 3,
                 embed_dim: int = 128, num_layers: int = 3, num_heads: int = 4,
                 max_sequence_length: int = 20, learning_rate: float = 0.0001,
                 target_return_weight: float = 1.0, dropout: float = 0.1):
        """
        Initialize Decision Transformer Agent
        
        Args:
            message_bus: Central message bus for communication
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            embed_dim: Embedding dimension for transformer
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            max_sequence_length: Maximum sequence length to process
            learning_rate: Learning rate for optimizer
            target_return_weight: Weight for target return in loss
            dropout: Dropout rate
        """
        self.message_bus = message_bus
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_sequence_length = max_sequence_length
        self.target_return_weight = target_return_weight
        
        # Initialize transformer model
        self.model = DecisionTransformer(
            state_dim=state_dim,
            action_dim=action_dim,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_length=max_sequence_length,
            dropout=dropout
        )
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Sequence buffer for collecting trajectories
        self.sequence_buffer = deque(maxlen=1000)
        self.current_sequence = {
            'states': [],
            'actions': [],
            'rewards': [],
            'returns_to_go': [],
            'timesteps': []
        }
        
        # Training metrics
        self.training_metrics = {
            'total_steps': 0,
            'episodes': 0,
            'avg_loss': 0.0,
            'avg_return': 0.0,
            'losses': deque(maxlen=100),
            'returns': deque(maxlen=100)
        }
        
        # Latest attention weights for visualization
        self.latest_attention_weights = None
        
        # Target return (adaptive based on performance)
        self.target_return = 100.0  # Initial target
        
        # Subscribe to message bus topics
        self.message_bus.subscribe('memory_insights', self._on_memory_insights)
        self.message_bus.subscribe('reward', self._on_reward)
        self.message_bus.subscribe('tuned_reward', self._on_tuned_reward)
        self.message_bus.subscribe('market_data', self._on_market_data)
        self.message_bus.subscribe('dt_action_request', self._on_action_request)
        self.message_bus.subscribe('parameter_adjustment', self._on_parameter_adjustment)
        
        # Current state for action prediction
        self.current_state = None
        self.last_action = None
        self.last_reward = 0.0
        
    def _on_memory_insights(self, insights: Dict[str, Any]) -> None:
        """Handle memory insights from strategic_memory_engine"""
        # Extract sequences from memory for training
        if 'decision_history' in insights:
            self._process_decision_history(insights['decision_history'])
            
    def _on_reward(self, reward_data: Dict[str, Any]) -> None:
        """Handle base reward signal"""
        reward = reward_data.get('reward', 0.0)
        self.last_reward = reward
        
        # Add to current sequence
        if self.last_action is not None:
            self.current_sequence['rewards'].append(reward)
            
    def _on_tuned_reward(self, reward_data: Dict[str, Any]) -> None:
        """Handle tuned reward from reward_tuner"""
        tuned_reward = reward_data.get('tuned_reward', 0.0)
        # Use tuned reward for training (more stable signal)
        self.last_reward = tuned_reward
        
    def _on_market_data(self, market_data: Dict[str, Any]) -> None:
        """Handle market data for state representation"""
        # Extract state from market data
        state = self._extract_state(market_data)
        self.current_state = state
        
    def _on_action_request(self, request: Dict[str, Any]) -> None:
        """Handle action request and provide DT prediction"""
        if self.current_state is not None:
            action, metrics = self.predict_action(
                self.current_state, 
                target_return=self.target_return
            )
            
            # Publish action
            self.message_bus.publish('dt_action', {
                'action': action,
                'action_type': self._decode_action(action),
                'confidence': metrics.get('confidence', 0.0),
                'target_return': self.target_return,
                'timestamp': time.time()
            })
            
            # Publish metrics with attention analysis
            self.message_bus.publish('dt_metrics', metrics)
            
    def _on_parameter_adjustment(self, params: Dict[str, Any]) -> None:
        """Handle adaptive parameter adjustments"""
        if 'dt_learning_rate' in params:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = params['dt_learning_rate']
                
        if 'dt_target_return_weight' in params:
            self.target_return_weight = params['dt_target_return_weight']
            
        if 'dt_sequence_length' in params:
            self.max_sequence_length = int(params['dt_sequence_length'])
            
    def _extract_state(self, market_data: Dict[str, Any]) -> np.ndarray:
        """Extract state vector from market data"""
        # Extract features: price, volume, indicators, portfolio state
        features = []
        
        # Price features
        features.append(market_data.get('close', 0.0))
        features.append(market_data.get('volume', 0.0))
        
        # Technical indicators
        features.append(market_data.get('rsi', 50.0))
        features.append(market_data.get('macd', 0.0))
        features.append(market_data.get('sma_20', 0.0))
        features.append(market_data.get('atr', 0.0))
        
        # Portfolio state (if available)
        features.append(market_data.get('portfolio_value', 10000.0))
        features.append(market_data.get('cash', 10000.0))
        features.append(market_data.get('position_size', 0.0))
        features.append(market_data.get('unrealized_pnl', 0.0))
        
        state = np.array(features, dtype=np.float32)
        
        # Normalize state (simple normalization, can be improved)
        state = state / (np.abs(state).max() + 1e-8)
        
        # Pad or truncate to state_dim
        if len(state) < self.state_dim:
            state = np.pad(state, (0, self.state_dim - len(state)))
        else:
            state = state[:self.state_dim]
            
        return state
        
    def _decode_action(self, action: np.ndarray) -> str:
        """Decode action vector to human-readable string"""
        action_idx = np.argmax(action)
        action_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
        return action_map.get(action_idx, 'HOLD')
        
    def _process_decision_history(self, history: List[Dict[str, Any]]) -> None:
        """Process decision history to extract training sequences"""
        if len(history) < 2:
            return
            
        # Convert history to sequence format
        for i in range(len(history) - self.max_sequence_length):
            sequence_slice = history[i:i + self.max_sequence_length]
            
            states = []
            actions = []
            rewards = []
            
            for entry in sequence_slice:
                # Extract state
                state = self._extract_state(entry.get('market_data', {}))
                states.append(state)
                
                # Extract action (one-hot encoded)
                action = entry.get('action', 'HOLD')
                action_vec = self._encode_action(action)
                actions.append(action_vec)
                
                # Extract reward
                reward = entry.get('reward', 0.0)
                rewards.append(reward)
            
            # Calculate return-to-go
            returns_to_go = self._calculate_returns_to_go(rewards)
            
            # Store sequence
            self.sequence_buffer.append({
                'states': np.array(states),
                'actions': np.array(actions),
                'rewards': np.array(rewards),
                'returns_to_go': np.array(returns_to_go),
                'timesteps': np.arange(len(states))
            })
            
    def _encode_action(self, action: str) -> np.ndarray:
        """Encode action string to one-hot vector"""
        action_map = {'HOLD': 0, 'BUY': 1, 'SELL': 2}
        action_idx = action_map.get(action, 0)
        action_vec = np.zeros(self.action_dim, dtype=np.float32)
        action_vec[action_idx] = 1.0
        return action_vec
        
    def _calculate_returns_to_go(self, rewards: List[float], gamma: float = 0.99) -> List[float]:
        """Calculate discounted return-to-go for each timestep"""
        returns_to_go = []
        running_return = 0.0
        
        for reward in reversed(rewards):
            running_return = reward + gamma * running_return
            returns_to_go.insert(0, running_return)
            
        return returns_to_go
        
    def predict_action(self, state: np.ndarray, target_return: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Predict action given current state and target return
        
        Args:
            state: Current state vector
            target_return: Desired return-to-go
            
        Returns:
            action: Predicted action vector
            metrics: Dictionary of prediction metrics
        """
        self.model.eval()
        
        with torch.no_grad():
            # Build sequence from current trajectory
            if len(self.current_sequence['states']) == 0:
                # Cold start - use zero padding
                seq_length = 1
                states = state.reshape(1, 1, -1)
                actions = np.zeros((1, 1, self.action_dim), dtype=np.float32)
                returns = np.array([[[target_return]]], dtype=np.float32)
                timesteps = np.array([[0]], dtype=np.int64)
            else:
                # Use recent trajectory
                seq_length = min(len(self.current_sequence['states']), self.max_sequence_length)
                
                states = self.current_sequence['states'][-seq_length:]
                states.append(state)
                states = np.array(states).reshape(1, seq_length + 1, -1)
                
                actions = self.current_sequence['actions'][-seq_length:]
                actions.append(np.zeros(self.action_dim))
                actions = np.array(actions).reshape(1, seq_length + 1, -1)
                
                # Return-to-go decreases over sequence
                returns = [target_return - sum(self.current_sequence['rewards'][-seq_length:])]
                returns = np.array(returns * (seq_length + 1)).reshape(1, seq_length + 1, 1)
                
                timesteps = np.arange(seq_length + 1).reshape(1, -1)
            
            # Convert to tensors
            states_t = torch.FloatTensor(states)
            actions_t = torch.FloatTensor(actions)
            returns_t = torch.FloatTensor(returns)
            timesteps_t = torch.LongTensor(timesteps)
            
            # Forward pass
            action_preds, attention_weights = self.model(returns_t, states_t, actions_t, timesteps_t)
            
            # Get last action prediction
            action = action_preds[0, -1, :].numpy()
            
            # Store attention weights for visualization
            self.latest_attention_weights = [w.numpy() for w in attention_weights]
            
            # Calculate confidence (using softmax entropy)
            action_probs = F.softmax(torch.FloatTensor(action), dim=0).numpy()
            entropy = -np.sum(action_probs * np.log(action_probs + 1e-8))
            confidence = 1.0 - entropy / np.log(self.action_dim)
            
        # Prepare metrics
        metrics = {
            'confidence': float(confidence),
            'action_probs': action_probs.tolist(),
            'attention_weights': [w.mean().item() for w in attention_weights],
            'sequence_length': seq_length,
            'target_return': target_return,
            'predicted_return': target_return  # Placeholder
        }
        
        return action, metrics
        
    def train_step(self, batch_size: int = 32) -> Dict[str, float]:
        """
        Perform a training step on a batch of sequences
        
        Args:
            batch_size: Number of sequences to sample
            
        Returns:
            Dictionary of training metrics
        """
        if len(self.sequence_buffer) < batch_size:
            return {'loss': 0.0, 'skipped': True}
            
        self.model.train()
        
        # Sample batch
        batch = [self.sequence_buffer[i] for i in np.random.choice(
            len(self.sequence_buffer), batch_size, replace=False
        )]
        
        # Prepare batch tensors
        max_len = max(len(seq['states']) for seq in batch)
        
        states_batch = []
        actions_batch = []
        returns_batch = []
        timesteps_batch = []
        action_targets = []
        
        for seq in batch:
            seq_len = len(seq['states'])
            
            # Pad sequences to max_len
            pad_len = max_len - seq_len
            
            states = np.concatenate([seq['states'], np.zeros((pad_len, self.state_dim))])
            actions = np.concatenate([seq['actions'], np.zeros((pad_len, self.action_dim))])
            returns = np.concatenate([seq['returns_to_go'], np.zeros(pad_len)])
            timesteps = np.concatenate([seq['timesteps'], np.zeros(pad_len, dtype=np.int64)])
            
            states_batch.append(states)
            actions_batch.append(actions)
            returns_batch.append(returns.reshape(-1, 1))
            timesteps_batch.append(timesteps)
            action_targets.append(actions)  # Actions are targets
        
        # Convert to tensors
        states_t = torch.FloatTensor(np.array(states_batch))
        actions_t = torch.FloatTensor(np.array(actions_batch))
        returns_t = torch.FloatTensor(np.array(returns_batch))
        timesteps_t = torch.LongTensor(np.array(timesteps_batch))
        targets_t = torch.FloatTensor(np.array(action_targets))
        
        # Forward pass
        action_preds, _ = self.model(returns_t, states_t, actions_t, timesteps_t)
        
        # Calculate loss (MSE between predicted and actual actions)
        loss = F.mse_loss(action_preds, targets_t)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # Update metrics
        loss_val = loss.item()
        self.training_metrics['losses'].append(loss_val)
        self.training_metrics['avg_loss'] = np.mean(self.training_metrics['losses'])
        self.training_metrics['total_steps'] += 1
        
        return {
            'loss': loss_val,
            'avg_loss': self.training_metrics['avg_loss'],
            'total_steps': self.training_metrics['total_steps'],
            'buffer_size': len(self.sequence_buffer)
        }
        
    def update_target_return(self, recent_returns: List[float]) -> None:
        """Update target return based on recent performance"""
        if len(recent_returns) > 0:
            # Set target as percentile of recent returns
            self.target_return = np.percentile(recent_returns, 75)
            
    def get_metrics(self) -> Dict[str, Any]:
        """Get current training and performance metrics"""
        return {
            'training_steps': self.training_metrics['total_steps'],
            'episodes': self.training_metrics['episodes'],
            'avg_loss': self.training_metrics['avg_loss'],
            'avg_return': self.training_metrics['avg_return'],
            'buffer_size': len(self.sequence_buffer),
            'target_return': self.target_return,
            'sequence_length': self.max_sequence_length,
            'attention_weights': self.latest_attention_weights
        }
