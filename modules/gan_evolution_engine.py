"""
GAN Evolution Engine - Generative Adversarial Network for agent evolution

Uses GAN to generate new agent candidates and evaluate them against
historical performance data. Integrates with meta_agent_evolution_engine.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


class Generator(nn.Module):
    """Generator network for creating agent parameter candidates"""
    
    def __init__(self, latent_dim: int = 64, output_dim: int = 16):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Tanh()  # Output in [-1, 1]
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.network(z)


class Discriminator(nn.Module):
    """Discriminator network for evaluating agent candidates"""
    
    def __init__(self, input_dim: int = 16):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class GANEvolutionEngine:
    """
    GAN Evolution Engine for agent parameter generation
    
    Uses Generative Adversarial Networks to:
    - Generate novel agent parameter configurations
    - Evaluate candidates against historical performance
    - Provide high-quality candidates for meta-agent evolution
    """
    
    def __init__(self, message_bus, latent_dim: int = 64, param_dim: int = 16,
                 generator_lr: float = 0.0002, discriminator_lr: float = 0.0002,
                 evolution_threshold: float = 0.7):
        self.message_bus = message_bus
        self.latent_dim = latent_dim
        self.param_dim = param_dim
        self.evolution_threshold = evolution_threshold
        
        # Networks
        self.generator = Generator(latent_dim, param_dim)
        self.discriminator = Discriminator(param_dim)
        
        # Optimizers
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=generator_lr, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=discriminator_lr, betas=(0.5, 0.999))
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Training history
        self.training_history = {
            'g_losses': [],
            'd_losses': [],
            'candidates_generated': 0,
            'candidates_accepted': 0
        }
        
        # Real agent performance data
        self.real_agent_data = []
        
        # Subscribe to topics
        self.message_bus.subscribe('agent_performance', self._handle_agent_performance)
        self.message_bus.subscribe('gan_generate_request', self._handle_generate_request)
        
    def _handle_agent_performance(self, data: Dict[str, Any]):
        """Collect real agent performance data for discriminator training"""
        if 'parameters' in data and 'performance' in data:
            # Only keep high-performing agents
            if data['performance'] > 0.5:
                self.real_agent_data.append(data['parameters'])
                # Keep buffer size manageable
                if len(self.real_agent_data) > 1000:
                    self.real_agent_data = self.real_agent_data[-1000:]
                    
    def _handle_generate_request(self, data: Dict[str, Any]):
        """Handle request to generate new candidates"""
        num_candidates = data.get('num_candidates', 1)
        candidates = self.generate_candidates(num_candidates)
        
        self.message_bus.publish('gan_candidates', {
            'candidates': candidates,
            'num_generated': num_candidates,
            'acceptance_rate': self.get_acceptance_rate()
        })
        
    def generate_candidates(self, num_candidates: int = 1) -> List[np.ndarray]:
        """
        Generate new agent parameter candidates
        
        Args:
            num_candidates: Number of candidates to generate
            
        Returns:
            List of parameter arrays
        """
        self.generator.eval()
        with torch.no_grad():
            # Sample from latent space
            z = torch.randn(num_candidates, self.latent_dim)
            # Generate parameters
            candidates = self.generator(z).numpy()
            
        self.training_history['candidates_generated'] += num_candidates
        
        # Filter by discriminator score
        valid_candidates = []
        for candidate in candidates:
            score = self.evaluate_candidate(candidate)
            if score > self.evolution_threshold:
                valid_candidates.append(candidate)
                self.training_history['candidates_accepted'] += 1
                
        return valid_candidates if valid_candidates else [candidates[0]]
        
    def evaluate_candidate(self, candidate: np.ndarray) -> float:
        """
        Evaluate a candidate using the discriminator
        
        Args:
            candidate: Parameter array to evaluate
            
        Returns:
            Score between 0 and 1 (higher is better)
        """
        self.discriminator.eval()
        with torch.no_grad():
            candidate_tensor = torch.FloatTensor(candidate).unsqueeze(0)
            score = self.discriminator(candidate_tensor).item()
        return score
        
    def train_step(self, batch_size: int = 32) -> Tuple[float, float]:
        """
        Perform one training step for both generator and discriminator
        
        Args:
            batch_size: Batch size for training
            
        Returns:
            Tuple of (generator_loss, discriminator_loss)
        """
        if len(self.real_agent_data) < batch_size:
            return 0.0, 0.0
            
        # Train Discriminator
        self.discriminator.train()
        self.d_optimizer.zero_grad()
        
        # Real data
        real_data = np.array(self.real_agent_data[-batch_size:])
        real_data = torch.FloatTensor(real_data)
        real_labels = torch.ones(batch_size, 1)
        
        real_output = self.discriminator(real_data)
        d_loss_real = self.criterion(real_output, real_labels)
        
        # Fake data
        z = torch.randn(batch_size, self.latent_dim)
        fake_data = self.generator(z)
        fake_labels = torch.zeros(batch_size, 1)
        
        fake_output = self.discriminator(fake_data.detach())
        d_loss_fake = self.criterion(fake_output, fake_labels)
        
        # Total discriminator loss
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.d_optimizer.step()
        
        # Train Generator
        self.generator.train()
        self.g_optimizer.zero_grad()
        
        z = torch.randn(batch_size, self.latent_dim)
        fake_data = self.generator(z)
        fake_output = self.discriminator(fake_data)
        
        # Generator wants discriminator to think fake data is real
        g_loss = self.criterion(fake_output, real_labels)
        g_loss.backward()
        self.g_optimizer.step()
        
        # Update history
        self.training_history['g_losses'].append(g_loss.item())
        self.training_history['d_losses'].append(d_loss.item())
        
        # Limit loss history to prevent memory leak (keep last 1000)
        if len(self.training_history['g_losses']) > 1000:
            self.training_history['g_losses'] = self.training_history['g_losses'][-1000:]
        if len(self.training_history['d_losses']) > 1000:
            self.training_history['d_losses'] = self.training_history['d_losses'][-1000:]
        
        # Publish metrics
        self.message_bus.publish('gan_metrics', {
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item(),
            'real_data_size': len(self.real_agent_data),
            'acceptance_rate': self.get_acceptance_rate()
        })
        
        return g_loss.item(), d_loss.item()
        
    def get_acceptance_rate(self) -> float:
        """Get rate of candidates that pass evolution threshold"""
        if self.training_history['candidates_generated'] == 0:
            return 0.0
        return self.training_history['candidates_accepted'] / self.training_history['candidates_generated']
        
    def update_parameters(self, parameters: Dict[str, Any]):
        """Update GAN parameters"""
        if 'generator_lr' in parameters:
            for param_group in self.g_optimizer.param_groups:
                param_group['lr'] = parameters['generator_lr']
                
        if 'discriminator_lr' in parameters:
            for param_group in self.d_optimizer.param_groups:
                param_group['lr'] = parameters['discriminator_lr']
                
        if 'evolution_threshold' in parameters:
            self.evolution_threshold = parameters['evolution_threshold']
            
    def get_metrics(self) -> Dict[str, Any]:
        """Get current training metrics"""
        return {
            'g_loss': np.mean(self.training_history['g_losses'][-100:]) if self.training_history['g_losses'] else 0.0,
            'd_loss': np.mean(self.training_history['d_losses'][-100:]) if self.training_history['d_losses'] else 0.0,
            'candidates_generated': self.training_history['candidates_generated'],
            'candidates_accepted': self.training_history['candidates_accepted'],
            'acceptance_rate': self.get_acceptance_rate(),
            'real_data_size': len(self.real_agent_data)
        }
        
    def save_model(self, path: str):
        """Save GAN models"""
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'g_optimizer': self.g_optimizer.state_dict(),
            'd_optimizer': self.d_optimizer.state_dict(),
            'training_history': self.training_history
        }, path)
        
    def load_model(self, path: str):
        """Load GAN models"""
        checkpoint = torch.load(path)
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer'])
        self.training_history = checkpoint['training_history']
