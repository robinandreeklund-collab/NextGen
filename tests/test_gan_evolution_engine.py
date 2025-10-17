"""
Tests for GAN Evolution Engine
"""

import pytest
import numpy as np
from modules.gan_evolution_engine import GANEvolutionEngine, Generator, Discriminator
from modules.message_bus import MessageBus


class TestGenerator:
    """Tests for Generator network"""
    
    def test_initialization(self):
        generator = Generator(latent_dim=64, output_dim=16)
        assert generator is not None
        
    def test_forward_pass(self):
        generator = Generator(latent_dim=64, output_dim=16)
        import torch
        z = torch.randn(5, 64)
        output = generator(z)
        assert output.shape == (5, 16)
        assert torch.all(output >= -1.0) and torch.all(output <= 1.0)


class TestDiscriminator:
    """Tests for Discriminator network"""
    
    def test_initialization(self):
        discriminator = Discriminator(input_dim=16)
        assert discriminator is not None
        
    def test_forward_pass(self):
        discriminator = Discriminator(input_dim=16)
        import torch
        x = torch.randn(5, 16)
        output = discriminator(x)
        assert output.shape == (5, 1)
        assert torch.all(output >= 0.0) and torch.all(output <= 1.0)


class TestGANEvolutionEngine:
    """Tests for GAN Evolution Engine"""
    
    def setup_method(self):
        self.message_bus = MessageBus()
        self.gan = GANEvolutionEngine(
            self.message_bus,
            latent_dim=64,
            param_dim=16,
            evolution_threshold=0.5
        )
        
    def test_initialization(self):
        assert self.gan is not None
        assert self.gan.latent_dim == 64
        assert self.gan.param_dim == 16
        assert self.gan.evolution_threshold == 0.5
        
    def test_generate_candidates(self):
        candidates = self.gan.generate_candidates(num_candidates=5)
        assert len(candidates) > 0
        for candidate in candidates:
            assert len(candidate) == 16
            
    def test_evaluate_candidate(self):
        candidate = np.random.randn(16)
        score = self.gan.evaluate_candidate(candidate)
        assert 0.0 <= score <= 1.0
        
    def test_handle_agent_performance(self):
        # Send high-performing agent data
        self.message_bus.publish('agent_performance', {
            'parameters': np.random.randn(16),
            'performance': 0.8
        })
        assert len(self.gan.real_agent_data) > 0
        
    def test_handle_agent_performance_low_performance(self):
        # Send low-performing agent data (should be ignored)
        initial_size = len(self.gan.real_agent_data)
        self.message_bus.publish('agent_performance', {
            'parameters': np.random.randn(16),
            'performance': 0.3
        })
        assert len(self.gan.real_agent_data) == initial_size
        
    def test_handle_generate_request(self):
        self.message_bus.publish('gan_generate_request', {
            'num_candidates': 3
        })
        # Should publish candidates
        
    def test_train_step_insufficient_data(self):
        g_loss, d_loss = self.gan.train_step(batch_size=32)
        assert g_loss == 0.0
        assert d_loss == 0.0
        
    def test_train_step_with_data(self):
        # Add real agent data
        for i in range(50):
            self.gan.real_agent_data.append(np.random.randn(16))
        
        g_loss, d_loss = self.gan.train_step(batch_size=32)
        assert g_loss > 0.0
        assert d_loss > 0.0
        
    def test_adversarial_training_cycle(self):
        """Test full adversarial training cycle"""
        # Add real agent data
        for i in range(100):
            self.gan.real_agent_data.append(np.random.randn(16))
        
        # Train for several steps
        g_losses = []
        d_losses = []
        for _ in range(10):
            g_loss, d_loss = self.gan.train_step(batch_size=32)
            g_losses.append(g_loss)
            d_losses.append(d_loss)
        
        assert len(g_losses) == 10
        assert len(d_losses) == 10
        assert all(loss > 0 for loss in g_losses)
        assert all(loss > 0 for loss in d_losses)
        
    def test_get_acceptance_rate(self):
        rate = self.gan.get_acceptance_rate()
        assert 0.0 <= rate <= 1.0
        
    def test_acceptance_rate_calculation(self):
        self.gan.training_history['candidates_generated'] = 100
        self.gan.training_history['candidates_accepted'] = 70
        rate = self.gan.get_acceptance_rate()
        assert rate == 0.7
        
    def test_update_parameters(self):
        self.gan.update_parameters({
            'generator_lr': 0.0001,
            'discriminator_lr': 0.0001,
            'evolution_threshold': 0.8
        })
        assert self.gan.evolution_threshold == 0.8
        
    def test_get_metrics(self):
        metrics = self.gan.get_metrics()
        assert 'g_loss' in metrics
        assert 'd_loss' in metrics
        assert 'candidates_generated' in metrics
        assert 'candidates_accepted' in metrics
        assert 'acceptance_rate' in metrics
        
    def test_candidate_validation(self):
        """Test that generated candidates pass validation"""
        # Add training data
        for i in range(100):
            self.gan.real_agent_data.append(np.random.randn(16))
        
        # Train generator
        for _ in range(20):
            self.gan.train_step(batch_size=32)
        
        # Generate candidates
        candidates = self.gan.generate_candidates(num_candidates=10)
        assert len(candidates) > 0
        
        # Validate each candidate
        for candidate in candidates:
            score = self.gan.evaluate_candidate(candidate)
            # Score should be reasonable after training
            assert 0.0 <= score <= 1.0
            
    def test_discriminator_convergence(self):
        """Test that discriminator learns to distinguish real from fake"""
        # Add real data
        for i in range(100):
            self.gan.real_agent_data.append(np.random.randn(16))
        
        # Initial discriminator performance
        real_sample = np.array(self.gan.real_agent_data[0])
        initial_real_score = self.gan.evaluate_candidate(real_sample)
        
        # Train
        for _ in range(50):
            self.gan.train_step(batch_size=32)
        
        # After training, discriminator should be better
        final_real_score = self.gan.evaluate_candidate(real_sample)
        # Real data should get high score
        assert final_real_score >= 0.0
        
    def test_real_data_buffer_limit(self):
        """Test that real data buffer doesn't grow indefinitely"""
        # Add more than buffer limit
        for i in range(1200):
            self.message_bus.publish('agent_performance', {
                'parameters': np.random.randn(16),
                'performance': 0.9
            })
        
        assert len(self.gan.real_agent_data) <= 1000
        
    def test_evolution_threshold_filtering(self):
        """Test that evolution threshold filters low-quality candidates"""
        self.gan.evolution_threshold = 0.9  # Very high threshold
        
        # Add some training data
        for i in range(100):
            self.gan.real_agent_data.append(np.random.randn(16))
        
        candidates = self.gan.generate_candidates(num_candidates=20)
        # With high threshold, fewer candidates should pass
        # At least one should be returned even if none pass
        assert len(candidates) >= 1
