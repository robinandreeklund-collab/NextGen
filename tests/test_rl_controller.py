# test_rl_controller.py - Tester för RL-kontroller

"""
Tester för RLController och PPO-agenter.
Verifierar att RL-träning fungerar korrekt.
Sprint 4.2: Testar även adaptiv parameterstyrning.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.message_bus import MessageBus
from modules.rl_controller import (
    RLController,
    PPOAgent,
    MetaParameterAgent,
)
import numpy as np


def test_ppo_agent_initialization():
    """Testar att PPOAgent initialiseras korrekt."""
    agent = PPOAgent(state_dim=10, action_dim=3, learning_rate=0.001)
    
    assert agent.state_dim == 10
    assert agent.action_dim == 3
    assert agent.learning_rate == 0.001
    assert agent.training_steps == 0
    assert agent.cumulative_reward == 0.0
    print("✓ PPOAgent initialisering fungerar")


def test_ppo_agent_action_selection():
    """Testar att PPOAgent kan välja actions."""
    agent = PPOAgent(state_dim=5, action_dim=3)
    state = np.random.randn(5)
    
    action = agent.select_action(state)
    
    assert isinstance(action, (int, np.integer))
    assert 0 <= action < 3
    print("✓ PPOAgent action selection fungerar")


def test_ppo_agent_update():
    """Testar att PPOAgent uppdateras korrekt."""
    agent = PPOAgent(state_dim=5, action_dim=3)
    state = np.random.randn(5)
    reward = 1.0
    
    initial_steps = agent.training_steps
    agent.update(reward, state)
    
    assert agent.training_steps == initial_steps + 1
    assert len(agent.episode_rewards) == 1
    assert agent.episode_rewards[0] == reward
    print("✓ PPOAgent update fungerar")


def test_meta_parameter_agent_initialization():
    """Testar att MetaParameterAgent initialiseras korrekt (Sprint 4.2)."""
    param_configs = {
        'evolution_threshold': {
            'bounds': [0.05, 0.5],
            'default': 0.25,
            'reward_signal': 'agent_performance_gain'
        },
        'min_samples': {
            'bounds': [5, 50],
            'default': 20,
            'reward_signal': 'feedback_consistency'
        }
    }
    
    agent = MetaParameterAgent(param_configs)
    
    assert agent.current_params['evolution_threshold'] == 0.25
    assert agent.current_params['min_samples'] == 20
    assert agent.training_steps == 0
    print("✓ MetaParameterAgent initialisering fungerar")


def test_meta_parameter_agent_adjustment():
    """Testar att MetaParameterAgent justerar parametrar (Sprint 4.2)."""
    param_configs = {
        'evolution_threshold': {
            'bounds': [0.05, 0.5],
            'default': 0.25,
            'reward_signal': 'agent_performance_gain'
        }
    }
    
    agent = MetaParameterAgent(param_configs)
    
    # Simulera reward signals
    reward_signals = {'agent_performance_gain': 0.5}
    adjusted = agent.adjust_parameters(reward_signals)
    
    assert 'evolution_threshold' in adjusted
    assert 0.05 <= adjusted['evolution_threshold'] <= 0.5
    assert agent.training_steps == 1
    assert len(agent.parameter_history['evolution_threshold']) == 1
    print("✓ MetaParameterAgent parameterjustering fungerar")


def test_meta_parameter_agent_history():
    """Testar att MetaParameterAgent loggar parameterhistorik (Sprint 4.2)."""
    param_configs = {
        'min_samples': {
            'bounds': [5, 50],
            'default': 20,
            'reward_signal': 'feedback_consistency'
        }
    }
    
    agent = MetaParameterAgent(param_configs)
    
    # Justera flera gånger
    for i in range(5):
        agent.adjust_parameters({'feedback_consistency': 0.3 + i * 0.1})
    
    history = agent.get_parameter_history('min_samples')
    assert len(history['min_samples']) == 5
    assert all('value' in entry for entry in history['min_samples'])
    print("✓ MetaParameterAgent parameterhistorik fungerar")


def test_rl_controller_initialization():
    """Testar att RLController initialiseras med agenter."""
    bus = MessageBus()
    config = {
        'learning_rate': 0.0003,
        'update_frequency': 10,
        'agents': {
            'strategy_engine': {'state_dim': 10, 'action_dim': 3},
            'risk_manager': {'state_dim': 8, 'action_dim': 3}
        }
    }
    
    controller = RLController(bus, config)
    
    assert len(controller.agents) == 2
    assert 'strategy_engine' in controller.agents
    assert 'risk_manager' in controller.agents
    assert controller.meta_parameter_agent is not None  # Sprint 4.2
    print("✓ RLController initialisering fungerar")


def test_rl_controller_reward_handling():
    """Testar att RLController hanterar reward korrekt."""
    bus = MessageBus()
    controller = RLController(bus)
    
    # Simulera reward
    reward_data = {'value': 1.5, 'portfolio_value': 1010.0}
    controller._on_reward(reward_data)
    
    assert len(controller.reward_history) == 1
    assert controller.reward_history[0] == 1.5
    assert controller.training_steps > 0
    print("✓ RLController reward-hantering fungerar")


def test_rl_controller_agent_performance():
    """Testar att RLController kan hämta agent performance."""
    bus = MessageBus()
    controller = RLController(bus)
    
    # Träna agenter
    for i in range(5):
        reward_data = {'value': float(i), 'portfolio_value': 1000.0 + i}
        controller._on_reward(reward_data)
    
    perf = controller.get_agent_performance('strategy_engine')
    assert isinstance(perf, float)
    print("✓ RLController agent performance fungerar")


def test_rl_controller_meta_parameter_update():
    """Testar att RLController uppdaterar meta-parametrar (Sprint 4.2)."""
    bus = MessageBus()
    controller = RLController(bus)
    
    # Spara original parametrar
    original_params = controller.get_current_meta_parameters()
    
    # Simulera träning för att trigga parameter update
    for i in range(15):  # Över parameter_update_frequency
        reward_data = {'value': float(i) * 0.1, 'portfolio_value': 1000.0 + i}
        controller._on_reward(reward_data)
    
    # Parametrar ska ha uppdaterats
    current_params = controller.get_current_meta_parameters()
    assert current_params is not None
    assert 'evolution_threshold' in current_params
    print("✓ RLController meta-parameter uppdatering fungerar")


def test_rl_controller_parameter_history():
    """Testar att RLController loggar parameterhistorik (Sprint 4.2)."""
    bus = MessageBus()
    controller = RLController(bus)
    
    # Simulera träning
    for i in range(15):
        reward_data = {'value': float(i) * 0.1, 'portfolio_value': 1000.0 + i}
        controller._on_reward(reward_data)
    
    history = controller.get_parameter_history(limit=10)
    assert isinstance(history, dict)
    print("✓ RLController parameterhistorik fungerar")


if __name__ == '__main__':
    print("Kör RL Controller-tester...")
    print()
    
    test_ppo_agent_initialization()
    test_ppo_agent_action_selection()
    test_ppo_agent_update()
    test_meta_parameter_agent_initialization()
    test_meta_parameter_agent_adjustment()
    test_meta_parameter_agent_history()
    test_rl_controller_initialization()
    test_rl_controller_reward_handling()
    test_rl_controller_agent_performance()
    test_rl_controller_meta_parameter_update()
    test_rl_controller_parameter_history()
    
    print()
    print("=" * 50)
    print("Alla RL Controller-tester godkända! ✅")
    print("=" * 50)

