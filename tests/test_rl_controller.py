# test_rl_controller.py - Tester för RL-kontroller

"""
Tester för RLController och PPO-agenter.
Verifierar att RL-träning fungerar korrekt.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.message_bus import MessageBus
from modules.rl_controller import RLController, PPOAgent
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


if __name__ == '__main__':
    print("Kör RL Controller-tester...")
    print()
    
    test_ppo_agent_initialization()
    test_ppo_agent_action_selection()
    test_ppo_agent_update()
    test_rl_controller_initialization()
    test_rl_controller_reward_handling()
    test_rl_controller_agent_performance()
    
    print()
    print("=" * 50)
    print("Alla RL Controller-tester godkända! ✅")
    print("=" * 50)

