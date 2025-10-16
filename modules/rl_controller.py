"""
rl_controller.py - RL-kontroller för agentträning

Beskrivning:
    Tränar PPO-agenter och distribuerar belöning till moduler med RL-kapacitet.
    Central för reinforcement learning och agentoptimering.

Roll:
    - Tar emot reward från portfolio_manager
    - Tar emot feedback från feedback_router
    - Tränar PPO-agenter för strategy_engine, risk_manager, decision_engine, execution_engine
    - Distribuerar agent_update till tränade moduler
    - Publicerar agent_status för introspektion

Inputs:
    - reward: Dict - Belöning från portfolio_manager
    - feedback: Dict - Feedback från feedback_router

Outputs:
    - agent_update: Dict - Uppdaterade agentparametrar
    - agent_status: Dict - Agentperformance och metrics

Publicerar till message_bus:
    - agent_status: Status för alla RL-agenter

Prenumererar på (från functions.yaml):
    - reward (från portfolio_manager)
    - feedback (från feedback_router)

Använder RL: Ja (från functions.yaml)
Tar emot feedback: Nej (från functions.yaml)

Anslutningar (från flowchart.yaml - rl_training):
    Från:
    - portfolio_manager (reward)
    - feedback_router (feedback)
    Till:
    - strategy_engine (agent_update)
    - risk_manager (agent_update)
    - decision_engine (agent_update)
    - execution_engine (agent_update)
    - meta_agent_evolution_engine (agent_status)

RL Response (från feedback_loop.yaml):
    Updates agents:
    - strategy_engine: Optimerar tradeförslag
    - risk_manager: Förbättrar riskbedömning
    - decision_engine: Optimerar beslutsfattande
    - execution_engine: Förbättrar execution timing
    
    Reward sources:
    - portfolio_manager: Portfolio value change
    - strategic_memory_engine: Historisk performance
    - feedback_router: Feedback quality metrics

Algoritm: PPO (Proximal Policy Optimization)
Hyperparametrar: config/rl_parameters.yaml

Används i Sprint: 2, 3, 4
"""

from typing import Dict, Any, List
import numpy as np


class PPOAgent:
    """Enkel PPO-agent implementation för Sprint 2."""
    
    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 0.0003):
        """
        Initialiserar PPO-agent.
        
        Args:
            state_dim: Dimensionalitet av state space
            action_dim: Antal möjliga actions
            learning_rate: Inlärningshastighet
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        
        # Stub: I produktion skulle detta vara neural networks
        # För Sprint 2, använd enkel Q-learning approximation
        self.policy_params = np.random.randn(state_dim, action_dim) * 0.1
        self.value_params = np.random.randn(state_dim) * 0.1
        
        self.training_steps = 0
        self.cumulative_reward = 0.0
        self.episode_rewards: List[float] = []
    
    def select_action(self, state: np.ndarray) -> int:
        """
        Väljer action baserat på nuvarande policy.
        
        Args:
            state: Nuvarande state
            
        Returns:
            Vald action (index)
        """
        # Stub: Enkel policy för demo
        logits = state @ self.policy_params
        probs = self._softmax(logits)
        action = np.random.choice(self.action_dim, p=probs)
        return action
    
    def update(self, reward: float, state: np.ndarray) -> None:
        """
        Uppdaterar agent med ny reward och state.
        
        Args:
            reward: Mottagen reward
            state: Nuvarande state
        """
        self.cumulative_reward += reward
        self.training_steps += 1
        
        # Stub: Enkel gradient descent update
        # I produktion: PPO med clipped objective
        
        # Säkerställ att state har rätt dimension
        if len(state) > self.state_dim:
            state = state[:self.state_dim]
        elif len(state) < self.state_dim:
            padded_state = np.zeros(self.state_dim)
            padded_state[:len(state)] = state
            state = padded_state
        
        # Enkel policy update med gradient descent approximation
        gradient = reward * state.reshape(-1, 1) * 0.01  # Skala ner för stabilitet
        if gradient.shape == self.policy_params.shape:
            self.policy_params += self.learning_rate * gradient
        
        # Lägg till reward i episodhistorik
        self.episode_rewards.append(reward)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax för action probabilities."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Hämtar performance metrics för agenten.
        
        Returns:
            Dict med metrics
        """
        recent_rewards = self.episode_rewards[-100:] if self.episode_rewards else [0]
        return {
            'training_steps': self.training_steps,
            'cumulative_reward': self.cumulative_reward,
            'average_reward': np.mean(recent_rewards),
            'recent_performance': np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else 0.0
        }


class RLController:
    """Tränar och distribuerar PPO-agenter för moduler med RL-kapacitet."""
    
    def __init__(self, message_bus, config: Dict[str, Any] = None):
        """
        Initialiserar RL-kontrollern.
        
        Args:
            message_bus: Referens till central message_bus
            config: RL-konfiguration (från rl_parameters.yaml)
        """
        self.message_bus = message_bus
        self.config = config or self._default_config()
        
        self.agents: Dict[str, PPOAgent] = {}
        self.reward_history: List[float] = []
        self.training_steps = 0
        self.update_counter = 0
        
        # Prenumerera på reward och feedback
        self.message_bus.subscribe('reward', self._on_reward)
        self.message_bus.subscribe('feedback_event', self._on_feedback)
        
        # Initiera agenter för kärnmoduler
        self.initialize_agents([
            'strategy_engine',
            'risk_manager', 
            'decision_engine',
            'execution_engine'
        ])
    
    def _default_config(self) -> Dict[str, Any]:
        """Standardkonfiguration om ingen tillhandahålls."""
        return {
            'learning_rate': 0.0003,
            'gamma': 0.99,
            'update_frequency': 10,
            'agents': {
                'strategy_engine': {'state_dim': 10, 'action_dim': 3},
                'risk_manager': {'state_dim': 8, 'action_dim': 3},
                'decision_engine': {'state_dim': 12, 'action_dim': 3},
                'execution_engine': {'state_dim': 6, 'action_dim': 2}
            }
        }
    
    def _on_reward(self, reward_data: Dict[str, Any]) -> None:
        """
        Callback för reward från portfolio_manager.
        
        Args:
            reward_data: Reward value och metadata
        """
        reward_value = reward_data.get('value', 0.0)
        self.reward_history.append(reward_value)
        self.update_counter += 1
        
        # Träna agenter med ny reward
        self.train_agents(reward_value, reward_data)
        
        # Publicera agent updates med viss frekvens
        if self.update_counter >= self.config.get('update_frequency', 10):
            self.publish_agent_updates()
            self.update_counter = 0
    
    def _on_feedback(self, feedback: Dict[str, Any]) -> None:
        """
        Callback för feedback från feedback_router.
        
        Args:
            feedback: Feedback event att använda för träning
        """
        # Sprint 2: Grundläggande feedback-hantering
        # Sprint 3: Avancerad reward shaping baserat på feedback
        source = feedback.get('source', '')
        triggers = feedback.get('triggers', [])
        
        # Logga feedback för framtida användning
        if 'slippage' in triggers:
            # Kan användas för att justera execution_engine reward
            pass
    
    def initialize_agents(self, module_names: List[str]) -> None:
        """
        Initialiserar PPO-agenter för angivna moduler.
        
        Args:
            module_names: Lista med modulnamn att skapa agenter för
        """
        for module_name in module_names:
            if module_name in self.config.get('agents', {}):
                agent_config = self.config['agents'][module_name]
                self.agents[module_name] = PPOAgent(
                    state_dim=agent_config['state_dim'],
                    action_dim=agent_config['action_dim'],
                    learning_rate=self.config.get('learning_rate', 0.0003)
                )
    
    def train_agents(self, reward: float, reward_data: Dict[str, Any]) -> None:
        """
        Tränar alla agenter baserat på reward.
        
        Args:
            reward: Reward value från portfolio_manager
            reward_data: Metadata om reward
        """
        self.training_steps += 1
        
        # Skapa state representation för RL
        # I produktion: state skulle komma från modulernas observationer
        state = self._create_state_representation(reward_data)
        
        # Uppdatera alla agenter med reward
        for module_name, agent in self.agents.items():
            agent.update(reward, state)
        
        # Publicera agent status
        self.publish_agent_status()
    
    def _create_state_representation(self, reward_data: Dict[str, Any]) -> np.ndarray:
        """
        Skapar state representation från reward data.
        
        Args:
            reward_data: Metadata från reward
            
        Returns:
            State som numpy array
        """
        # Stub: Enkel state för demo
        # I produktion: kombinera indikatordata, portfolio status, etc.
        portfolio_value = reward_data.get('portfolio_value', 1000.0)
        normalized_value = portfolio_value / 1000.0  # Normalisera
        
        # Generera state med rätt dimension (max state_dim från alla agenter)
        state_dim = max(config['state_dim'] for config in self.config['agents'].values())
        state = np.zeros(state_dim)
        state[0] = normalized_value
        state[1] = reward_data.get('value', 0.0)
        
        return state
    
    def publish_agent_updates(self) -> None:
        """Publicerar agent_update till alla moduler med RL."""
        for module_name, agent in self.agents.items():
            metrics = agent.get_performance_metrics()
            self.message_bus.publish('agent_update', {
                'module': module_name,
                'training_step': self.training_steps,
                'metrics': metrics,
                'policy_updated': True
            })
    
    def publish_agent_status(self) -> None:
        """Publicerar agent_status för introspektion."""
        agents_metrics = {}
        for module_name, agent in self.agents.items():
            agents_metrics[module_name] = agent.get_performance_metrics()
        
        status = {
            'agents': agents_metrics,
            'training_steps': self.training_steps,
            'reward_history': self.reward_history[-100:],  # Senaste 100
            'average_reward': np.mean(self.reward_history[-10:]) if len(self.reward_history) >= 10 else 0.0,
            'total_agents': len(self.agents)
        }
        self.message_bus.publish('agent_status', status)
    
    def get_agent_performance(self, module_name: str) -> float:
        """
        Hämtar performance för en specifik agent.
        
        Args:
            module_name: Modulnamn att hämta performance för
            
        Returns:
            Performance metric för agenten
        """
        if module_name in self.agents:
            metrics = self.agents[module_name].get_performance_metrics()
            return metrics['average_reward']
        return 0.0

