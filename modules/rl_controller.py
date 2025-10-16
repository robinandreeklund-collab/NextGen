"""
rl_controller.py - RL-kontroller för agentträning

Beskrivning:
    Tränar PPO-agenter och distribuerar belöning till moduler med RL-kapacitet.
    Central för reinforcement learning och agentoptimering.
    Sprint 4.2: Styr även meta-parametrar som evolution_threshold, min_samples, update_frequency
    och agent_entropy_threshold via MetaParameterAgent.

Roll:
    - Tar emot reward från portfolio_manager
    - Tar emot feedback från feedback_router
    - Tränar PPO-agenter för strategy_engine, risk_manager, decision_engine, execution_engine
    - Distribuerar agent_update till tränade moduler
    - Publicerar agent_status för introspektion
    - Justerar meta-parametrar via MetaParameterAgent (Sprint 4.2)
    - Publicerar parameter_adjustment för systemmoduler

Inputs:
    - reward: Dict - Belöning från portfolio_manager
    - feedback: Dict - Feedback från feedback_router

Outputs:
    - agent_update: Dict - Uppdaterade agentparametrar
    - agent_status: Dict - Agentperformance och metrics
    - parameter_adjustment: Dict - Justerade meta-parametrar (Sprint 4.2)

Publicerar till message_bus:
    - agent_status: Status för alla RL-agenter
    - parameter_adjustment: Justerade meta-parametrar

Prenumererar på (från functions_v2.yaml):
    - reward (från portfolio_manager)
    - feedback_event (från feedback_router)

Använder RL: Ja (från functions_v2.yaml)
Tar emot feedback: Nej (från functions_v2.yaml)

Anslutningar (från flowchart_v2.yaml - rl_training):
    Från:
    - portfolio_manager (reward)
    - feedback_router (feedback)
    Till:
    - strategy_engine (agent_update)
    - risk_manager (agent_update)
    - decision_engine (agent_update)
    - execution_engine (agent_update)
    - meta_agent_evolution_engine (agent_status, parameter_adjustment)
    - strategic_memory_engine (parameter_adjustment)
    - agent_manager (parameter_adjustment)

RL Response (från feedback_loop_v2.yaml):
    Updates agents:
    - strategy_engine: Optimerar tradeförslag
    - risk_manager: Förbättrar riskbedömning
    - decision_engine: Optimerar beslutsfattande
    - execution_engine: Förbättrar execution timing
    
    Updates meta-parameters (Sprint 4.2):
    - evolution_threshold: Dynamisk evolutionströskel
    - min_samples: Adaptivt antal samples för analys
    - update_frequency: Adaptiv uppdateringsfrekvens
    - agent_entropy_threshold: Dynamisk entropitröskel
    
    Reward sources:
    - portfolio_manager: Portfolio value change
    - strategic_memory_engine: Historisk performance
    - feedback_router: Feedback quality metrics
    - meta_agent_evolution_engine: Agent improvement signals (Sprint 4.2)

Algoritm: PPO (Proximal Policy Optimization)
Hyperparametrar: config/rl_parameters.yaml
Adaptive Parameters: docs/adaptive_parameters_v2.yaml

Används i Sprint: 2, 3, 4, 4.2
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


class MetaParameterAgent:
    """
    PPO-agent för adaptiv meta-parameterstyrning (Sprint 4.2).
    
    Styr meta-parametrar som evolution_threshold, min_samples, update_frequency
    och agent_entropy_threshold baserat på belöningssignaler.
    """
    
    def __init__(self, parameter_configs: Dict[str, Dict[str, Any]], learning_rate: float = 0.0001):
        """
        Initialiserar MetaParameterAgent.
        
        Args:
            parameter_configs: Konfiguration för varje parameter (bounds, default, etc.)
            learning_rate: Inlärningshastighet
        """
        self.parameter_configs = parameter_configs
        self.learning_rate = learning_rate
        
        # Nuvarande parametervärden (börja med defaults)
        self.current_params = {
            name: config.get('default', (config['bounds'][0] + config['bounds'][1]) / 2)
            for name, config in parameter_configs.items()
        }
        
        # History för varje parameter
        self.parameter_history: Dict[str, List[Dict[str, Any]]] = {
            name: [] for name in parameter_configs.keys()
        }
        
        # Reward tracking för meta-parameter optimization
        self.reward_history: List[float] = []
        self.training_steps = 0
        
        # Performance metrics per parameter
        self.parameter_performance: Dict[str, List[float]] = {
            name: [] for name in parameter_configs.keys()
        }
    
    def adjust_parameters(self, reward_signals: Dict[str, float]) -> Dict[str, float]:
        """
        Justerar meta-parametrar baserat på reward signals.
        
        Args:
            reward_signals: Dict med reward för olika aspekter
                (agent_performance_gain, feedback_density, reward_volatility, etc.)
        
        Returns:
            Dict med justerade parametervärden
        """
        import time
        
        self.training_steps += 1
        
        # Kombinera reward signals till total reward
        total_reward = sum(reward_signals.values()) / len(reward_signals) if reward_signals else 0.0
        self.reward_history.append(total_reward)
        
        adjustments = {}
        
        # Justera varje parameter baserat på relevanta reward signals
        for param_name, config in self.parameter_configs.items():
            # Hämta relevant reward signal för denna parameter
            signal_name = config.get('reward_signal', 'agent_performance_gain')
            reward = reward_signals.get(signal_name, 0.0)
            
            # Enkel gradient-baserad justering
            current_value = self.current_params[param_name]
            bounds = config['bounds']
            
            # Beräkna justering baserat på reward (enkel proportionell kontroll)
            adjustment_rate = self.learning_rate * (bounds[1] - bounds[0])
            delta = adjustment_rate * reward
            
            # Applicera justering med bounds-kontroll
            new_value = np.clip(current_value + delta, bounds[0], bounds[1])
            
            # Lägg till lite exploration noise
            if self.training_steps % 10 == 0:
                noise = np.random.randn() * adjustment_rate * 0.1
                new_value = np.clip(new_value + noise, bounds[0], bounds[1])
            
            adjustments[param_name] = new_value
            self.current_params[param_name] = new_value
            
            # Logga i history
            self.parameter_history[param_name].append({
                'value': new_value,
                'reward': reward,
                'timestamp': time.time(),
                'training_step': self.training_steps
            })
            
            # Spåra performance för denna parameter
            self.parameter_performance[param_name].append(reward)
        
        return adjustments
    
    def get_current_parameters(self) -> Dict[str, float]:
        """
        Hämtar nuvarande parametervärden.
        
        Returns:
            Dict med aktuella parametervärden
        """
        return self.current_params.copy()
    
    def get_parameter_history(self, param_name: str = None, limit: int = 100) -> Dict[str, List[Dict[str, Any]]]:
        """
        Hämtar parameterhistorik.
        
        Args:
            param_name: Specifik parameter (None för alla)
            limit: Max antal entries per parameter
        
        Returns:
            Dict med parameterhistorik
        """
        if param_name:
            return {param_name: self.parameter_history.get(param_name, [])[-limit:]}
        return {name: history[-limit:] for name, history in self.parameter_history.items()}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Hämtar performance metrics för meta-parameter agent.
        
        Returns:
            Dict med metrics
        """
        recent_rewards = self.reward_history[-100:] if self.reward_history else [0]
        
        # Beräkna trend för varje parameter
        param_trends = {}
        for param_name, perf_history in self.parameter_performance.items():
            if len(perf_history) >= 10:
                recent = np.mean(perf_history[-10:])
                older = np.mean(perf_history[-20:-10]) if len(perf_history) >= 20 else recent
                trend = 'improving' if recent > older else 'declining' if recent < older else 'stable'
            else:
                trend = 'insufficient_data'
            
            param_trends[param_name] = {
                'trend': trend,
                'current_value': self.current_params[param_name],
                'recent_performance': np.mean(perf_history[-10:]) if perf_history else 0.0
            }
        
        return {
            'training_steps': self.training_steps,
            'average_reward': np.mean(recent_rewards),
            'recent_performance': np.mean(self.reward_history[-10:]) if len(self.reward_history) >= 10 else 0.0,
            'parameter_trends': param_trends,
            'total_adjustments': sum(len(h) for h in self.parameter_history.values())
        }


class RLController:
    """Tränar och distribuerar PPO-agenter för moduler med RL-kapacitet samt styr meta-parametrar (Sprint 4.2)."""
    
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
        
        # Sprint 4.2: Meta-parameter agent
        self.meta_parameter_agent = self._initialize_meta_parameter_agent()
        self.parameter_update_counter = 0
        
        # Prenumerera på reward och feedback
        self.message_bus.subscribe('reward', self._on_reward)
        self.message_bus.subscribe('feedback_event', self._on_feedback)
        
        # Sprint 4.2: Prenumerera på agent_status för meta-parameter belöning
        self.message_bus.subscribe('agent_status', self._on_agent_status_for_meta)
        
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
            'parameter_update_frequency': 10,  # Sprint 4.2
            'agents': {
                'strategy_engine': {'state_dim': 10, 'action_dim': 3},
                'risk_manager': {'state_dim': 8, 'action_dim': 3},
                'decision_engine': {'state_dim': 12, 'action_dim': 3},
                'execution_engine': {'state_dim': 6, 'action_dim': 2}
            }
        }
    
    def _initialize_meta_parameter_agent(self) -> MetaParameterAgent:
        """
        Initialiserar meta-parameter agent (Sprint 4.2).
        
        Returns:
            MetaParameterAgent instance
        """
        # Läs konfiguration från adaptive_parameters_v2.yaml
        parameter_configs = {
            'evolution_threshold': {
                'bounds': [0.05, 0.5],
                'default': 0.25,
                'reward_signal': 'agent_performance_gain'
            },
            'min_samples': {
                'bounds': [5, 50],
                'default': 20,
                'reward_signal': 'feedback_consistency'
            },
            'update_frequency': {
                'bounds': [1, 100],
                'default': 10,
                'reward_signal': 'reward_volatility'
            },
            'agent_entropy_threshold': {
                'bounds': [0.1, 0.9],
                'default': 0.3,
                'reward_signal': 'decision_diversity'
            }
        }
        
        return MetaParameterAgent(
            parameter_configs=parameter_configs,
            learning_rate=0.0001
        )
    
    def _on_reward(self, reward_data: Dict[str, Any]) -> None:
        """
        Callback för reward från portfolio_manager.
        
        Args:
            reward_data: Reward value och metadata
        """
        reward_value = reward_data.get('value', 0.0)
        self.reward_history.append(reward_value)
        self.update_counter += 1
        self.parameter_update_counter += 1
        
        # Träna agenter med ny reward
        self.train_agents(reward_value, reward_data)
        
        # Publicera agent updates med viss frekvens
        if self.update_counter >= self.config.get('update_frequency', 10):
            self.publish_agent_updates()
            self.update_counter = 0
        
        # Sprint 4.2: Uppdatera meta-parametrar med viss frekvens
        param_update_freq = self.config.get('parameter_update_frequency', 10)
        if self.parameter_update_counter >= param_update_freq:
            self.update_meta_parameters()
            self.parameter_update_counter = 0
    
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
    
    def _on_agent_status_for_meta(self, status: Dict[str, Any]) -> None:
        """
        Callback för agent_status - används för meta-parameter belöningssignaler (Sprint 4.2).
        
        Args:
            status: Agent status med performance metrics
        """
        # Spara för meta-parameter reward calculation
        if not hasattr(self, '_recent_agent_statuses'):
            self._recent_agent_statuses = []
        
        self._recent_agent_statuses.append(status)
        
        # Behåll endast senaste 20 statuses
        if len(self._recent_agent_statuses) > 20:
            self._recent_agent_statuses = self._recent_agent_statuses[-20:]
    
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
        
        # Sprint 4.2: Inkludera meta-parameter metrics
        meta_metrics = self.meta_parameter_agent.get_performance_metrics()
        
        status = {
            'agents': agents_metrics,
            'training_steps': self.training_steps,
            'reward_history': self.reward_history[-100:],  # Senaste 100
            'average_reward': np.mean(self.reward_history[-10:]) if len(self.reward_history) >= 10 else 0.0,
            'total_agents': len(self.agents),
            'meta_parameters': self.meta_parameter_agent.get_current_parameters(),  # Sprint 4.2
            'meta_parameter_metrics': meta_metrics  # Sprint 4.2
        }
        self.message_bus.publish('agent_status', status)
    
    def update_meta_parameters(self) -> None:
        """
        Uppdaterar meta-parametrar via MetaParameterAgent (Sprint 4.2).
        
        Beräknar reward signals och justerar parametrar.
        """
        import time
        
        # Beräkna reward signals från olika källor
        reward_signals = self._calculate_meta_parameter_rewards()
        
        # Justera parametrar
        adjusted_params = self.meta_parameter_agent.adjust_parameters(reward_signals)
        
        # Publicera parameter_adjustment
        parameter_adjustment = {
            'timestamp': time.time(),
            'training_step': self.training_steps,
            'adjusted_parameters': adjusted_params,
            'reward_signals': reward_signals,
            'source': 'rl_controller'
        }
        
        self.message_bus.publish('parameter_adjustment', parameter_adjustment)
    
    def _calculate_meta_parameter_rewards(self) -> Dict[str, float]:
        """
        Beräknar reward signals för meta-parameter optimization (Sprint 4.2).
        
        Returns:
            Dict med reward signals
        """
        rewards = {
            'agent_performance_gain': 0.0,
            'feedback_consistency': 0.0,
            'reward_volatility': 0.0,
            'decision_diversity': 0.0,
            'overfitting_penalty': 0.0
        }
        
        # 1. Agent performance gain
        if hasattr(self, '_recent_agent_statuses') and len(self._recent_agent_statuses) >= 2:
            recent = self._recent_agent_statuses[-5:]
            older = self._recent_agent_statuses[-10:-5] if len(self._recent_agent_statuses) >= 10 else []
            
            if recent and older:
                recent_avg = np.mean([s.get('average_reward', 0) for s in recent])
                older_avg = np.mean([s.get('average_reward', 0) for s in older])
                
                if older_avg != 0:
                    rewards['agent_performance_gain'] = (recent_avg - older_avg) / abs(older_avg)
                else:
                    rewards['agent_performance_gain'] = 0.0
        
        # 2. Feedback consistency - baserat på reward volatility
        if len(self.reward_history) >= 10:
            recent_rewards = self.reward_history[-10:]
            volatility = np.std(recent_rewards) if len(recent_rewards) > 1 else 0.0
            # Lägre volatility = högre consistency reward
            rewards['feedback_consistency'] = 1.0 - min(volatility, 1.0)
        
        # 3. Reward volatility - används för update_frequency parameter
        if len(self.reward_history) >= 20:
            volatility = np.std(self.reward_history[-20:])
            rewards['reward_volatility'] = -volatility  # Negativ reward för hög volatility
        
        # 4. Decision diversity - estimera från agent entropy
        if hasattr(self, '_recent_agent_statuses') and self._recent_agent_statuses:
            # Approximera diversity från spread i agent performance
            performances = [s.get('average_reward', 0) for s in self._recent_agent_statuses[-10:]]
            if len(performances) > 1:
                diversity = np.std(performances)
                rewards['decision_diversity'] = diversity
        
        # 5. Overfitting penalty - om performance är för stabil kan det tyda på overfitting
        if len(self.reward_history) >= 50:
            recent = self.reward_history[-10:]
            historical = self.reward_history[-50:-10]
            
            recent_std = np.std(recent) if len(recent) > 1 else 0.0
            historical_std = np.std(historical) if len(historical) > 1 else 0.0
            
            # Om recent volatility är mycket lägre kan det tyda på overfitting
            if historical_std > 0 and recent_std < historical_std * 0.3:
                rewards['overfitting_penalty'] = -0.5
        
        return rewards
    
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
    
    def get_current_meta_parameters(self) -> Dict[str, float]:
        """
        Hämtar nuvarande meta-parametervärden (Sprint 4.2).
        
        Returns:
            Dict med aktuella meta-parametrar
        """
        return self.meta_parameter_agent.get_current_parameters()
    
    def get_parameter_history(self, param_name: str = None, limit: int = 100) -> Dict[str, List[Dict[str, Any]]]:
        """
        Hämtar parameterhistorik (Sprint 4.2).
        
        Args:
            param_name: Specifik parameter (None för alla)
            limit: Max antal entries
        
        Returns:
            Dict med parameterhistorik
        """
        return self.meta_parameter_agent.get_parameter_history(param_name, limit)

