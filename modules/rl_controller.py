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


class RLController:
    """Tränar och distribuerar PPO-agenter för moduler med RL-kapacitet."""
    
    def __init__(self, message_bus):
        """
        Initialiserar RL-kontrollern.
        
        Args:
            message_bus: Referens till central message_bus
        """
        self.message_bus = message_bus
        self.agents: Dict[str, Any] = {}  # module_name -> PPO agent
        self.reward_history: List[float] = []
        self.training_steps = 0
        
        # Prenumerera på reward och feedback
        self.message_bus.subscribe('reward', self._on_reward)
        self.message_bus.subscribe('feedback_event', self._on_feedback)
    
    def _on_reward(self, reward_data: Dict[str, Any]) -> None:
        """
        Callback för reward från portfolio_manager.
        
        Args:
            reward_data: Reward value och metadata
        """
        reward_value = reward_data.get('value', 0.0)
        self.reward_history.append(reward_value)
        
        # Stub: I Sprint 2 kommer PPO-träning implementeras här
        self.train_agents(reward_value)
    
    def _on_feedback(self, feedback: Dict[str, Any]) -> None:
        """
        Callback för feedback från feedback_router.
        
        Args:
            feedback: Feedback event att använda för träning
        """
        # Stub: Används i Sprint 3 för mer avancerad reward shaping
        pass
    
    def initialize_agents(self, module_names: List[str]) -> None:
        """
        Initialiserar PPO-agenter för angivna moduler.
        
        Args:
            module_names: Lista med modulnamn att skapa agenter för
        """
        # Stub: I Sprint 2 kommer PPO-agenter initialiseras här
        for module_name in module_names:
            self.agents[module_name] = {
                'type': 'PPO',
                'state': 'initialized',
                'performance': 0.0
            }
    
    def train_agents(self, reward: float) -> None:
        """
        Tränar alla agenter baserat på reward.
        
        Args:
            reward: Reward value från portfolio_manager
        """
        # Stub: I Sprint 2 kommer faktisk PPO-träning implementeras
        self.training_steps += 1
        
        # Uppdatera agent status
        for module_name in self.agents:
            self.agents[module_name]['performance'] = sum(self.reward_history[-10:]) / min(10, len(self.reward_history))
        
        # Publicera agent updates till moduler
        self.publish_agent_updates()
        
        # Publicera agent status
        self.publish_agent_status()
    
    def publish_agent_updates(self) -> None:
        """Publicerar agent_update till alla moduler med RL."""
        for module_name, agent_data in self.agents.items():
            self.message_bus.publish('agent_update', {
                'module': module_name,
                'agent_data': agent_data,
                'training_step': self.training_steps
            })
    
    def publish_agent_status(self) -> None:
        """Publicerar agent_status för introspektion."""
        status = {
            'agents': self.agents,
            'training_steps': self.training_steps,
            'reward_history': self.reward_history[-100:],  # Senaste 100
            'average_reward': sum(self.reward_history[-10:]) / min(10, len(self.reward_history)) if self.reward_history else 0.0
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
            return self.agents[module_name].get('performance', 0.0)
        return 0.0

