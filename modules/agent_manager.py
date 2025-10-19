"""
agent_manager.py - Agenthanterare

Beskrivning:
    Hanterar agentprofiler, versioner och rolljusteringar.
    Sprint 4.2: Spårar även meta-parameterversioner från RL-controller.

Roll:
    - Tar emot agent_update från meta_agent_evolution_engine
    - Hanterar versioner av agenter
    - Lagrar agentprofiler och roller
    - Publicerar agent_profile
    - Loggar parameterhistorik (Sprint 4.2)

Inputs:
    - agent_update: Dict - Uppdaterad agentlogik
    - parameter_adjustment: Dict - Parameterjusteringar från RL-controller (Sprint 4.2)

Outputs:
    - agent_profile: Dict - Agentprofiler och konfiguration

Publicerar till message_bus:
    - agent_profile: För systemet

Prenumererar på (från functions_v2.yaml):
    - agent_update (från meta_agent_evolution_engine)
    - parameter_adjustment (från rl_controller) - Sprint 4.2

Använder RL: Nej
Tar emot feedback: Nej

Används i Sprint: 4, 4.2
"""

from typing import Dict, Any, List
import time


class AgentManager:
    """Hanterar agentprofiler och versioner."""
    
    def __init__(self, message_bus):
        """
        Initialiserar agent manager.
        
        Args:
            message_bus: Referens till central message_bus
        """
        self.message_bus = message_bus
        self.agent_profiles: Dict[str, Dict[str, Any]] = {}
        self.agent_versions: Dict[str, List[Dict[str, Any]]] = {}
        self.active_agents: Dict[str, str] = {}  # agent_id -> current_version
        
        # Sprint 4.2: Parameter history
        self.parameter_history: List[Dict[str, Any]] = []
        
        # Prenumerera på agent updates
        self.message_bus.subscribe('agent_update', self._on_agent_update)
        
        # Sprint 4.2: Prenumerera på parameter_adjustment
        self.message_bus.subscribe('parameter_adjustment', self._on_parameter_adjustment)
        
        # Initiera default agent profiles
        self._initialize_default_profiles()
    
    def _initialize_default_profiles(self) -> None:
        """Initialiserar default agentprofiler för systemet."""
        default_agents = {
            'strategy_agent': {
                'name': 'Strategy Agent',
                'module': 'strategy_engine',
                'role': 'Generate trade proposals based on indicators',
                'state_dim': 10,
                'action_dim': 3,
                'uses_rl': True,
                'version': '1.0.0',
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'risk_agent': {
                'name': 'Risk Management Agent',
                'module': 'risk_manager',
                'role': 'Assess risk levels and adjust positions',
                'state_dim': 8,
                'action_dim': 3,
                'uses_rl': True,
                'version': '1.0.0',
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'decision_agent': {
                'name': 'Decision Agent',
                'module': 'decision_engine',
                'role': 'Make final trading decisions',
                'state_dim': 12,
                'action_dim': 3,
                'uses_rl': True,
                'version': '1.0.0',
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'execution_agent': {
                'name': 'Execution Agent',
                'module': 'execution_engine',
                'role': 'Execute trades with optimal timing',
                'state_dim': 6,
                'action_dim': 2,
                'uses_rl': True,
                'version': '1.0.0',
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'dt_agent': {
                'name': 'Decision Transformer Agent',
                'module': 'decision_transformer_agent',
                'role': 'Sequence-based decision making with transformer architecture',
                'state_dim': 10,
                'action_dim': 3,
                'uses_rl': True,
                'architecture': 'transformer',
                'sequence_length': 20,
                'version': '1.0.0',
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        for agent_id, profile in default_agents.items():
            self.agent_profiles[agent_id] = profile
            self.active_agents[agent_id] = profile['version']
            
            # Initiera versionshistorik
            if agent_id not in self.agent_versions:
                self.agent_versions[agent_id] = []
            self.agent_versions[agent_id].append({
                'version': profile['version'],
                'timestamp': time.time(),
                'changes': 'Initial version',
                'status': 'active'
            })
            
            # Limit version history per agent to prevent memory leak (keep last 100)
            if len(self.agent_versions[agent_id]) > 100:
                self.agent_versions[agent_id] = self.agent_versions[agent_id][-100:]
    
    def _on_agent_update(self, update: Dict[str, Any]) -> None:
        """
        Callback för agent updates från meta_agent_evolution_engine.
        
        Args:
            update: Agent update event
        """
        update_type = update.get('update_type')
        
        if update_type == 'evolution_suggestion':
            self._handle_evolution_suggestion(update)
        elif update_type == 'system_wide_evolution':
            self._handle_system_wide_evolution(update)
        elif update_type == 'version_update':
            self._handle_version_update(update)
        else:
            # Log unknown update type
            pass
    
    def _handle_evolution_suggestion(self, update: Dict[str, Any]) -> None:
        """
        Hanterar evolutionsförslag för en specifik agent.
        
        Args:
            update: Evolution suggestion
        """
        agent_id = update.get('agent_id')
        evolution_suggestion = update.get('evolution_suggestion', {})
        
        if agent_id not in self.agent_profiles:
            return
        
        # Skapa ny version baserat på suggestion
        current_version = self.agent_profiles[agent_id].get('version', '1.0.0')
        new_version = self._increment_version(current_version)
        
        # Uppdatera agent profile
        self.agent_profiles[agent_id]['version'] = new_version
        self.agent_profiles[agent_id]['last_evolution'] = {
            'timestamp': time.time(),
            'reason': evolution_suggestion.get('analysis', {}).get('reason'),
            'suggestions': evolution_suggestion.get('suggestions', [])
        }
        
        # Lägg till version i historik
        version_entry = {
            'version': new_version,
            'timestamp': time.time(),
            'changes': f"Evolution triggered: {evolution_suggestion.get('analysis', {}).get('reason')}",
            'suggestions': evolution_suggestion.get('suggestions', []),
            'status': 'active'
        }
        
        if agent_id not in self.agent_versions:
            self.agent_versions[agent_id] = []
        
        # Markera tidigare version som inaktiv
        if self.agent_versions[agent_id]:
            self.agent_versions[agent_id][-1]['status'] = 'inactive'
        
        self.agent_versions[agent_id].append(version_entry)
        self.active_agents[agent_id] = new_version
        
        # Publicera uppdaterad profile
        self._publish_agent_profile(agent_id)
        
        # Skicka feedback till meta_agent_evolution_engine
        feedback = {
            'agent_id': agent_id,
            'action': 'evolution_applied',
            'new_version': new_version,
            'timestamp': time.time()
        }
        self.message_bus.publish('agent_update', {
            'update_type': 'evolution_feedback',
            'feedback': feedback
        })
    
    def _handle_system_wide_evolution(self, update: Dict[str, Any]) -> None:
        """
        Hanterar systemövergripande evolution.
        
        Args:
            update: System-wide evolution update
        """
        evolution_suggestion = update.get('evolution_suggestion', {})
        
        # Applicera evolution på alla RL-agenter
        for agent_id, profile in self.agent_profiles.items():
            if profile.get('uses_rl', False):
                # Skapa minor version update för alla agenter
                current_version = profile.get('version', '1.0.0')
                new_version = self._increment_version(current_version, minor=True)
                
                profile['version'] = new_version
                profile['last_system_evolution'] = {
                    'timestamp': time.time(),
                    'reason': evolution_suggestion.get('analysis', {}).get('reason')
                }
                
                # Lägg till i versionshistorik
                version_entry = {
                    'version': new_version,
                    'timestamp': time.time(),
                    'changes': 'System-wide evolution applied',
                    'status': 'active'
                }
                
                if agent_id in self.agent_versions and self.agent_versions[agent_id]:
                    self.agent_versions[agent_id][-1]['status'] = 'inactive'
                
                if agent_id not in self.agent_versions:
                    self.agent_versions[agent_id] = []
                self.agent_versions[agent_id].append(version_entry)
                
                # Limit version history per agent to prevent memory leak (keep last 100)
                if len(self.agent_versions[agent_id]) > 100:
                    self.agent_versions[agent_id] = self.agent_versions[agent_id][-100:]
                
                self.active_agents[agent_id] = new_version
                
                # Publicera uppdaterad profile
                self._publish_agent_profile(agent_id)
    
    def _handle_version_update(self, update: Dict[str, Any]) -> None:
        """
        Hanterar manuell versionsuppdatering.
        
        Args:
            update: Version update
        """
        agent_id = update.get('agent_id')
        new_config = update.get('config', {})
        
        if agent_id not in self.agent_profiles:
            return
        
        # Uppdatera konfiguration
        self.agent_profiles[agent_id].update(new_config)
        
        # Skapa ny version
        current_version = self.agent_profiles[agent_id].get('version', '1.0.0')
        new_version = self._increment_version(current_version)
        
        self.agent_profiles[agent_id]['version'] = new_version
        
        # Lägg till i historik
        version_entry = {
            'version': new_version,
            'timestamp': time.time(),
            'changes': 'Manual configuration update',
            'config_changes': list(new_config.keys()),
            'status': 'active'
        }
        
        if agent_id in self.agent_versions and self.agent_versions[agent_id]:
            self.agent_versions[agent_id][-1]['status'] = 'inactive'
        
        if agent_id not in self.agent_versions:
            self.agent_versions[agent_id] = []
        self.agent_versions[agent_id].append(version_entry)
        
        # Limit version history per agent to prevent memory leak (keep last 100)
        if len(self.agent_versions[agent_id]) > 100:
            self.agent_versions[agent_id] = self.agent_versions[agent_id][-100:]
        
        self.active_agents[agent_id] = new_version
        
        self._publish_agent_profile(agent_id)
    
    def _increment_version(self, version: str, minor: bool = False) -> str:
        """
        Inkrementerar versionsnummer.
        
        Args:
            version: Nuvarande version (ex: "1.0.0")
            minor: Om True, inkrementera minor version, annars patch
            
        Returns:
            Ny version
        """
        try:
            parts = version.split('.')
            major, minor_v, patch = int(parts[0]), int(parts[1]), int(parts[2])
            
            if minor:
                minor_v += 1
                patch = 0
            else:
                patch += 1
            
            return f"{major}.{minor_v}.{patch}"
        except (ValueError, IndexError):
            return "1.0.1"
    
    def _publish_agent_profile(self, agent_id: str) -> None:
        """
        Publicerar agent profile till message_bus.
        
        Args:
            agent_id: Agent att publicera
        """
        if agent_id not in self.agent_profiles:
            return
        
        profile = {
            'agent_id': agent_id,
            'profile': self.agent_profiles[agent_id],
            'version_history': self.agent_versions.get(agent_id, []),
            'active_version': self.active_agents.get(agent_id),
            'timestamp': time.time(),
            'parameter_history': self.parameter_history[-10:]  # Sprint 4.2: senaste 10
        }
        
        self.message_bus.publish('agent_profile', profile)
    
    def _on_parameter_adjustment(self, adjustment: Dict[str, Any]) -> None:
        """
        Callback för parameter adjustments från RL-controller (Sprint 4.2).
        
        Args:
            adjustment: Parameterjusteringar
        """
        param_entry = {
            **adjustment,
            'timestamp': time.time(),
            'logged_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        self.parameter_history.append(param_entry)
        
        # Limit history to prevent memory leak (keep last 1000)
        if len(self.parameter_history) > 1000:
            self.parameter_history = self.parameter_history[-1000:]
    
    def get_agent_profile(self, agent_id: str) -> Dict[str, Any]:
        """
        Hämtar agent profile.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Agent profile eller tom dict
        """
        return self.agent_profiles.get(agent_id, {})
    
    def get_all_profiles(self) -> Dict[str, Dict[str, Any]]:
        """
        Hämtar alla agent profiles.
        
        Returns:
            Dict med alla profiles
        """
        return self.agent_profiles.copy()
    
    def get_version_history(self, agent_id: str) -> List[Dict[str, Any]]:
        """
        Hämtar versionshistorik för en agent.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Lista med versioner
        """
        return self.agent_versions.get(agent_id, [])
    
    def rollback_version(self, agent_id: str, version: str) -> bool:
        """
        Rullar tillbaka till en tidigare version.
        
        Args:
            agent_id: Agent ID
            version: Version att återställa till
            
        Returns:
            True om lyckad, False annars
        """
        if agent_id not in self.agent_versions:
            return False
        
        # Leta efter versionen i historik
        target_version = None
        for v in self.agent_versions[agent_id]:
            if v['version'] == version:
                target_version = v
                break
        
        if not target_version:
            return False
        
        # Markera nuvarande som inaktiv
        if self.agent_versions[agent_id]:
            self.agent_versions[agent_id][-1]['status'] = 'inactive'
        
        # Skapa ny entry för rollback
        rollback_entry = {
            'version': version,
            'timestamp': time.time(),
            'changes': f'Rolled back to version {version}',
            'status': 'active',
            'is_rollback': True
        }
        
        self.agent_versions[agent_id].append(rollback_entry)
        
        # Limit version history per agent to prevent memory leak (keep last 100)
        if len(self.agent_versions[agent_id]) > 100:
            self.agent_versions[agent_id] = self.agent_versions[agent_id][-100:]
        
        self.active_agents[agent_id] = version
        self.agent_profiles[agent_id]['version'] = version
        
        self._publish_agent_profile(agent_id)
        
        return True
    
    def get_evolution_tree(self) -> Dict[str, Any]:
        """
        Genererar ett evolutionsträd för alla agenter.
        
        Returns:
            Evolution tree struktur
        """
        tree = {
            'total_agents': len(self.agent_profiles),
            'agents': {},
            'parameter_adjustments': len(self.parameter_history)  # Sprint 4.2
        }
        
        for agent_id, profile in self.agent_profiles.items():
            tree['agents'][agent_id] = {
                'name': profile.get('name'),
                'current_version': profile.get('version'),
                'total_versions': len(self.agent_versions.get(agent_id, [])),
                'version_history': self.agent_versions.get(agent_id, []),
                'uses_rl': profile.get('uses_rl', False)
            }
        
        return tree
    
    def get_parameter_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Hämtar parameterhistorik (Sprint 4.2).
        
        Args:
            limit: Max antal entries
        
        Returns:
            Lista med parameterjusteringar
        """
        return self.parameter_history[-limit:]

