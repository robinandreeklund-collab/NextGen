"""
meta_agent_evolution_engine.py - Meta-agent evolution

Beskrivning:
    Utvärderar agentperformance och föreslår logikjusteringar.
    Evolutionär utveckling av agentbeteende baserat på feedback och performance.
    Sprint 4.2: Tar emot adaptiva meta-parametrar från RL-controller.

Roll:
    - Analyserar agent_status från rl_controller
    - Analyserar feedback_insight från feedback_analyzer
    - Identifierar förbättringsmöjligheter i agentlogik
    - Föreslår evolution_suggestion för agenter
    - Skickar agent_update till agent_manager
    - Använder adaptiva parametrar från RL-controller (Sprint 4.2)

Inputs:
    - agent_status: Dict - Performance metrics från rl_controller
    - feedback_insight: Dict - Analyserade mönster från feedback_analyzer
    - parameter_adjustment: Dict - Adaptiva parametrar från RL-controller (Sprint 4.2)

Outputs:
    - evolution_suggestion: Dict - Förslag på agentförbättringar
    - agent_update: Dict - Uppdaterad agentlogik

Publicerar till message_bus:
    - agent_update: För agent_manager

Prenumererar på (från functions_v2.yaml):
    - agent_status (från rl_controller)
    - feedback_insight (från feedback_analyzer)
    - parameter_adjustment (från rl_controller) - Sprint 4.2

Använder RL: Ja (från functions_v2.yaml)
Tar emot feedback: Ja (från agent_manager)

Anslutningar (från flowchart_v2.yaml - agent_evolution):
    Från:
    - rl_controller (agent_status, parameter_adjustment)
    - feedback_analyzer (feedback_insight)
    Till: agent_manager (agent_update)

Används i Sprint: 4, 4.2
"""

from typing import Dict, Any, List
import time


class MetaAgentEvolutionEngine:
    """Utvecklar och förbättrar agenter baserat på performance."""
    
    def __init__(self, message_bus):
        """
        Initialiserar meta-agent evolutionsmotorn.
        
        Args:
            message_bus: Referens till central message_bus
        """
        self.message_bus = message_bus
        self.rl_agent = None
        
        # Performance tracking för agenter
        self.agent_performance_history: Dict[str, List[Dict[str, Any]]] = {}
        self.feedback_insights: List[Dict[str, Any]] = []
        self.evolution_history: List[Dict[str, Any]] = []
        
        # Sprint 4.2: Adaptiva parametrar från RL-controller
        self.current_parameters = {
            'evolution_threshold': 0.25,  # Default, uppdateras från RL
            'min_samples': 20  # Default, uppdateras från RL
        }
        
        # Sprint 4.2: Parameter history
        self.parameter_history: List[Dict[str, Any]] = []
        
        # Evolution thresholds (kommer justeras adaptivt i Sprint 4.2)
        self.performance_threshold = self.current_parameters['evolution_threshold']
        self.min_samples_for_evolution = int(self.current_parameters['min_samples'])
        
        # Prenumerera på relevanta events
        self.message_bus.subscribe('agent_status', self._on_agent_status)
        self.message_bus.subscribe('feedback_insight', self._on_feedback_insight)
        
        # Sprint 4.2: Prenumerera på parameter_adjustment
        self.message_bus.subscribe('parameter_adjustment', self._on_parameter_adjustment)
    
    def _on_agent_status(self, status: Dict[str, Any]) -> None:
        """
        Callback för agent status från rl_controller.
        
        Args:
            status: Agent performance status
        """
        agent_id = status.get('agent_id', 'unknown')
        
        if agent_id not in self.agent_performance_history:
            self.agent_performance_history[agent_id] = []
        
        # Lägg till status med timestamp
        status_entry = {
            **status,
            'timestamp': time.time(),
            'logged_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        self.agent_performance_history[agent_id].append(status_entry)
        
        # Limit history per agent to prevent memory leak (keep last 1000)
        if len(self.agent_performance_history[agent_id]) > 1000:
            self.agent_performance_history[agent_id] = self.agent_performance_history[agent_id][-1000:]
        
        # Analysera om evolution behövs
        if len(self.agent_performance_history[agent_id]) >= self.min_samples_for_evolution:
            self._analyze_agent_evolution_need(agent_id)
    
    def _on_feedback_insight(self, insight: Dict[str, Any]) -> None:
        """
        Callback för feedback insights från feedback_analyzer.
        
        Args:
            insight: Analyserade mönster och insights
        """
        insight_entry = {
            **insight,
            'timestamp': time.time(),
            'logged_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        self.feedback_insights.append(insight_entry)
        
        # Analysera om insights triggar evolution
        self._analyze_insight_for_evolution(insight_entry)
    
    def _on_parameter_adjustment(self, adjustment: Dict[str, Any]) -> None:
        """
        Callback för parameter adjustments från RL-controller (Sprint 4.2).
        
        Args:
            adjustment: Justerade parametervärden
        """
        adjusted_params = adjustment.get('adjusted_parameters', {})
        
        # Uppdatera lokala parametrar
        if 'evolution_threshold' in adjusted_params:
            self.current_parameters['evolution_threshold'] = adjusted_params['evolution_threshold']
            self.performance_threshold = adjusted_params['evolution_threshold']
        
        if 'min_samples' in adjusted_params:
            self.current_parameters['min_samples'] = adjusted_params['min_samples']
            self.min_samples_for_evolution = int(adjusted_params['min_samples'])
        
        # Logga parameter change
        param_entry = {
            **adjustment,
            'timestamp': time.time(),
            'logged_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'applied_parameters': {
                'evolution_threshold': self.performance_threshold,
                'min_samples': self.min_samples_for_evolution
            }
        }
        self.parameter_history.append(param_entry)
        
        # Limit history to prevent memory leak (keep last 1000)
        if len(self.parameter_history) > 1000:
            self.parameter_history = self.parameter_history[-1000:]
    
    def _analyze_agent_evolution_need(self, agent_id: str) -> None:
        """
        Analyserar om en agent behöver evolutionär uppdatering.
        
        Args:
            agent_id: ID för agenten att analysera
        """
        history = self.agent_performance_history[agent_id]
        
        if len(history) < self.min_samples_for_evolution:
            return
        
        # Jämför första och andra halvan av historik
        mid_point = len(history) // 2
        first_half = history[:mid_point]
        second_half = history[mid_point:]
        
        # Beräkna genomsnittlig performance
        first_performance = self._calculate_average_performance(first_half)
        second_performance = self._calculate_average_performance(second_half)
        
        # Detektera performance degradation
        if first_performance > 0:
            degradation = (first_performance - second_performance) / first_performance
            
            if degradation > self.performance_threshold:
                # Trigga evolution
                self._suggest_agent_evolution(agent_id, {
                    'reason': 'performance_degradation',
                    'degradation_percentage': degradation * 100,
                    'first_half_performance': first_performance,
                    'second_half_performance': second_performance,
                    'severity': 'high' if degradation > 0.25 else 'medium'
                })
    
    def _analyze_insight_for_evolution(self, insight: Dict[str, Any]) -> None:
        """
        Analyserar om feedback insight triggar evolution.
        
        Args:
            insight: Feedback insight att analysera
        """
        patterns = insight.get('patterns', [])
        recommendations = insight.get('recommendations', [])
        
        # Kolla om patterns indikerar behov av evolution
        for pattern in patterns:
            if pattern.get('type') == 'agent_drift':
                affected_agent = pattern.get('agent_id')
                if affected_agent:
                    self._suggest_agent_evolution(affected_agent, {
                        'reason': 'agent_drift_detected',
                        'pattern': pattern,
                        'severity': pattern.get('severity', 'medium')
                    })
            
            elif pattern.get('type') == 'low_success_rate':
                # Föreslå evolution för alla relevanta agenter
                self._suggest_system_wide_evolution({
                    'reason': 'low_success_rate',
                    'pattern': pattern,
                    'recommendations': recommendations
                })
    
    def _calculate_average_performance(self, status_list: List[Dict[str, Any]]) -> float:
        """
        Beräknar genomsnittlig performance från en lista av agent status.
        
        Args:
            status_list: Lista med agent status
            
        Returns:
            Genomsnittlig performance
        """
        if not status_list:
            return 0.0
        
        total_performance = 0.0
        count = 0
        
        for status in status_list:
            # Försök hämta olika performance-metriker
            performance = (status.get('reward', 0.0) + 
                          status.get('success_rate', 0.0) +
                          status.get('performance', 0.0))
            if performance != 0.0:
                total_performance += performance
                count += 1
        
        return total_performance / count if count > 0 else 0.0
    
    def _suggest_agent_evolution(self, agent_id: str, analysis: Dict[str, Any]) -> None:
        """
        Föreslår evolution för en specifik agent.
        
        Args:
            agent_id: ID för agenten
            analysis: Analysresultat
        """
        evolution_suggestion = {
            'agent_id': agent_id,
            'timestamp': time.time(),
            'logged_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'analysis': analysis,
            'suggestions': []
        }
        
        # Generera specifika förslag baserat på analys
        if analysis['reason'] == 'performance_degradation':
            evolution_suggestion['suggestions'].extend([
                'Justera learning rate',
                'Öka exploration rate',
                'Återställ till tidigare version om tillgänglig',
                'Analysera state representation'
            ])
        
        elif analysis['reason'] == 'agent_drift_detected':
            evolution_suggestion['suggestions'].extend([
                'Minska learning rate',
                'Öka batch size för stabilitet',
                'Implementera early stopping',
                'Granska reward shaping'
            ])
        
        # Logga evolution suggestion
        self.evolution_history.append(evolution_suggestion)
        
        # Limit history to prevent memory leak (keep last 1000)
        if len(self.evolution_history) > 1000:
            self.evolution_history = self.evolution_history[-1000:]
        
        # Skapa agent_update för agent_manager
        agent_update = {
            'agent_id': agent_id,
            'update_type': 'evolution_suggestion',
            'timestamp': time.time(),
            'evolution_suggestion': evolution_suggestion,
            'action_required': True
        }
        
        # Publicera till agent_manager
        self.message_bus.publish('agent_update', agent_update)
    
    def _suggest_system_wide_evolution(self, analysis: Dict[str, Any]) -> None:
        """
        Föreslår systemövergripande evolution.
        
        Args:
            analysis: Systemanalys
        """
        evolution_suggestion = {
            'scope': 'system_wide',
            'timestamp': time.time(),
            'logged_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'analysis': analysis,
            'suggestions': [
                'Granska reward function',
                'Justera indicator weights',
                'Utvärdera trading strategy',
                'Analysera market conditions'
            ]
        }
        
        self.evolution_history.append(evolution_suggestion)
        
        # Publicera system-wide update
        agent_update = {
            'update_type': 'system_wide_evolution',
            'timestamp': time.time(),
            'evolution_suggestion': evolution_suggestion,
            'action_required': True
        }
        
        self.message_bus.publish('agent_update', agent_update)
    
    def receive_feedback(self, feedback: Dict[str, Any]) -> None:
        """
        Tar emot feedback från agent_manager om evolution results.
        
        Args:
            feedback: Feedback om genomförda evolutioner
        """
        feedback_entry = {
            **feedback,
            'timestamp': time.time(),
            'logged_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Logga feedback för framtida analys
        if 'evolution_feedback' not in self.__dict__:
            self.evolution_feedback = []
        self.evolution_feedback.append(feedback_entry)
    
    def update_from_rl(self, agent_update: Dict[str, Any]) -> None:
        """
        Uppdaterar evolution logic baserat på RL-feedback.
        
        Args:
            agent_update: RL-baserad agentuppdatering
        """
        # Uppdatera RL-agent om implementerad
        if self.rl_agent:
            # Framtida implementation: Använd RL för att optimera evolution strategy
            pass
    
    def get_evolution_history(self, agent_id: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Hämtar evolutionshistorik.
        
        Args:
            agent_id: Specifik agent (None för alla)
            limit: Max antal entries
            
        Returns:
            Lista med evolutionshändelser
        """
        if agent_id:
            filtered = [e for e in self.evolution_history 
                       if e.get('agent_id') == agent_id or e.get('scope') == 'system_wide']
            return filtered[-limit:]
        return self.evolution_history[-limit:]
    
    def get_agent_performance_trend(self, agent_id: str) -> Dict[str, Any]:
        """
        Analyserar performance-trend för en agent.
        
        Args:
            agent_id: Agent att analysera
            
        Returns:
            Trend-analys
        """
        if agent_id not in self.agent_performance_history:
            return {'trend': 'unknown', 'data_points': 0}
        
        history = self.agent_performance_history[agent_id]
        
        if len(history) < 3:
            return {'trend': 'insufficient_data', 'data_points': len(history)}
        
        # Beräkna trend
        recent = history[-5:]
        recent_avg = self._calculate_average_performance(recent)
        
        older = history[:-5] if len(history) > 5 else history[:len(history)//2]
        older_avg = self._calculate_average_performance(older)
        
        if recent_avg > older_avg * 1.1:
            trend = 'improving'
        elif recent_avg < older_avg * 0.9:
            trend = 'declining'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'recent_performance': recent_avg,
            'historical_performance': older_avg,
            'data_points': len(history),
            'change_percentage': ((recent_avg - older_avg) / older_avg * 100) if older_avg > 0 else 0
        }
    
    def generate_evolution_tree(self) -> Dict[str, Any]:
        """
        Genererar ett evolutionsträd som visar agenternas utveckling över tid.
        
        Returns:
            Evolution tree struktur
        """
        tree = {
            'total_evolution_events': len(self.evolution_history),
            'agents': {},
            'system_wide_events': []
        }
        
        for event in self.evolution_history:
            if event.get('scope') == 'system_wide':
                tree['system_wide_events'].append(event)
            else:
                agent_id = event.get('agent_id')
                if agent_id:
                    if agent_id not in tree['agents']:
                        tree['agents'][agent_id] = {
                            'evolution_count': 0,
                            'events': []
                        }
                    tree['agents'][agent_id]['evolution_count'] += 1
                    tree['agents'][agent_id]['events'].append(event)
        
        return tree
    
    def get_current_parameters(self) -> Dict[str, Any]:
        """
        Hämtar nuvarande adaptiva parametrar (Sprint 4.2).
        
        Returns:
            Dict med aktuella parametervärden
        """
        return {
            'evolution_threshold': self.performance_threshold,
            'min_samples': self.min_samples_for_evolution,
            'parameter_source': 'adaptive_rl' if self.parameter_history else 'default'
        }
    
    def get_parameter_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Hämtar parameterhistorik (Sprint 4.2).
        
        Args:
            limit: Max antal entries
        
        Returns:
            Lista med parameterjusteringar
        """
        return self.parameter_history[-limit:]

