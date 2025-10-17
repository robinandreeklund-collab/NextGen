"""
introspection_panel.py - Introspektionspanel

Beskrivning:
    Visualiserar modulstatus, RL-performance och kommunikation.
    Dashboard för transparens och debugging av systemet.
    Sprint 4.2: Visualiserar även parameterhistorik och trends.
    Sprint 7: Indikatorvisualisering, resursallokering och teamdynamik.

Roll:
    - Prenumererar på agent_status, feedback_event, indicator_data
    - Visualiserar feedback flow mellan moduler
    - Visar agent adaptation och RL-performance
    - Visar reward trends över tid
    - Renderar dashboard för Dash-applikation
    - Visualiserar parameterhistorik och trends (Sprint 4.2)
    - Visualiserar indikator-trender och korrelationer (Sprint 7)
    - Visualiserar resursallokering och effektivitet (Sprint 7)
    - Visualiserar teamdynamik och synergier (Sprint 7)

Inputs:
    - agent_status: Dict - Status från rl_controller
    - feedback_event: Dict - Feedback från feedback_router
    - indicator_data: Dict - Indikatorer från indicator_registry
    - parameter_adjustment: Dict - Parameterjusteringar från RL-controller (Sprint 4.2)
    - resource_allocation: Dict - Resursallokering från resource_planner (Sprint 7)
    - team_metrics: Dict - Teammetrik från team_dynamics_engine (Sprint 7)

Outputs:
    - dashboard_render: Dict - Data för Dash-visualisering

Publicerar till message_bus:
    - Ingen (konsumerar endast data)

Prenumererar på (från functions_v2.yaml):
    - agent_status (från rl_controller)
    - feedback_event (från feedback_router)
    - indicator_data (från indicator_registry)
    - parameter_adjustment (från rl_controller) - Sprint 4.2
    - resource_allocation (från resource_planner) - Sprint 7
    - team_metrics (från team_dynamics_engine) - Sprint 7

Använder RL: Nej (från functions_v2.yaml)
Tar emot feedback: Nej (från functions_v2.yaml)

Anslutningar (från flowchart_sprint1_7.yaml - visualization):
    Från:
    - rl_controller (agent_status, parameter_adjustment)
    - feedback_router (feedback_event)
    - indicator_registry (indicator_data)
    - portfolio_manager (portfolio_status)
    - strategic_memory_engine (memory_insights)
    - decision_simulator (simulation_result)
    - resource_planner (resource_allocation) - Sprint 7
    - team_dynamics_engine (team_metrics) - Sprint 7
    Till: dashboard_render

Visualisering (från feedback_loop_sprint1_7.yaml):
    Displays:
    - feedback flow: Visuell representation av feedback mellan moduler
    - agent adaptation: Hur agenter anpassar sig över tid
    - reward trends: Utveckling av reward över episodes
    - parameter history: Meta-parametrar över tid (Sprint 4.2)
    - parameter impact: Korrelation mellan parametrar och performance (Sprint 4.2)

Används i Sprint: 3, 4.2, 7
"""

from typing import Dict, Any, List


class IntrospectionPanel:
    """Visualiserar system status och modul-kommunikation."""
    
    def __init__(self, message_bus):
        """
        Initialiserar introspektionspanelen.
        
        Args:
            message_bus: Referens till central message_bus
        """
        self.message_bus = message_bus
        self.agent_status_history: List[Dict[str, Any]] = []
        self.feedback_events: List[Dict[str, Any]] = []
        self.indicator_snapshots: List[Dict[str, Any]] = []
        
        # Sprint 4.2: Parameter adjustment history
        self.parameter_adjustments: List[Dict[str, Any]] = []
        
        # Prenumerera på relevanta topics
        self.message_bus.subscribe('agent_status', self._on_agent_status)
        self.message_bus.subscribe('feedback_event', self._on_feedback)
        self.message_bus.subscribe('indicator_data', self._on_indicators)
        self.message_bus.subscribe('portfolio_status', self._on_portfolio)
        
        # Sprint 4.2: Prenumerera på parameter_adjustment
        self.message_bus.subscribe('parameter_adjustment', self._on_parameter_adjustment)
        
        # Sprint 4.4: Prenumerera på reward metrics från RewardTunerAgent
        self.message_bus.subscribe('reward_metrics', self._on_reward_metrics)
        
        # Sprint 4.4: Reward metrics history
        self.reward_metrics_history: List[Dict[str, Any]] = []
        
        # Sprint 7: Prenumerera på resource och team metrics
        self.message_bus.subscribe('resource_allocation', self._on_resource_allocation)
        self.message_bus.subscribe('team_metrics', self._on_team_metrics)
        
        # Sprint 7: Resource och team history
        self.resource_allocations: List[Dict[str, Any]] = []
        self.team_metrics_history: List[Dict[str, Any]] = []
    
    def _on_agent_status(self, status: Dict[str, Any]) -> None:
        """Callback för agent status från rl_controller."""
        self.agent_status_history.append(status)
        # Behåll senaste 100
        if len(self.agent_status_history) > 100:
            self.agent_status_history = self.agent_status_history[-100:]
    
    def _on_feedback(self, feedback: Dict[str, Any]) -> None:
        """Callback för feedback events."""
        self.feedback_events.append(feedback)
        # Behåll senaste 100
        if len(self.feedback_events) > 100:
            self.feedback_events = self.feedback_events[-100:]
    
    def _on_indicators(self, indicators: Dict[str, Any]) -> None:
        """Callback för indikatordata."""
        self.indicator_snapshots.append(indicators)
        # Behåll senaste 50
        if len(self.indicator_snapshots) > 50:
            self.indicator_snapshots = self.indicator_snapshots[-50:]
    
    def _on_portfolio(self, status: Dict[str, Any]) -> None:
        """Callback för portföljstatus."""
        # Stub: Lagras för visualisering
        pass
    
    def _on_parameter_adjustment(self, adjustment: Dict[str, Any]) -> None:
        """
        Callback för parameter adjustments från RL-controller (Sprint 4.2).
        
        Args:
            adjustment: Parameterjusteringar
        """
        self.parameter_adjustments.append(adjustment)
        # Behåll senaste 100
        if len(self.parameter_adjustments) > 100:
            self.parameter_adjustments = self.parameter_adjustments[-100:]
    
    def _on_reward_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Callback för reward metrics från RewardTunerAgent (Sprint 4.4).
        
        Args:
            metrics: Reward transformation metrics
        """
        self.reward_metrics_history.append(metrics)
        # Limit history to prevent memory leak (keep last 100)
        if len(self.reward_metrics_history) > 100:
            self.reward_metrics_history = self.reward_metrics_history[-100:]
    
    def _on_resource_allocation(self, allocation: Dict[str, Any]) -> None:
        """
        Callback för resource allocations från ResourcePlanner (Sprint 7).
        
        Args:
            allocation: Resource allocation event
        """
        self.resource_allocations.append(allocation)
        # Behåll senaste 100
        if len(self.resource_allocations) > 100:
            self.resource_allocations = self.resource_allocations[-100:]
    
    def _on_team_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Callback för team metrics från TeamDynamicsEngine (Sprint 7).
        
        Args:
            metrics: Team performance metrics
        """
        self.team_metrics_history.append(metrics)
        # Behåll senaste 100
        if len(self.team_metrics_history) > 100:
            self.team_metrics_history = self.team_metrics_history[-100:]
    
    def render_dashboard(self) -> Dict[str, Any]:
        """
        Genererar dashboard-data för visualisering.
        
        Returns:
            Dict med alla data för dashboard
        """
        # Sprint 3: Faktisk Dash-integration med rik data
        
        # Extrahera reward trends från agent_status
        reward_trends = self._extract_reward_trends()
        
        # Analysera modul-kommunikation
        module_connections = self.get_module_connections()
        
        # Sammanställ feedback flow metrics
        feedback_metrics = self._calculate_feedback_metrics()
        
        # Beräkna agent adaptation metrics
        agent_adaptation = self._calculate_agent_adaptation()
        
        # Sprint 4.4: Add reward flow visualization
        reward_flow = self._extract_reward_flow()
        
        dashboard_data = {
            'agent_status': self.agent_status_history[-10:] if self.agent_status_history else [],
            'feedback_flow': self.feedback_events[-20:] if self.feedback_events else [],
            'indicators': self.indicator_snapshots[-10:] if self.indicator_snapshots else [],
            'reward_trends': reward_trends,
            'module_connections': module_connections,
            'feedback_metrics': feedback_metrics,
            'agent_adaptation': agent_adaptation,
            'reward_flow': reward_flow,  # Sprint 4.4
            'timestamp': self._get_timestamp()
        }
        
        return dashboard_data
    
    def _extract_reward_trends(self) -> List[Dict[str, Any]]:
        """
        Extraherar reward trends från agent_status history.
        
        Returns:
            Lista med reward trends över tid
        """
        trends = []
        
        for i, status in enumerate(self.agent_status_history):
            if 'reward' in status:
                trends.append({
                    'step': i,
                    'reward': status['reward'],
                    'module': status.get('module', 'unknown'),
                    'timestamp': status.get('timestamp', i)
                })
        
        return trends
    
    def _calculate_feedback_metrics(self) -> Dict[str, Any]:
        """
        Beräknar metrics för feedback flow.
        
        Returns:
            Dict med feedback metrics
        """
        if not self.feedback_events:
            return {
                'total_events': 0,
                'by_source': {},
                'by_priority': {},
                'avg_per_minute': 0
            }
        
        # Räkna events per källa
        by_source = {}
        by_priority = {}
        
        for event in self.feedback_events:
            source = event.get('source', 'unknown')
            priority = event.get('priority', 'medium')
            
            by_source[source] = by_source.get(source, 0) + 1
            by_priority[priority] = by_priority.get(priority, 0) + 1
        
        # Beräkna rate (om vi har timestamps)
        events_with_time = [
            e for e in self.feedback_events 
            if 'timestamp' in e or 'route_timestamp' in e
        ]
        
        avg_per_minute = 0
        if len(events_with_time) > 1:
            first_time = events_with_time[0].get('timestamp') or events_with_time[0].get('route_timestamp', 0)
            last_time = events_with_time[-1].get('timestamp') or events_with_time[-1].get('route_timestamp', 0)
            time_span_minutes = (last_time - first_time) / 60
            if time_span_minutes > 0:
                avg_per_minute = len(events_with_time) / time_span_minutes
        
        return {
            'total_events': len(self.feedback_events),
            'by_source': by_source,
            'by_priority': by_priority,
            'avg_per_minute': avg_per_minute
        }
    
    def _calculate_agent_adaptation(self) -> Dict[str, Any]:
        """
        Beräknar hur agenter anpassar sig över tid.
        
        Returns:
            Dict med agent adaptation metrics
        """
        if len(self.agent_status_history) < 2:
            return {
                'adaptation_rate': 0,
                'performance_trend': 'stable',
                'learning_progress': 0
            }
        
        # Analysera performance över tid
        recent = self.agent_status_history[-10:]
        performances = [
            s.get('performance', 0.5) 
            for s in recent 
            if 'performance' in s
        ]
        
        if len(performances) < 2:
            return {
                'adaptation_rate': 0,
                'performance_trend': 'stable',
                'learning_progress': 0
            }
        
        # Beräkna trend
        first_half_avg = sum(performances[:len(performances)//2]) / (len(performances)//2)
        second_half_avg = sum(performances[len(performances)//2:]) / (len(performances) - len(performances)//2)
        
        adaptation_rate = (second_half_avg - first_half_avg) / max(first_half_avg, 0.01)
        
        if adaptation_rate > 0.1:
            trend = 'improving'
        elif adaptation_rate < -0.1:
            trend = 'declining'
        else:
            trend = 'stable'
        
        return {
            'adaptation_rate': adaptation_rate,
            'performance_trend': trend,
            'learning_progress': second_half_avg,
            'recent_performances': performances[-5:]
        }
    
    def _get_timestamp(self) -> float:
        """Returnerar nuvarande timestamp."""
        import time
        return time.time()
    
    def get_module_connections(self) -> List[Dict[str, Any]]:
        """
        Analyserar feedback för att visa modul-kopplingar.
        
        Returns:
            Lista med modul-till-modul kopplingar
        """
        # Sprint 3: Nätverksanalys av kommunikation baserat på feedback flow
        
        connections = []
        connection_counts = {}
        
        # Analysera feedback_events för att identifiera kopplingar
        for event in self.feedback_events:
            source = event.get('source', 'unknown')
            
            # Identifiera mottagare baserat på triggers och routing
            targets = self._identify_targets(event)
            
            for target in targets:
                key = f"{source}->{target}"
                connection_counts[key] = connection_counts.get(key, 0) + 1
        
        # Konvertera till lista med detaljer
        for connection_key, count in connection_counts.items():
            parts = connection_key.split('->')
            if len(parts) == 2:
                connections.append({
                    'source': parts[0],
                    'target': parts[1],
                    'count': count,
                    'strength': min(count / 10, 1.0)  # Normaliserad styrka 0-1
                })
        
        return sorted(connections, key=lambda x: x['count'], reverse=True)
    
    def _identify_targets(self, event: Dict[str, Any]) -> List[str]:
        """
        Identifierar målmoduler för ett feedback event.
        
        Args:
            event: Feedback event
            
        Returns:
            Lista med modulnamn som är mottagare
        """
        targets = []
        triggers = event.get('triggers', [])
        source = event.get('source', 'unknown')
        
        # Enligt feedback_loop.yaml routing:
        # execution_engine → rl_controller, feedback_analyzer, strategic_memory
        # portfolio_manager → rl_controller, feedback_analyzer, strategic_memory
        # strategic_memory_engine → feedback_analyzer, meta_agent_evolution
        
        if source in ['execution_engine', 'portfolio_manager']:
            targets.extend(['rl_controller', 'feedback_analyzer', 'strategic_memory_engine'])
        
        if source == 'strategic_memory_engine':
            targets.extend(['feedback_analyzer', 'meta_agent_evolution_engine'])
        
        # Trigger-baserad routing
        if 'slippage' in triggers or 'trade_result' in triggers:
            if 'rl_controller' not in targets:
                targets.append('rl_controller')
        
        if 'capital_change' in triggers:
            if 'portfolio_manager' not in targets:
                targets.append('portfolio_manager')
        
        if 'indicator_correlation' in triggers:
            if 'meta_agent_evolution_engine' not in targets:
                targets.append('meta_agent_evolution_engine')
        
        return targets
    
    def get_parameter_visualization_data(self) -> Dict[str, Any]:
        """
        Genererar data för visualisering av parameterhistorik (Sprint 4.2).
        
        Returns:
            Dict med parameter visualization data
        """
        if not self.parameter_adjustments:
            return {
                'parameter_trends': {},
                'parameter_correlation': {},
                'recent_adjustments': []
            }
        
        # Extrahera trends för varje parameter
        parameter_trends = {
            'evolution_threshold': [],
            'min_samples': [],
            'update_frequency': [],
            'agent_entropy_threshold': []
        }
        
        for adjustment in self.parameter_adjustments:
            adjusted = adjustment.get('adjusted_parameters', {})
            timestamp = adjustment.get('timestamp', 0)
            
            for param_name in parameter_trends.keys():
                if param_name in adjusted:
                    parameter_trends[param_name].append({
                        'timestamp': timestamp,
                        'value': adjusted[param_name],
                        'reward_signals': adjustment.get('reward_signals', {})
                    })
        
        # Beräkna parameter correlation med performance
        parameter_correlation = {}
        for param_name, trend_data in parameter_trends.items():
            if len(trend_data) >= 3:
                values = [d['value'] for d in trend_data]
                parameter_correlation[param_name] = {
                    'min': min(values),
                    'max': max(values),
                    'current': values[-1],
                    'trend': 'increasing' if values[-1] > values[0] else 'decreasing' if values[-1] < values[0] else 'stable',
                    'volatility': max(values) - min(values)
                }
        
        return {
            'parameter_trends': parameter_trends,
            'parameter_correlation': parameter_correlation,
            'recent_adjustments': self.parameter_adjustments[-10:],
            'total_adjustments': len(self.parameter_adjustments)
        }
    
    def _extract_reward_flow(self) -> Dict[str, Any]:
        """
        Extraherar reward flow visualization data (Sprint 4.4).
        
        Returns:
            Dict med reward transformation data
        """
        if not self.reward_metrics_history:
            return {
                'base_rewards': [],
                'tuned_rewards': [],
                'transformation_ratios': [],
                'volatility': [],
                'overfitting_events': []
            }
        
        base_rewards = []
        tuned_rewards = []
        transformation_ratios = []
        volatility = []
        overfitting_events = []
        
        for i, metrics in enumerate(self.reward_metrics_history):
            base_rewards.append({
                'step': i,
                'value': metrics.get('base_reward', 0.0)
            })
            tuned_rewards.append({
                'step': i,
                'value': metrics.get('tuned_reward', 0.0)
            })
            transformation_ratios.append({
                'step': i,
                'ratio': metrics.get('transformation_ratio', 1.0)
            })
            volatility.append({
                'step': i,
                'value': metrics.get('volatility', 0.0)
            })
            if metrics.get('overfitting_detected', False):
                overfitting_events.append({
                    'step': i,
                    'score': metrics.get('overfitting_score', 0.0)
                })
        
        return {
            'base_rewards': base_rewards,
            'tuned_rewards': tuned_rewards,
            'transformation_ratios': transformation_ratios,
            'volatility': volatility,
            'overfitting_events': overfitting_events,
            'current_parameters': self.reward_metrics_history[-1].get('reward_scaling_factor', 1.0) if self.reward_metrics_history else 1.0
        }
    
    def get_comprehensive_dashboard_data(self) -> Dict[str, Any]:
        """
        Genererar omfattande dashboard data inklusive parametrar (Sprint 4.2+4.4) och Sprint 7 data.
        
        Returns:
            Dict med all dashboard data
        """
        base_data = self.get_dashboard_data()
        parameter_data = self.get_parameter_visualization_data()
        sprint7_data = self.get_sprint7_visualization_data()
        
        return {
            **base_data,
            'parameter_visualization': parameter_data,
            'sprint7_visualization': sprint7_data
        }
    
    def get_sprint7_visualization_data(self) -> Dict[str, Any]:
        """
        Genererar Sprint 7 visualization data (Sprint 7).
        
        Returns:
            Dict med Sprint 7 visualization data
        """
        return {
            'resource_dashboard': self._extract_resource_data(),
            'team_dashboard': self._extract_team_data(),
            'indicator_correlation': self._extract_indicator_correlation(),
            'system_overview': self._extract_system_overview()
        }
    
    def _extract_resource_data(self) -> Dict[str, Any]:
        """
        Extraherar resource allocation visualization data.
        
        Returns:
            Dict med resource data
        """
        if not self.resource_allocations:
            return {
                'allocations_over_time': [],
                'utilization_by_module': {},
                'efficiency_metrics': {},
                'bottleneck_alerts': []
            }
        
        # Allocation timeline
        allocations_over_time = []
        module_allocations = {}
        
        for i, allocation in enumerate(self.resource_allocations):
            allocations_over_time.append({
                'step': i,
                'module_id': allocation.get('module_id', 'unknown'),
                'resource_type': allocation.get('resource_type', 'unknown'),
                'amount': allocation.get('allocated_amount', 0),
                'score': allocation.get('allocation_score', 0.0),
                'timestamp': allocation.get('timestamp', 0)
            })
            
            # Aggregate by module
            module_id = allocation.get('module_id', 'unknown')
            if module_id not in module_allocations:
                module_allocations[module_id] = {'compute': 0, 'memory': 0, 'training': 0}
            
            resource_type = allocation.get('resource_type', 'compute')
            module_allocations[module_id][resource_type] += allocation.get('allocated_amount', 0)
        
        return {
            'allocations_over_time': allocations_over_time,
            'utilization_by_module': module_allocations,
            'total_allocations': len(self.resource_allocations),
            'recent_allocations': self.resource_allocations[-10:]
        }
    
    def _extract_team_data(self) -> Dict[str, Any]:
        """
        Extraherar team dynamics visualization data.
        
        Returns:
            Dict med team data
        """
        if not self.team_metrics_history:
            return {
                'team_performance': [],
                'synergy_trends': [],
                'coordination_trends': [],
                'active_teams': []
            }
        
        team_performance = []
        synergy_by_team = {}
        coordination_by_team = {}
        
        for i, metrics in enumerate(self.team_metrics_history):
            team_id = metrics.get('team_id', 'unknown')
            
            team_performance.append({
                'step': i,
                'team_id': team_id,
                'synergy_score': metrics.get('synergy_score', 0.0),
                'coordination_score': metrics.get('coordination_score', 0.0),
                'decisions_made': metrics.get('decisions_made', 0)
            })
            
            # Track trends per team
            if team_id not in synergy_by_team:
                synergy_by_team[team_id] = []
                coordination_by_team[team_id] = []
            
            synergy_by_team[team_id].append(metrics.get('synergy_score', 0.0))
            coordination_by_team[team_id].append(metrics.get('coordination_score', 0.0))
        
        return {
            'team_performance': team_performance,
            'synergy_trends': synergy_by_team,
            'coordination_trends': coordination_by_team,
            'total_teams': len(synergy_by_team),
            'recent_metrics': self.team_metrics_history[-10:]
        }
    
    def _extract_indicator_correlation(self) -> Dict[str, Any]:
        """
        Extraherar indicator correlation visualization data.
        
        Returns:
            Dict med indicator correlation data
        """
        if len(self.indicator_snapshots) < 2:
            return {
                'indicator_trends': {},
                'correlation_matrix': {},
                'effectiveness_ranking': []
            }
        
        # Extract indicator trends
        indicator_trends = {}
        
        for snapshot in self.indicator_snapshots:
            for indicator_name, value in snapshot.items():
                if indicator_name not in ['symbol', 'timestamp']:
                    if indicator_name not in indicator_trends:
                        indicator_trends[indicator_name] = []
                    
                    if isinstance(value, (int, float)):
                        indicator_trends[indicator_name].append(value)
        
        return {
            'indicator_trends': indicator_trends,
            'total_indicators': len(indicator_trends),
            'recent_snapshots': self.indicator_snapshots[-10:]
        }
    
    def _extract_system_overview(self) -> Dict[str, Any]:
        """
        Extraherar system overview data.
        
        Returns:
            Dict med system overview
        """
        return {
            'total_agent_updates': len(self.agent_status_history),
            'total_feedback_events': len(self.feedback_events),
            'total_indicator_snapshots': len(self.indicator_snapshots),
            'total_parameter_adjustments': len(self.parameter_adjustments),
            'total_resource_allocations': len(self.resource_allocations),
            'total_team_metrics': len(self.team_metrics_history),
            'system_active': True
        }

