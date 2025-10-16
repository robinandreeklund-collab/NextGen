"""
introspection_panel.py - Introspektionspanel

Beskrivning:
    Visualiserar modulstatus, RL-performance och kommunikation.
    Dashboard för transparens och debugging av systemet.

Roll:
    - Prenumererar på agent_status, feedback_event, indicator_data
    - Visualiserar feedback flow mellan moduler
    - Visar agent adaptation och RL-performance
    - Visar reward trends över tid
    - Renderar dashboard för Dash-applikation

Inputs:
    - agent_status: Dict - Status från rl_controller
    - feedback_event: Dict - Feedback från feedback_router
    - indicator_data: Dict - Indikatorer från indicator_registry

Outputs:
    - dashboard_render: Dict - Data för Dash-visualisering

Publicerar till message_bus:
    - Ingen (konsumerar endast data)

Prenumererar på (från functions.yaml):
    - agent_status (från rl_controller)
    - feedback_event (från feedback_router)
    - indicator_data (från indicator_registry)

Använder RL: Nej (från functions.yaml)
Tar emot feedback: Nej (från functions.yaml)

Anslutningar (från flowchart.yaml - visualization):
    Från:
    - rl_controller (agent_status)
    - feedback_router (feedback_event)
    - indicator_registry (indicator_data)
    - portfolio_manager (portfolio_status)
    - strategic_memory_engine (memory_insights)
    - decision_simulator (simulation_result)
    Till: dashboard_render

Visualisering (från feedback_loop.yaml):
    Displays:
    - feedback flow: Visuell representation av feedback mellan moduler
    - agent adaptation: Hur agenter anpassar sig över tid
    - reward trends: Utveckling av reward över episodes

Används i Sprint: 3, 7
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
        
        # Prenumerera på relevanta topics
        self.message_bus.subscribe('agent_status', self._on_agent_status)
        self.message_bus.subscribe('feedback_event', self._on_feedback)
        self.message_bus.subscribe('indicator_data', self._on_indicators)
        self.message_bus.subscribe('portfolio_status', self._on_portfolio)
    
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
        
        dashboard_data = {
            'agent_status': self.agent_status_history[-10:] if self.agent_status_history else [],
            'feedback_flow': self.feedback_events[-20:] if self.feedback_events else [],
            'indicators': self.indicator_snapshots[-10:] if self.indicator_snapshots else [],
            'reward_trends': reward_trends,
            'module_connections': module_connections,
            'feedback_metrics': feedback_metrics,
            'agent_adaptation': agent_adaptation,
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

