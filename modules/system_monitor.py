"""
system_monitor.py - Systemövervakare

Beskrivning:
    Visar systemöversikt, indikatortrender och agentrespons.

Roll:
    - Prenumererar på dashboard_data från olika moduler
    - Aggregerar systemstatus
    - Visar översikt av hela systemet
    - Returnerar system_view via metod, publicerar inte till topic

Inputs:
    - dashboard_data: Dict - Data från alla moduler

Outputs:
    - system_view: Dict - Komplett systemöversikt (returneras från metod)

Publicerar till message_bus:
    - Ingen (konsumerar endast)

Prenumererar på (från functions.yaml):
    - dashboard_data (från olika källor)

Använder RL: Nej
Tar emot feedback: Nej

Används i Sprint: 7
"""

from typing import Dict, Any, List
import time


class SystemMonitor:
    """Övervakar och visualiserar hela systemet."""
    
    def __init__(self, message_bus):
        self.message_bus = message_bus
        self.system_metrics: Dict[str, Any] = {}
        self.module_status: Dict[str, Dict[str, Any]] = {}
        self.performance_history: List[Dict[str, Any]] = []
        
        # Subscribe to various dashboard data sources
        self.message_bus.subscribe('dashboard_data', self._on_dashboard_data)
        self.message_bus.subscribe('agent_status', self._on_agent_status)
        self.message_bus.subscribe('portfolio_status', self._on_portfolio_status)
        self.message_bus.subscribe('timeline_insight', self._on_timeline_insight)
        self.message_bus.subscribe('chain_execution', self._on_chain_execution)
        
        # Subscribe to other module activities to track them
        self.message_bus.subscribe('decision_vote', self._on_decision_activity)
        self.message_bus.subscribe('final_decision', self._on_final_decision_activity)
        self.message_bus.subscribe('execution_result', self._on_execution_activity)
        self.message_bus.subscribe('decision_proposal', self._on_strategy_activity)
        self.message_bus.subscribe('vote_matrix', self._on_vote_activity)
        self.message_bus.subscribe('tuned_reward', self._on_reward_tuner_activity)
        
        # Subscribe to feedback_router and risk_manager activities
        self.message_bus.subscribe('high_priority_feedback', self._on_feedback_router_activity)
        self.message_bus.subscribe('risk_profile', self._on_risk_manager_activity)
        
        # Sprint 8: Subscribe to DQN, GAN, GNN activities
        self.message_bus.subscribe('dqn_metrics', self._on_dqn_activity)
        self.message_bus.subscribe('dqn_action_response', self._on_dqn_activity)
        self.message_bus.subscribe('gan_metrics', self._on_gan_activity)
        self.message_bus.subscribe('gan_candidates', self._on_gan_activity)
        self.message_bus.subscribe('gnn_analysis_response', self._on_gnn_activity)
        
        # Initialize system metrics
        self._initialize_metrics()
    
    def _initialize_metrics(self):
        """Initialize system metrics tracking."""
        self.system_metrics = {
            'start_time': time.time(),
            'total_decisions': 0,
            'total_executions': 0,
            'total_rewards': 0,
            'agent_updates': 0,
            'chain_executions': 0,
            'timeline_events': 0,
            'portfolio_value': 1000.0,  # Initial capital
            'last_update': time.time()
        }
        
        # Initialize expected modules so they show up in dashboard
        # Sprint 1-7 modules
        expected_modules = [
            'strategy_engine', 'portfolio_manager', 'rl_controller',
            'reward_tuner', 'decision_engine', 'consensus_engine',
            'vote_engine', 'execution_engine', 'timespan_tracker',
            'action_chain_engine', 'risk_manager', 'feedback_router'
        ]
        # Sprint 8 modules
        expected_modules.extend(['dqn_controller', 'gan_evolution', 'gnn_analyzer'])
        
        # Initialize all expected modules with a placeholder
        current_time = time.time()
        for module_name in expected_modules:
            if module_name not in self.module_status:
                self.module_status[module_name] = {
                    'first_seen': current_time,
                    'last_update': current_time,
                    'update_count': 0,
                    'initialized': True
                }
    
    def _on_dashboard_data(self, data: Dict[str, Any]):
        """Handle dashboard data from various modules."""
        source = data.get('source', 'unknown')
        
        if source not in self.module_status:
            self.module_status[source] = {
                'first_seen': time.time(),
                'update_count': 0
            }
        
        self.module_status[source]['last_update'] = time.time()
        self.module_status[source]['update_count'] += 1
        self.module_status[source]['latest_data'] = data
        
        self.system_metrics['last_update'] = time.time()
    
    def _on_agent_status(self, data: Dict[str, Any]):
        """Handle agent status updates."""
        self.system_metrics['agent_updates'] += 1
        
        # Track rl_controller as active module
        self._track_module_activity('rl_controller')
        
        # Track agent performance
        if 'performance' in data:
            perf_entry = {
                'timestamp': time.time(),
                'type': 'agent_performance',
                'agent': data.get('agent', 'unknown'),
                'performance': data['performance']
            }
            self.performance_history.append(perf_entry)
            
            # Limit performance history to prevent memory leak (keep last 1000)
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]
    
    def _on_portfolio_status(self, data: Dict[str, Any]):
        """Handle portfolio status updates."""
        # Track portfolio_manager as active module
        self._track_module_activity('portfolio_manager')
        
        if 'portfolio_value' in data:
            self.system_metrics['portfolio_value'] = data['portfolio_value']
            
            perf_entry = {
                'timestamp': time.time(),
                'type': 'portfolio',
                'value': data['portfolio_value'],
                'pnl': data.get('pnl', 0)
            }
            self.performance_history.append(perf_entry)
            
            # Limit performance history to prevent memory leak (keep last 1000)
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]
    
    def _on_timeline_insight(self, data: Dict[str, Any]):
        """Handle timeline insights."""
        self.system_metrics['timeline_events'] += 1
        # Track timespan_tracker as active module
        self._track_module_activity('timespan_tracker')
    
    def _on_chain_execution(self, data: Dict[str, Any]):
        """Handle chain execution events."""
        self.system_metrics['chain_executions'] += 1
        self.system_metrics['total_executions'] += 1
        # Track action_chain_engine as active module
        self._track_module_activity('action_chain_engine')
    
    def _track_module_activity(self, module_name: str):
        """Track that a module is active."""
        if module_name not in self.module_status:
            self.module_status[module_name] = {
                'first_seen': time.time(),
                'update_count': 0
            }
        
        self.module_status[module_name]['last_update'] = time.time()
        self.module_status[module_name]['update_count'] += 1
    
    def _on_decision_activity(self, data: Dict[str, Any]):
        """Track decision_engine activity via decision_vote."""
        self._track_module_activity('decision_engine')
    
    def _on_final_decision_activity(self, data: Dict[str, Any]):
        """Track final decision from consensus_engine."""
        self._track_module_activity('consensus_engine')
    
    def _on_execution_activity(self, data: Dict[str, Any]):
        """Track execution_engine activity."""
        self._track_module_activity('execution_engine')
    
    def _on_strategy_activity(self, data: Dict[str, Any]):
        """Track strategy_engine activity."""
        self._track_module_activity('strategy_engine')
    
    def _on_vote_activity(self, data: Dict[str, Any]):
        """Track vote_engine activity."""
        self._track_module_activity('vote_engine')
    
    def _on_reward_tuner_activity(self, data: Dict[str, Any]):
        """Track reward_tuner activity."""
        self._track_module_activity('reward_tuner')
    
    def _on_dqn_activity(self, data: Dict[str, Any]):
        """Track DQN controller activity (Sprint 8)."""
        self._track_module_activity('dqn_controller')
        self.system_metrics['last_update'] = time.time()
    
    def _on_gan_activity(self, data: Dict[str, Any]):
        """Track GAN evolution engine activity (Sprint 8)."""
        self._track_module_activity('gan_evolution')
        self.system_metrics['last_update'] = time.time()
    
    def _on_gnn_activity(self, data: Dict[str, Any]):
        """Track GNN timespan analyzer activity (Sprint 8)."""
        self._track_module_activity('gnn_analyzer')
        self.system_metrics['last_update'] = time.time()
    
    def _on_feedback_router_activity(self, data: Dict[str, Any]):
        """Track feedback router activity."""
        self._track_module_activity('feedback_router')
        self.system_metrics['last_update'] = time.time()
    
    def _on_risk_manager_activity(self, data: Dict[str, Any]):
        """Track risk manager activity."""
        self._track_module_activity('risk_manager')
        self.system_metrics['last_update'] = time.time()
    
    def get_system_view(self) -> Dict[str, Any]:
        """
        Get complete system overview.
        
        Returns:
            Dict with comprehensive system status
        """
        uptime = time.time() - self.system_metrics['start_time']
        
        # Calculate health score
        # Increased timeout from 60s to 300s (5 minutes) to handle event-driven modules
        # that only activate when there's trading activity or market data
        active_modules = len([m for m in self.module_status.values() 
                             if time.time() - m.get('last_update', 0) < 300])
        total_modules = len(self.module_status)
        health_score = (active_modules / total_modules) if total_modules > 0 else 0
        
        return {
            'system_status': 'operational' if health_score > 0.5 else 'degraded',
            'health_score': health_score,
            'uptime_seconds': uptime,
            'metrics': self.system_metrics,
            'active_modules': active_modules,
            'total_modules': total_modules,
            'module_status': self.module_status,
            'performance_snapshot': self.performance_history[-10:] if self.performance_history else [],
            'timestamp': time.time()
        }
    
    def get_module_status(self, module_name: str = None) -> Dict[str, Any]:
        """
        Get status for a specific module or all modules.
        
        Args:
            module_name: Optional module name, if None returns all
        
        Returns:
            Module status information
        """
        if module_name:
            return self.module_status.get(module_name, {})
        return self.module_status
    
    def get_performance_metrics(self, time_window: float = 300) -> Dict[str, Any]:
        """
        Get performance metrics within a time window.
        
        Args:
            time_window: Time window in seconds (default: 5 minutes)
        
        Returns:
            Performance metrics summary
        """
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        recent_perf = [p for p in self.performance_history if p['timestamp'] >= cutoff_time]
        
        # Calculate metrics by type
        agent_perfs = [p for p in recent_perf if p['type'] == 'agent_performance']
        portfolio_perfs = [p for p in recent_perf if p['type'] == 'portfolio']
        
        return {
            'time_window': time_window,
            'total_entries': len(recent_perf),
            'agent_updates': len(agent_perfs),
            'portfolio_updates': len(portfolio_perfs),
            'latest_portfolio_value': portfolio_perfs[-1]['value'] if portfolio_perfs else 0,
            'portfolio_change': (portfolio_perfs[-1]['value'] - portfolio_perfs[0]['value']) if len(portfolio_perfs) > 1 else 0
        }
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get detailed system health metrics.
        
        Returns:
            Health metrics for all subsystems
        """
        current_time = time.time()
        
        # Check module freshness
        stale_modules = []
        active_modules = []
        
        for module_name, status in self.module_status.items():
            last_update = status.get('last_update', 0)
            if current_time - last_update > 120:  # 2 minute stale threshold (increased from 60s)
                stale_modules.append(module_name)
            else:
                active_modules.append(module_name)
        
        # Calculate overall health
        total_modules = len(self.module_status)
        health_score = len(active_modules) / total_modules if total_modules > 0 else 0
        
        return {
            'health_score': health_score,
            'status': 'healthy' if health_score > 0.8 else 'warning' if health_score > 0.5 else 'critical',
            'active_modules': active_modules,
            'stale_modules': stale_modules,
            'total_modules': total_modules,
            'uptime': current_time - self.system_metrics['start_time'],
            'timestamp': current_time
        }

