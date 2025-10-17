# test_system_monitor.py - Tester för systemövervakare

import pytest
import time
from modules.message_bus import MessageBus
from modules.system_monitor import SystemMonitor


class TestSystemMonitor:
    """Tests for SystemMonitor module."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.message_bus = MessageBus()
        self.monitor = SystemMonitor(self.message_bus)
    
    def test_initialization(self):
        """Test monitor initialization."""
        assert self.monitor.message_bus is not None
        assert isinstance(self.monitor.system_metrics, dict)
        assert isinstance(self.monitor.module_status, dict)
        assert 'start_time' in self.monitor.system_metrics
    
    def test_dashboard_data_tracking(self):
        """Test that dashboard data is tracked by source."""
        dashboard_data = {
            'source': 'test_module',
            'metric': 'test_value',
            'value': 42
        }
        
        self.message_bus.publish('dashboard_data', dashboard_data)
        time.sleep(0.01)
        
        assert 'test_module' in self.monitor.module_status
        assert self.monitor.module_status['test_module']['update_count'] == 1
    
    def test_agent_status_tracking(self):
        """Test that agent status updates are tracked."""
        initial_count = self.monitor.system_metrics['agent_updates']
        
        agent_status = {
            'agent': 'strategy_agent',
            'performance': 0.85
        }
        
        self.message_bus.publish('agent_status', agent_status)
        time.sleep(0.01)
        
        assert self.monitor.system_metrics['agent_updates'] > initial_count
        assert len(self.monitor.performance_history) > 0
    
    def test_portfolio_status_tracking(self):
        """Test that portfolio status is tracked."""
        portfolio_data = {
            'portfolio_value': 1500.0,
            'pnl': 500.0
        }
        
        self.message_bus.publish('portfolio_status', portfolio_data)
        time.sleep(0.01)
        
        assert self.monitor.system_metrics['portfolio_value'] == 1500.0
        assert len(self.monitor.performance_history) > 0
    
    def test_timeline_insight_tracking(self):
        """Test that timeline insights are tracked."""
        initial_count = self.monitor.system_metrics['timeline_events']
        
        self.message_bus.publish('timeline_insight', {'test': True})
        time.sleep(0.01)
        
        assert self.monitor.system_metrics['timeline_events'] > initial_count
    
    def test_chain_execution_tracking(self):
        """Test that chain executions are tracked."""
        initial_count = self.monitor.system_metrics['chain_executions']
        
        self.message_bus.publish('chain_execution', {'chain': 'test'})
        time.sleep(0.01)
        
        assert self.monitor.system_metrics['chain_executions'] > initial_count
        assert self.monitor.system_metrics['total_executions'] > 0
    
    def test_get_system_view(self):
        """Test getting system view."""
        # Add some data
        self.message_bus.publish('dashboard_data', {'source': 'module1'})
        self.message_bus.publish('agent_status', {'agent': 'test', 'performance': 0.8})
        time.sleep(0.01)
        
        view = self.monitor.get_system_view()
        
        assert 'system_status' in view
        assert 'health_score' in view
        assert 'uptime_seconds' in view
        assert 'metrics' in view
        assert 'active_modules' in view
        assert view['uptime_seconds'] >= 0
    
    def test_health_score_calculation(self):
        """Test that health score is calculated correctly."""
        # Add active module
        self.message_bus.publish('dashboard_data', {'source': 'active_module'})
        time.sleep(0.01)
        
        view = self.monitor.get_system_view()
        assert view['health_score'] > 0
        assert view['system_status'] in ['operational', 'degraded']
    
    def test_get_module_status_specific(self):
        """Test getting status for specific module."""
        self.message_bus.publish('dashboard_data', {
            'source': 'test_module',
            'data': 'test'
        })
        time.sleep(0.01)
        
        status = self.monitor.get_module_status('test_module')
        assert 'update_count' in status
        assert status['update_count'] == 1
    
    def test_get_module_status_all(self):
        """Test getting status for all modules."""
        self.message_bus.publish('dashboard_data', {'source': 'module1'})
        self.message_bus.publish('dashboard_data', {'source': 'module2'})
        time.sleep(0.01)
        
        all_status = self.monitor.get_module_status()
        assert isinstance(all_status, dict)
        assert 'module1' in all_status
        assert 'module2' in all_status
    
    def test_get_performance_metrics(self):
        """Test getting performance metrics."""
        # Add performance data
        self.message_bus.publish('agent_status', {
            'agent': 'test',
            'performance': 0.9
        })
        self.message_bus.publish('portfolio_status', {
            'portfolio_value': 1200.0,
            'pnl': 200.0
        })
        time.sleep(0.01)
        
        metrics = self.monitor.get_performance_metrics(time_window=300)
        
        assert 'time_window' in metrics
        assert 'total_entries' in metrics
        assert 'agent_updates' in metrics
        assert 'portfolio_updates' in metrics
    
    def test_get_system_health(self):
        """Test getting detailed system health."""
        # Add some modules
        self.message_bus.publish('dashboard_data', {'source': 'module1'})
        self.message_bus.publish('dashboard_data', {'source': 'module2'})
        time.sleep(0.01)
        
        health = self.monitor.get_system_health()
        
        assert 'health_score' in health
        assert 'status' in health
        assert 'active_modules' in health
        assert 'total_modules' in health
        assert health['status'] in ['healthy', 'warning', 'critical']
    
    def test_stale_module_detection(self):
        """Test detection of stale modules."""
        # Add a module and let it become stale
        self.message_bus.publish('dashboard_data', {'source': 'stale_module'})
        time.sleep(0.01)
        
        # Manually set last_update to old time
        if 'stale_module' in self.monitor.module_status:
            self.monitor.module_status['stale_module']['last_update'] = time.time() - 120
        
        health = self.monitor.get_system_health()
        assert 'stale_modules' in health
    
    def test_multiple_module_tracking(self):
        """Test tracking multiple modules simultaneously."""
        modules = ['strategy', 'risk', 'decision', 'execution']
        
        for module in modules:
            self.message_bus.publish('dashboard_data', {'source': module})
        
        time.sleep(0.01)
        
        view = self.monitor.get_system_view()
        assert view['total_modules'] >= len(modules)
    
    def test_performance_history_accumulation(self):
        """Test that performance history accumulates correctly."""
        initial_len = len(self.monitor.performance_history)
        
        # Add multiple performance entries
        for i in range(5):
            self.message_bus.publish('agent_status', {
                'agent': f'agent_{i}',
                'performance': 0.8 + i * 0.02
            })
        
        time.sleep(0.01)
        
        assert len(self.monitor.performance_history) >= initial_len + 5
    
    def test_system_metrics_update(self):
        """Test that system metrics are updated correctly."""
        initial_time = self.monitor.system_metrics['last_update']
        
        self.message_bus.publish('dashboard_data', {'source': 'test'})
        time.sleep(0.01)
        
        assert self.monitor.system_metrics['last_update'] > initial_time
