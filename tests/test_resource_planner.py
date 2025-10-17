"""
Tests for Resource Planner - Sprint 7
"""

import pytest
import time
from modules.message_bus import MessageBus
from modules.resource_planner import ResourcePlanner


class TestResourcePlanner:
    """Test Resource Planner functionality"""
    
    def setup_method(self):
        """Setup for each test"""
        self.message_bus = MessageBus()
        self.resource_planner = ResourcePlanner(self.message_bus)
    
    def test_initialization(self):
        """Test resource planner initialization"""
        assert self.resource_planner.compute_pool['total'] == 100
        assert self.resource_planner.memory_pool['total'] == 100
        assert self.resource_planner.training_budget['total'] == 100
        
        # Check default allocations were made
        assert len(self.resource_planner.compute_pool['allocated']) > 0
        assert len(self.resource_planner.memory_pool['allocated']) > 0
    
    def test_resource_request_handling(self):
        """Test handling resource requests"""
        # Request resources
        self.message_bus.publish('resource_request', {
            'module_id': 'test_module',
            'resource_type': 'compute',
            'amount_requested': 10,
            'priority': 'high'
        })
        
        time.sleep(0.1)
        
        # Check allocation
        status = self.resource_planner.get_resource_status()
        assert 'test_module' in status['compute']['allocated']
    
    def test_priority_based_allocation(self):
        """Test priority-based resource allocation"""
        # High priority request
        self.message_bus.publish('resource_request', {
            'module_id': 'high_priority_module',
            'resource_type': 'compute',
            'amount_requested': 10,
            'priority': 'critical'
        })
        
        # Low priority request
        self.message_bus.publish('resource_request', {
            'module_id': 'low_priority_module',
            'resource_type': 'compute',
            'amount_requested': 10,
            'priority': 'low'
        })
        
        time.sleep(0.1)
        
        # High priority should be allocated
        status = self.resource_planner.get_resource_status()
        assert 'high_priority_module' in status['compute']['allocated']
    
    def test_performance_metric_tracking(self):
        """Test performance metric tracking"""
        # Submit performance metric
        self.message_bus.publish('performance_metric', {
            'module_id': 'test_module',
            'resource_consumed': 10,
            'performance_achieved': 1.5
        })
        
        time.sleep(0.1)
        
        # Check performance tracking
        perf = self.resource_planner.get_module_performance('test_module')
        assert perf['efficiency_score'] > 0
    
    def test_resource_release(self):
        """Test resource release when module completes"""
        # Allocate resources
        self.message_bus.publish('resource_request', {
            'module_id': 'temp_module',
            'resource_type': 'memory',
            'amount_requested': 15,
            'priority': 'medium'
        })
        
        time.sleep(0.1)
        
        # Release resources
        self.message_bus.publish('module_completed', {
            'module_id': 'temp_module'
        })
        
        time.sleep(0.1)
        
        # Check resources released
        status = self.resource_planner.get_resource_status()
        assert 'temp_module' not in status['memory']['allocated']
    
    def test_allocation_history(self):
        """Test allocation history logging"""
        initial_size = len(self.resource_planner.allocation_history)
        
        # Make allocation
        self.message_bus.publish('resource_request', {
            'module_id': 'history_test',
            'resource_type': 'training',
            'amount_requested': 5,
            'priority': 'medium'
        })
        
        time.sleep(0.1)
        
        # Check history updated
        assert len(self.resource_planner.allocation_history) > initial_size
    
    def test_get_resource_status(self):
        """Test getting resource status"""
        status = self.resource_planner.get_resource_status()
        
        assert 'compute' in status
        assert 'memory' in status
        assert 'training' in status
        assert status['compute']['total'] == 100
        assert 'allocated' in status['compute']
    
    def test_allocation_score_calculation(self):
        """Test allocation score calculation"""
        # Set performance metrics
        self.resource_planner.module_performance['test_mod'] = {
            'efficiency_score': 0.8,
            'performance_gain': 0.5,
            'utilization_rate': 0.7
        }
        
        # Calculate score
        score = self.resource_planner._calculate_allocation_score('test_mod', 'high')
        
        assert 0.0 <= score <= 1.0
    
    def test_reallocation_logic(self):
        """Test resource reallocation from low to high priority"""
        # Allocate to low priority
        self.resource_planner.compute_pool['available'] = 0  # Force reallocation
        self.resource_planner.compute_pool['allocated']['low_mod'] = 50
        
        # Request with higher priority
        self.message_bus.publish('resource_request', {
            'module_id': 'high_mod',
            'resource_type': 'compute',
            'amount_requested': 20,
            'priority': 'critical'
        })
        
        time.sleep(0.1)
        
        # High priority should get allocation
        status = self.resource_planner.get_resource_status()
        assert 'high_mod' in status['compute']['allocated']
    
    def test_dashboard_data(self):
        """Test dashboard data generation"""
        dashboard = self.resource_planner.get_dashboard_data()
        
        assert dashboard['module'] == 'resource_planner'
        assert dashboard['status'] == 'active'
        assert 'metrics' in dashboard
        assert 'compute_utilization' in dashboard['metrics']
        assert 'resource_status' in dashboard
    
    def test_multiple_resource_types(self):
        """Test allocation across multiple resource types"""
        # Request compute
        self.message_bus.publish('resource_request', {
            'module_id': 'multi_module',
            'resource_type': 'compute',
            'amount_requested': 10,
            'priority': 'high'
        })
        
        # Request memory
        self.message_bus.publish('resource_request', {
            'module_id': 'multi_module',
            'resource_type': 'memory',
            'amount_requested': 15,
            'priority': 'high'
        })
        
        time.sleep(0.1)
        
        status = self.resource_planner.get_resource_status()
        assert 'multi_module' in status['compute']['allocated']
        assert 'multi_module' in status['memory']['allocated']
    
    def test_efficiency_calculation(self):
        """Test efficiency calculation from performance metrics"""
        # Submit good performance
        self.message_bus.publish('performance_metric', {
            'module_id': 'efficient_module',
            'resource_consumed': 10,
            'performance_achieved': 2.0
        })
        
        time.sleep(0.1)
        
        perf = self.resource_planner.get_module_performance('efficient_module')
        assert perf['efficiency_score'] == 2.0 / 10
    
    def test_default_allocation_strategy(self):
        """Test default allocation strategy for known modules"""
        status = self.resource_planner.get_resource_status()
        
        # Strategy agent should get highest compute allocation
        assert 'strategy_agent' in status['compute']['allocated']
        
        # Strategic memory should get highest memory allocation
        assert 'strategic_memory' in status['memory']['allocated']
