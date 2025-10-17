"""
Resource Planner Module - Sprint 7
Hanterar resursallokering mellan agenter och moduler
"""

import time
from typing import Dict, List, Optional, Any
from collections import defaultdict


class ResourcePlanner:
    """Planerar och allokerar resurser mellan moduler och agenter"""
    
    def __init__(self, message_bus):
        self.message_bus = message_bus
        
        # Resource pools
        self.compute_pool = {
            'total': 100,
            'available': 100,
            'allocated': {}
        }
        
        self.memory_pool = {
            'total': 100,
            'available': 100,
            'allocated': {}
        }
        
        self.training_budget = {
            'total': 100,
            'available': 100,
            'allocated': {}
        }
        
        # Resource allocation history
        self.allocation_history = []
        
        # Performance tracking
        self.module_performance = defaultdict(lambda: {
            'efficiency_score': 1.0,
            'utilization_rate': 0.0,
            'performance_gain': 0.0
        })
        
        # Resource requests queue
        self.pending_requests = []
        
        # Priority weights
        self.priority_weights = {
            'critical': 1.0,
            'high': 0.75,
            'medium': 0.5,
            'low': 0.25
        }
        
        # Default allocations based on Sprint 7 evolution_matrix
        self.default_allocations = {
            'compute': {
                'strategy_agent': 0.25,
                'risk_agent': 0.20,
                'decision_agent': 0.20,
                'execution_agent': 0.15,
                'meta_parameter_agent': 0.10,
                'reward_tuner_agent': 0.10
            },
            'memory': {
                'strategic_memory': 0.30,
                'indicator_registry': 0.20,
                'introspection_panel': 0.20,
                'system_monitor': 0.15,
                'resource_planner': 0.10,
                'team_dynamics': 0.05
            }
        }
        
        # Initialize default allocations
        self._initialize_default_allocations()
        
        # Subscribe to relevant topics
        self.message_bus.subscribe('resource_request', self.handle_resource_request)
        self.message_bus.subscribe('performance_metric', self.handle_performance_metric)
        self.message_bus.subscribe('module_completed', self.release_resources)
        
        print("[ResourcePlanner] Initialized with compute:100, memory:100, training:100")
    
    def _initialize_default_allocations(self):
        """Initialize default resource allocations for known modules"""
        for module_id, allocation in self.default_allocations['compute'].items():
            amount = int(self.compute_pool['total'] * allocation)
            if amount > 0:
                self._allocate_from_pool(self.compute_pool, module_id, amount)
        
        for module_id, allocation in self.default_allocations['memory'].items():
            amount = int(self.memory_pool['total'] * allocation)
            if amount > 0:
                self._allocate_from_pool(self.memory_pool, module_id, amount)
    
    def handle_resource_request(self, data: Dict[str, Any]):
        """Handle resource request from a module"""
        module_id = data.get('module_id', 'unknown')
        resource_type = data.get('resource_type', 'compute')
        amount = data.get('amount_requested', 10)
        priority = data.get('priority', 'medium')
        
        # Add to pending requests
        request = {
            'module_id': module_id,
            'resource_type': resource_type,
            'amount': amount,
            'priority': priority,
            'timestamp': time.time()
        }
        self.pending_requests.append(request)
        
        # Process request immediately
        self._process_request(request)
    
    def _process_request(self, request: Dict[str, Any]):
        """Process a resource request"""
        module_id = request['module_id']
        resource_type = request['resource_type']
        amount = request['amount']
        priority = request['priority']
        
        # Select appropriate pool
        pool = self._get_pool(resource_type)
        if not pool:
            return
        
        # Calculate allocation score
        allocation_score = self._calculate_allocation_score(module_id, priority)
        
        # Check if we can allocate
        if pool['available'] >= amount:
            allocated = self._allocate_from_pool(pool, module_id, amount)
            
            # Publish allocation
            self.message_bus.publish('resource_allocation', {
                'module_id': module_id,
                'resource_type': resource_type,
                'allocated_amount': allocated,
                'allocation_score': allocation_score,
                'timestamp': time.time()
            })
            
            # Log allocation
            self._log_allocation(module_id, resource_type, allocated, allocation_score)
        else:
            # Try to reallocate from low-priority modules
            self._attempt_reallocation(pool, module_id, amount, priority)
    
    def _get_pool(self, resource_type: str) -> Optional[Dict]:
        """Get resource pool by type"""
        if resource_type == 'compute':
            return self.compute_pool
        elif resource_type == 'memory':
            return self.memory_pool
        elif resource_type == 'training':
            return self.training_budget
        return None
    
    def _calculate_allocation_score(self, module_id: str, priority: str) -> float:
        """Calculate allocation score for a module"""
        perf = self.module_performance[module_id]
        
        # Weighted factors
        priority_score = self.priority_weights.get(priority, 0.5)
        efficiency_score = perf['efficiency_score']
        performance_score = max(0.0, min(1.0, perf['performance_gain'] + 1.0))
        
        # Weighted sum
        score = (
            priority_score * 0.40 +
            efficiency_score * 0.35 +
            performance_score * 0.25
        )
        
        return score
    
    def _allocate_from_pool(self, pool: Dict, module_id: str, amount: int) -> int:
        """Allocate resources from a pool"""
        allocated = min(amount, pool['available'])
        pool['available'] -= allocated
        pool['allocated'][module_id] = pool['allocated'].get(module_id, 0) + allocated
        return allocated
    
    def _attempt_reallocation(self, pool: Dict, module_id: str, amount: int, priority: str):
        """Attempt to reallocate resources from lower priority modules"""
        # Find modules with lower priority
        candidates = []
        for mod_id, allocated in pool['allocated'].items():
            mod_score = self._calculate_allocation_score(mod_id, 'low')
            req_score = self._calculate_allocation_score(module_id, priority)
            
            if mod_score < req_score and allocated >= amount:
                candidates.append((mod_id, allocated, mod_score))
        
        # Sort by score (lowest first)
        candidates.sort(key=lambda x: x[2])
        
        if candidates:
            # Take from lowest priority module
            source_module, allocated, _ = candidates[0]
            pool['allocated'][source_module] -= amount
            pool['allocated'][module_id] = pool['allocated'].get(module_id, 0) + amount
            
            # Publish reallocation
            self.message_bus.publish('resource_reallocation', {
                'from_module': source_module,
                'to_module': module_id,
                'amount': amount,
                'timestamp': time.time()
            })
    
    def handle_performance_metric(self, data: Dict[str, Any]):
        """Update module performance metrics"""
        module_id = data.get('module_id', 'unknown')
        resource_consumed = data.get('resource_consumed', 0)
        performance_achieved = data.get('performance_achieved', 0.0)
        
        # Calculate efficiency
        efficiency = performance_achieved / max(resource_consumed, 1)
        
        # Update performance tracking
        perf = self.module_performance[module_id]
        perf['efficiency_score'] = efficiency
        perf['performance_gain'] = performance_achieved
        
        # Calculate utilization
        total_allocated = sum([
            self.compute_pool['allocated'].get(module_id, 0),
            self.memory_pool['allocated'].get(module_id, 0),
            self.training_budget['allocated'].get(module_id, 0)
        ])
        perf['utilization_rate'] = resource_consumed / max(total_allocated, 1)
    
    def release_resources(self, data: Dict[str, Any]):
        """Release resources when module completes"""
        module_id = data.get('module_id', 'unknown')
        
        # Release from all pools
        for pool in [self.compute_pool, self.memory_pool, self.training_budget]:
            if module_id in pool['allocated']:
                amount = pool['allocated'][module_id]
                pool['available'] += amount
                del pool['allocated'][module_id]
    
    def _log_allocation(self, module_id: str, resource_type: str, amount: int, score: float):
        """Log allocation to history"""
        entry = {
            'timestamp': time.time(),
            'module_id': module_id,
            'resource_type': resource_type,
            'amount': amount,
            'allocation_score': score
        }
        self.allocation_history.append(entry)
        
        # Keep last 1000 entries
        if len(self.allocation_history) > 1000:
            self.allocation_history = self.allocation_history[-1000:]
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource status"""
        return {
            'compute': {
                'total': self.compute_pool['total'],
                'available': self.compute_pool['available'],
                'allocated': dict(self.compute_pool['allocated'])
            },
            'memory': {
                'total': self.memory_pool['total'],
                'available': self.memory_pool['available'],
                'allocated': dict(self.memory_pool['allocated'])
            },
            'training': {
                'total': self.training_budget['total'],
                'available': self.training_budget['available'],
                'allocated': dict(self.training_budget['allocated'])
            },
            'pending_requests': len(self.pending_requests),
            'allocation_history_size': len(self.allocation_history)
        }
    
    def get_module_performance(self, module_id: Optional[str] = None) -> Dict[str, Any]:
        """Get performance metrics for a module or all modules"""
        if module_id:
            return dict(self.module_performance.get(module_id, {}))
        return {k: dict(v) for k, v in self.module_performance.items()}
    
    def get_allocation_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get allocation history"""
        return self.allocation_history[-limit:]
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get dashboard data for introspection panel"""
        status = self.get_resource_status()
        
        # Calculate utilization rates
        compute_util = 1.0 - (status['compute']['available'] / status['compute']['total'])
        memory_util = 1.0 - (status['memory']['available'] / status['memory']['total'])
        training_util = 1.0 - (status['training']['available'] / status['training']['total'])
        
        # Get efficiency scores
        avg_efficiency = 0.0
        if self.module_performance:
            avg_efficiency = sum(p['efficiency_score'] for p in self.module_performance.values()) / len(self.module_performance)
        
        return {
            'module': 'resource_planner',
            'status': 'active',
            'metrics': {
                'compute_utilization': compute_util,
                'memory_utilization': memory_util,
                'training_utilization': training_util,
                'avg_efficiency': avg_efficiency,
                'pending_requests': len(self.pending_requests),
                'active_modules': len(self.module_performance)
            },
            'resource_status': status,
            'timestamp': time.time()
        }
