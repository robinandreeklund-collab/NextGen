"""
action_chain_engine.py - Action chain-motor

Beskrivning:
    Definierar återanvändbara beslutskedjor.

Roll:
    - Tar emot chain_definition
    - Kör fördefinierade action chains
    - Publicerar chain_execution

Inputs:
    - chain_definition: Dict - Definition av action chain

Outputs:
    - chain_execution: Dict - Resultat av kedjan

Publicerar till message_bus:
    - chain_execution

Prenumererar på:
    - Ingen (från functions.yaml)

Använder RL: Nej
Tar emot feedback: Nej

Används i Sprint: 6
"""

from typing import Dict, Any, List, Callable
import time


class ActionChainEngine:
    """Hanterar återanvändbara beslutskedjor."""
    
    def __init__(self, message_bus):
        self.message_bus = message_bus
        self.chains: Dict[str, Any] = {}
        self.chain_executions: List[Dict[str, Any]] = []
        self.chain_templates: Dict[str, List[str]] = {}
        
        # Subscribe to chain definition requests
        self.message_bus.subscribe('chain_definition', self._on_chain_definition)
        self.message_bus.subscribe('execute_chain', self._on_execute_chain)
        
        # Define some standard chain templates
        self._initialize_standard_chains()
    
    def _initialize_standard_chains(self):
        """Initialize standard action chain templates."""
        # Standard trading chain: analyze → decide → execute
        self.chain_templates['standard_trade'] = [
            'indicator_analysis',
            'risk_assessment',
            'strategy_decision',
            'consensus_vote',
            'execution'
        ]
        
        # Risk-averse chain: extra risk checks
        self.chain_templates['risk_averse'] = [
            'indicator_analysis',
            'risk_assessment',
            'secondary_risk_check',
            'strategy_decision',
            'consensus_vote',
            'final_risk_verification',
            'execution'
        ]
        
        # Aggressive chain: faster decision path
        self.chain_templates['aggressive'] = [
            'indicator_analysis',
            'strategy_decision',
            'execution'
        ]
        
        # Analysis chain: no execution, just learning
        self.chain_templates['analysis_only'] = [
            'indicator_analysis',
            'risk_assessment',
            'strategy_decision',
            'simulation',
            'memory_storage'
        ]
    
    def _on_chain_definition(self, data: Dict[str, Any]):
        """Handle chain definition requests."""
        chain_id = data.get('chain_id')
        if not chain_id:
            return
        
        self.chains[chain_id] = {
            'definition': data,
            'created_at': time.time(),
            'executions': 0
        }
    
    def _on_execute_chain(self, data: Dict[str, Any]):
        """Execute a predefined action chain."""
        chain_name = data.get('chain_name', 'standard_trade')
        context = data.get('context', {})
        
        if chain_name in self.chain_templates:
            self._execute_template_chain(chain_name, context)
        elif chain_name in self.chains:
            self._execute_custom_chain(chain_name, context)
        else:
            # Default to standard trade chain
            self._execute_template_chain('standard_trade', context)
    
    def _execute_template_chain(self, chain_name: str, context: Dict[str, Any]):
        """Execute a template-based chain."""
        steps = self.chain_templates.get(chain_name, [])
        
        execution = {
            'chain_name': chain_name,
            'chain_type': 'template',
            'steps': steps,
            'context': context,
            'started_at': time.time(),
            'status': 'executing',
            'results': {}
        }
        
        # Publish chain execution event
        self.message_bus.publish('chain_execution', {
            'chain_name': chain_name,
            'steps': steps,
            'step_count': len(steps),
            'context': context,
            'timestamp': time.time()
        })
        
        execution['completed_at'] = time.time()
        execution['status'] = 'completed'
        execution['duration'] = execution['completed_at'] - execution['started_at']
        
        self.chain_executions.append(execution)
        
        # Keep only last 100 executions
        if len(self.chain_executions) > 100:
            self.chain_executions = self.chain_executions[-100:]
    
    def _execute_custom_chain(self, chain_id: str, context: Dict[str, Any]):
        """Execute a custom defined chain."""
        chain = self.chains.get(chain_id)
        if not chain:
            return
        
        chain['executions'] += 1
        
        execution = {
            'chain_id': chain_id,
            'chain_type': 'custom',
            'context': context,
            'started_at': time.time(),
            'status': 'executing'
        }
        
        # Publish chain execution event
        self.message_bus.publish('chain_execution', {
            'chain_id': chain_id,
            'chain_type': 'custom',
            'context': context,
            'timestamp': time.time()
        })
        
        execution['completed_at'] = time.time()
        execution['status'] = 'completed'
        execution['duration'] = execution['completed_at'] - execution['started_at']
        
        self.chain_executions.append(execution)
    
    def define_chain(self, chain_id: str, steps: List[str], metadata: Dict[str, Any] = None) -> bool:
        """
        Define a new action chain programmatically.
        
        Args:
            chain_id: Unique identifier for the chain
            steps: List of step names in the chain
            metadata: Optional metadata about the chain
        
        Returns:
            True if chain was defined successfully
        """
        if not chain_id or not steps:
            return False
        
        self.chains[chain_id] = {
            'definition': {
                'chain_id': chain_id,
                'steps': steps,
                'metadata': metadata or {}
            },
            'created_at': time.time(),
            'executions': 0
        }
        
        return True
    
    def execute_chain(self, chain_name: str, context: Dict[str, Any] = None) -> bool:
        """
        Execute a chain by name.
        
        Args:
            chain_name: Name of chain template or custom chain ID
            context: Execution context
        
        Returns:
            True if chain execution started successfully
        """
        context = context or {}
        
        if chain_name in self.chain_templates or chain_name in self.chains:
            self._on_execute_chain({
                'chain_name': chain_name,
                'context': context
            })
            return True
        
        return False
    
    def get_chain_statistics(self) -> Dict[str, Any]:
        """Get statistics about chain executions."""
        total_executions = len(self.chain_executions)
        
        # Count by chain type
        template_executions = sum(1 for e in self.chain_executions if e.get('chain_type') == 'template')
        custom_executions = sum(1 for e in self.chain_executions if e.get('chain_type') == 'custom')
        
        # Average duration
        durations = [e['duration'] for e in self.chain_executions if 'duration' in e]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        return {
            'total_chains_defined': len(self.chains),
            'total_templates': len(self.chain_templates),
            'total_executions': total_executions,
            'template_executions': template_executions,
            'custom_executions': custom_executions,
            'avg_execution_duration': avg_duration,
            'available_templates': list(self.chain_templates.keys()),
            'custom_chain_ids': list(self.chains.keys())
        }
    
    def get_chain_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent chain execution history.
        
        Args:
            limit: Maximum number of executions to return
        
        Returns:
            List of recent chain executions
        """
        return self.chain_executions[-limit:]

