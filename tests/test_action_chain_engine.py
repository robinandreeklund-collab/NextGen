# test_action_chain_engine.py - Tester för åtgärdskedja-motor

import pytest
import time
from modules.message_bus import MessageBus
from modules.action_chain_engine import ActionChainEngine


class TestActionChainEngine:
    """Tests for ActionChainEngine module."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.message_bus = MessageBus()
        self.engine = ActionChainEngine(self.message_bus)
    
    def test_initialization(self):
        """Test engine initialization."""
        assert self.engine.message_bus is not None
        assert isinstance(self.engine.chains, dict)
        assert isinstance(self.engine.chain_templates, dict)
        assert len(self.engine.chain_templates) > 0  # Should have standard templates
    
    def test_standard_chain_templates_exist(self):
        """Test that standard chain templates are initialized."""
        expected_templates = ['standard_trade', 'risk_averse', 'aggressive', 'analysis_only']
        
        for template in expected_templates:
            assert template in self.engine.chain_templates
            assert isinstance(self.engine.chain_templates[template], list)
            assert len(self.engine.chain_templates[template]) > 0
    
    def test_define_custom_chain(self):
        """Test defining a custom chain."""
        chain_id = 'test_chain'
        steps = ['step1', 'step2', 'step3']
        metadata = {'description': 'Test chain'}
        
        result = self.engine.define_chain(chain_id, steps, metadata)
        
        assert result is True
        assert chain_id in self.engine.chains
        assert self.engine.chains[chain_id]['definition']['steps'] == steps
    
    def test_define_chain_validation(self):
        """Test chain definition validation."""
        # Empty chain_id should fail
        result = self.engine.define_chain('', ['step1'])
        assert result is False
        
        # Empty steps should fail
        result = self.engine.define_chain('test', [])
        assert result is False
    
    def test_execute_template_chain(self):
        """Test executing a template-based chain."""
        executions_received = []
        
        def capture_execution(data):
            executions_received.append(data)
        
        self.message_bus.subscribe('chain_execution', capture_execution)
        
        context = {'symbol': 'AAPL', 'action': 'BUY'}
        result = self.engine.execute_chain('standard_trade', context)
        
        time.sleep(0.01)
        
        assert result is True
        assert len(executions_received) > 0
        assert executions_received[0]['chain_name'] == 'standard_trade'
        assert len(self.engine.chain_executions) > 0
    
    def test_execute_custom_chain(self):
        """Test executing a custom chain."""
        # Define custom chain
        chain_id = 'custom_test'
        steps = ['custom_step1', 'custom_step2']
        self.engine.define_chain(chain_id, steps)
        
        executions_received = []
        self.message_bus.subscribe('chain_execution', lambda d: executions_received.append(d))
        
        result = self.engine.execute_chain(chain_id, {'test': True})
        time.sleep(0.01)
        
        assert result is True
        assert len(executions_received) > 0
        assert self.engine.chains[chain_id]['executions'] == 1
    
    def test_execute_nonexistent_chain_falls_back(self):
        """Test that executing non-existent chain returns False."""
        executions_received = []
        self.message_bus.subscribe('chain_execution', lambda d: executions_received.append(d))
        
        result = self.engine.execute_chain('nonexistent_chain', {})
        time.sleep(0.01)
        
        # Should return False for non-existent chain
        assert result is False
    
    def test_chain_definition_via_message_bus(self):
        """Test defining chain via message bus."""
        chain_data = {
            'chain_id': 'message_chain',
            'steps': ['step1', 'step2']
        }
        
        self.message_bus.publish('chain_definition', chain_data)
        time.sleep(0.01)
        
        assert 'message_chain' in self.engine.chains
    
    def test_execute_chain_via_message_bus(self):
        """Test executing chain via message bus."""
        executions_received = []
        self.message_bus.subscribe('chain_execution', lambda d: executions_received.append(d))
        
        self.message_bus.publish('execute_chain', {
            'chain_name': 'aggressive',
            'context': {'symbol': 'TSLA'}
        })
        time.sleep(0.01)
        
        assert len(executions_received) > 0
        assert executions_received[0]['chain_name'] == 'aggressive'
    
    def test_get_chain_statistics(self):
        """Test getting chain statistics."""
        # Execute some chains
        self.engine.execute_chain('standard_trade', {})
        self.engine.execute_chain('aggressive', {})
        time.sleep(0.01)
        
        stats = self.engine.get_chain_statistics()
        
        assert 'total_templates' in stats
        assert 'total_executions' in stats
        assert 'avg_execution_duration' in stats
        assert 'available_templates' in stats
        assert stats['total_executions'] >= 2
    
    def test_get_chain_history(self):
        """Test getting chain execution history."""
        # Execute chains
        for i in range(3):
            self.engine.execute_chain('standard_trade', {'index': i})
        
        time.sleep(0.01)
        
        history = self.engine.get_chain_history(limit=10)
        assert len(history) >= 3
        assert all('started_at' in e for e in history)
    
    def test_chain_execution_tracking(self):
        """Test that chain executions are properly tracked."""
        initial_count = len(self.engine.chain_executions)
        
        self.engine.execute_chain('risk_averse', {})
        time.sleep(0.01)
        
        assert len(self.engine.chain_executions) == initial_count + 1
        latest = self.engine.chain_executions[-1]
        assert 'started_at' in latest
        assert 'completed_at' in latest
        assert 'duration' in latest
        assert latest['status'] == 'completed'
    
    def test_chain_execution_history_size_limit(self):
        """Test that execution history is limited to prevent unbounded growth."""
        # Execute many chains
        for i in range(150):
            self.engine.execute_chain('aggressive', {'index': i})
            if i % 50 == 0:
                time.sleep(0.01)
        
        # Should be capped at 100
        assert len(self.engine.chain_executions) <= 100
    
    def test_different_chain_templates(self):
        """Test executing different chain templates."""
        templates = ['standard_trade', 'risk_averse', 'aggressive', 'analysis_only']
        
        for template in templates:
            result = self.engine.execute_chain(template, {'test': template})
            assert result is True
        
        time.sleep(0.01)
        assert len(self.engine.chain_executions) >= len(templates)
    
    def test_chain_context_preservation(self):
        """Test that context is preserved in chain execution."""
        executions_received = []
        self.message_bus.subscribe('chain_execution', lambda d: executions_received.append(d))
        
        context = {
            'symbol': 'AAPL',
            'action': 'BUY',
            'quantity': 100,
            'custom_field': 'test_value'
        }
        
        self.engine.execute_chain('standard_trade', context)
        time.sleep(0.01)
        
        assert len(executions_received) > 0
        assert executions_received[0]['context'] == context

