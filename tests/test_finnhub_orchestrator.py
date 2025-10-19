"""
Test suite for Finnhub Orchestrator and its submodules
"""

import pytest
import time
from modules.message_bus import MessageBus
from modules.finnhub_orchestrator import FinnhubOrchestrator
from modules.indicator_synth_engine import IndicatorSynthEngine
from modules.symbol_rotation_engine import SymbolRotationEngine
from modules.rotation_strategy_engine import RotationStrategyEngine
from modules.stream_strategy_agent import StreamStrategyAgent
from modules.stream_replay_engine import StreamReplayEngine
from modules.stream_ontology_mapper import StreamOntologyMapper


class TestIndicatorSynthEngine:
    """Tests for IndicatorSynthEngine"""
    
    def test_initialization(self):
        """Test engine initialization"""
        bus = MessageBus()
        engine = IndicatorSynthEngine(bus)
        
        assert engine.message_bus == bus
        assert len(engine.recipes) > 0
    
    def test_synthesize_indicators(self):
        """Test indicator synthesis"""
        bus = MessageBus()
        engine = IndicatorSynthEngine(bus)
        
        # Update cache with mock data
        engine.update_indicator_cache('AAPL', {
            'RSI': 65.0,
            'MACD': 1.5,
            'Stochastic': 70.0,
            'ATR': 2.3,
            'Bollinger_Width': 0.15
        })
        
        # Synthesize
        result = engine.synthesize(['AAPL'])
        
        assert 'symbols' in result
        assert 'AAPL' in result['symbols']
        assert 'momentum_composite' in result['symbols']['AAPL']
    
    def test_add_custom_recipe(self):
        """Test adding custom recipe"""
        bus = MessageBus()
        engine = IndicatorSynthEngine(bus)
        
        recipe = {
            'inputs': ['RSI', 'MACD'],
            'weights': [0.5, 0.5],
            'operation': 'weighted_average'
        }
        
        engine.add_recipe('custom_indicator', recipe)
        
        assert 'custom_indicator' in engine.get_available_recipes()


class TestSymbolRotationEngine:
    """Tests for SymbolRotationEngine"""
    
    def test_initialization(self):
        """Test engine initialization"""
        bus = MessageBus()
        engine = SymbolRotationEngine(bus, rotation_interval=300)
        
        assert engine.rotation_interval == 300
        assert len(engine.symbol_pool) > 0
    
    def test_rotate_by_priority(self):
        """Test priority-based rotation"""
        bus = MessageBus()
        engine = SymbolRotationEngine(bus)
        
        current = ['AAPL', 'TSLA', 'MSFT']
        priorities = {'AAPL': 0.8, 'TSLA': 0.3, 'MSFT': 0.5}
        strategy = {'type': 'top_priority', 'rotation_rate': 0.3}
        
        new_symbols = engine.rotate_symbols(current, priorities, strategy, max_symbols=5)
        
        assert len(new_symbols) > 0
        assert 'AAPL' in new_symbols  # High priority should be kept
    
    def test_rotate_random(self):
        """Test random rotation"""
        bus = MessageBus()
        engine = SymbolRotationEngine(bus)
        
        current = ['AAPL', 'TSLA', 'MSFT']
        priorities = {'AAPL': 0.5, 'TSLA': 0.5, 'MSFT': 0.5}
        strategy = {'type': 'random', 'rotation_rate': 0.5}
        
        new_symbols = engine.rotate_symbols(current, priorities, strategy, max_symbols=5)
        
        assert len(new_symbols) > 0
    
    def test_add_symbols_to_pool(self):
        """Test adding symbols to pool"""
        bus = MessageBus()
        engine = SymbolRotationEngine(bus)
        
        initial_count = len(engine.symbol_pool)
        engine.add_symbols_to_pool(['TEST1', 'TEST2'])
        
        assert len(engine.symbol_pool) == initial_count + 2


class TestRotationStrategyEngine:
    """Tests for RotationStrategyEngine"""
    
    def test_initialization(self):
        """Test engine initialization"""
        bus = MessageBus()
        engine = RotationStrategyEngine(bus)
        
        assert engine.current_strategy['type'] == 'top_priority'
    
    def test_compute_strategy(self):
        """Test strategy computation"""
        bus = MessageBus()
        engine = RotationStrategyEngine(bus)
        
        priorities = {'AAPL': 0.6, 'TSLA': 0.4}
        metrics = {}
        current = ['AAPL', 'TSLA']
        
        strategy = engine.compute_rotation_strategy(priorities, metrics, current)
        
        assert 'type' in strategy
        assert 'rotation_rate' in strategy
    
    def test_process_feedback(self):
        """Test feedback processing"""
        bus = MessageBus()
        engine = RotationStrategyEngine(bus)
        
        feedback = {
            'strategy': 'top_priority',
            'reward': 0.5
        }
        
        engine.process_feedback(feedback)
        
        assert len(engine.feedback_buffer) == 1


class TestStreamStrategyAgent:
    """Tests for StreamStrategyAgent"""
    
    def test_initialization(self):
        """Test agent initialization"""
        bus = MessageBus()
        agent = StreamStrategyAgent(bus)
        
        assert agent.model is not None
    
    def test_get_symbol_scores(self):
        """Test symbol scoring"""
        bus = MessageBus()
        agent = StreamStrategyAgent(bus)
        
        symbols = ['AAPL', 'TSLA']
        metrics = {}
        
        scores = agent.get_symbol_scores(symbols, metrics)
        
        assert len(scores) == 2
        assert 'AAPL' in scores
        assert 0.0 <= scores['AAPL'] <= 1.0
    
    def test_update_strategy(self):
        """Test strategy update"""
        bus = MessageBus()
        agent = StreamStrategyAgent(bus)
        
        metrics = {}
        priorities = {'AAPL': 0.7, 'TSLA': 0.3}
        
        strategy = agent.update_strategy(metrics, priorities)
        
        assert 'batch_size' in strategy
        assert 'resource_allocation' in strategy


class TestStreamReplayEngine:
    """Tests for StreamReplayEngine"""
    
    def test_initialization(self):
        """Test engine initialization"""
        bus = MessageBus()
        engine = StreamReplayEngine(bus)
        
        assert not engine.is_replaying
        assert engine.replay_speed == 1.0
    
    def test_start_stop_replay(self):
        """Test starting and stopping replay"""
        bus = MessageBus()
        engine = StreamReplayEngine(bus)
        
        config = {
            'mode': 'synthetic',
            'speed': 2.0,
            'symbols': ['AAPL']
        }
        
        engine.start_replay(config)
        assert engine.is_replaying
        
        time.sleep(0.5)
        
        engine.stop_replay()
        assert not engine.is_replaying
    
    def test_replay_speed_adjustment(self):
        """Test replay speed adjustment"""
        bus = MessageBus()
        engine = StreamReplayEngine(bus)
        
        engine.set_replay_speed(3.0)
        assert engine.replay_speed == 3.0


class TestStreamOntologyMapper:
    """Tests for StreamOntologyMapper"""
    
    def test_initialization(self):
        """Test mapper initialization"""
        bus = MessageBus()
        mapper = StreamOntologyMapper(bus)
        
        assert len(mapper.mapping_rules) > 0
    
    def test_map_finnhub_data(self):
        """Test mapping Finnhub data"""
        bus = MessageBus()
        mapper = StreamOntologyMapper(bus)
        
        raw_data = {
            'p': 150.5,
            's': 'AAPL',
            't': 1234567890000,
            'v': 1000000
        }
        
        mapped = mapper.map_data(raw_data, 'finnhub')
        
        assert mapped is not None
        assert mapped['price'] == 150.5
        assert mapped['symbol'] == 'AAPL'
    
    def test_batch_mapping(self):
        """Test batch data mapping"""
        bus = MessageBus()
        mapper = StreamOntologyMapper(bus)
        
        raw_data_list = [
            {'p': 150.5, 's': 'AAPL', 't': 1234567890000, 'v': 1000000},
            {'p': 200.0, 's': 'TSLA', 't': 1234567890000, 'v': 2000000}
        ]
        
        mapped_list = mapper.batch_map(raw_data_list, 'finnhub')
        
        assert len(mapped_list) == 2


class TestFinnhubOrchestrator:
    """Tests for FinnhubOrchestrator"""
    
    def test_initialization(self):
        """Test orchestrator initialization"""
        bus = MessageBus()
        orchestrator = FinnhubOrchestrator(
            api_key='test_key',
            message_bus=bus,
            live_mode=False
        )
        
        assert orchestrator.api_key == 'test_key'
        assert not orchestrator.live_mode
        assert orchestrator.message_bus == bus
    
    def test_nasdaq_symbols_loading(self):
        """Test that NASDAQ 100 symbols are loaded"""
        bus = MessageBus()
        orchestrator = FinnhubOrchestrator(
            api_key='test_key',
            message_bus=bus,
            live_mode=False
        )
        
        # Check that default_symbols has been populated
        default_symbols = orchestrator.config.get('default_symbols', [])
        assert len(default_symbols) > 0
        
        # If NASDAQ file exists, should have many symbols
        # Otherwise falls back to 7 default symbols
        assert len(default_symbols) >= 7
    
    def test_detailed_metrics_tracking(self):
        """Test detailed metrics tracking"""
        bus = MessageBus()
        orchestrator = FinnhubOrchestrator(
            api_key='test_key',
            message_bus=bus,
            live_mode=False
        )
        
        orchestrator.start()
        time.sleep(1)
        
        # Check that detailed metrics are being tracked
        assert 'active_subscriptions' in orchestrator.detailed_metrics
        assert 'websocket_usage_pct' in orchestrator.detailed_metrics
        assert 'historical_symbols_used' in orchestrator.detailed_metrics
        assert 'submodule_details' in orchestrator.detailed_metrics
        
        orchestrator.stop()
    
    def test_start_stop(self):
        """Test starting and stopping orchestrator"""
        bus = MessageBus()
        orchestrator = FinnhubOrchestrator(
            api_key='test_key',
            message_bus=bus,
            live_mode=False
        )
        
        orchestrator.start()
        assert orchestrator.is_running
        assert len(orchestrator.active_symbols) > 0
        
        time.sleep(1)
        
        orchestrator.stop()
        assert not orchestrator.is_running
    
    def test_config_update(self):
        """Test dynamic configuration update"""
        bus = MessageBus()
        orchestrator = FinnhubOrchestrator(
            api_key='test_key',
            message_bus=bus
        )
        
        new_config = {
            'rotation_interval': 600,
            'max_concurrent_streams': 15
        }
        
        orchestrator.update_config(new_config)
        
        assert orchestrator.config['rotation_interval'] == 600
        assert orchestrator.config['max_concurrent_streams'] == 15
    
    def test_get_status(self):
        """Test getting orchestrator status"""
        bus = MessageBus()
        orchestrator = FinnhubOrchestrator(
            api_key='test_key',
            message_bus=bus
        )
        
        status = orchestrator.get_status()
        
        assert 'is_running' in status
        assert 'active_symbols' in status
        assert 'priorities' in status
    
    def test_replay_mode(self):
        """Test enabling/disabling replay mode"""
        bus = MessageBus()
        orchestrator = FinnhubOrchestrator(
            api_key='test_key',
            message_bus=bus
        )
        
        replay_config = {
            'mode': 'synthetic',
            'speed': 2.0
        }
        
        orchestrator.enable_replay_mode(replay_config)
        time.sleep(0.5)
        orchestrator.disable_replay_mode()
    
    def test_message_bus_integration(self):
        """Test message bus integration"""
        bus = MessageBus()
        orchestrator = FinnhubOrchestrator(
            api_key='test_key',
            message_bus=bus
        )
        
        # Track published messages
        published_messages = []
        
        def capture_message(data):
            published_messages.append(data)
        
        bus.subscribe('orchestrator_status', capture_message)
        
        orchestrator.start()
        time.sleep(1)
        orchestrator.stop()
        
        # Should have published at least one status message
        assert len(published_messages) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
