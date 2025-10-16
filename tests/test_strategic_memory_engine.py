"""
Tester för strategic_memory_engine.py

Testar:
- Initialization
- Decision logging
- Indicator storage
- Execution result processing
- Correlation analysis
- Insight generation
- Feedback storage
"""

import pytest
from modules.strategic_memory_engine import StrategicMemoryEngine
from modules.message_bus import MessageBus


class TestStrategicMemoryEngine:
    """Tester för StrategicMemoryEngine."""
    
    def setup_method(self):
        """Initiera test environment."""
        self.message_bus = MessageBus()
        self.memory_engine = StrategicMemoryEngine(self.message_bus)
    
    def test_initialization(self):
        """Testar att memory engine initialiseras korrekt."""
        assert self.memory_engine is not None
        assert self.memory_engine.decision_history == []
        assert self.memory_engine.indicator_history == {}
        assert self.memory_engine.execution_history == []
        assert self.memory_engine.correlation_cache == {}
    
    def test_decision_logging(self):
        """Testar att beslut loggas korrekt."""
        decision = {
            'symbol': 'AAPL',
            'action': 'BUY',
            'quantity': 10,
            'price': 150.0
        }
        
        self.message_bus.publish('final_decision', decision)
        
        assert len(self.memory_engine.decision_history) == 1
        assert self.memory_engine.decision_history[0]['symbol'] == 'AAPL'
        assert 'timestamp' in self.memory_engine.decision_history[0]
        assert 'logged_at' in self.memory_engine.decision_history[0]
    
    def test_indicator_storage(self):
        """Testar att indikatorer lagras per symbol."""
        indicators1 = {
            'symbol': 'AAPL',
            'RSI': 65.0,
            'MACD': 0.5
        }
        
        indicators2 = {
            'symbol': 'GOOGL',
            'RSI': 30.0,
            'MACD': -0.3
        }
        
        self.message_bus.publish('indicator_data', indicators1)
        self.message_bus.publish('indicator_data', indicators2)
        
        assert 'AAPL' in self.memory_engine.indicator_history
        assert 'GOOGL' in self.memory_engine.indicator_history
        assert len(self.memory_engine.indicator_history['AAPL']) == 1
        assert len(self.memory_engine.indicator_history['GOOGL']) == 1
    
    def test_execution_result_processing(self):
        """Testar att execution results processas och kopplas till beslut."""
        # Publicera beslut först
        decision = {
            'symbol': 'AAPL',
            'action': 'BUY',
            'quantity': 10
        }
        self.message_bus.publish('final_decision', decision)
        
        # Publicera execution result
        execution = {
            'symbol': 'AAPL',
            'action': 'BUY',
            'success': True,
            'profit': 50.0
        }
        self.message_bus.publish('execution_result', execution)
        
        assert len(self.memory_engine.execution_history) == 1
        assert self.memory_engine.execution_history[0]['success'] is True
        
        # Verifiera att execution är kopplat till beslut
        assert 'execution_result' in self.memory_engine.decision_history[0]
    
    def test_correlation_analysis(self):
        """Testar att korrelationsanalys skapas mellan indikatorer och resultat."""
        # Publicera indikatorer
        indicators = {
            'symbol': 'AAPL',
            'RSI': 65.0,
            'MACD': 0.5
        }
        self.message_bus.publish('indicator_data', indicators)
        
        # Publicera execution
        execution = {
            'symbol': 'AAPL',
            'action': 'BUY',
            'success': True,
            'profit': 50.0
        }
        self.message_bus.publish('execution_result', execution)
        
        # Verifiera korrelationscache
        assert 'AAPL' in self.memory_engine.correlation_cache
        cache = self.memory_engine.correlation_cache['AAPL']
        assert cache['total_trades'] == 1
        assert cache['successful_trades'] == 1
        assert 'RSI' in cache['indicator_success']
        assert 'MACD' in cache['indicator_success']
    
    def test_insight_generation_basic(self):
        """Testar basic insight generation."""
        insights = self.memory_engine.generate_insights()
        
        assert 'total_decisions' in insights
        assert 'total_executions' in insights
        assert 'success_rate' in insights
        assert 'patterns' in insights
        assert 'recommendations' in insights
    
    def test_insight_generation_with_data(self):
        """Testar insight generation med faktisk data."""
        # Lägg till flera executions
        for i in range(10):
            execution = {
                'symbol': 'AAPL',
                'action': 'BUY',
                'success': i % 2 == 0,  # 50% success rate
                'profit': 10.0 if i % 2 == 0 else -5.0
            }
            self.message_bus.publish('execution_result', execution)
        
        insights = self.memory_engine.generate_insights()
        
        assert insights['total_executions'] == 10
        assert insights['success_rate'] == 0.5
        assert insights['average_profit'] == 2.5  # (5*10 + 5*(-5)) / 10
    
    def test_best_indicators_identification(self):
        """Testar att bästa indikatorer identifieras."""
        # Skapa correlation data
        for i in range(10):
            indicators = {
                'symbol': 'AAPL',
                'RSI': 65.0,
                'MACD': 0.5
            }
            self.message_bus.publish('indicator_data', indicators)
            
            execution = {
                'symbol': 'AAPL',
                'action': 'BUY',
                'success': True,  # Hög success rate
                'profit': 10.0
            }
            self.message_bus.publish('execution_result', execution)
        
        insights = self.memory_engine.generate_insights()
        
        assert len(insights['best_indicators']) > 0
        assert insights['best_indicators'][0]['success_rate'] > 0.6
    
    def test_get_decision_history(self):
        """Testar att decision history kan hämtas med limit."""
        # Lägg till 150 beslut
        for i in range(150):
            decision = {
                'symbol': 'AAPL',
                'action': 'BUY',
                'id': i
            }
            self.message_bus.publish('final_decision', decision)
        
        # Hämta senaste 100
        history = self.memory_engine.get_decision_history(limit=100)
        assert len(history) == 100
        assert history[-1]['id'] == 149  # Senaste
    
    def test_get_correlation_analysis(self):
        """Testar att korrelationsanalys kan hämtas."""
        # Skapa correlation data
        indicators = {
            'symbol': 'AAPL',
            'RSI': 65.0
        }
        self.message_bus.publish('indicator_data', indicators)
        
        execution = {
            'symbol': 'AAPL',
            'action': 'BUY',
            'success': True,
            'profit': 10.0
        }
        self.message_bus.publish('execution_result', execution)
        
        # Hämta för specifik symbol
        correlation = self.memory_engine.get_correlation_analysis('AAPL')
        assert correlation['total_trades'] == 1
        
        # Hämta alla
        all_correlations = self.memory_engine.get_correlation_analysis()
        assert 'AAPL' in all_correlations
    
    def test_performance_summary(self):
        """Testar att performance summary genereras."""
        # Lägg till data
        self.message_bus.publish('final_decision', {'symbol': 'AAPL'})
        self.message_bus.publish('execution_result', {'symbol': 'AAPL', 'success': True})
        self.message_bus.publish('feedback_event', {'source': 'test'})
        
        summary = self.memory_engine.get_performance_summary()
        
        assert summary['total_decisions'] == 1
        assert summary['total_executions'] == 1
        assert summary['feedback_events'] >= 1  # execution_result genererar också feedback
        # Sprint 4.2
        assert 'parameter_adjustments' in summary
    
    def test_parameter_adjustment_logging(self):
        """Testar att parameter adjustments loggas (Sprint 4.2)."""
        adjustment = {
            'adjusted_parameters': {
                'evolution_threshold': 0.3,
                'min_samples': 25
            },
            'reward_signals': {
                'agent_performance_gain': 0.5
            }
        }
        
        self.message_bus.publish('parameter_adjustment', adjustment)
        
        assert len(self.memory_engine.parameter_history) == 1
        assert self.memory_engine.parameter_history[0]['adjusted_parameters']['evolution_threshold'] == 0.3
    
    def test_parameter_context_in_decisions(self):
        """Testar att parameter context läggs till beslut (Sprint 4.2)."""
        # Lägg till beslut
        self.message_bus.publish('final_decision', {'symbol': 'AAPL', 'action': 'BUY'})
        
        # Lägg till parameter adjustment
        adjustment = {
            'adjusted_parameters': {
                'evolution_threshold': 0.3
            },
            'reward_signals': {
                'agent_performance_gain': 0.5
            }
        }
        self.message_bus.publish('parameter_adjustment', adjustment)
        
        # Senaste beslutet ska ha parameter context
        assert len(self.memory_engine.decision_history) > 0
        decision = self.memory_engine.decision_history[-1]
        assert 'parameter_context' in decision
        assert decision['parameter_context']['adjusted_parameters']['evolution_threshold'] == 0.3
    
    def test_get_parameter_history(self):
        """Testar hämtning av parameterhistorik (Sprint 4.2)."""
        # Simulera flera parameter adjustments
        for i in range(5):
            adjustment = {
                'adjusted_parameters': {
                    'evolution_threshold': 0.2 + (i * 0.02)
                }
            }
            self.message_bus.publish('parameter_adjustment', adjustment)
        
        history = self.memory_engine.get_parameter_history(limit=3)
        
        assert len(history) == 3  # Limited to 3
        assert all('adjusted_parameters' in entry for entry in history)

