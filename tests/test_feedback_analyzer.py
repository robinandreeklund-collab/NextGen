"""
test_feedback_analyzer.py - Tester för feedback-analys

Testar feedback_analyzer, feedback_router och introspection_panel för Sprint 3.

Testmål:
- Feedbackanalys identifierar mönster korrekt
- Router prioriterar och routar feedback
- Introspection panel genererar dashboard-data
- Performance pattern detection fungerar
- Indicator mismatch detection fungerar
- Agent drift detection fungerar
"""

import pytest
import sys
import os

# Lägg till projektroot till path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.message_bus import MessageBus
from modules.feedback_analyzer import FeedbackAnalyzer
from modules.feedback_router import FeedbackRouter
from modules.introspection_panel import IntrospectionPanel


class TestFeedbackAnalyzer:
    """Tester för FeedbackAnalyzer-modulen."""
    
    def setup_method(self):
        """Setup före varje test."""
        self.bus = MessageBus()
        self.analyzer = FeedbackAnalyzer(self.bus)
    
    def test_feedback_analyzer_initialization(self):
        """Testar att analyzer initialiseras korrekt."""
        assert self.analyzer.message_bus is not None
        assert len(self.analyzer.feedback_buffer) == 0
        assert self.analyzer.analysis_count == 0
    
    def test_feedback_reception(self):
        """Testar att analyzer tar emot feedback events."""
        feedback = {
            'source': 'execution_engine',
            'triggers': ['trade_result'],
            'data': {'success': True}
        }
        
        self.bus.publish('feedback_event', feedback)
        
        assert len(self.analyzer.feedback_buffer) == 1
        assert self.analyzer.feedback_buffer[0]['source'] == 'execution_engine'
    
    def test_performance_pattern_detection(self):
        """Testar detektering av performance patterns."""
        # Simulera flera slippage events
        for i in range(5):
            feedback = {
                'source': 'execution_engine',
                'triggers': ['slippage'],
                'data': {'slippage': 0.004}  # 0.4%
            }
            self.analyzer.feedback_buffer.append(feedback)
        
        patterns = self.analyzer.detect_performance_patterns()
        
        assert len(patterns) > 0
        assert any(p['type'] == 'high_slippage' for p in patterns)
    
    def test_trade_success_rate_pattern(self):
        """Testar detektering av trade success rate."""
        # Simulera trades med låg success rate
        for i in range(8):
            feedback = {
                'source': 'execution_engine',
                'triggers': ['trade_result'],
                'data': {'success': i < 3}  # 3/8 = 37.5% success
            }
            self.analyzer.feedback_buffer.append(feedback)
        
        patterns = self.analyzer.detect_performance_patterns()
        
        # Borde hitta både success_rate och low_success_rate patterns
        assert any(p['type'] == 'trade_success_rate' for p in patterns)
        assert any(p['type'] == 'low_success_rate' for p in patterns)
    
    def test_capital_change_pattern(self):
        """Testar detektering av capital change patterns."""
        # Simulera kapitalförändringar
        for i in range(5):
            feedback = {
                'source': 'portfolio_manager',
                'triggers': ['capital_change'],
                'data': {'change': 10.0 if i % 2 == 0 else -5.0}
            }
            self.analyzer.feedback_buffer.append(feedback)
        
        patterns = self.analyzer.detect_performance_patterns()
        
        assert any(p['type'] == 'avg_capital_change' for p in patterns)
    
    def test_agent_drift_detection(self):
        """Testar detektering av agent drift."""
        # Simulera performance degradation
        for i in range(15):
            feedback = {
                'source': 'rl_controller',
                'triggers': ['agent_update'],
                'data': {'performance': 0.8 if i < 7 else 0.5}  # Performance drops
            }
            self.analyzer.feedback_buffer.append(feedback)
        
        drift_cases = self.analyzer.detect_agent_drift()
        
        assert len(drift_cases) > 0
        assert drift_cases[0]['type'] == 'performance_degradation'
    
    def test_analyze_feedback_generates_insights(self):
        """Testar att analyze_feedback genererar insights."""
        # Lägg till tillräckligt med feedback för analys
        for i in range(12):
            feedback = {
                'source': 'execution_engine',
                'triggers': ['trade_result'],
                'data': {'success': True}
            }
            self.analyzer.feedback_buffer.append(feedback)
        
        insight = self.analyzer.analyze_feedback()
        
        assert 'timestamp' in insight
        assert 'patterns' in insight
        assert 'anomalies' in insight
        assert 'recommendations' in insight
        assert insight['samples_analyzed'] >= 10


class TestFeedbackRouter:
    """Tester för FeedbackRouter-modulen."""
    
    def setup_method(self):
        """Setup före varje test."""
        self.bus = MessageBus()
        self.router = FeedbackRouter(self.bus)
    
    def test_feedback_router_initialization(self):
        """Testar att router initialiseras korrekt."""
        assert self.router.message_bus is not None
        assert len(self.router.feedback_log) == 0
    
    def test_feedback_logging(self):
        """Testar att feedback loggas."""
        feedback = {
            'source': 'execution_engine',
            'triggers': ['slippage'],
            'data': {'slippage': 0.002}
        }
        
        # Använd route_feedback direkt istället för via message_bus
        self.router.route_feedback(feedback)
        
        assert len(self.router.feedback_log) >= 1
    
    def test_priority_calculation_critical(self):
        """Testar kritisk prioritet för stora förluster."""
        feedback = {
            'source': 'portfolio_manager',
            'triggers': ['capital_change'],
            'data': {'change': -150}
        }
        
        priority = self.router._calculate_priority(feedback)
        assert priority == 'critical'
    
    def test_priority_calculation_high(self):
        """Testar hög prioritet för slippage."""
        feedback = {
            'source': 'execution_engine',
            'triggers': ['slippage'],
            'data': {'slippage': 0.006}  # > 0.5%
        }
        
        priority = self.router._calculate_priority(feedback)
        assert priority == 'high'
    
    def test_priority_calculation_medium(self):
        """Testar medium prioritet för standard feedback."""
        feedback = {
            'source': 'execution_engine',
            'triggers': ['trade_result'],
            'data': {'success': True}
        }
        
        priority = self.router._calculate_priority(feedback)
        assert priority == 'medium'
    
    def test_feedback_enrichment(self):
        """Testar att feedback berikas med metadata."""
        feedback = {
            'source': 'execution_engine',
            'triggers': ['trade_result'],
            'data': {}
        }
        
        # Använd route_feedback direkt
        enriched = self.router.route_feedback(feedback)
        
        assert enriched is not None
        assert 'priority' in enriched
        assert 'routed_by' in enriched
        assert 'route_timestamp' in enriched


class TestIntrospectionPanel:
    """Tester för IntrospectionPanel-modulen."""
    
    def setup_method(self):
        """Setup före varje test."""
        self.bus = MessageBus()
        self.panel = IntrospectionPanel(self.bus)
    
    def test_introspection_panel_initialization(self):
        """Testar att panel initialiseras korrekt."""
        assert self.panel.message_bus is not None
        assert len(self.panel.agent_status_history) == 0
        assert len(self.panel.feedback_events) == 0
    
    def test_agent_status_reception(self):
        """Testar att panel tar emot agent status."""
        status = {
            'module': 'strategy_engine',
            'performance': 0.75,
            'reward': 10.5
        }
        
        self.bus.publish('agent_status', status)
        
        assert len(self.panel.agent_status_history) == 1
    
    def test_feedback_event_reception(self):
        """Testar att panel tar emot feedback events."""
        feedback = {
            'source': 'execution_engine',
            'triggers': ['trade_result']
        }
        
        self.bus.publish('feedback_event', feedback)
        
        assert len(self.panel.feedback_events) == 1
    
    def test_render_dashboard_structure(self):
        """Testar att dashboard data har rätt struktur."""
        dashboard = self.panel.render_dashboard()
        
        assert 'agent_status' in dashboard
        assert 'feedback_flow' in dashboard
        assert 'indicators' in dashboard
        assert 'reward_trends' in dashboard
        assert 'module_connections' in dashboard
        assert 'feedback_metrics' in dashboard
        assert 'agent_adaptation' in dashboard
    
    def test_reward_trends_extraction(self):
        """Testar extraktion av reward trends."""
        # Lägg till agent status med rewards
        for i in range(5):
            status = {
                'module': 'strategy_engine',
                'reward': 10.0 + i * 2,
                'timestamp': i
            }
            self.panel.agent_status_history.append(status)
        
        trends = self.panel._extract_reward_trends()
        
        assert len(trends) == 5
        assert all('reward' in t for t in trends)
        assert all('step' in t for t in trends)
    
    def test_feedback_metrics_calculation(self):
        """Testar beräkning av feedback metrics."""
        # Lägg till feedback events
        for i in range(10):
            event = {
                'source': 'execution_engine' if i % 2 == 0 else 'portfolio_manager',
                'priority': 'high' if i < 3 else 'medium',
                'timestamp': i * 60  # 1 minut mellan events
            }
            self.panel.feedback_events.append(event)
        
        metrics = self.panel._calculate_feedback_metrics()
        
        assert metrics['total_events'] == 10
        assert 'execution_engine' in metrics['by_source']
        assert 'portfolio_manager' in metrics['by_source']
        assert 'high' in metrics['by_priority']
        assert 'medium' in metrics['by_priority']
    
    def test_agent_adaptation_calculation(self):
        """Testar beräkning av agent adaptation."""
        # Lägg till performance data som förbättras
        for i in range(10):
            status = {
                'performance': 0.5 + (i * 0.03),  # Improving performance
                'timestamp': i
            }
            self.panel.agent_status_history.append(status)
        
        adaptation = self.panel._calculate_agent_adaptation()
        
        assert 'adaptation_rate' in adaptation
        assert 'performance_trend' in adaptation
        assert adaptation['performance_trend'] == 'improving'
    
    def test_module_connections_detection(self):
        """Testar detektering av modulkopplingar."""
        # Lägg till feedback från olika källor
        sources = ['execution_engine', 'portfolio_manager', 'execution_engine']
        for source in sources:
            event = {
                'source': source,
                'triggers': ['trade_result']
            }
            self.panel.feedback_events.append(event)
        
        connections = self.panel.get_module_connections()
        
        assert len(connections) > 0
        assert all('source' in c for c in connections)
        assert all('target' in c for c in connections)
        assert all('count' in c for c in connections)


class TestIntegratedFeedbackSystem:
    """Integration-tester för hela feedback-systemet."""
    
    def setup_method(self):
        """Setup före varje test."""
        self.bus = MessageBus()
        self.router = FeedbackRouter(self.bus)
        self.analyzer = FeedbackAnalyzer(self.bus)
        self.panel = IntrospectionPanel(self.bus)
    
    def test_end_to_end_feedback_flow(self):
        """Testar komplett feedback-flöde från source till analys."""
        # Simulera feedback från execution_engine
        feedback = {
            'source': 'execution_engine',
            'triggers': ['trade_result', 'slippage'],
            'data': {
                'success': True,
                'slippage': 0.003
            }
        }
        
        # Publicera feedback direkt till message_bus (inte via router för att undvika loop)
        self.bus.publish('feedback_event', feedback)
        
        # Verifiera att modulerna tog emot feedback
        # Router loggar via callback
        assert len(self.router.feedback_log) >= 1
        # Analyzer tar emot via callback
        assert len(self.analyzer.feedback_buffer) >= 1
        # Panel tar emot via callback
        assert len(self.panel.feedback_events) >= 1
    
    def test_feedback_analysis_triggers_insights(self):
        """Testar att feedback-analys genererar insights."""
        # Lägg till många feedback events för att trigga analys
        for i in range(15):
            feedback = {
                'source': 'execution_engine',
                'triggers': ['trade_result'],
                'data': {'success': i % 3 == 0}
            }
            self.bus.publish('feedback_event', feedback)
        
        # Analyzer borde ha analyserat och publicerat insights
        log = self.bus.get_message_log()
        insights = [msg for msg in log if msg['topic'] == 'feedback_insight']
        
        assert len(insights) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

