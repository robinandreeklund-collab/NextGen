# test_timespan_tracker.py - Tester för tidsspann-spårare

import pytest
import time
from modules.message_bus import MessageBus
from modules.timespan_tracker import TimespanTracker


class TestTimespanTracker:
    """Tests for TimespanTracker module."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.message_bus = MessageBus()
        self.tracker = TimespanTracker(self.message_bus)
    
    def test_initialization(self):
        """Test tracker initialization."""
        assert self.tracker.message_bus is not None
        assert isinstance(self.tracker.timeline, list)
        assert len(self.tracker.timeline) == 0
        assert isinstance(self.tracker.indicator_history, dict)
    
    def test_decision_event_tracking(self):
        """Test that decision events are tracked in timeline."""
        decision_data = {
            'symbol': 'AAPL',
            'action': 'BUY',
            'confidence': 0.8
        }
        
        self.message_bus.publish('decision_vote', decision_data)  # Changed from decision_event
        time.sleep(0.01)  # Allow callback to process
        
        assert len(self.tracker.timeline) > 0
        assert len(self.tracker.decision_events) > 0
        assert self.tracker.timeline[0]['type'] == 'decision'
        assert self.tracker.timeline[0]['data'] == decision_data
    
    def test_indicator_data_tracking(self):
        """Test that indicator data is tracked by symbol."""
        indicator_data = {
            'symbol': 'AAPL',
            'rsi': 65,
            'sma': 150.5
        }
        
        self.message_bus.publish('indicator_data', indicator_data)
        time.sleep(0.01)
        
        assert 'AAPL' in self.tracker.indicator_history
        assert len(self.tracker.indicator_history['AAPL']) == 1
    
    def test_final_decision_tracking(self):
        """Test that final decisions are tracked."""
        final_decision = {
            'symbol': 'AAPL',
            'action': 'BUY',
            'quantity': 10
        }
        
        self.message_bus.publish('final_decision', final_decision)
        time.sleep(0.01)
        
        assert len(self.tracker.timeline) > 0
        final_events = [e for e in self.tracker.timeline if e['type'] == 'final_decision']
        assert len(final_events) == 1
    
    def test_timeline_analysis(self):
        """Test that timeline analysis generates insights."""
        insights_received = []
        
        def capture_insight(data):
            insights_received.append(data)
        
        self.message_bus.subscribe('timeline_insight', capture_insight)
        
        # Generate multiple decision events
        for i in range(3):
            self.message_bus.publish('decision_vote', {'id': i})  # Changed from decision_event
            time.sleep(0.01)
        
        # Should have generated timeline insights
        assert len(insights_received) > 0
        insight = insights_received[-1]
        assert 'avg_time_between_decisions' in insight
        assert 'decision_count' in insight
    
    def test_timeline_summary(self):
        """Test getting timeline summary."""
        # Add some events
        self.message_bus.publish('decision_vote', {'test': 1})  # Changed from decision_event
        self.message_bus.publish('final_decision', {'test': 2})
        self.message_bus.publish('indicator_data', {'symbol': 'AAPL', 'rsi': 50})
        time.sleep(0.01)
        
        summary = self.tracker.get_timeline_summary()
        assert 'total_events' in summary
        assert 'decision_events' in summary
        assert 'final_decisions' in summary
        assert 'AAPL' in summary['symbols_tracked']
    
    def test_get_decision_timeline_with_window(self):
        """Test getting decisions within time window."""
        # Add decision
        self.message_bus.publish('decision_vote', {'action': 'BUY'})  # Changed from decision_event
        time.sleep(0.01)
        
        # Get recent decisions (5 minute window)
        recent = self.tracker.get_decision_timeline(time_window=300)
        assert len(recent) > 0
        
        # Get very old decisions (should be empty)
        old = self.tracker.get_decision_timeline(time_window=0.001)
        assert len(old) == 0
    
    def test_get_indicator_timeline(self):
        """Test getting indicator timeline for a symbol."""
        # Add indicators
        for i in range(3):
            self.message_bus.publish('indicator_data', {
                'symbol': 'AAPL',
                'rsi': 50 + i,
                'index': i
            })
            time.sleep(0.01)
        
        timeline = self.tracker.get_indicator_timeline('AAPL', time_window=300)
        assert len(timeline) == 3
        
        # Test non-existent symbol
        empty = self.tracker.get_indicator_timeline('INVALID', time_window=300)
        assert len(empty) == 0
    
    def test_timeline_size_management(self):
        """Test that timeline doesn't grow unbounded."""
        # Add many events (more than 500)
        for i in range(550):
            self.message_bus.publish('final_decision', {'id': i})
            if i % 100 == 0:
                time.sleep(0.01)
        
        # Timeline should be capped at 500
        assert len(self.tracker.timeline) <= 500
    
    def test_indicator_history_size_management(self):
        """Test that indicator history per symbol is capped."""
        # Add many indicators for one symbol
        for i in range(150):
            self.message_bus.publish('indicator_data', {
                'symbol': 'AAPL',
                'index': i
            })
            if i % 50 == 0:
                time.sleep(0.01)
        
        # Should be capped at 100 per symbol
        assert len(self.tracker.indicator_history['AAPL']) <= 100
    
    def test_multiple_symbols_tracking(self):
        """Test tracking multiple symbols simultaneously."""
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        
        for symbol in symbols:
            self.message_bus.publish('indicator_data', {
                'symbol': symbol,
                'rsi': 50
            })
        
        time.sleep(0.01)
        
        for symbol in symbols:
            assert symbol in self.tracker.indicator_history
            assert len(self.tracker.indicator_history[symbol]) == 1

