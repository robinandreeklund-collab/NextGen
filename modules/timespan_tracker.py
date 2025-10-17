"""
timespan_tracker.py - Tidsspårare

Beskrivning:
    Synkroniserar beslut och indikatorer över tid.

Roll:
    - Spårar beslut och events över tidsspann
    - Synkroniserar olika tidsskalor
    - Publicerar timeline_insight

Inputs:
    - decision_vote: Dict - Beslutshändelser från decision_engine
    - indicator_data: Dict - Indikatordata över tid
    - final_decision: Dict - Slutgiltiga beslut

Outputs:
    - timeline_insight: Dict - Tidssynkroniserade insights

Publicerar till message_bus:
    - timeline_insight

Prenumererar på:
    - decision_vote (from decision_engine)
    - indicator_data (from indicator_registry)
    - final_decision (from consensus_engine)

Använder RL: Nej
Tar emot feedback: Nej

Används i Sprint: 6
"""

from typing import Dict, Any
import time


class TimespanTracker:
    """Spårar och synkroniserar events över tid."""
    
    def __init__(self, message_bus):
        self.message_bus = message_bus
        self.timeline: list = []
        self.decision_events: list = []
        self.indicator_history: Dict[str, list] = {}
        
        # Subscribe to relevant topics
        self.message_bus.subscribe('decision_vote', self._on_decision_event)  # decision_vote from decision_engine
        self.message_bus.subscribe('indicator_data', self._on_indicator_data)
        self.message_bus.subscribe('final_decision', self._on_final_decision)
    
    def _on_decision_event(self, data: Dict[str, Any]):
        """Handle decision events and add to timeline."""
        event = {
            'timestamp': time.time(),
            'type': 'decision',
            'data': data
        }
        self.timeline.append(event)
        self.decision_events.append(event)
        
        # Trigger timeline analysis
        self._analyze_timeline()
    
    def _on_indicator_data(self, data: Dict[str, Any]):
        """Handle indicator data and track over time."""
        symbol = data.get('symbol', 'unknown')
        if symbol not in self.indicator_history:
            self.indicator_history[symbol] = []
        
        self.indicator_history[symbol].append({
            'timestamp': time.time(),
            'data': data
        })
        
        # Keep only last 100 entries per symbol
        if len(self.indicator_history[symbol]) > 100:
            self.indicator_history[symbol] = self.indicator_history[symbol][-100:]
    
    def _on_final_decision(self, data: Dict[str, Any]):
        """Track final decisions for timeline correlation."""
        event = {
            'timestamp': time.time(),
            'type': 'final_decision',
            'data': data
        }
        self.timeline.append(event)
        
        # Keep timeline manageable
        if len(self.timeline) > 500:
            self.timeline = self.timeline[-500:]
    
    def _analyze_timeline(self):
        """Analyze timeline and generate insights."""
        if len(self.timeline) < 2:
            return
        
        # Calculate time spans between decisions
        recent_decisions = [e for e in self.timeline[-20:] if e['type'] == 'decision']
        if len(recent_decisions) >= 2:
            time_deltas = []
            for i in range(1, len(recent_decisions)):
                delta = recent_decisions[i]['timestamp'] - recent_decisions[i-1]['timestamp']
                time_deltas.append(delta)
            
            avg_time_between_decisions = sum(time_deltas) / len(time_deltas) if time_deltas else 0
            
            # Publish timeline insight
            insight = {
                'type': 'timeline_analysis',
                'total_events': len(self.timeline),
                'decision_count': len(recent_decisions),
                'avg_time_between_decisions': avg_time_between_decisions,
                'time_window': self.timeline[-1]['timestamp'] - self.timeline[0]['timestamp'] if self.timeline else 0,
                'timestamp': time.time()
            }
            
            self.message_bus.publish('timeline_insight', insight)
    
    def get_timeline_summary(self) -> Dict[str, Any]:
        """Get summary of timeline events."""
        decision_events = [e for e in self.timeline if e['type'] == 'decision']
        final_decisions = [e for e in self.timeline if e['type'] == 'final_decision']
        
        return {
            'total_events': len(self.timeline),
            'decision_events': len(decision_events),
            'final_decisions': len(final_decisions),
            'symbols_tracked': list(self.indicator_history.keys()),
            'time_span': self.timeline[-1]['timestamp'] - self.timeline[0]['timestamp'] if len(self.timeline) > 1 else 0
        }
    
    def get_decision_timeline(self, time_window: float = 300) -> list:
        """
        Get decisions within a time window.
        
        Args:
            time_window: Time window in seconds (default: 300 = 5 minutes)
        
        Returns:
            List of decision events within the time window
        """
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        return [
            e for e in self.timeline 
            if e['timestamp'] >= cutoff_time and e['type'] in ['decision', 'final_decision']
        ]
    
    def get_indicator_timeline(self, symbol: str, time_window: float = 300) -> list:
        """
        Get indicator data for a symbol within a time window.
        
        Args:
            symbol: Trading symbol
            time_window: Time window in seconds
        
        Returns:
            List of indicator data within the time window
        """
        if symbol not in self.indicator_history:
            return []
        
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        return [
            entry for entry in self.indicator_history[symbol]
            if entry['timestamp'] >= cutoff_time
        ]

