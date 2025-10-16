"""
feedback_analyzer.py - Feedback-analys

Beskrivning:
    Identifierar mönster i feedbackflöden och föreslår förbättringar.
    Analyserar feedback från olika källor för att upptäcka trends och anomalier.

Roll:
    - Analyserar feedback_event från feedback_router
    - Identifierar performance patterns
    - Upptäcker indicator-response mismatch
    - Detekterar agent drift
    - Genererar feedback_insight för andra moduler

Inputs:
    - feedback_event: Dict - Feedback från feedback_router

Outputs:
    - feedback_insight: Dict - Analyserade mönster och förbättringsförslag

Publicerar till message_bus:
    - feedback_insight: Insights för meta_agent_evolution_engine

Prenumererar på (från functions.yaml):
    - feedback_event (från feedback_router)

Använder RL: Nej (från functions.yaml)
Tar emot feedback: Ja (från meta_agent_evolution_engine)

Anslutningar (från flowchart.yaml - feedback_flow):
    Från: feedback_router (feedback_event)
    Till:
    - meta_agent_evolution_engine (feedback_insight)
    - strategic_memory_engine (patterns)

Analysis (från feedback_loop.yaml):
    Detects:
    - performance patterns: Återkommande mönster i performance
    - indicator-response mismatch: När indikator och respons inte stämmer
    - agent drift: När agenter börjar avvika från optimal beteende
    
    Emits:
    - feedback_insight: Analyserade mönster och rekommendationer

Indikatorer från indicator_map.yaml:
    Använder:
    - News Sentiment: market mood and reaction

Används i Sprint: 3, 4
"""

from typing import Dict, Any, List
from collections import defaultdict


class FeedbackAnalyzer:
    """Analyserar feedback och identifierar mönster."""
    
    def __init__(self, message_bus):
        """
        Initialiserar feedback-analysatorn.
        
        Args:
            message_bus: Referens till central message_bus
        """
        self.message_bus = message_bus
        self.feedback_buffer: List[Dict[str, Any]] = []
        self.pattern_cache: Dict[str, Any] = {}
        self.analysis_count = 0
        
        # Prenumerera på feedback events
        self.message_bus.subscribe('feedback_event', self._on_feedback_event)
    
    def _on_feedback_event(self, feedback: Dict[str, Any]) -> None:
        """
        Callback för feedback events.
        
        Args:
            feedback: Feedback event att analysera
        """
        self.feedback_buffer.append(feedback)
        
        # Analysera när vi har tillräckligt med data
        if len(self.feedback_buffer) >= 10:
            self.analyze_feedback()
    
    def analyze_feedback(self) -> Dict[str, Any]:
        """
        Analyserar feedback buffer för mönster.
        
        Returns:
            Dict med feedback_insight
        """
        import time
        
        insight = {
            'timestamp': time.time(),
            'samples_analyzed': len(self.feedback_buffer),
            'patterns': [],
            'anomalies': [],
            'recommendations': []
        }
        
        # Analysera sources
        sources = defaultdict(int)
        triggers_by_source = defaultdict(list)
        
        for fb in self.feedback_buffer:
            source = fb.get('source', 'unknown')
            sources[source] += 1
            triggers = fb.get('triggers', [])
            triggers_by_source[source].extend(triggers)
        
        # Sprint 3: Identifiera performance patterns
        performance_patterns = self.detect_performance_patterns()
        insight['patterns'].extend(performance_patterns)
        
        # Identifiera hög aktivitet från specifika moduler
        for source, count in sources.items():
            if count > 5:
                insight['patterns'].append({
                    'type': 'high_activity',
                    'source': source,
                    'count': count,
                    'description': f'Hög aktivitet från {source} ({count} events)'
                })
        
        # Sprint 3: Identifiera indicator-response mismatch
        mismatch_patterns = self.detect_indicator_mismatch()
        insight['patterns'].extend(mismatch_patterns)
        
        # Sprint 3: Identifiera agent drift
        drift_patterns = self.detect_agent_drift()
        if drift_patterns:
            insight['anomalies'].extend(drift_patterns)
        
        # Generera rekommendationer baserat på mönster
        if len(insight['patterns']) > 3:
            insight['recommendations'].append({
                'type': 'pattern_overload',
                'description': 'Många mönster detekterade - överväg justeringar',
                'action': 'review_module_thresholds'
            })
        
        if any(p['type'] == 'high_slippage' for p in insight['patterns']):
            insight['recommendations'].append({
                'type': 'execution_quality',
                'description': 'Hög slippage detekterad - justera execution timing',
                'action': 'adjust_execution_strategy'
            })
        
        self.analysis_count += 1
        
        # Cacha insights för historisk analys
        self.pattern_cache[f'analysis_{self.analysis_count}'] = insight
        
        # Publicera insight
        self.publish_insight(insight)
        
        # Rensa gamla feedback (behåll senaste 50)
        if len(self.feedback_buffer) > 50:
            self.feedback_buffer = self.feedback_buffer[-50:]
        
        return insight
    
    def publish_insight(self, insight: Dict[str, Any]) -> None:
        """
        Publicerar feedback insight till message_bus.
        
        Args:
            insight: Feedback insight att publicera
        """
        self.message_bus.publish('feedback_insight', insight)
    
    def detect_performance_patterns(self) -> List[Dict[str, Any]]:
        """
        Identifierar återkommande performance patterns.
        
        Returns:
            Lista med identifierade mönster
        """
        patterns = []
        
        # Analysera slippage patterns
        slippage_events = [
            fb for fb in self.feedback_buffer 
            if 'slippage' in fb.get('triggers', [])
        ]
        
        if len(slippage_events) > 3:
            avg_slippage = sum(
                fb.get('data', {}).get('slippage', 0) 
                for fb in slippage_events
            ) / len(slippage_events)
            
            if avg_slippage > 0.003:  # > 0.3%
                patterns.append({
                    'type': 'high_slippage',
                    'count': len(slippage_events),
                    'avg_value': avg_slippage,
                    'description': f'Hög genomsnittlig slippage: {avg_slippage:.4f}'
                })
        
        # Analysera trade_result patterns
        trade_results = [
            fb for fb in self.feedback_buffer 
            if 'trade_result' in fb.get('triggers', [])
        ]
        
        if len(trade_results) > 5:
            successful = sum(
                1 for fb in trade_results 
                if fb.get('data', {}).get('success', False)
            )
            success_rate = successful / len(trade_results)
            
            patterns.append({
                'type': 'trade_success_rate',
                'success_rate': success_rate,
                'total_trades': len(trade_results),
                'description': f'Trade success rate: {success_rate:.2%}'
            })
            
            if success_rate < 0.5:
                patterns.append({
                    'type': 'low_success_rate',
                    'success_rate': success_rate,
                    'description': 'Låg trade success rate - överväg strategi-justering'
                })
        
        # Analysera capital_change patterns
        capital_changes = [
            fb for fb in self.feedback_buffer 
            if 'capital_change' in fb.get('triggers', [])
        ]
        
        if len(capital_changes) > 3:
            changes = [
                fb.get('data', {}).get('change', 0) 
                for fb in capital_changes
            ]
            avg_change = sum(changes) / len(changes)
            
            patterns.append({
                'type': 'avg_capital_change',
                'avg_change': avg_change,
                'total_events': len(capital_changes),
                'description': f'Genomsnittlig kapitalförändring: ${avg_change:.2f}'
            })
        
        return patterns
    
    def detect_indicator_mismatch(self) -> List[Dict[str, Any]]:
        """
        Identifierar när indikator och respons inte stämmer.
        
        Returns:
            Lista med mismatch-fall
        """
        mismatches = []
        
        # Analysera korrelation mellan indikator-signaler och outcomes
        # Detta kräver tillgång till både indicator_data och trade outcomes
        
        # Exempel: Analysera om RSI-signaler resulterar i förväntade outcomes
        # För Sprint 3, implementera grundläggande korrelationsanalys
        
        # Samla feedback med indikator-information
        indicator_related = [
            fb for fb in self.feedback_buffer 
            if fb.get('data', {}).get('indicators') is not None
        ]
        
        if len(indicator_related) > 5:
            # Analysera om indikator-baserade beslut ger bra resultat
            good_outcomes = sum(
                1 for fb in indicator_related 
                if fb.get('data', {}).get('outcome', 'neutral') == 'positive'
            )
            
            if good_outcomes < len(indicator_related) * 0.4:  # < 40% success
                mismatches.append({
                    'type': 'low_indicator_correlation',
                    'success_rate': good_outcomes / len(indicator_related),
                    'description': 'Indikatorer korrelerar dåligt med outcomes'
                })
        
        return mismatches
    
    def detect_agent_drift(self) -> List[Dict[str, Any]]:
        """
        Identifierar agent drift från optimal beteende.
        
        Returns:
            Lista med drift-fall
        """
        drift_cases = []
        
        # Analysera agent performance över tid för att detektera drift
        # Drift = när agent behavior förändras på ett sätt som minskar performance
        
        # Samla agent-relaterade feedback
        agent_feedback = [
            fb for fb in self.feedback_buffer 
            if fb.get('source') in ['rl_controller', 'strategy_engine', 'decision_engine', 'risk_manager']
        ]
        
        if len(agent_feedback) > 10:
            # Dela upp i första halvan och andra halvan
            mid = len(agent_feedback) // 2
            first_half = agent_feedback[:mid]
            second_half = agent_feedback[mid:]
            
            # Jämför performance metrics
            def get_avg_performance(feedback_list):
                perfs = [
                    fb.get('data', {}).get('performance', 0.5) 
                    for fb in feedback_list 
                    if fb.get('data', {}).get('performance') is not None
                ]
                return sum(perfs) / len(perfs) if perfs else 0.5
            
            first_perf = get_avg_performance(first_half)
            second_perf = get_avg_performance(second_half)
            
            # Om performance sjunker signifikant, indikera drift
            if first_perf > 0.6 and second_perf < first_perf * 0.85:
                drift_cases.append({
                    'type': 'performance_degradation',
                    'first_half_perf': first_perf,
                    'second_half_perf': second_perf,
                    'degradation': (first_perf - second_perf) / first_perf,
                    'description': f'Agent performance sjönk från {first_perf:.2f} till {second_perf:.2f}'
                })
        
        return drift_cases
    
    def receive_feedback(self, feedback: Dict[str, Any]) -> None:
        """
        Tar emot feedback från meta_agent_evolution_engine.
        
        Args:
            feedback: Feedback om analyzer performance
        """
        # Stub: Implementeras i Sprint 4
        pass

