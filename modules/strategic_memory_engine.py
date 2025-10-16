"""
strategic_memory_engine.py - Strategiskt minne

Beskrivning:
    Loggar beslut, indikatorer, utfall och reward för historisk analys.
    Identifierar mönster och ger insights till decision_engine.

Roll:
    - Loggar alla beslut och deras utfall
    - Lagrar indikatordata kopplat till beslut
    - Analyserar historisk performance
    - Genererar memory_insights för decision_engine
    - Genererar feedback om decision outcomes och indicator correlations

Inputs:
    - final_decision: Dict - Beslut från decision_engine
    - indicator_data: Dict - Indikatorer från indicator_registry
    - execution_result: Dict - Resultat från execution_engine

Outputs:
    - memory_insights: Dict - Historiska lärdomar och mönster

Publicerar till message_bus:
    - memory_insights: Insights för decision_engine
    - feedback_event: Feedback om decision outcomes

Prenumererar på (från functions.yaml):
    - final_decision (från decision_engine)
    - indicator_data (från indicator_registry)
    - execution_result (från execution_engine)

Använder RL: Nej (från functions.yaml)
Tar emot feedback: Ja (från feedback_analyzer, meta_agent_evolution_engine)

Anslutningar (från flowchart.yaml - memory_flow):
    Från:
    - decision_engine (final_decision)
    - indicator_registry (indicator_data)
    - execution_engine (execution_result)
    - risk_manager (risk_profile)
    Till:
    - decision_engine (memory_insights)
    - feedback_analyzer (historical data)
    - introspection_panel (för visualisering)

Feedback-generering (från feedback_loop.yaml):
    Triggers:
    - decision_outcome: Om beslut ledde till vinst/förlust
    - indicator_correlation: Vilka indikatorer korrelerade med utfall
    
    Stores:
    - feedback_event: Från andra moduler
    - feedback_insight: Från feedback_analyzer
    - agent_response: Från RL-agenter

Används i Sprint: 4, 6
"""

from typing import Dict, Any, List, Tuple
import time


class StrategicMemoryEngine:
    """Loggar och analyserar historiska beslut och utfall."""
    
    def __init__(self, message_bus):
        """
        Initialiserar strategiskt minne.
        
        Args:
            message_bus: Referens till central message_bus
        """
        self.message_bus = message_bus
        self.decision_history: List[Dict[str, Any]] = []
        self.indicator_history: Dict[str, List[Dict[str, Any]]] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.feedback_storage: List[Dict[str, Any]] = []
        self.insight_storage: List[Dict[str, Any]] = []
        self.agent_responses: List[Dict[str, Any]] = []
        
        # Korrelationsanalys cache
        self.correlation_cache: Dict[str, Dict[str, Any]] = {}
        
        # Prenumerera på relevanta events
        self.message_bus.subscribe('final_decision', self._on_decision)
        self.message_bus.subscribe('indicator_data', self._on_indicators)
        self.message_bus.subscribe('execution_result', self._on_execution)
        self.message_bus.subscribe('risk_profile', self._on_risk_profile)
        self.message_bus.subscribe('feedback_event', self._on_feedback)
        self.message_bus.subscribe('feedback_insight', self._on_insight)
        self.message_bus.subscribe('agent_status', self._on_agent_response)
    
    def _on_decision(self, decision: Dict[str, Any]) -> None:
        """
        Callback för beslut från decision_engine.
        
        Args:
            decision: Handelsbeslut att logga
        """
        decision_entry = {
            **decision,
            'timestamp': time.time(),
            'logged_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        self.decision_history.append(decision_entry)
    
    def _on_indicators(self, indicators: Dict[str, Any]) -> None:
        """
        Callback för indikatordata.
        
        Args:
            indicators: Indikatordata att lagra
        """
        symbol = indicators.get('symbol', 'UNKNOWN')
        if symbol not in self.indicator_history:
            self.indicator_history[symbol] = []
        
        indicator_entry = {
            **indicators,
            'timestamp': time.time(),
            'logged_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        self.indicator_history[symbol].append(indicator_entry)
    
    def _on_execution(self, result: Dict[str, Any]) -> None:
        """
        Callback för execution results.
        
        Args:
            result: Execution result att analysera
        """
        execution_entry = {
            **result,
            'timestamp': time.time(),
            'logged_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        self.execution_history.append(execution_entry)
        
        # Koppla execution till beslut
        self._link_execution_to_decision(execution_entry)
        
        # Generera feedback om decision outcome
        self.generate_decision_feedback(execution_entry)
        
        # Analysera indicator correlation
        self._analyze_indicator_correlation(execution_entry)
    
    def _on_risk_profile(self, risk_profile: Dict[str, Any]) -> None:
        """
        Callback för risk profile.
        
        Args:
            risk_profile: Riskprofil att lagra
        """
        # Lägg till risk profile i senaste beslut om det finns
        if self.decision_history:
            self.decision_history[-1]['risk_profile'] = risk_profile
    
    def _on_feedback(self, feedback: Dict[str, Any]) -> None:
        """
        Callback för feedback events (från feedback_loop.yaml).
        
        Args:
            feedback: Feedback event att lagra
        """
        feedback_entry = {
            **feedback,
            'timestamp': time.time(),
            'logged_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        self.feedback_storage.append(feedback_entry)
    
    def _on_insight(self, insight: Dict[str, Any]) -> None:
        """
        Callback för feedback insights (från feedback_loop.yaml).
        
        Args:
            insight: Feedback insight att lagra
        """
        insight_entry = {
            **insight,
            'timestamp': time.time(),
            'logged_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        self.insight_storage.append(insight_entry)
    
    def _on_agent_response(self, agent_status: Dict[str, Any]) -> None:
        """
        Callback för agent responses (från feedback_loop.yaml).
        
        Args:
            agent_status: Agent status att lagra
        """
        agent_entry = {
            **agent_status,
            'timestamp': time.time(),
            'logged_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        self.agent_responses.append(agent_entry)
    
    def _link_execution_to_decision(self, execution: Dict[str, Any]) -> None:
        """
        Kopplar execution result till motsvarande beslut.
        
        Args:
            execution: Execution result
        """
        symbol = execution.get('symbol')
        action = execution.get('action')
        
        # Hitta senaste beslut för denna symbol och action
        for decision in reversed(self.decision_history):
            if (decision.get('symbol') == symbol and 
                decision.get('action') == action and
                'execution_result' not in decision):
                decision['execution_result'] = execution
                break
    
    def _analyze_indicator_correlation(self, execution: Dict[str, Any]) -> None:
        """
        Analyserar korrelation mellan indikatorer och execution outcome.
        
        Args:
            execution: Execution result att analysera
        """
        symbol = execution.get('symbol')
        success = execution.get('success', False)
        profit = execution.get('profit', 0.0)
        
        # Hämta senaste indikatorer för denna symbol
        if symbol not in self.indicator_history or not self.indicator_history[symbol]:
            return
        
        recent_indicators = self.indicator_history[symbol][-1]
        
        # Uppdatera korrelationscache
        if symbol not in self.correlation_cache:
            self.correlation_cache[symbol] = {
                'total_trades': 0,
                'successful_trades': 0,
                'indicator_success': {},
                'indicator_profit': {}
            }
        
        cache = self.correlation_cache[symbol]
        cache['total_trades'] += 1
        if success:
            cache['successful_trades'] += 1
        
        # Analysera varje indikator
        for indicator_name, indicator_value in recent_indicators.items():
            if indicator_name in ['symbol', 'timestamp', 'logged_at']:
                continue
            
            if indicator_name not in cache['indicator_success']:
                cache['indicator_success'][indicator_name] = {'success': 0, 'total': 0}
                cache['indicator_profit'][indicator_name] = {'total_profit': 0.0, 'count': 0}
            
            cache['indicator_success'][indicator_name]['total'] += 1
            if success:
                cache['indicator_success'][indicator_name]['success'] += 1
            
            cache['indicator_profit'][indicator_name]['total_profit'] += profit
            cache['indicator_profit'][indicator_name]['count'] += 1
    
    def generate_insights(self) -> Dict[str, Any]:
        """
        Genererar insights baserat på historisk data.
        
        Returns:
            Dict med memory_insights
        """
        insights = {
            'total_decisions': len(self.decision_history),
            'total_executions': len(self.execution_history),
            'success_rate': 0.0,
            'average_profit': 0.0,
            'patterns': [],
            'recommendations': [],
            'indicator_correlations': {},
            'best_indicators': [],
            'worst_indicators': []
        }
        
        # Beräkna success rate och average profit
        if self.execution_history:
            successful = sum(1 for e in self.execution_history if e.get('success', False))
            insights['success_rate'] = successful / len(self.execution_history)
            
            total_profit = sum(e.get('profit', 0.0) for e in self.execution_history)
            insights['average_profit'] = total_profit / len(self.execution_history)
        
        # Analysera mönster från korrelationscache
        for symbol, cache in self.correlation_cache.items():
            if cache['total_trades'] < 5:  # Kräver minst 5 trades för analys
                continue
            
            symbol_success_rate = (cache['successful_trades'] / cache['total_trades'] 
                                  if cache['total_trades'] > 0 else 0.0)
            
            # Analysera indikatorprestanda
            for indicator, data in cache['indicator_success'].items():
                if data['total'] > 0:
                    indicator_success_rate = data['success'] / data['total']
                    
                    profit_data = cache['indicator_profit'][indicator]
                    avg_profit = (profit_data['total_profit'] / profit_data['count'] 
                                 if profit_data['count'] > 0 else 0.0)
                    
                    insights['indicator_correlations'][indicator] = {
                        'success_rate': indicator_success_rate,
                        'average_profit': avg_profit,
                        'sample_size': data['total']
                    }
                    
                    # Identifiera bästa och sämsta indikatorer
                    if indicator_success_rate > 0.6 and avg_profit > 0:
                        insights['best_indicators'].append({
                            'name': indicator,
                            'success_rate': indicator_success_rate,
                            'average_profit': avg_profit
                        })
                    elif indicator_success_rate < 0.4:
                        insights['worst_indicators'].append({
                            'name': indicator,
                            'success_rate': indicator_success_rate,
                            'average_profit': avg_profit
                        })
        
        # Sortera bästa och sämsta indikatorer
        insights['best_indicators'].sort(key=lambda x: x['success_rate'], reverse=True)
        insights['worst_indicators'].sort(key=lambda x: x['success_rate'])
        
        # Generera rekommendationer baserat på analys
        if insights['best_indicators']:
            insights['recommendations'].append(
                f"Fokusera på indikatorer: {', '.join(i['name'] for i in insights['best_indicators'][:3])}"
            )
        
        if insights['worst_indicators']:
            insights['recommendations'].append(
                f"Minska vikt på indikatorer: {', '.join(i['name'] for i in insights['worst_indicators'][:3])}"
            )
        
        if insights['success_rate'] < 0.5:
            insights['recommendations'].append(
                "Övervaka strategier - success rate under 50%"
            )
        
        # Identifiera patterns från historik
        if len(self.execution_history) >= 10:
            recent_executions = self.execution_history[-10:]
            recent_success = sum(1 for e in recent_executions if e.get('success', False))
            recent_success_rate = recent_success / len(recent_executions)
            
            if recent_success_rate < insights['success_rate'] - 0.15:
                insights['patterns'].append({
                    'type': 'performance_degradation',
                    'severity': 'high',
                    'description': 'Senaste performance sämre än genomsnitt'
                })
            elif recent_success_rate > insights['success_rate'] + 0.15:
                insights['patterns'].append({
                    'type': 'performance_improvement',
                    'severity': 'low',
                    'description': 'Senaste performance bättre än genomsnitt'
                })
        
        return insights
    
    def publish_insights(self) -> None:
        """Publicerar memory insights till message_bus."""
        insights = self.generate_insights()
        self.message_bus.publish('memory_insights', insights)
    
    def generate_decision_feedback(self, execution_result: Dict[str, Any]) -> None:
        """
        Genererar feedback om decision outcome (från feedback_loop.yaml).
        
        Args:
            execution_result: Execution result att analysera
        """
        success = execution_result.get('success', False)
        profit = execution_result.get('profit', 0.0)
        
        feedback = {
            'source': 'strategic_memory_engine',
            'triggers': ['decision_outcome'],
            'data': {
                'outcome': 'success' if success else 'failure',
                'symbol': execution_result.get('symbol'),
                'action': execution_result.get('action'),
                'profit': profit,
                'success': success
            }
        }
        
        # Analysera indicator correlation
        symbol = execution_result.get('symbol')
        if symbol in self.correlation_cache:
            cache = self.correlation_cache[symbol]
            feedback['triggers'].append('indicator_correlation')
            feedback['data']['indicator_correlation'] = {
                'total_trades': cache['total_trades'],
                'success_rate': (cache['successful_trades'] / cache['total_trades'] 
                                if cache['total_trades'] > 0 else 0.0),
                'top_indicators': list(cache['indicator_success'].keys())[:5]
            }
        
        # Publicera feedback
        self.message_bus.publish('feedback_event', feedback)
    
    def receive_feedback(self, feedback: Dict[str, Any]) -> None:
        """
        Tar emot feedback från feedback_analyzer.
        
        Args:
            feedback: Feedback om memory insights quality
        """
        # Lagra feedback för framtida analys
        self._on_feedback(feedback)
    
    def get_decision_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Hämtar beslutshistorik.
        
        Args:
            limit: Max antal beslut att returnera
            
        Returns:
            Lista med senaste besluten
        """
        return self.decision_history[-limit:]
    
    def get_correlation_analysis(self, symbol: str = None) -> Dict[str, Any]:
        """
        Hämtar korrelationsanalys för en specifik symbol eller alla symboler.
        
        Args:
            symbol: Symbol att analysera (None för alla)
            
        Returns:
            Dict med korrelationsdata
        """
        if symbol:
            return self.correlation_cache.get(symbol, {})
        return self.correlation_cache
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Genererar en sammanfattning av performance över tid.
        
        Returns:
            Dict med performance-metriker
        """
        summary = {
            'total_decisions': len(self.decision_history),
            'total_executions': len(self.execution_history),
            'feedback_events': len(self.feedback_storage),
            'insights_received': len(self.insight_storage),
            'agent_responses': len(self.agent_responses),
            'symbols_tracked': len(self.indicator_history),
            'correlations_analyzed': len(self.correlation_cache)
        }
        
        return summary

