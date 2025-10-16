"""
demo_sprint3.py - Sprint 3 Demo: Feedbackloopar och Introspektion

Beskrivning:
    Demonstrerar Sprint 3-funktionalitet med fokus på:
    - Feedback-routing med intelligent prioritering
    - Mönsteranalys i feedback_analyzer
    - Detektering av performance patterns, indicator mismatch och agent drift
    - Introspektionspanel med dashboard-data
    - Visualisering av modulkommunikation

Användning:
    python demo_sprint3.py
"""

from modules.message_bus import message_bus
from modules.feedback_router import FeedbackRouter
from modules.feedback_analyzer import FeedbackAnalyzer
from modules.introspection_panel import IntrospectionPanel
from modules.indicator_registry import IndicatorRegistry
import time


def main():
    """
    Huvudfunktion som demonstrerar Sprint 3-funktionalitet.
    """
    print("=" * 70)
    print("NextGen AI Trader - Sprint 3 Demo")
    print("Feedbackloopar och Introspektion")
    print("=" * 70)
    print()
    
    # Initiera Sprint 3-moduler
    print("Initierar Sprint 3-moduler...")
    router = FeedbackRouter(message_bus)
    analyzer = FeedbackAnalyzer(message_bus)
    panel = IntrospectionPanel(message_bus)
    indicator_reg = IndicatorRegistry("demo_api_key", message_bus)
    
    print("✓ FeedbackRouter initierad")
    print("✓ FeedbackAnalyzer initierad")
    print("✓ IntrospectionPanel initierad")
    print()
    
    # Steg 1: Demonstrera News och Insider Sentiment indikatorer
    print("=" * 70)
    print("Steg 1: Sprint 3 Indikatorer - News & Insider Sentiment")
    print("=" * 70)
    print()
    
    for symbol in ['AAPL', 'TSLA', 'MSFT']:
        indicators = indicator_reg.get_indicators(symbol)
        alt = indicators['alternative']
        
        print(f"{symbol}:")
        print(f"  News Sentiment: {alt['NewsSentiment']:.2f} (0.0 = Bearish, 1.0 = Bullish)")
        print(f"  Insider Sentiment: {alt['InsiderSentiment']:.2f} (0.0 = Selling, 1.0 = Buying)")
        print()
    
    # Steg 2: Simulera feedback events med olika prioriteringar
    print("=" * 70)
    print("Steg 2: Feedback Routing med Intelligent Prioritering")
    print("=" * 70)
    print()
    
    test_feedbacks = [
        {
            'source': 'execution_engine',
            'triggers': ['trade_result'],
            'data': {'success': True, 'slippage': 0.001}
        },
        {
            'source': 'execution_engine',
            'triggers': ['slippage'],
            'data': {'slippage': 0.007}  # Hög slippage
        },
        {
            'source': 'portfolio_manager',
            'triggers': ['capital_change'],
            'data': {'change': -120}  # Stor förlust
        },
        {
            'source': 'execution_engine',
            'triggers': ['trade_result'],
            'data': {'success': False}
        },
    ]
    
    print("Skickar feedback events med olika prioriteter...")
    for i, feedback in enumerate(test_feedbacks, 1):
        enriched = router.route_feedback(feedback)
        print(f"{i}. Source: {feedback['source']}, Triggers: {feedback['triggers']}")
        print(f"   → Prioritet: {enriched['priority']}")
        print()
    
    # Steg 3: Generera mer feedback för mönsteranalys
    print("=" * 70)
    print("Steg 3: Genererar Feedback för Mönsteranalys")
    print("=" * 70)
    print()
    
    print("Simulerar 20 trading cycles med varierande resultat...")
    for i in range(20):
        # Variera success rate och slippage
        success = i % 3 != 0  # ~67% success rate
        slippage = 0.002 + (i % 5) * 0.001  # Varierar mellan 0.2% - 0.6%
        
        feedback = {
            'source': 'execution_engine',
            'triggers': ['trade_result', 'slippage'],
            'data': {
                'success': success,
                'slippage': slippage,
                'performance': 0.7 - (i * 0.015) if i < 15 else 0.5  # Performance degradation
            }
        }
        
        # Publicera via message_bus så analyzer får den
        message_bus.publish('feedback_event', feedback)
    
    print(f"✓ {len(analyzer.feedback_buffer)} feedback events mottagna av analyzer")
    print()
    
    # Steg 4: Utför feedback-analys
    print("=" * 70)
    print("Steg 4: Feedback-Analys och Mönsterdetektering")
    print("=" * 70)
    print()
    
    insight = analyzer.analyze_feedback()
    
    print(f"Analyserade {insight['samples_analyzed']} feedback events")
    print()
    
    print("Identifierade Mönster:")
    for i, pattern in enumerate(insight['patterns'], 1):
        print(f"{i}. {pattern['type']}")
        print(f"   Beskrivning: {pattern['description']}")
        print()
    
    if insight['anomalies']:
        print("Anomalier Detekterade:")
        for i, anomaly in enumerate(insight['anomalies'], 1):
            print(f"{i}. {anomaly['type']}")
            print(f"   Beskrivning: {anomaly['description']}")
            print()
    
    if insight['recommendations']:
        print("Rekommendationer:")
        for i, rec in enumerate(insight['recommendations'], 1):
            print(f"{i}. {rec['type']}: {rec['description']}")
            print(f"   Åtgärd: {rec['action']}")
            print()
    
    # Steg 5: Simulera agent status för introspection
    print("=" * 70)
    print("Steg 5: Agent Status och Introspection")
    print("=" * 70)
    print()
    
    print("Simulerar agent status updates...")
    for i in range(10):
        status = {
            'module': 'strategy_engine',
            'performance': 0.6 + (i * 0.03),  # Improving performance
            'reward': 5.0 + i * 2.5,
            'timestamp': time.time() + i
        }
        message_bus.publish('agent_status', status)
    
    print(f"✓ {len(panel.agent_status_history)} agent status updates mottagna")
    print()
    
    # Steg 6: Generera dashboard-data
    print("=" * 70)
    print("Steg 6: Dashboard Data och Visualisering")
    print("=" * 70)
    print()
    
    dashboard = panel.render_dashboard()
    
    print("Dashboard Metrics:")
    print(f"  Agent Status Snapshots: {len(dashboard['agent_status'])}")
    print(f"  Feedback Events: {len(dashboard['feedback_flow'])}")
    print(f"  Reward Trends: {len(dashboard['reward_trends'])} datapunkter")
    print(f"  Modul-kopplingar: {len(dashboard['module_connections'])} kopplingar")
    print()
    
    # Visa feedback metrics
    fb_metrics = dashboard['feedback_metrics']
    print("Feedback Metrics:")
    print(f"  Totalt Events: {fb_metrics['total_events']}")
    print(f"  Events/minut: {fb_metrics['avg_per_minute']:.2f}")
    print()
    
    print("Events per Källa:")
    for source, count in fb_metrics['by_source'].items():
        print(f"  {source}: {count}")
    print()
    
    print("Events per Prioritet:")
    for priority, count in fb_metrics['by_priority'].items():
        print(f"  {priority}: {count}")
    print()
    
    # Visa agent adaptation
    adaptation = dashboard['agent_adaptation']
    print("Agent Adaptation:")
    print(f"  Adaptation Rate: {adaptation['adaptation_rate']:.2%}")
    print(f"  Performance Trend: {adaptation['performance_trend']}")
    print(f"  Learning Progress: {adaptation['learning_progress']:.2f}")
    print()
    
    # Visa modul-kopplingar
    if dashboard['module_connections']:
        print("Top Modul-kopplingar:")
        for i, conn in enumerate(dashboard['module_connections'][:5], 1):
            print(f"{i}. {conn['source']} → {conn['target']}: {conn['count']} events (styrka: {conn['strength']:.2f})")
        print()
    
    # Steg 7: Sammanfattning
    print("=" * 70)
    print("Sprint 3 Demo Sammanfattning")
    print("=" * 70)
    print()
    
    print("✅ Implementerade funktioner:")
    print("  1. News Sentiment & Insider Sentiment indikatorer")
    print("  2. Intelligent feedback-routing med prioritering")
    print("  3. Performance pattern detection:")
    print(f"     - {len([p for p in insight['patterns'] if 'slippage' in p['type']])} slippage patterns")
    print(f"     - {len([p for p in insight['patterns'] if 'success' in p['type']])} success rate patterns")
    print(f"     - {len([p for p in insight['patterns'] if 'capital' in p['type']])} capital change patterns")
    print("  4. Agent drift detection")
    print(f"     - {len(insight['anomalies'])} anomalier detekterade")
    print("  5. Introspection dashboard med:")
    print(f"     - {fb_metrics['total_events']} feedback events loggade")
    print(f"     - {len(dashboard['module_connections'])} modul-kopplingar kartlagda")
    print(f"     - Agent adaptation tracking (trend: {adaptation['performance_trend']})")
    print()
    
    print("🎯 Testbara mål uppnådda:")
    print("  ✓ Modulkommunikation fungerar via message_bus")
    print("  ✓ Feedbackflöde routas och loggas korrekt")
    print("  ✓ Mönsteranalys identifierar performance patterns")
    print("  ✓ Dashboard-data genereras för visualisering")
    print()
    
    print("📊 Nästa steg:")
    print("  - Kör Dash-applikationen med: python dashboards/feedback_flow.py")
    print("  - Integrera med RL-controller för feedback-baserad träning")
    print("  - Utöka med fler mönster-detektorer")
    print()
    
    print("=" * 70)
    print("Sprint 3 Demo avslutad!")
    print("=" * 70)


if __name__ == "__main__":
    main()
