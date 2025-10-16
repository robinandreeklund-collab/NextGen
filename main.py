"""
NextGen AI Trader - Huvudstartpunkt

Detta är huvudapplikationen för NextGen AI Trader-systemet.
Sprint 4 inkluderar strategiskt minne och agentutveckling.

Moduler i Sprint 1-4:
- Sprint 1: Grundläggande trading med RSI
- Sprint 2: RL-träning och avancerade indikatorer
- Sprint 3: Feedbackloopar och introspektion
- Sprint 4: Strategiskt minne och agentutveckling

Användning:
    python main.py
"""

from modules.message_bus import message_bus
from modules.data_ingestion import DataIngestion
from modules.indicator_registry import IndicatorRegistry
from modules.strategy_engine import StrategyEngine
from modules.risk_manager import RiskManager
from modules.decision_engine import DecisionEngine
from modules.execution_engine import ExecutionEngine
from modules.portfolio_manager import PortfolioManager
from modules.rl_controller import RLController
from modules.feedback_router import FeedbackRouter
from modules.feedback_analyzer import FeedbackAnalyzer
from modules.strategic_memory_engine import StrategicMemoryEngine  # Sprint 4
from modules.meta_agent_evolution_engine import MetaAgentEvolutionEngine  # Sprint 4
from modules.agent_manager import AgentManager  # Sprint 4
import yaml


def load_rl_config():
    """Laddar RL-konfiguration från config/rl_parameters.yaml."""
    try:
        with open('config/rl_parameters.yaml', 'r') as f:
            return yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError):
        # Fallback till default config
        return None


def main():
    """
    Huvudfunktion som demonstrerar Sprint 1-4 flödet.
    
    Flöde:
    1. Initiera alla moduler (Sprint 1-4)
    2. Hämta indikatorer (tekniska, fundamentala, alternativa)
    3. Generera tradeförslag med RL-förstärkning
    4. Riskbedöm med ATR och Analyst Ratings
    5. Fatta beslut med RL-optimering och funds validation
    6. Exekvera och uppdatera portfölj
    7. Träna RL-agenter baserat på reward
    8. Logga i strategic memory och analysera performance
    9. Agent evolution baserat på degradation
    10. Versionshantering av agenter
    """
    print("=" * 80)
    print("NextGen AI Trader - Sprint 1-4 Full System")
    print("Inkluderar: RL, Feedback, Strategiskt Minne och Agentutveckling")
    print("=" * 80)
    print()
    
    # Initiera alla Sprint 1-4 moduler
    print("Initierar alla moduler (Sprint 1-4)...")
    
    # API-nyckel (placeholder för demo)
    api_key = "demo_api_key"
    
    # Ladda RL-config
    rl_config = load_rl_config()
    
    # Sprint 1-2 moduler
    data_ingest = DataIngestion(api_key, message_bus)
    indicator_reg = IndicatorRegistry(api_key, message_bus)
    strategy = StrategyEngine(message_bus)
    risk_mgr = RiskManager(message_bus)
    decision = DecisionEngine(message_bus)
    execution = ExecutionEngine(message_bus, simulation_mode=True)
    portfolio = PortfolioManager(
        message_bus, 
        start_capital=1000.0,
        transaction_fee=0.0025
    )
    rl_ctrl = RLController(message_bus, rl_config)
    
    # Sprint 3 moduler
    feedback_rtr = FeedbackRouter(message_bus)
    feedback_analyzer = FeedbackAnalyzer(message_bus)
    
    # Sprint 4 moduler
    strategic_memory = StrategicMemoryEngine(message_bus)
    agent_evolution = MetaAgentEvolutionEngine(message_bus)
    agent_mgr = AgentManager(message_bus)
    
    print("✓ Sprint 1-2: Data, Indikatorer, RL, Portfolio")
    print("✓ Sprint 3: Feedback loops och introspection")
    print("✓ Sprint 4: Strategic memory, Agent evolution, Versionshantering")
    print(f"✓ RL-agenter: {list(rl_ctrl.agents.keys())}")
    print(f"✓ Managed agents: {list(agent_mgr.agent_profiles.keys())}")
    print(f"✓ Evolution threshold: {agent_evolution.performance_threshold*100:.0f}%")
    print()
    
    # Steg 1: Hämta trending symboler
    print("Steg 1: Hämtar trending symboler...")
    symbols = data_ingest.fetch_trending_symbols()
    print(f"✓ Hittade symboler: {', '.join(symbols)}")
    print()
    
    # Steg 2: Hämta indikatorer för symboler (med nya Sprint 2-indikatorer)
    print("Steg 2: Hämtar indikatorer (inkl. MACD, ATR, Analyst Ratings)...")
    for symbol in symbols:
        indicators = indicator_reg.get_indicators(symbol)
        print(f"  {symbol}:")
        print(f"    RSI: {indicators['technical']['RSI']:.1f}")
        print(f"    MACD Histogram: {indicators['technical']['MACD']['histogram']:.2f}")
        print(f"    ATR: {indicators['technical']['ATR']:.1f}")
        print(f"    Analyst Consensus: {indicators['fundamental']['AnalystRatings']['consensus']}")
    print()
    
    # Steg 3: Kör flera trading cycles för att träna RL-agenter
    print("Steg 3: Kör trading cycles med RL-träning...")
    print()
    
    for cycle in range(3):
        print(f"--- Trading Cycle {cycle + 1} ---")
        
        # Använd TSLA (låg RSI, negativ MACD) för demo
        symbol = 'TSLA'
        
        # Generera tradeförslag
        proposal = strategy.generate_proposal(symbol)
        print(f"  Tradeförslag: {proposal['action']} {proposal['quantity']} @ {symbol}")
        print(f"    Motivering: {proposal['reasoning']}")
        print(f"    Confidence: {proposal['confidence']:.2f}")
        print(f"    RL Enabled: {proposal.get('rl_enabled', False)}")
        strategy.publish_proposal(proposal)
        
        # Riskbedömning
        risk_profile = risk_mgr.assess_risk(symbol)
        print(f"  Riskprofil: {risk_profile['risk_level']}")
        print(f"    Volatilitet: {risk_profile['volatility']:.2f}")
        print(f"    Rekommendationer: {', '.join(risk_profile['recommendations'][:2])}")
        risk_mgr.publish_risk_profile(risk_profile)
        
        # Fatta beslut
        final_decision = decision.make_decision(symbol)
        print(f"  Slutgiltigt beslut: {final_decision['action']} {final_decision['quantity']} @ {symbol}")
        print(f"    Confidence: {final_decision['confidence']:.2f}")
        print(f"    RL Enabled: {final_decision.get('rl_enabled', False)}")
        decision.publish_decision(final_decision)
        
        print()
    
    # Steg 4: Visa RL-statistik
    print("Steg 4: RL-träning och Agent Performance")
    print(f"  Training steps: {rl_ctrl.training_steps}")
    print(f"  Average reward: {rl_ctrl.reward_history[-1] if rl_ctrl.reward_history else 0:.4f}")
    print(f"  Reward history: {[f'{r:.2f}' for r in rl_ctrl.reward_history[-3:]]}")
    print()
    
    for module_name in rl_ctrl.agents.keys():
        perf = rl_ctrl.get_agent_performance(module_name)
        print(f"  {module_name}: {perf:.4f}")
    print()
    
    # Steg 5: Visa portföljstatus
    print("Steg 5: Portföljstatus")
    status = portfolio.get_status()
    print(f"  Kapital (cash): ${status['cash']:.2f}")
    print(f"  Totalt värde: ${status['total_value']:.2f}")
    print(f"  P&L: ${status['pnl']:.2f} ({status['pnl_pct']:.2f}%)")
    print(f"  Antal trades: {status['num_trades']}")
    print()
    
    # Steg 6: Visa feedback-statistik
    print("Steg 6: Feedback-statistik")
    feedback_log = feedback_rtr.get_feedback_log()
    print(f"  Totalt antal feedback events: {len(feedback_log)}")
    if feedback_log:
        print(f"  Senaste feedback:")
        for fb in feedback_log[-3:]:
            print(f"    - Källa: {fb.get('source')}, Triggers: {fb.get('triggers')}")
    print()
    
    # Visa message_bus-statistik
    print("Steg 7: Message Bus-statistik")
    log = message_bus.get_message_log()
    print(f"  Totalt antal meddelanden: {len(log)}")
    
    # Räkna topics
    topics = {}
    for msg in log:
        topic = msg['topic']
        topics[topic] = topics.get(topic, 0) + 1
    
    print("  Meddelanden per topic:")
    for topic, count in sorted(topics.items()):
        print(f"    {topic}: {count}")
    
    print()
    print("=" * 60)
    print("Sprint 2 Demo avslutad")
    print("RL-agenter är nu tränade och förbättrar strategier!")
    print("=" * 60)


if __name__ == "__main__":
    main()

