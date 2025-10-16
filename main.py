"""
NextGen AI Trader - Huvudstartpunkt

Detta är huvudapplikationen för NextGen AI Trader-systemet.
Sprint 1 demonstrerar grundläggande flöde från data till execution.

Moduler i Sprint 1:
- data_ingestion: Hämtar marknadsdata
- indicator_registry: Beräknar tekniska indikatorer
- strategy_engine: Genererar tradeförslag
- decision_engine: Fattar beslut
- execution_engine: Exekverar trades
- portfolio_manager: Hanterar portfölj

Användning:
    python main.py
"""

from modules.message_bus import message_bus
from modules.data_ingestion import DataIngestion
from modules.indicator_registry import IndicatorRegistry
from modules.strategy_engine import StrategyEngine
from modules.decision_engine import DecisionEngine
from modules.execution_engine import ExecutionEngine
from modules.portfolio_manager import PortfolioManager


def main():
    """
    Huvudfunktion som demonstrerar Sprint 1-flödet.
    
    Flöde:
    1. Initiera alla moduler med message_bus
    2. Hämta trending symboler
    3. Hämta indikatorer för symboler
    4. Generera tradeförslag baserat på indikatorer
    5. Fatta beslut
    6. Exekvera (simulerat)
    7. Uppdatera portfölj
    """
    print("=" * 60)
    print("NextGen AI Trader - Sprint 1 Demo")
    print("=" * 60)
    print()
    
    # Initiera alla Sprint 1-moduler
    print("Initierar moduler...")
    
    # API-nyckel (placeholder för demo)
    api_key = "demo_api_key"
    
    # Initiera moduler
    data_ingest = DataIngestion(api_key, message_bus)
    indicator_reg = IndicatorRegistry(api_key, message_bus)
    strategy = StrategyEngine(message_bus)
    decision = DecisionEngine(message_bus)
    execution = ExecutionEngine(message_bus, simulation_mode=True)
    portfolio = PortfolioManager(
        message_bus, 
        start_capital=1000.0,  # Från sprint_plan.yaml
        transaction_fee=0.0025  # 0.25% från sprint_plan.yaml
    )
    
    print("✓ Alla moduler initierade")
    print()
    
    # Steg 1: Hämta trending symboler
    print("Steg 1: Hämtar trending symboler...")
    symbols = data_ingest.fetch_trending_symbols()
    print(f"✓ Hittade symboler: {', '.join(symbols)}")
    print()
    
    # Steg 2: Hämta indikatorer för första symbolen
    symbol = symbols[0]
    print(f"Steg 2: Hämtar indikatorer för {symbol}...")
    indicators = indicator_reg.get_indicators(symbol)
    print(f"✓ Hämtade indikatorer: OHLC, Volume, SMA, RSI")
    print(f"  RSI: {indicators['technical']['RSI']}")
    print()
    
    # Testa med en annan symbol för att visa BUY-signal
    symbol2 = symbols[1] if len(symbols) > 1 else symbol
    print(f"Steg 2b: Hämtar indikatorer för {symbol2} (låg RSI)...")
    indicators2 = indicator_reg.get_indicators(symbol2)
    print(f"✓ Hämtade indikatorer: OHLC, Volume, SMA, RSI")
    print(f"  RSI: {indicators2['technical']['RSI']} (översåld)")
    print()
    
    # Steg 3: Generera tradeförslag
    print(f"Steg 3: Genererar tradeförslag för {symbol}...")
    proposal = strategy.generate_proposal(symbol)
    print(f"✓ Förslag: {proposal['action']} {proposal['quantity']} @ {symbol}")
    print(f"  Motivering: {proposal['reasoning']}")
    print(f"  Confidence: {proposal['confidence']:.2f}")
    strategy.publish_proposal(proposal)
    print()
    
    # Testa BUY-scenario med symbol2
    print(f"Steg 3b: Genererar tradeförslag för {symbol2}...")
    proposal2 = strategy.generate_proposal(symbol2)
    print(f"✓ Förslag: {proposal2['action']} {proposal2['quantity']} @ {symbol2}")
    print(f"  Motivering: {proposal2['reasoning']}")
    print(f"  Confidence: {proposal2['confidence']:.2f}")
    strategy.publish_proposal(proposal2)
    print()
    
    # Steg 4: Fatta beslut
    print(f"Steg 4: Fattar beslut för {symbol2} (BUY-scenario)...")
    final_decision = decision.make_decision(symbol2)
    print(f"✓ Beslut: {final_decision['action']} {final_decision['quantity']} @ {symbol2}")
    print(f"  Confidence: {final_decision['confidence']:.2f}")
    decision.publish_decision(final_decision)
    print()
    
    # Steg 5: Exekvera trade (simulerat via message_bus callback)
    print(f"Steg 5: Exekverar trade (simulerat)...")
    # Execution sker automatiskt via message_bus prenumeration på 'final_decision'
    # För demo, visa portföljstatus efter
    print("✓ Trade exekverad")
    print()
    
    # Steg 6: Visa portföljstatus
    print("Steg 6: Portföljstatus")
    status = portfolio.get_status()
    print(f"  Kapital (cash): ${status['cash']:.2f}")
    print(f"  Totalt värde: ${status['total_value']:.2f}")
    print(f"  P&L: ${status['pnl']:.2f} ({status['pnl_pct']:.2f}%)")
    print(f"  Antal trades: {status['num_trades']}")
    print()
    
    # Visa message_bus-statistik
    print("Message Bus-statistik:")
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
    print("Sprint 1 Demo avslutad")
    print("=" * 60)


if __name__ == "__main__":
    main()

