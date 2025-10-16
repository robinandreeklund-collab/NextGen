"""
websocket_test.py - Realtidstest av hela systemet med Finnhub WebSocket

Beskrivning:
    Testar hela NextGen AI Trader-systemet med live data från Finnhub WebSocket.
    Kör 10 fasta NASDAQ 100-aktier och visar realtidsflöde genom alla moduler.

Funktioner:
    - Ansluter till Finnhub WebSocket med 10 fasta symboler
    - Visar dataflöde genom strategic_memory, meta_agent_evolution, agent_manager
    - Realtidsanalys av korrelationer, agent performance och evolution
    - Live feedback-routing och mönsteranalys
    - Kontinuerlig körning tills användaren stoppar

API Begränsningar:
    - Max 60 REST API calls/min (vi använder endast WebSocket)
    - WebSocket: Max 50 symboler samtidigt (vi använder 10)

Användning:
    python websocket_test.py
    
Stoppa med Ctrl+C för att avsluta och visa sammanfattning.
"""

import asyncio
import json
import time
import signal
import sys
from typing import Dict, Any, List
from datetime import datetime
import websocket

# Importera våra moduler
from modules.message_bus import MessageBus
from modules.strategic_memory_engine import StrategicMemoryEngine
from modules.meta_agent_evolution_engine import MetaAgentEvolutionEngine
from modules.agent_manager import AgentManager
from modules.feedback_router import FeedbackRouter
from modules.feedback_analyzer import FeedbackAnalyzer
from modules.introspection_panel import IntrospectionPanel
from modules.indicator_registry import IndicatorRegistry
from modules.strategy_engine import StrategyEngine
from modules.risk_manager import RiskManager
from modules.decision_engine import DecisionEngine
from modules.execution_engine import ExecutionEngine
from modules.portfolio_manager import PortfolioManager
from modules.rl_controller import RLController


class WebSocketTester:
    """Testar hela systemet med live Finnhub WebSocket data."""
    
    def __init__(self, api_key: str):
        """
        Initialiserar WebSocket-testet.
        
        Args:
            api_key: Finnhub API-nyckel
        """
        self.api_key = api_key
        self.ws = None
        self.running = False
        
        # 10 fasta NASDAQ 100-symboler
        self.symbols = [
            'AAPL',   # Apple
            'MSFT',   # Microsoft
            'GOOGL',  # Alphabet
            'AMZN',   # Amazon
            'TSLA',   # Tesla
            'NVDA',   # NVIDIA
            'META',   # Meta
            'NFLX',   # Netflix
            'AMD',    # AMD
            'INTC'    # Intel
        ]
        
        # Initialisera alla moduler
        self.message_bus = MessageBus()
        self.setup_modules()
        
        # Statistik
        self.stats = {
            'messages_received': 0,
            'trades_processed': 0,
            'decisions_made': 0,
            'buy_count': 0,
            'sell_count': 0,
            'hold_count': 0,
            'evolution_events': 0,
            'start_time': time.time(),
            'symbol_counts': {symbol: 0 for symbol in self.symbols}
        }
        
        # Signal handler för graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Debug mode för detaljerad loggning
        self.debug_mode = True
        self.debug_counter = 0
    
    def setup_modules(self) -> None:
        """Initialiserar alla Sprint 1-4 moduler."""
        print("🔧 Initialiserar moduler...")
        
        # Sprint 4 moduler
        self.strategic_memory = StrategicMemoryEngine(self.message_bus)
        self.meta_evolution = MetaAgentEvolutionEngine(self.message_bus)
        self.agent_manager = AgentManager(self.message_bus)
        
        # Sprint 3 moduler
        self.feedback_router = FeedbackRouter(self.message_bus)
        self.feedback_analyzer = FeedbackAnalyzer(self.message_bus)
        self.introspection_panel = IntrospectionPanel(self.message_bus)
        
        # Sprint 2 moduler
        self.rl_controller = RLController(self.message_bus)
        
        # Sprint 1 moduler
        self.indicator_registry = IndicatorRegistry(self.api_key, self.message_bus)
        self.strategy_engine = StrategyEngine(self.message_bus)
        self.risk_manager = RiskManager(self.message_bus)
        self.decision_engine = DecisionEngine(self.message_bus)
        self.execution_engine = ExecutionEngine(self.message_bus)
        self.portfolio_manager = PortfolioManager(
            message_bus=self.message_bus,
            start_capital=1000.0,
            transaction_fee=0.0025
        )
        
        print("✅ Alla moduler initialiserade")
    
    def on_message(self, ws, message: str) -> None:
        """
        Callback för WebSocket-meddelanden.
        
        Args:
            ws: WebSocket-instans
            message: Mottaget meddelande (JSON)
        """
        try:
            data = json.loads(message)
            
            # Hantera trade-data
            if data.get('type') == 'trade':
                for trade in data.get('data', []):
                    self.process_trade(trade)
            
            self.stats['messages_received'] += 1
            
            # Visa progress varje 10:e meddelande
            if self.stats['messages_received'] % 10 == 0:
                self.print_progress()
        
        except Exception as e:
            print(f"⚠️  Fel vid meddelandehantering: {e}")
    
    def process_trade(self, trade: Dict[str, Any]) -> None:
        """
        Processar en trade genom hela systemet.
        
        Args:
            trade: Trade-data från Finnhub
                {
                    's': 'AAPL',      # Symbol
                    'p': 150.25,      # Price
                    'v': 100,         # Volume
                    't': 1634567890   # Timestamp (ms)
                }
        """
        symbol = trade.get('s')
        price = trade.get('p')
        volume = trade.get('v')
        timestamp = trade.get('t')
        
        if not all([symbol, price, volume, timestamp]):
            return
        
        # Uppdatera statistik
        self.stats['trades_processed'] += 1
        if symbol in self.stats['symbol_counts']:
            self.stats['symbol_counts'][symbol] += 1
        
        # Debug: Visa första trades för varje symbol
        if self.debug_mode and self.stats['symbol_counts'][symbol] <= 2:
            print(f"\n📊 Trade #{self.stats['trades_processed']}: {symbol} @ ${price:.2f} (vol: {volume})")
        
        # 1. Publicera market data
        self.message_bus.publish('market_data', {
            'symbol': symbol,
            'price': price,
            'volume': volume,
            'timestamp': timestamp
        })
        
        # 2. Hämta indikatorer (cachar och uppdaterar var 5:e trade)
        if self.stats['trades_processed'] % 5 == 0:
            indicators = self.indicator_registry.get_indicators(symbol)
            if self.debug_mode and self.debug_counter < 3:
                print(f"   📈 Indikatorer hämtade för {symbol}")
                print(f"      RSI: {indicators.get('technical', {}).get('RSI', 'N/A')}")
                print(f"      MACD: {indicators.get('technical', {}).get('MACD', {}).get('histogram', 'N/A')}")
                self.debug_counter += 1
        
        # 3. Strategibeslut (varje 10:e trade för att simulera thoughtful decisions)
        if self.stats['trades_processed'] % 10 == 0:
            self.make_trading_decision(symbol, price)
    
    def make_trading_decision(self, symbol: str, price: float) -> None:
        """
        Fattar handelsbeslut för en symbol.
        
        Args:
            symbol: Aktiesymbol
            price: Aktuellt pris
        """
        # Spara priset för execution_engine
        self.current_prices = getattr(self, 'current_prices', {})
        self.current_prices[symbol] = price
        
        # Strategy engine genererar förslag (tar symbol)
        proposal = self.strategy_engine.generate_proposal(symbol)
        
        # Debug: Visa decision flow för första besluten
        if self.debug_mode and self.stats['decisions_made'] < 5:
            print(f"\n🤔 Beslut #{self.stats['trades_processed']//10} för {symbol}:")
            print(f"   💡 Strategy proposal: {proposal.get('action')} "
                  f"(confidence: {proposal.get('confidence', 0):.2f})")
            print(f"      Reasoning: {proposal.get('reasoning', 'N/A')}")
        
        # Publicera förslag till message_bus så decision_engine kan ta emot det
        self.strategy_engine.publish_proposal(proposal)
        
        # Risk manager bedömer risk (tar symbol)
        risk_profile = self.risk_manager.assess_risk(symbol)
        
        if self.debug_mode and self.stats['decisions_made'] < 5:
            print(f"   ⚠️  Risk assessment: {risk_profile.get('risk_level', 'N/A')} "
                  f"(score: {risk_profile.get('risk_score', 0):.2f})")
        
        # Publicera riskprofil till message_bus
        self.risk_manager.publish_risk_profile(risk_profile)
        
        # Decision engine fattar beslut (tar symbol och price för insufficient funds check)
        # Nu har den både proposal och risk_profile via message_bus
        decision = self.decision_engine.make_decision(symbol, current_price=price)
        
        if self.debug_mode and self.stats['decisions_made'] < 5:
            print(f"   ⚖️  Final decision: {decision.get('action')} "
                  f"(confidence: {decision.get('confidence', 0):.2f})")
        
        if decision and decision.get('action') != 'HOLD':
            # Sätt aktuellt pris för execution
            decision['current_price'] = price
            
            # Räkna beslut
            action = decision.get('action')
            if action == 'BUY':
                self.stats['buy_count'] += 1
            elif action == 'SELL':
                self.stats['sell_count'] += 1
            
            # Execution engine exekverar
            execution_result = self.execution_engine.execute_trade(decision)
            
            # Debug: Visa execution detaljer
            if self.debug_mode and self.stats['decisions_made'] < 10:
                print(f"\n   🔨 EXECUTION #{self.stats['decisions_made'] + 1}:")
                print(f"      {execution_result.get('action')} {execution_result.get('quantity')} {symbol}")
                print(f"      @ ${execution_result.get('executed_price', 0):.2f} "
                      f"(market: ${execution_result.get('market_price', 0):.2f})")
                print(f"      Cost: ${execution_result.get('total_cost', 0):.2f}")
                print(f"      Slippage: {execution_result.get('slippage', 0)*100:.3f}%")
            
            # Portfolio manager uppdaterar
            self.portfolio_manager.update_portfolio(execution_result)
            
            if self.debug_mode and self.stats['decisions_made'] < 10:
                portfolio = self.portfolio_manager.get_status(self.current_prices)
                print(f"   💰 Portfolio: ${portfolio.get('cash', 0):.2f} cash, "
                      f"${portfolio.get('total_value', 0):.2f} total")
                
                # Visa alla positioner
                if portfolio.get('positions'):
                    print(f"   📊 Innehav ({len(portfolio['positions'])} positioner):")
                    for sym, pos in portfolio['positions'].items():
                        current_price = self.current_prices.get(sym, pos['avg_price'])
                        current_value = pos['quantity'] * current_price
                        pnl = current_value - (pos['quantity'] * pos['avg_price'])
                        print(f"      {sym}: {pos['quantity']} @ avg ${pos['avg_price']:.2f} "
                              f"(nuv: ${current_price:.2f}, P&L: ${pnl:.2f})")
            
            self.stats['decisions_made'] += 1
            
            # Logga till strategic memory
            self.strategic_memory.log_decision({
                'symbol': symbol,
                'action': decision.get('action'),
                'price': price,
                'execution_result': execution_result
            })
            
            # Kontrollera evolution events
            if len(self.meta_evolution.evolution_history) > self.stats['evolution_events']:
                self.stats['evolution_events'] = len(self.meta_evolution.evolution_history)
                print(f"\n🧬 Evolution event detekterad!")
        else:
            # HOLD decision
            self.stats['hold_count'] += 1
            if self.debug_mode and self.stats['decisions_made'] < 5:
                print(f"   ⏸️  Decision: HOLD - ingen trade")
    
    def print_progress(self) -> None:
        """Skriver ut progress-information."""
        runtime = time.time() - self.stats['start_time']
        minutes = int(runtime // 60)
        seconds = int(runtime % 60)
        
        print(f"\n{'='*80}")
        print(f"⏱️  Runtime: {minutes}m {seconds}s")
        print(f"📨 Meddelanden: {self.stats['messages_received']}")
        print(f"💹 Trades processade: {self.stats['trades_processed']}")
        print(f"🎯 Beslut fattade: {self.stats['decisions_made']}")
        print(f"🧬 Evolution events: {self.stats['evolution_events']}")
        
        # Visa status om beslut med BUY/SELL/HOLD percentages
        if self.stats['trades_processed'] > 50:
            total_decision_points = self.stats['trades_processed'] // 10
            buy_pct = (self.stats['buy_count'] / total_decision_points * 100) if total_decision_points > 0 else 0
            sell_pct = (self.stats['sell_count'] / total_decision_points * 100) if total_decision_points > 0 else 0
            hold_pct = (self.stats['hold_count'] / total_decision_points * 100) if total_decision_points > 0 else 0
            
            print(f"\n📊 Beslutsdistribution ({total_decision_points} totalt):")
            print(f"   🟢 BUY:  {self.stats['buy_count']} ({buy_pct:.1f}%)")
            print(f"   🔴 SELL: {self.stats['sell_count']} ({sell_pct:.1f}%)")
            print(f"   ⚪ HOLD: {self.stats['hold_count']} ({hold_pct:.1f}%)")
        
        # Portfolio status med current prices
        portfolio_status = self.portfolio_manager.get_status(self.current_prices if hasattr(self, 'current_prices') else None)
        print(f"\n💰 Portfolio:")
        print(f"   Kapital: ${portfolio_status.get('cash', 0):.2f}")
        print(f"   Värde: ${portfolio_status.get('total_value', 0):.2f}")
        print(f"   P&L: ${portfolio_status.get('pnl', 0):.2f} ({portfolio_status.get('pnl_pct', 0):.1f}%)")
        
        # Visa innehav
        if portfolio_status.get('positions'):
            print(f"   Positioner: {len(portfolio_status['positions'])}")
        
        # Top 3 mest aktiva symboler
        top_symbols = sorted(
            self.stats['symbol_counts'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        print(f"\n📊 Mest aktiva symboler:")
        for symbol, count in top_symbols:
            print(f"   {symbol}: {count} trades")
        
        # Strategic Memory insights
        insights = self.strategic_memory.generate_insights()
        if insights['total_decisions'] > 0:
            print(f"\n🧠 Strategic Memory:")
            print(f"   Beslut: {insights['total_decisions']}")
            print(f"   Success rate: {insights['success_rate']*100:.1f}%")
            if insights['best_indicators']:
                top_indicator = insights['best_indicators'][0]
                print(f"   Bästa indikator: {top_indicator['name']} "
                      f"({top_indicator['success_rate']*100:.1f}%)")
        
        # Agent status
        agent_profiles = self.agent_manager.get_all_profiles()
        print(f"\n🤖 Agenter:")
        for agent_id, profile in list(agent_profiles.items())[:2]:
            print(f"   {profile['name']}: v{profile['version']}")
        
        print(f"{'='*80}\n")
    
    def print_final_summary(self) -> None:
        """Skriver ut slutlig sammanfattning."""
        print("\n" + "="*80)
        print("📊 SLUTLIG SAMMANFATTNING")
        print("="*80)
        
        runtime = time.time() - self.stats['start_time']
        print(f"\n⏱️  Total körtid: {int(runtime//60)}m {int(runtime%60)}s")
        print(f"📨 Totalt meddelanden: {self.stats['messages_received']}")
        print(f"💹 Totalt trades: {self.stats['trades_processed']}")
        print(f"🎯 Totalt beslut: {self.stats['decisions_made']}")
        print(f"🧬 Evolution events: {self.stats['evolution_events']}")
        
        # Beslutsdistribution
        total_decision_points = self.stats['trades_processed'] // 10 if self.stats['trades_processed'] > 0 else 1
        buy_pct = (self.stats['buy_count'] / total_decision_points * 100)
        sell_pct = (self.stats['sell_count'] / total_decision_points * 100)
        hold_pct = (self.stats['hold_count'] / total_decision_points * 100)
        
        print(f"\n📊 BESLUTSDISTRIBUTION:")
        print(f"   🟢 BUY:  {self.stats['buy_count']} ({buy_pct:.1f}%)")
        print(f"   🔴 SELL: {self.stats['sell_count']} ({sell_pct:.1f}%)")
        print(f"   ⚪ HOLD: {self.stats['hold_count']} ({hold_pct:.1f}%)")
        
        # Portfolio resultat med current prices
        portfolio_status = self.portfolio_manager.get_status(self.current_prices if hasattr(self, 'current_prices') else None)
        final_value = portfolio_status.get('total_value', 1000)
        profit = final_value - 1000
        roi = (profit / 1000) * 100
        
        print(f"\n💰 PORTFOLIO RESULTAT:")
        print(f"   Start kapital: $1000.00")
        print(f"   Slutligt värde: ${final_value:.2f}")
        print(f"   Profit/Loss: ${profit:.2f}")
        print(f"   ROI: {roi:.2f}%")
        
        # Visa innehav
        if portfolio_status.get('positions'):
            print(f"\n   📊 SLUTLIGA INNEHAV ({len(portfolio_status['positions'])} positioner):")
            for symbol, pos in portfolio_status['positions'].items():
                current_price = self.current_prices.get(symbol, pos['avg_price']) if hasattr(self, 'current_prices') else pos['avg_price']
                current_value = pos['quantity'] * current_price
                pnl = current_value - (pos['quantity'] * pos['avg_price'])
                print(f"      {symbol}: {pos['quantity']} @ ${pos['avg_price']:.2f} "
                      f"→ ${current_price:.2f} (P&L: ${pnl:.2f})")
        
        # Strategic Memory sammanfattning
        insights = self.strategic_memory.generate_insights()
        print(f"\n🧠 STRATEGIC MEMORY:")
        print(f"   Totalt beslut: {insights['total_decisions']}")
        print(f"   Success rate: {insights['success_rate']*100:.1f}%")
        print(f"   Genomsnittlig profit: ${insights['average_profit']:.2f}")
        
        if insights['best_indicators']:
            print(f"\n   📈 Bästa indikatorer:")
            for i, indicator in enumerate(insights['best_indicators'][:3], 1):
                print(f"      {i}. {indicator['name']}: "
                      f"{indicator['success_rate']*100:.1f}% success, "
                      f"${indicator['average_profit']:.2f} avg profit")
        
        if insights['recommendations']:
            print(f"\n   💡 Rekommendationer:")
            for rec in insights['recommendations']:
                print(f"      - {rec}")
        
        # Agent Evolution sammanfattning
        evolution_tree = self.meta_evolution.generate_evolution_tree()
        print(f"\n🧬 AGENT EVOLUTION:")
        print(f"   Total evolution events: {evolution_tree['total_evolution_events']}")
        
        for agent_id, data in evolution_tree['agents'].items():
            if data['evolution_count'] > 0:
                print(f"   {agent_id}: {data['evolution_count']} evolutioner")
        
        # Agent versioner
        print(f"\n🤖 AGENT VERSIONER:")
        agent_profiles = self.agent_manager.get_all_profiles()
        for agent_id, profile in agent_profiles.items():
            versions = self.agent_manager.get_version_history(agent_id)
            print(f"   {profile['name']}: v{profile['version']} "
                  f"({len(versions)} versioner)")
        
        # Trades per symbol
        print(f"\n📊 TRADES PER SYMBOL:")
        sorted_symbols = sorted(
            self.stats['symbol_counts'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for symbol, count in sorted_symbols:
            percentage = (count / self.stats['trades_processed'] * 100) if self.stats['trades_processed'] > 0 else 0
            print(f"   {symbol}: {count} trades ({percentage:.1f}%)")
        
        # Modul diagnostik
        print(f"\n🔍 MODUL DIAGNOSTIK:")
        
        # Strategy engine
        strategy_indicators = len(self.strategy_engine.current_indicators)
        print(f"   Strategy Engine: {strategy_indicators} symboler med indikatorer")
        
        # Risk manager
        risk_indicators = len(self.risk_manager.current_indicators)
        print(f"   Risk Manager: {risk_indicators} symboler med riskdata")
        
        # Decision engine
        decision_proposals = len(self.decision_engine.trade_proposals)
        print(f"   Decision Engine: {decision_proposals} aktiva förslag")
        
        # Strategic memory
        memory_summary = self.strategic_memory.get_performance_summary()
        print(f"   Strategic Memory: {memory_summary['total_decisions']} beslut loggade")
        print(f"   Strategic Memory: {memory_summary['total_executions']} executions loggade")
        
        # Feedback router
        feedback_count = len(self.feedback_router.feedback_log)
        print(f"   Feedback Router: {feedback_count} feedback events")
        
        # Execution engine och portfolio status
        portfolio = self.portfolio_manager.get_status()
        print(f"\n📊 EXECUTION & PORTFOLIO:")
        print(f"   Totalt beslut (BUY/SELL): {self.stats['decisions_made']}")
        print(f"   Portfolio cash: ${portfolio.get('cash', 0):.2f}")
        print(f"   Portfolio värde: ${portfolio.get('total_value', 0):.2f}")
        print(f"   Antal positioner: {len(portfolio.get('positions', {}))}")
        
        if self.stats['decisions_made'] == 0 and self.stats['trades_processed'] > 0:
            print(f"\n⚠️  DIAGNOS: Inga TRADES exekverade (alla beslut = HOLD)!")
            print(f"   Detta är NORMALT beteende - systemet handlar endast vid tydliga signaler.")
            print(f"\n   📊 Beslutskriterier för BUY:")
            print(f"      - RSI < 30 (översåld) ELLER")
            print(f"      - RSI < 50 + MACD > 0.5 + Analyst BUY")
            print(f"\n   📊 Beslutskriterier för SELL:")
            print(f"      - RSI > 70 (överköpt) ELLER")
            print(f"      - RSI > 50 + MACD < -0.5 + Analyst SELL")
            print(f"\n   💡 Indikatorer genereras nu DYNAMISKT:")
            print(f"      - RSI varierar 20-80 baserat på symbol + tidsstämpel")
            print(f"      - MACD varierar -3 till +3")
            print(f"      - Analyst ratings varierar (BUY/HOLD/SELL)")
            print(f"      - Värden ändras över tid (per minut för RSI)")
            print(f"\n   🔧 För att se fler trades:")
            print(f"      1. Kör längre tid (RSI oscillerar, passerar <30 eller >70)")
            print(f"      2. Live WebSocket-data + tid = större variation")
            print(f"      3. Inga fasta stub-värden längre - allt beräknas dynamiskt")
        elif portfolio.get('total_value', 1000) == 1000 and self.stats['decisions_made'] > 0:
            print(f"\n⚠️  OBS: Portfolio värde oförändrat trots {self.stats['decisions_made']} trades!")
            print(f"   Möjliga orsaker:")
            print(f"   - Execution prices använder live market data nu")
            print(f"   - Strategic memory loggar nu alla trades korrekt")
            print(f"   - Portfolio uppdateras efter varje execution")
        
        print("\n" + "="*80)
        print("Tack för att du testade NextGen AI Trader! 🚀")
        print("="*80 + "\n")
    
    def on_error(self, ws, error: Any) -> None:
        """Callback för WebSocket-fel."""
        print(f"❌ WebSocket-fel: {error}")
    
    def on_close(self, ws, close_status_code, close_msg) -> None:
        """Callback när WebSocket stängs."""
        print(f"\n🔌 WebSocket-anslutning stängd")
        if close_status_code or close_msg:
            print(f"   Status: {close_status_code}, Meddelande: {close_msg}")
    
    def on_open(self, ws) -> None:
        """Callback när WebSocket öppnas."""
        print(f"\n✅ WebSocket-anslutning etablerad!")
        print(f"📡 Prenumererar på {len(self.symbols)} symboler...")
        
        # Prenumerera på alla symboler
        for symbol in self.symbols:
            subscribe_message = {
                'type': 'subscribe',
                'symbol': symbol
            }
            ws.send(json.dumps(subscribe_message))
            print(f"   ✓ {symbol}")
        
        print(f"\n🚀 Live trading-systemet körs nu!")
        print(f"⏹️  Tryck Ctrl+C för att stoppa och visa sammanfattning")
        print(f"\n💡 DEBUG MODE: Aktiv - visar detaljerad info för första trades och beslut")
        print(f"   Beslut fattas var 10:e trade, indikatorer uppdateras var 5:e trade\n")
    
    def signal_handler(self, sig, frame) -> None:
        """Hanterar Ctrl+C för graceful shutdown."""
        print("\n\n⏹️  Stoppar systemet...")
        self.running = False
        
        if self.ws:
            self.ws.close()
        
        self.print_final_summary()
        sys.exit(0)
    
    def run(self) -> None:
        """Kör WebSocket-testet."""
        print("\n" + "="*80)
        print("🚀 NextGen AI Trader - Live WebSocket Test")
        print("="*80)
        print(f"\n📡 Ansluter till Finnhub WebSocket...")
        print(f"🎯 Symboler: {', '.join(self.symbols)}")
        print(f"💰 Start kapital: $1000.00")
        print(f"💵 Transaktionsavgift: 0.25%\n")
        
        self.running = True
        
        # Skapa WebSocket-anslutning
        websocket_url = f"wss://ws.finnhub.io?token={self.api_key}"
        
        self.ws = websocket.WebSocketApp(
            websocket_url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open
        )
        
        # Kör WebSocket (blockerande)
        try:
            self.ws.run_forever()
        except KeyboardInterrupt:
            self.signal_handler(None, None)
        except Exception as e:
            print(f"❌ Oväntat fel: {e}")
            self.print_final_summary()


def main():
    """Huvudfunktion för WebSocket-test."""
    # Finnhub API-nyckel
    API_KEY = "d3in10hr01qmn7fkr2a0d3in10hr01qmn7fkr2ag"
    
    # Skapa och kör tester
    tester = WebSocketTester(API_KEY)
    tester.run()


if __name__ == "__main__":
    main()
