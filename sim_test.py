"""
sim_test.py - Simulerad test av hela systemet med aggressiv data

Beskrivning:
    Testar hela NextGen AI Trader-systemet med simulerad aggressiv data istället för live WebSocket.
    Detta möjliggör kontrollerad testning med extrema marknadsrörelser för att trigga många beslut.
    
    Sprint 4.3: Visar alla adaptiva parametrar i realtid inklusive nya modulparametrar.

Funktioner:
    - Simulerar aggressiva prisrörelser åt båda håll
    - Visar dataflöde genom alla moduler
    - Realtidsvisning av ALLA adaptiva parametrar (Sprint 4.3)
    - Kontinuerlig körning med konfigurerbar hastighet

Användning:
    python sim_test.py
    
Stoppa med Ctrl+C för att avsluta och visa sammanfattning.
"""

import time
import random
import signal
import sys
from typing import Dict, Any, List
from datetime import datetime
import math

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
from modules.vote_engine import VoteEngine


class SimulatedTester:
    """Testar hela systemet med simulerad aggressiv data."""
    
    def __init__(self):
        """Initialiserar den simulerade testern."""
        self.running = False
        
        # Symboler för test
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        
        # Initiala priser
        self.base_prices = {
            'AAPL': 150.0,
            'MSFT': 300.0,
            'GOOGL': 120.0,
            'AMZN': 140.0,
            'TSLA': 200.0
        }
        
        self.current_prices = self.base_prices.copy()
        self.price_trends = {symbol: 0.0 for symbol in self.symbols}
        
        # Initialisera alla moduler
        self.message_bus = MessageBus()
        self.setup_modules()
        
        # Statistik
        self.stats = {
            'iterations': 0,
            'trades_processed': 0,
            'decisions_made': 0,  # Alla beslut (BUY/SELL/HOLD)
            'buy_decisions': 0,   # BUY-beslut (före execution)
            'sell_decisions': 0,  # SELL-beslut (före execution)
            'hold_decisions': 0,  # HOLD-beslut
            'buy_executions': 0,  # Genomförda BUY i portfolio
            'sell_executions': 0, # Genomförda SELL i portfolio
            'insufficient_funds_count': 0,  # BUY blockerade
            'insufficient_holdings_count': 0,  # SELL blockerade
            'start_time': time.time(),
            'execution_log': []  # Endast genomförda trades
        }
        
        # Signal handler för graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Sprint 4.3: Spåra alla adaptiva parametrar
        self.parameter_history = []
        self.sprint43_params_history = []
    
    def setup_modules(self) -> None:
        """Initialiserar alla Sprint 1-4.3 moduler."""
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
        
        # Sprint 1 moduler (utan API key för simulering)
        self.strategy_engine = StrategyEngine(self.message_bus)
        self.risk_manager = RiskManager(self.message_bus)
        self.decision_engine = DecisionEngine(self.message_bus)
        self.execution_engine = ExecutionEngine(self.message_bus)
        self.portfolio_manager = PortfolioManager(
            message_bus=self.message_bus,
            start_capital=1000.0,
            transaction_fee=0.0025
        )
        
        # Sprint 4.3 modul
        self.vote_engine = VoteEngine(self.message_bus)
        
        # Sprint 4.3: Prenumerera på parameter_adjustment
        self.message_bus.subscribe('parameter_adjustment', self._on_parameter_adjustment)
        
        print("✅ Alla moduler initialiserade")
    
    def _on_parameter_adjustment(self, adjustment: Dict[str, Any]) -> None:
        """
        Callback för parameter adjustments (Sprint 4.3).
        Loggar både meta-parametrar och modulparametrar.
        """
        # Hantera både 'parameters' och 'adjusted_parameters' keys
        params = adjustment.get('parameters', adjustment.get('adjusted_parameters', {}))
        
        param_entry = {
            'timestamp': time.time(),
            'parameters': params,
            'source': adjustment.get('source', 'unknown')
        }
        self.parameter_history.append(param_entry)
        
        # Behåll senaste 100
        if len(self.parameter_history) > 100:
            self.parameter_history = self.parameter_history[-100:]
    
    def generate_aggressive_price_movement(self, symbol: str) -> float:
        """
        Genererar aggressiva prisrörelser för att trigga många beslut.
        
        Args:
            symbol: Aktiesymbol
            
        Returns:
            Nytt pris med aggressiv rörelse
        """
        # Ändra trend ibland för volatilitet
        if random.random() < 0.1:  # 10% chans att byta trend
            self.price_trends[symbol] = random.choice([-1, 1]) * random.uniform(0.02, 0.05)
        
        # Lägg till lite noise
        noise = random.uniform(-0.02, 0.02)
        
        # Uppdatera pris med trend + noise
        price_change_pct = self.price_trends[symbol] + noise
        new_price = self.current_prices[symbol] * (1 + price_change_pct)
        
        # Håll priserna inom rimliga gränser (50% till 150% av base)
        min_price = self.base_prices[symbol] * 0.5
        max_price = self.base_prices[symbol] * 1.5
        new_price = max(min_price, min(max_price, new_price))
        
        self.current_prices[symbol] = new_price
        return new_price
    
    def generate_aggressive_indicators(self, symbol: str, iteration: int) -> Dict[str, Any]:
        """
        Genererar aggressiva indikatorer som varierar kraftigt.
        
        Args:
            symbol: Aktiesymbol
            iteration: Nuvarande iteration för variation
            
        Returns:
            Indikatordata med extrema värden
        """
        # RSI oscillerar mellan extremer (10-90)
        rsi_base = 50 + 40 * math.sin(iteration * 0.3 + hash(symbol) % 10)
        rsi = max(10, min(90, rsi_base))
        
        # MACD varierar kraftigt
        macd_hist = 3 * math.sin(iteration * 0.2 + hash(symbol) % 5)
        
        # ATR för volatilitet
        price_volatility = abs(self.price_trends.get(symbol, 0))
        atr = 2.0 + price_volatility * 50
        
        # Analyst ratings baserat på trend
        if rsi < 30:
            analyst = 'STRONG_BUY'
        elif rsi > 70:
            analyst = 'SELL'
        elif rsi < 45:
            analyst = 'BUY'
        else:
            analyst = 'HOLD'
        
        return {
            'symbol': symbol,
            'technical': {
                'RSI': rsi,
                'MACD': {'histogram': macd_hist},
                'ATR': atr,
                'Volume': random.randint(500000, 3000000)
            },
            'fundamental': {
                'AnalystRatings': {'consensus': analyst}
            }
        }
    
    def simulate_iteration(self) -> None:
        """Simulerar en iteration av trading."""
        self.stats['iterations'] += 1
        iteration = self.stats['iterations']
        
        # Process varje symbol
        for symbol in self.symbols:
            # Generera aggressiv prisrörelse
            new_price = self.generate_aggressive_price_movement(symbol)
            
            # Publicera market data
            self.message_bus.publish('market_data', {
                'symbol': symbol,
                'price': new_price,
                'volume': random.randint(100, 1000),
                'timestamp': int(time.time() * 1000)
            })
            
            # Generera och publicera aggressiva indikatorer
            indicators = self.generate_aggressive_indicators(symbol, iteration)
            self.message_bus.publish('indicator_data', indicators)
            
            # Uppdatera modulernas indikatordata direkt
            self.strategy_engine.current_indicators[symbol] = indicators
            self.risk_manager.current_indicators[symbol] = indicators
            
            self.stats['trades_processed'] += 1
            
            # Fatta beslut varje iteration (aggressivt)
            self.make_trading_decision(symbol, new_price)
    
    def make_trading_decision(self, symbol: str, price: float) -> None:
        """
        Fattar handelsbeslut för en symbol.
        
        Args:
            symbol: Aktiesymbol
            price: Aktuellt pris
        """
        # Strategy engine genererar förslag
        proposal = self.strategy_engine.generate_proposal(symbol)
        self.strategy_engine.publish_proposal(proposal)
        
        # Risk manager bedömer risk
        risk_profile = self.risk_manager.assess_risk(symbol)
        self.risk_manager.publish_risk_profile(risk_profile)
        
        # Decision engine fattar beslut
        decision = self.decision_engine.make_decision(symbol, current_price=price)
        
        # Räkna alla beslut
        self.stats['decisions_made'] += 1
        
        if decision and decision.get('action') != 'HOLD':
            # Räkna beslut per typ (före execution)
            if decision.get('action') == 'BUY':
                self.stats['buy_decisions'] += 1
            elif decision.get('action') == 'SELL':
                self.stats['sell_decisions'] += 1
            
            # Sätt aktuellt pris för execution
            decision['current_price'] = price
            original_action = decision.get('action')
            
            # Execution engine exekverar
            execution_result = self.execution_engine.execute_trade(decision)
            
            # Räkna vad som hände efter execution
            if execution_result.get('action') == 'HOLD':
                # Beslut blockerat - räknas som blockerad
                if original_action == 'BUY':
                    self.stats['insufficient_funds_count'] += 1
                elif original_action == 'SELL':
                    self.stats['insufficient_holdings_count'] += 1
            elif execution_result.get('success'):
                # Genomförd execution - hamnar i portfolio
                executed_action = execution_result.get('action')
                self.stats['execution_log'].append({
                    'symbol': symbol,
                    'action': executed_action,
                    'quantity': execution_result.get('quantity'),
                    'price': execution_result.get('executed_price'),
                    'cost': execution_result.get('total_cost'),
                    'timestamp': time.time()
                })
                
                # Räkna genomförda executions
                if executed_action == 'BUY':
                    self.stats['buy_executions'] += 1
                elif executed_action == 'SELL':
                    self.stats['sell_executions'] += 1
            
            # Portfolio manager uppdaterar
            self.portfolio_manager.update_portfolio(execution_result)
            
            # Logga till strategic memory
            self.strategic_memory.log_decision({
                'symbol': symbol,
                'action': decision.get('action'),
                'price': price,
                'execution_result': execution_result
            })
            
            # Publicera reward för att trigga RL parameter adjustments
            # Reward baseras på portfolio performance
            portfolio = self.portfolio_manager.get_status(self.current_prices)
            portfolio_value = portfolio.get('total_value', 1000)
            reward_value = (portfolio_value - 1000) / 1000  # Normalized reward
            
            self.message_bus.publish('reward', {
                'value': reward_value,
                'portfolio_value': portfolio_value,
                'timestamp': time.time()
            })
        else:
            # HOLD beslut
            self.stats['hold_decisions'] += 1
    
    def print_progress(self) -> None:
        """Skriver ut progress med Sprint 4.3 adaptiva parametrar."""
        runtime = time.time() - self.stats['start_time']
        
        print(f"\n{'='*90}")
        print(f"⏱️  Iteration: {self.stats['iterations']} | Runtime: {int(runtime)}s")
        print(f"💹 Trades processade: {self.stats['trades_processed']}")
        
        # Decisions (alla beslut fattade av decision_engine)
        print(f"\n🎯 Decisions (Beslut):")
        print(f"   Totalt: {self.stats['decisions_made']}")
        print(f"   🟢 BUY:  {self.stats['buy_decisions']}")
        print(f"   🔴 SELL: {self.stats['sell_decisions']}")
        print(f"   ⚪ HOLD: {self.stats['hold_decisions']}")
        
        # Executions (endast de som verkligen hamnade i portfolio)
        total_executions = self.stats['buy_executions'] + self.stats['sell_executions']
        print(f"\n💼 Executions (I Portfolio):")
        print(f"   Totalt: {total_executions}")
        print(f"   ✅ BUY:  {self.stats['buy_executions']} genomförda köp")
        print(f"   ✅ SELL: {self.stats['sell_executions']} genomförda försäljningar")
        if self.stats['insufficient_funds_count'] + self.stats['insufficient_holdings_count'] > 0:
            print(f"   ⚠️  Blockerade: {self.stats['insufficient_funds_count']} BUY (no funds), "
                  f"{self.stats['insufficient_holdings_count']} SELL (no holdings)")
        
        # Portfolio
        portfolio = self.portfolio_manager.get_status(self.current_prices)
        pnl = portfolio.get('total_value', 1000) - 1000
        print(f"\n💰 Portfolio:")
        print(f"   Cash: ${portfolio.get('cash', 0):.2f}")
        print(f"   Total värde: ${portfolio.get('total_value', 0):.2f}")
        print(f"   P&L: ${pnl:.2f} ({(pnl/1000*100):+.1f}%)")
        print(f"   Executions i portfolio: {total_executions} trades")
        
        # Positioner
        if portfolio.get('positions'):
            print(f"   📊 {len(portfolio['positions'])} positioner:")
            for sym, pos in list(portfolio['positions'].items())[:3]:
                current_price = self.current_prices.get(sym, pos['avg_price'])
                pnl = (current_price - pos['avg_price']) * pos['quantity']
                print(f"      {sym}: {pos['quantity']}@${pos['avg_price']:.2f} "
                      f"→ ${current_price:.2f} (P&L: ${pnl:.2f})")
        
        # Priser
        print(f"\n📈 Nuvarande priser:")
        for sym in self.symbols:
            change_pct = ((self.current_prices[sym] - self.base_prices[sym]) / self.base_prices[sym]) * 100
            trend_emoji = "📈" if change_pct > 0 else "📉"
            print(f"   {sym}: ${self.current_prices[sym]:.2f} {trend_emoji} ({change_pct:+.1f}%)")
        
        # Sprint 4.3: ALLA ADAPTIVA PARAMETRAR
        print(f"\n{'='*90}")
        print(f"🔧 ADAPTIVA PARAMETRAR (Sprint 4.3) - {len(self.parameter_history)} adjustments totalt")
        
        # Diagnostik om inga adjustments
        if len(self.parameter_history) == 0 and self.stats['buy_executions'] + self.stats['sell_executions'] > 0:
            reward_count = len(self.rl_controller.reward_history) if hasattr(self.rl_controller, 'reward_history') else 0
            param_update_freq = self.rl_controller.config.get('parameter_update_frequency', 10)
            print(f"   ℹ️  Väntar på parameter adjustment (krävs {param_update_freq} reward events)")
            print(f"   📊 Reward events mottagna: {reward_count}/{param_update_freq}")
        
        print(f"{'='*90}")
        
        # Strategy Engine parametrar
        print(f"\n📊 Strategy Engine:")
        print(f"   signal_threshold:     {self.strategy_engine.signal_threshold:.4f} (bounds: 0.1-0.9)")
        print(f"   indicator_weighting:  {self.strategy_engine.indicator_weighting:.4f} (bounds: 0.0-1.0)")
        print(f"   → Reward: trade_success_rate, cumulative_reward")
        
        # Risk Manager parametrar
        print(f"\n⚠️  Risk Manager:")
        print(f"   risk_tolerance:       {self.risk_manager.risk_tolerance:.4f} (bounds: 0.01-0.5)")
        print(f"   max_drawdown:         {self.risk_manager.max_drawdown:.4f} (bounds: 0.01-0.3)")
        print(f"   → Reward: drawdown_avoidance, portfolio_stability")
        
        # Decision Engine parametrar
        print(f"\n⚖️  Decision Engine:")
        print(f"   consensus_threshold:  {self.decision_engine.consensus_threshold:.4f} (bounds: 0.5-1.0)")
        print(f"   memory_weighting:     {self.decision_engine.memory_weighting:.4f} (bounds: 0.0-1.0)")
        print(f"   → Reward: decision_accuracy, historical_alignment")
        
        # Execution Engine parametrar
        print(f"\n🔨 Execution Engine:")
        print(f"   execution_delay:      {self.execution_engine.execution_delay:.1f}s (bounds: 0-10)")
        print(f"   slippage_tolerance:   {self.execution_engine.slippage_tolerance:.4f} (bounds: 0.001-0.05)")
        print(f"   → Reward: slippage_reduction, execution_efficiency")
        
        # Vote Engine parametrar
        print(f"\n🗳️  Vote Engine:")
        print(f"   agent_vote_weight:    {self.vote_engine.agent_vote_weight:.4f} (bounds: 0.1-2.0)")
        print(f"   → Reward: agent_hit_rate")
        
        # RL Controller meta-parametrar (Sprint 4.2)
        current_meta = self.rl_controller.get_current_meta_parameters()
        print(f"\n🤖 RL Controller (Meta-parametrar Sprint 4.2):")
        if current_meta:
            if 'evolution_threshold' in current_meta:
                print(f"   evolution_threshold:      {current_meta['evolution_threshold']:.4f}")
            if 'min_samples' in current_meta:
                print(f"   min_samples:              {int(current_meta['min_samples'])}")
            if 'update_frequency' in current_meta:
                print(f"   update_frequency:         {int(current_meta['update_frequency'])}")
            if 'agent_entropy_threshold' in current_meta:
                print(f"   agent_entropy_threshold:  {current_meta['agent_entropy_threshold']:.4f}")
        
        # Visa trend om vi har historik
        if len(self.parameter_history) >= 2:
            print(f"\n📈 Parameter Trends (senaste vs första):")
            first_params = self.parameter_history[0]['parameters']
            latest_params = self.parameter_history[-1]['parameters']
            
            all_params = set(first_params.keys()) | set(latest_params.keys())
            for param in sorted(all_params):
                if param in first_params and param in latest_params:
                    first_val = first_params[param]
                    latest_val = latest_params[param]
                    diff = latest_val - first_val
                    if abs(diff) > 0.0001:
                        trend = "↑" if diff > 0 else "↓"
                        if isinstance(diff, float):
                            print(f"   {param}: {trend} ({diff:+.4f})")
                        else:
                            print(f"   {param}: {trend} ({int(diff):+d})")
        
        print(f"{'='*90}\n")
    
    def print_final_summary(self) -> None:
        """Skriver ut slutlig sammanfattning."""
        print(f"\n{'='*90}")
        print(f"📊 SLUTLIG SAMMANFATTNING")
        print(f"{'='*90}")
        
        runtime = time.time() - self.stats['start_time']
        print(f"\n⏱️  Total körtid: {int(runtime)}s")
        print(f"🔄 Iterationer: {self.stats['iterations']}")
        print(f"💹 Trades processade: {self.stats['trades_processed']}")
        print(f"🎯 Beslut fattade: {self.stats['decisions_made']}")
        
        # Decision distribution
        print(f"\n🎯 DECISION DISTRIBUTION (Alla beslut):")
        print(f"   Totalt beslut: {self.stats['decisions_made']}")
        print(f"   🟢 BUY beslut:  {self.stats['buy_decisions']}")
        print(f"   🔴 SELL beslut: {self.stats['sell_decisions']}")
        print(f"   ⚪ HOLD beslut: {self.stats['hold_decisions']}")
        
        # Execution distribution (endast de som hamnade i portfolio)
        total_executions = self.stats['buy_executions'] + self.stats['sell_executions']
        print(f"\n💼 EXECUTION DISTRIBUTION (I Portfolio):")
        print(f"   Totalt executions: {total_executions}")
        print(f"   ✅ BUY:  {self.stats['buy_executions']} genomförda köp")
        print(f"   ✅ SELL: {self.stats['sell_executions']} genomförda försäljningar")
        print(f"   ⚠️  Blockerade: {self.stats['insufficient_funds_count']} BUY (no funds), "
              f"{self.stats['insufficient_holdings_count']} SELL (no holdings)")
        
        # Portfolio resultat
        portfolio = self.portfolio_manager.get_status(self.current_prices)
        final_value = portfolio.get('total_value', 1000)
        profit = final_value - 1000
        roi = (profit / 1000) * 100
        
        print(f"\n💰 PORTFOLIO RESULTAT:")
        print(f"   Start: $1000.00")
        print(f"   Slut:  ${final_value:.2f}")
        print(f"   P&L:   ${profit:.2f}")
        print(f"   ROI:   {roi:.2f}%")
        
        if portfolio.get('positions'):
            print(f"\n   📊 SLUTLIGA INNEHAV:")
            for symbol, pos in portfolio['positions'].items():
                current_price = self.current_prices.get(symbol, pos['avg_price'])
                pnl = (current_price - pos['avg_price']) * pos['quantity']
                print(f"      {symbol}: {pos['quantity']}@${pos['avg_price']:.2f} "
                      f"→ ${current_price:.2f} (P&L: ${pnl:.2f})")
        
        # Strategic Memory
        insights = self.strategic_memory.generate_insights()
        print(f"\n🧠 STRATEGIC MEMORY:")
        print(f"   Beslut: {insights['total_decisions']}")
        print(f"   Executions: {insights['total_executions']}")
        if insights['total_executions'] > 0:
            print(f"   Success rate: {insights['success_rate']*100:.1f}%")
        
        # Sprint 4.3: Parameter Evolution Summary
        print(f"\n{'='*90}")
        print(f"🔧 PARAMETER EVOLUTION SAMMANFATTNING (Sprint 4.3)")
        print(f"{'='*90}")
        print(f"\nTotal parameter adjustments: {len(self.parameter_history)}")
        
        if len(self.parameter_history) >= 2:
            first = self.parameter_history[0]['parameters']
            last = self.parameter_history[-1]['parameters']
            
            print(f"\n📊 Parameter Changes (Start → Slut):")
            all_params = set(first.keys()) | set(last.keys())
            
            for param in sorted(all_params):
                if param in first and param in last:
                    start_val = first[param]
                    end_val = last[param]
                    change = end_val - start_val
                    change_pct = (change / start_val * 100) if start_val != 0 else 0
                    
                    if isinstance(start_val, float):
                        print(f"   {param}:")
                        print(f"      {start_val:.4f} → {end_val:.4f} "
                              f"(Δ {change:+.4f}, {change_pct:+.1f}%)")
                    else:
                        print(f"   {param}:")
                        print(f"      {int(start_val)} → {int(end_val)} "
                              f"(Δ {int(change):+d}, {change_pct:+.1f}%)")
        
        print(f"\n{'='*90}")
        print(f"Tack för att du testade NextGen AI Trader med Sprint 4.3! 🚀")
        print(f"{'='*90}\n")
    
    def signal_handler(self, sig, frame) -> None:
        """Hanterar Ctrl+C för graceful shutdown."""
        print("\n\n⏹️  Stoppar systemet...")
        self.running = False
        self.print_final_summary()
        sys.exit(0)
    
    def run(self, iterations: int = 100, delay: float = 0.5) -> None:
        """
        Kör simulerad test.
        
        Args:
            iterations: Antal iterationer att köra (0 = oändligt)
            delay: Fördröjning mellan iterationer i sekunder
        """
        print(f"\n{'='*90}")
        print(f"🚀 NextGen AI Trader - Simulerad Test (Sprint 4.3)")
        print(f"{'='*90}")
        print(f"\n🎯 Symboler: {', '.join(self.symbols)}")
        print(f"💰 Start kapital: $1000.00")
        print(f"⚡ Aggressiv data: Ja (stora prisrörelser)")
        print(f"🔧 Sprint 4.3: Alla adaptiva parametrar synliga")
        print(f"\n⏹️  Tryck Ctrl+C för att stoppa\n")
        
        self.running = True
        iteration_count = 0
        
        try:
            while self.running:
                self.simulate_iteration()
                iteration_count += 1
                
                # Visa progress varje 10:e iteration
                if iteration_count % 10 == 0:
                    self.print_progress()
                
                # Stoppa efter iterations (om satt)
                if iterations > 0 and iteration_count >= iterations:
                    break
                
                time.sleep(delay)
        
        except KeyboardInterrupt:
            self.signal_handler(None, None)
        except Exception as e:
            print(f"❌ Oväntat fel: {e}")
            import traceback
            traceback.print_exc()
            self.print_final_summary()


def main():
    """Huvudfunktion för simulerad test."""
    # Skapa och kör tester
    tester = SimulatedTester()
    
    # Kör 200 iterationer med 0.3s delay (60 sekunder totalt)
    # Sätt iterations=0 för oändlig körning
    tester.run(iterations=200, delay=0.3)


if __name__ == "__main__":
    main()
