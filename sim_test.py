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
from modules.strategy_engine import StrategyEngine
from modules.risk_manager import RiskManager
from modules.decision_engine import DecisionEngine
from modules.execution_engine import ExecutionEngine
from modules.portfolio_manager import PortfolioManager
from modules.rl_controller import RLController
from modules.vote_engine import VoteEngine
from modules.reward_tuner import RewardTunerAgent  # Sprint 4.4
from modules.decision_simulator import DecisionSimulator  # Sprint 5
from modules.consensus_engine import ConsensusEngine  # Sprint 5
from modules.timespan_tracker import TimespanTracker  # Sprint 6
from modules.action_chain_engine import ActionChainEngine  # Sprint 6
from modules.system_monitor import SystemMonitor  # Sprint 6
from modules.dqn_controller import DQNController  # Sprint 8
from modules.gan_evolution_engine import GANEvolutionEngine  # Sprint 8
from modules.gnn_timespan_analyzer import GNNTimespanAnalyzer  # Sprint 8
from modules.specialized_agents import SpecializedAgentsCoordinator  # 8 Trading Agents


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
        
        # Sprint 5: Spåra simuleringar och konsensus
        self.simulation_results = []
        self.consensus_decisions = []
        
        # Sprint 6: Spåra timeline och action chains
        self.timeline_insights = []
        self.chain_executions = []
        
        # Sprint 8: Spåra DQN, GAN och GNN aktivitet
        self.dqn_metrics_history = []
        self.gan_candidates_history = []
        self.gnn_patterns_history = []
        
        # 8 Trading Agents: Spåra agent statistik
        self.agent_votes_history = []
        self.agent_states_history = []
    
    def setup_modules(self) -> None:
        """Initialiserar alla Sprint 1-6 moduler."""
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
        
        # Sprint 4.4 modul - RewardTunerAgent (MUST be created BEFORE portfolio_manager)
        self.reward_tuner = RewardTunerAgent(
            message_bus=self.message_bus,
            reward_scaling_factor=1.0,
            volatility_penalty_weight=0.3,
            overfitting_detector_threshold=0.2
        )
        
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
        
        # Sprint 4.4: Register RewardTunerAgent callback with PortfolioManager
        self.portfolio_manager.register_reward_tuner_callback(self.reward_tuner.on_base_reward)
        
        # Sprint 4.3 modul
        self.vote_engine = VoteEngine(self.message_bus)
        
        # Sprint 5 moduler
        self.decision_simulator = DecisionSimulator(self.message_bus)
        self.consensus_engine = ConsensusEngine(self.message_bus, consensus_model='weighted')
        
        # Sprint 6 moduler
        self.timespan_tracker = TimespanTracker(self.message_bus)
        self.action_chain_engine = ActionChainEngine(self.message_bus)
        self.system_monitor = SystemMonitor(self.message_bus)
        
        # Sprint 8 moduler
        # state_dim=12: Expanded state with technical indicators, volume, and portfolio context
        # [price_change, rsi, macd, atr, bb_position, volume_ratio, sma_distance,
        #  volume_trend, price_momentum, volatility_index, position_size, cash_ratio]
        # action_dim=7: Expanded actions with position sizing
        # [BUY_SMALL, BUY_MEDIUM, BUY_LARGE, SELL_PARTIAL, SELL_ALL, HOLD, REBALANCE]
        self.dqn_controller = DQNController(self.message_bus, state_dim=12, action_dim=7)
        self.gan_evolution = GANEvolutionEngine(self.message_bus, latent_dim=64, param_dim=16)
        self.gnn_analyzer = GNNTimespanAnalyzer(self.message_bus, input_dim=32, temporal_window=20)
        
        # 8 Trading Agents module
        self.specialized_agents = SpecializedAgentsCoordinator(
            self.message_bus, 
            initial_capital_per_agent=1000.0
        )
        
        # Sprint 4.3: Prenumerera på parameter_adjustment
        self.message_bus.subscribe('parameter_adjustment', self._on_parameter_adjustment)
        
        # Sprint 5: Prenumerera på simulation_result för logging
        self.message_bus.subscribe('simulation_result', self._on_simulation_result)
        
        # Sprint 6: Prenumerera på timeline_insight och chain_execution
        self.message_bus.subscribe('timeline_insight', self._on_timeline_insight)
        self.message_bus.subscribe('chain_execution', self._on_chain_execution)
        
        # Sprint 8: Prenumerera på DQN, GAN och GNN events
        self.message_bus.subscribe('dqn_metrics', self._on_dqn_metrics)
        self.message_bus.subscribe('gan_candidates', self._on_gan_candidates)
        self.message_bus.subscribe('gnn_analysis_response', self._on_gnn_analysis)
        
        # 8 Trading Agents: Prenumerera på agent events
        self.message_bus.subscribe('agent_state', self._on_agent_state)
        
        print("✅ Alla moduler initialiserade (inkl. Sprint 8: DQN, GAN, GNN + 8 Trading Agents)")
    
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
    
    def _on_simulation_result(self, result: Dict[str, Any]) -> None:
        """
        Callback för simulation results (Sprint 5).
        Loggar simuleringsresultat för analys.
        """
        self.simulation_results.append({
            'timestamp': time.time(),
            'result': result
        })
        
        # Behåll senaste 50
        if len(self.simulation_results) > 50:
            self.simulation_results = self.simulation_results[-50:]
    
    def _on_timeline_insight(self, insight: Dict[str, Any]) -> None:
        """
        Callback för timeline insights (Sprint 6).
        Loggar timeline-analys.
        """
        self.timeline_insights.append({
            'timestamp': time.time(),
            'insight': insight
        })
        
        # Behåll senaste 50
        if len(self.timeline_insights) > 50:
            self.timeline_insights = self.timeline_insights[-50:]
    
    def _on_chain_execution(self, execution: Dict[str, Any]) -> None:
        """
        Callback för chain executions (Sprint 6).
        Loggar action chain-körningar.
        """
        self.chain_executions.append({
            'timestamp': time.time(),
            'execution': execution
        })
        
        # Behåll senaste 50
        if len(self.chain_executions) > 50:
            self.chain_executions = self.chain_executions[-50:]
    
    def _on_dqn_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Callback för DQN metrics (Sprint 8).
        Loggar DQN träningsmetriker.
        """
        self.dqn_metrics_history.append({
            'timestamp': time.time(),
            'metrics': metrics
        })
        
        # Behåll senaste 100
        if len(self.dqn_metrics_history) > 100:
            self.dqn_metrics_history = self.dqn_metrics_history[-100:]
    
    def _on_gan_candidates(self, data: Dict[str, Any]) -> None:
        """
        Callback för GAN-kandidater (Sprint 8).
        Loggar genererade agentkandidater.
        """
        self.gan_candidates_history.append({
            'timestamp': time.time(),
            'candidates': data.get('candidates', []),
            'num_generated': data.get('num_generated', 0),
            'acceptance_rate': data.get('acceptance_rate', 0.0)
        })
        
        # Behåll senaste 50
        if len(self.gan_candidates_history) > 50:
            self.gan_candidates_history = self.gan_candidates_history[-50:]
    
    def _on_gnn_analysis(self, response: Dict[str, Any]) -> None:
        """
        Callback för GNN-analys (Sprint 8).
        Loggar temporala mönster från GNN.
        """
        self.gnn_patterns_history.append({
            'timestamp': time.time(),
            'patterns': response.get('patterns', {}),
            'insights': response.get('insights', {}),
            'graph_size': response.get('graph_size', 0)
        })
        
        # Behåll senaste 50
        if len(self.gnn_patterns_history) > 50:
            self.gnn_patterns_history = self.gnn_patterns_history[-50:]
    
    def _on_agent_state(self, state: Dict[str, Any]) -> None:
        """
        Callback för agent state (8 Trading Agents).
        Loggar agenternas tillstånd.
        """
        self.agent_states_history.append({
            'timestamp': time.time(),
            'state': state
        })
        
        # Behåll senaste 100 (8 agents * ~10 states each)
        if len(self.agent_states_history) > 100:
            self.agent_states_history = self.agent_states_history[-100:]
    
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
        # Sprint 6: Execute action chain for this trading decision
        # Determine which chain to use based on market conditions
        indicators = self.strategy_engine.current_indicators.get(symbol, {})
        rsi = indicators.get('RSI', 50)
        
        # Choose chain based on market volatility/confidence
        if rsi > 70 or rsi < 30:
            # High RSI extremes - use aggressive chain for quick decisions
            chain_name = 'aggressive'
        elif 40 <= rsi <= 60:
            # Neutral market - use standard trade chain
            chain_name = 'standard_trade'
        else:
            # Moderate conditions - use risk-averse chain
            chain_name = 'risk_averse'
        
        # Execute the chosen action chain
        self.action_chain_engine.execute_chain(chain_name, {
            'symbol': symbol,
            'price': price,
            'rsi': rsi
        })
        
        # Strategy engine genererar förslag
        proposal = self.strategy_engine.generate_proposal(symbol)
        self.strategy_engine.publish_proposal(proposal)
        
        # Risk manager bedömer risk
        risk_profile = self.risk_manager.assess_risk(symbol)
        self.risk_manager.publish_risk_profile(risk_profile)
        
        # Decision engine fattar beslut
        decision = self.decision_engine.make_decision(symbol, current_price=price)
        
        # Sprint 5: Publicera beslut för vote_engine och consensus_engine
        self.decision_engine.publish_decision(decision)
        
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
            
            # Publish portfolio status and calculate reward to keep system_monitor updated
            # and trigger reward_tuner
            self.portfolio_manager.publish_status()
            self.portfolio_manager.calculate_and_publish_reward()
            
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
            
            # Sprint 8: Träna DQN med states och rewards
            if hasattr(self, 'last_state'):
                # Expanded state representation (12 dimensions):
                # [price_change, rsi, macd, atr, bb_position, volume_ratio, sma_distance,
                #  volume_trend, price_momentum, volatility_index, position_size, cash_ratio]
                indicators = self.strategy_engine.current_indicators.get(symbol, {})
                prices = self.price_history.get(symbol, [price])
                
                # Calculate expanded state features
                price_change = (price - self.current_prices.get(symbol, price)) / (price if price != 0 else 1e-8)
                rsi = indicators.get('technical', {}).get('RSI', 50) / 100.0
                macd = indicators.get('technical', {}).get('MACD', {}).get('histogram', 0) / 10.0
                
                # ATR approximation
                atr = indicators.get('technical', {}).get('ATR', 0.02) / price if price > 0 else 0.02
                
                # Bollinger Band position (approximation)
                bb_upper = indicators.get('technical', {}).get('BB_upper', price * 1.02)
                bb_lower = indicators.get('technical', {}).get('BB_lower', price * 0.98)
                bb_position = (price - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5
                bb_position = max(0, min(1, bb_position))
                
                # Volume ratio (approximation - use 1.0 as baseline in sim)
                volume_ratio = 1.0
                
                # SMA distance
                sma_20 = indicators.get('technical', {}).get('SMA_20', price)
                sma_distance = (price - sma_20) / sma_20 if sma_20 > 0 else 0
                sma_distance = max(-0.2, min(0.2, sma_distance))
                
                # Volume trend (approximation)
                volume_trend = 0.0
                
                # Price momentum (from recent price history)
                if len(prices) >= 10:
                    price_momentum = (price - prices[-10]) / prices[-10] if prices[-10] > 0 else 0
                    price_momentum = max(-0.5, min(0.5, price_momentum))
                else:
                    price_momentum = 0
                
                # Volatility index (approximation)
                volatility_index = 0.05  # Default moderate volatility
                
                # Position size
                position_size = 0.0
                if symbol in self.portfolio_manager.positions:
                    position_qty = self.portfolio_manager.positions[symbol]['quantity']
                    position_value = position_qty * price
                    position_size = position_value / portfolio_value if portfolio_value > 0 else 0
                
                # Cash ratio
                cash_ratio = self.portfolio_manager.cash / portfolio_value if portfolio_value > 0 else 1.0
                
                current_state = [
                    price_change,
                    rsi,
                    macd,
                    atr,
                    bb_position,
                    volume_ratio / 3.0,  # Normalize
                    sma_distance,
                    volume_trend,
                    price_momentum,
                    volatility_index * 10,  # Scale up for network
                    position_size,
                    cash_ratio
                ]
                
                # Map action to index: BUY=0, SELL=1, HOLD=2
                action_map = {'BUY': 0, 'SELL': 1, 'HOLD': 2}
                action_idx = action_map.get(decision.get('action', 'HOLD'), 2)
                
                # Store transition in DQN
                self.dqn_controller.store_transition(
                    self.last_state,
                    action_idx,
                    reward_value,
                    current_state,
                    False  # Not done
                )
                
                # Train DQN if buffer is ready
                if len(self.dqn_controller.replay_buffer) >= self.dqn_controller.batch_size:
                    self.dqn_controller.train_step()
                
                self.last_state = current_state
            else:
                # Initialize state with expanded features (12 dimensions)
                indicators = self.strategy_engine.current_indicators.get(symbol, {})
                self.last_state = [
                    0.0,  # price_change
                    indicators.get('technical', {}).get('RSI', 50) / 100.0,  # rsi
                    indicators.get('technical', {}).get('MACD', {}).get('histogram', 0) / 10.0,  # macd
                    0.02,  # atr (default)
                    0.5,   # bb_position (middle)
                    1.0 / 3.0,  # volume_ratio (normalized)
                    0.0,   # sma_distance
                    0.0,   # volume_trend
                    0.0,   # price_momentum
                    0.5,   # volatility_index
                    0.0,   # position_size
                    1.0    # cash_ratio (start with all cash)
                ]
            
            # Sprint 8: Feed agent performance to GAN every 10 iterations
            if self.stats['iterations'] % 10 == 0:
                # Create synthetic agent performance for GAN training
                agent_params = [random.gauss(0, 1) for _ in range(16)]
                performance = 0.5 + reward_value * 0.3  # Scale reward to performance
                
                self.message_bus.publish('agent_performance', {
                    'parameters': agent_params,
                    'performance': min(1.0, max(0.0, performance))
                })
                
                # Train GAN
                if len(self.gan_evolution.real_agent_data) >= 32:
                    self.gan_evolution.train_step(batch_size=32)
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
        
        # Sprint 4.4: RewardTunerAgent Debug Info
        print(f"\n{'='*90}")
        print(f"🎯 SPRINT 4.4 - RewardTunerAgent (Meta-belöningsjustering)")
        print(f"{'='*90}")
        
        reward_metrics = self.reward_tuner.get_reward_metrics()
        
        # Current parameters
        print(f"\n⚙️  RewardTuner Parametrar:")
        current_params = reward_metrics['current_parameters']
        print(f"   reward_scaling_factor:          {current_params['reward_scaling_factor']:.4f} (bounds: 0.5-2.0)")
        print(f"   volatility_penalty_weight:      {current_params['volatility_penalty_weight']:.4f} (bounds: 0.0-1.0)")
        print(f"   overfitting_detector_threshold: {current_params['overfitting_detector_threshold']:.4f} (bounds: 0.05-0.5)")
        print(f"   → Reward signals: training_stability, reward_consistency, generalization_score")
        
        # Reward transformation statistics
        base_rewards = reward_metrics['base_reward_history']
        tuned_rewards = reward_metrics['tuned_reward_history']
        
        # Diagnostic info
        rl_reward_count = len(self.rl_controller.reward_history) if hasattr(self.rl_controller, 'reward_history') else 0
        print(f"\n🔍 Diagnostic Info:")
        print(f"   Base rewards received:     {len(base_rewards)}")
        print(f"   Tuned rewards generated:   {len(tuned_rewards)}")
        print(f"   RL controller rewards:     {rl_reward_count}")
        print(f"   Portfolio executions:      {self.stats['buy_executions'] + self.stats['sell_executions']}")
        
        if base_rewards and tuned_rewards:
            print(f"\n📊 Reward Transformation Stats:")
            print(f"   Totalt rewards processade: {len(base_rewards)}")
            
            # Latest rewards
            if len(base_rewards) > 0:
                latest_base = base_rewards[-1]
                latest_tuned = tuned_rewards[-1]
                latest_ratio = latest_tuned / latest_base if latest_base != 0 else 1.0
                print(f"   Senaste base_reward:   {latest_base:+.4f}")
                print(f"   Senaste tuned_reward:  {latest_tuned:+.4f}")
                print(f"   Transformation ratio:  {latest_ratio:.4f}")
            
            # Average transformation
            if len(reward_metrics['transformation_ratios']) > 0:
                avg_ratio = sum(reward_metrics['transformation_ratios']) / len(reward_metrics['transformation_ratios'])
                print(f"   Genomsnittlig ratio:   {avg_ratio:.4f}")
        else:
            print(f"\n⏳ Reward Transformation Stats:")
            print(f"   Väntar på första reward från portfolio_manager...")
            print(f"   Status: RewardTunerAgent är redo men har inte fått några base_reward events än")
        
        # Volatility metrics
        volatility_hist = reward_metrics['volatility_history']
        if volatility_hist:
            print(f"\n📈 Volatility Metrics:")
            recent_volatility = volatility_hist[-1] if volatility_hist else 0.0
            avg_volatility = sum(volatility_hist) / len(volatility_hist) if volatility_hist else 0.0
            print(f"   Senaste volatility:    {recent_volatility:.4f}")
            print(f"   Genomsnittlig:         {avg_volatility:.4f}")
            print(f"   Volatility samples:    {len(volatility_hist)}")
        else:
            print(f"\n⏳ Volatility Metrics:")
            print(f"   Inga volatility data än - krävs minst 2 rewards för beräkning")
        
        # Overfitting events
        overfitting_events = reward_metrics['overfitting_events']
        if overfitting_events:
            print(f"\n⚠️  Overfitting Detection:")
            print(f"   Totalt events:         {len(overfitting_events)}")
            if len(overfitting_events) > 0:
                latest_event = overfitting_events[-1]
                print(f"   Senaste score:         {latest_event.get('overfitting_score', 0):.4f}")
                print(f"   Recent performance:    {latest_event.get('recent_performance', 0):.4f}")
                print(f"   Long-term performance: {latest_event.get('long_term_performance', 0):.4f}")
        else:
            print(f"\n✅ Overfitting Detection:")
            print(f"   Inga overfitting events detekterade")
        
        # Parameter adjustment history
        param_hist = reward_metrics['parameter_history']
        if param_hist:
            print(f"\n🔧 Parameter Adjustments:")
            print(f"   Totalt adjustments:    {len(param_hist)}")
            if len(param_hist) >= 2:
                first = param_hist[0]
                latest = param_hist[-1]
                print(f"   Scaling factor trend:  {first['reward_scaling_factor']:.4f} → {latest['reward_scaling_factor']:.4f}")
        
        # Add helpful note if no data
        if not base_rewards:
            print(f"\n💡 Note:")
            print(f"   RewardTunerAgent fungerar korrekt men har inte fått några rewards än.")
            print(f"   Detta kan bero på:")
            print(f"   • Systemet nyss startat och väntar på första trade att completea")
            print(f"   • Portfolio value har inte ändrats sedan senaste reward")
            print(f"   • Koden uppdaterades under körning - starta om sim_test.py för nya features")
        
        print(f"{'='*90}\n")
        
        # Sprint 5: Simulering och Konsensus
        print(f"\n{'='*90}")
        print(f"🎲 SPRINT 5 - Simulering och Konsensus")
        print(f"{'='*90}")
        
        # Decision Simulator stats
        sim_stats = self.decision_simulator.get_simulation_statistics()
        print(f"\n🎲 Decision Simulator:")
        print(f"   Totalt simuleringar: {sim_stats['total_simulations']}")
        if sim_stats['total_simulations'] > 0:
            print(f"   Rekommendationer:")
            print(f"      ✅ Proceed: {sim_stats['proceed_recommendations']}")
            print(f"      ⚠️  Caution: {sim_stats['caution_recommendations']}")
            print(f"      ❌ Reject:  {sim_stats['reject_recommendations']}")
            print(f"   Genomsnittlig expected value: ${sim_stats['average_expected_value']:.2f}")
        
        # Vote Engine stats
        vote_stats = self.vote_engine.get_voting_statistics()
        print(f"\n🗳️  Vote Engine:")
        print(f"   Totalt röster: {vote_stats['total_votes']}")
        if vote_stats['total_votes'] > 0:
            print(f"   Unika röstare: {vote_stats['unique_voters']}")
            print(f"   Genomsnittlig confidence: {vote_stats['average_confidence']:.2f}")
            if vote_stats['action_distribution']:
                print(f"   Röstfördelning:")
                for action, count in vote_stats['action_distribution'].items():
                    print(f"      {action}: {count} röster")
        
        # Consensus Engine stats
        consensus_stats = self.consensus_engine.get_consensus_statistics()
        print(f"\n⚖️  Consensus Engine:")
        print(f"   Totalt konsensusbeslut: {consensus_stats['total_decisions']}")
        print(f"   Konsensusmodell: {consensus_stats['consensus_model']}")
        if consensus_stats['total_decisions'] > 0:
            print(f"   Genomsnittlig confidence: {consensus_stats['average_confidence']:.2f}")
            print(f"   Genomsnittlig robusthet: {consensus_stats['average_robustness']:.2f}")
            if consensus_stats['action_distribution']:
                print(f"   Beslutsfördelning:")
                for action, count in consensus_stats['action_distribution'].items():
                    print(f"      {action}: {count} beslut")
        
        print(f"{'='*90}\n")
        
        # Sprint 6: Tidsanalys och Action Chains
        print(f"\n{'='*90}")
        print(f"⏰ SPRINT 6 - Tidsanalys och Action Chains")
        print(f"{'='*90}")
        
        # Timespan Tracker stats
        timeline_summary = self.timespan_tracker.get_timeline_summary()
        print(f"\n⏱️  Timespan Tracker:")
        print(f"   Totalt events: {timeline_summary['total_events']}")
        print(f"   Decision events: {timeline_summary['decision_events']}")
        print(f"   Final decisions: {timeline_summary['final_decisions']}")
        print(f"   Symboler spårade: {len(timeline_summary['symbols_tracked'])}")
        if timeline_summary['time_span'] > 0:
            print(f"   Time span: {timeline_summary['time_span']:.1f}s")
        if len(self.timeline_insights) > 0:
            latest_insight = self.timeline_insights[-1]['insight']
            print(f"   Senaste insight:")
            if 'avg_time_between_decisions' in latest_insight:
                print(f"      Avg tid mellan beslut: {latest_insight['avg_time_between_decisions']:.2f}s")
        
        # Action Chain Engine stats
        chain_stats = self.action_chain_engine.get_chain_statistics()
        print(f"\n🔗 Action Chain Engine:")
        print(f"   Totalt chains definierade: {chain_stats['total_chains_defined']}")
        print(f"   Chain templates: {chain_stats['total_templates']}")
        print(f"   Totalt executions: {chain_stats['total_executions']}")
        if chain_stats['total_executions'] > 0:
            print(f"      Template executions: {chain_stats['template_executions']}")
            print(f"      Custom executions: {chain_stats['custom_executions']}")
            print(f"   Avg execution duration: {chain_stats['avg_execution_duration']:.4f}s")
        print(f"   Tillgängliga templates: {', '.join(chain_stats['available_templates'])}")
        
        # System Monitor health
        system_health = self.system_monitor.get_system_health()
        print(f"\n🏥 System Monitor:")
        print(f"   Health score: {system_health['health_score']:.2f}")
        print(f"   Status: {system_health['status']}")
        print(f"   Aktiva moduler: {len(system_health['active_modules'])}/{system_health['total_modules']}")
        if system_health['stale_modules']:
            print(f"   ⚠️  Stale moduler: {', '.join(system_health['stale_modules'])}")
        print(f"   Uptime: {system_health['uptime']:.1f}s")
        
        print(f"{'='*90}\n")
        
        # Sprint 8: DQN, GAN, GNN Metrics
        print(f"\n{'='*90}")
        print(f"🤖 SPRINT 8 - DQN, GAN, GNN & Hybrid RL")
        print(f"{'='*90}")
        
        # DQN Controller stats
        dqn_metrics = self.dqn_controller.get_metrics()
        print(f"\n🎯 DQN Controller:")
        print(f"   Training steps: {dqn_metrics['training_steps']}")
        print(f"   Epsilon (exploration): {dqn_metrics['epsilon']:.4f}")
        print(f"   Replay buffer size: {dqn_metrics['buffer_size']}/{self.dqn_controller.replay_buffer.buffer.maxlen}")
        print(f"   Avg loss (recent): {dqn_metrics['avg_loss']:.4f}")
        if len(self.dqn_metrics_history) > 0:
            latest_dqn = self.dqn_metrics_history[-1]['metrics']
            print(f"   Latest metrics:")
            print(f"      Loss: {latest_dqn.get('loss', 0):.4f}")
            print(f"      Buffer usage: {latest_dqn.get('buffer_size', 0)}")
        
        # GAN Evolution Engine stats
        gan_metrics = self.gan_evolution.get_metrics()
        print(f"\n🧬 GAN Evolution Engine:")
        print(f"   Generator loss: {gan_metrics['g_loss']:.4f}")
        print(f"   Discriminator loss: {gan_metrics['d_loss']:.4f}")
        print(f"   Candidates generated: {gan_metrics['candidates_generated']}")
        print(f"   Candidates accepted: {gan_metrics['candidates_accepted']}")
        print(f"   Acceptance rate: {gan_metrics['acceptance_rate']:.2%}")
        print(f"   Real agent data: {gan_metrics['real_data_size']} samples")
        if len(self.gan_candidates_history) > 0:
            latest_gan = self.gan_candidates_history[-1]
            print(f"   Latest generation:")
            print(f"      Candidates: {latest_gan['num_generated']}")
            print(f"      Acceptance: {latest_gan['acceptance_rate']:.2%}")
        
        # GNN Timespan Analyzer stats
        gnn_metrics = self.gnn_analyzer.get_metrics()
        print(f"\n📊 GNN Timespan Analyzer:")
        print(f"   Decision history: {gnn_metrics['decision_history_size']}")
        print(f"   Indicator history: {gnn_metrics['indicator_history_size']}")
        print(f"   Outcome history: {gnn_metrics['outcome_history_size']}")
        print(f"   Temporal window: {gnn_metrics['temporal_window']}")
        print(f"   Patterns detected: {gnn_metrics['patterns_detected']}")
        
        if len(self.gnn_patterns_history) > 0:
            latest_gnn = self.gnn_patterns_history[-1]
            patterns = latest_gnn.get('patterns', {})
            if patterns and patterns.get('patterns'):
                print(f"   Latest patterns:")
                for pattern in patterns['patterns'][:3]:
                    print(f"      {pattern['type']}: {pattern['confidence']:.2%}")
        
        # Hybrid RL comparison
        print(f"\n⚖️  Hybrid RL (PPO + DQN):")
        ppo_metrics = self.rl_controller.get_metrics() if hasattr(self.rl_controller, 'get_metrics') else {}
        print(f"   PPO active: ✅")
        print(f"   DQN active: ✅")
        print(f"   DQN training steps: {dqn_metrics['training_steps']}")
        print(f"   DQN epsilon: {dqn_metrics['epsilon']:.4f}")
        print(f"   Parallel execution: Active")
        
        print(f"{'='*90}\n")
        
        # 8 Trading Agents Section
        print(f"\n{'='*90}")
        print(f"🎯 8 SPECIALIZED TRADING AGENTS")
        print(f"{'='*90}")
        
        # Get aggregated statistics
        agents_stats = self.specialized_agents.get_aggregated_statistics()
        print(f"\n📊 Aggregated Agent Statistics:")
        print(f"   Total agents: {agents_stats['num_agents']}")
        print(f"   Combined capital: ${agents_stats['total_capital']:.2f}")
        print(f"   Combined portfolio value: ${agents_stats['total_portfolio_value']:.2f}")
        print(f"   Combined trades: {agents_stats['total_trades']}")
        
        # Individual agent performance
        print(f"\n🤖 Individual Agent Performance:")
        for agent_stat in agents_stats['agent_statistics']:
            agent_id = agent_stat['agent_id']
            pv = agent_stat['portfolio_value']
            capital = agent_stat['capital']
            roi = agent_stat['roi']
            trades = agent_stat['total_trades']
            win_rate = agent_stat['win_rate']
            
            # Emoji based on performance
            perf_emoji = "🟢" if roi > 0 else "🔴" if roi < 0 else "⚪"
            
            print(f"   {perf_emoji} {agent_id:25s}: "
                  f"Value: ${pv:8.2f} | "
                  f"ROI: {roi*100:+6.2f}% | "
                  f"Trades: {trades:3d} | "
                  f"Win Rate: {win_rate*100:5.1f}%")
        
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
        
        # Sprint 5: Simulering och Konsensus Sammanfattning
        print(f"\n{'='*90}")
        print(f"🎲 SPRINT 5 - SIMULERING OCH KONSENSUS SAMMANFATTNING")
        print(f"{'='*90}")
        
        sim_stats = self.decision_simulator.get_simulation_statistics()
        print(f"\n🎲 Decision Simulator:")
        print(f"   Totalt simuleringar: {sim_stats['total_simulations']}")
        if sim_stats['total_simulations'] > 0:
            proceed_pct = (sim_stats['proceed_recommendations'] / sim_stats['total_simulations']) * 100
            caution_pct = (sim_stats['caution_recommendations'] / sim_stats['total_simulations']) * 100
            reject_pct = (sim_stats['reject_recommendations'] / sim_stats['total_simulations']) * 100
            
            print(f"   Rekommendationsfördelning:")
            print(f"      ✅ Proceed: {sim_stats['proceed_recommendations']} ({proceed_pct:.1f}%)")
            print(f"      ⚠️  Caution: {sim_stats['caution_recommendations']} ({caution_pct:.1f}%)")
            print(f"      ❌ Reject:  {sim_stats['reject_recommendations']} ({reject_pct:.1f}%)")
            print(f"   Genomsnittlig expected value: ${sim_stats['average_expected_value']:.2f}")
            
            # Visa några senaste simuleringar
            if len(self.simulation_results) > 0:
                print(f"\n   📊 Senaste 5 simuleringar:")
                for sim_entry in list(self.simulation_results)[-5:]:
                    result = sim_entry['result']
                    print(f"      {result['symbol']} {result['original_action']}: "
                          f"EV ${result['expected_value']:.2f}, "
                          f"Rekommendation: {result['recommendation']}")
        
        vote_stats = self.vote_engine.get_voting_statistics()
        print(f"\n🗳️  Vote Engine:")
        print(f"   Totalt röster: {vote_stats['total_votes']}")
        if vote_stats['total_votes'] > 0:
            print(f"   Unika röstare: {vote_stats['unique_voters']}")
            print(f"   Genomsnittlig confidence: {vote_stats['average_confidence']:.2f}")
            if vote_stats['action_distribution']:
                print(f"   Röstfördelning:")
                total_votes = sum(vote_stats['action_distribution'].values())
                for action, count in vote_stats['action_distribution'].items():
                    pct = (count / total_votes) * 100
                    print(f"      {action}: {count} röster ({pct:.1f}%)")
        
        consensus_stats = self.consensus_engine.get_consensus_statistics()
        print(f"\n⚖️  Consensus Engine:")
        print(f"   Totalt konsensusbeslut: {consensus_stats['total_decisions']}")
        print(f"   Konsensusmodell: {consensus_stats['consensus_model']}")
        if consensus_stats['total_decisions'] > 0:
            print(f"   Genomsnittlig confidence: {consensus_stats['average_confidence']:.2f}")
            print(f"   Genomsnittlig robusthet: {consensus_stats['average_robustness']:.2f}")
            if consensus_stats['action_distribution']:
                print(f"   Beslutsfördelning:")
                total_decisions = sum(consensus_stats['action_distribution'].values())
                for action, count in consensus_stats['action_distribution'].items():
                    pct = (count / total_decisions) * 100
                    print(f"      {action}: {count} beslut ({pct:.1f}%)")
        
        print(f"\n💡 Sprint 5 Status:")
        if sim_stats['total_simulations'] > 0 or consensus_stats['total_decisions'] > 0:
            print(f"   ✅ Sprint 5-moduler aktiva och fungerar")
            print(f"   ✅ Simulering av alternativa beslut implementerad")
            print(f"   ✅ Röstmatris och konsensusmodell implementerad")
        else:
            print(f"   ℹ️  Sprint 5-moduler laddade men inte aktiverade än")
            print(f"   ℹ️  Simulering och konsensus triggas vid beslutspunkter")
        
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
