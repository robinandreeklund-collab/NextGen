"""
analyzer_debug.py - Comprehensive Debug and Analysis Dashboard

Beskrivning:
    Fullständig debug- och analysdashboard för NextGen AI Trader.
    Integrerar alla systemkomponenter för realtidsvisualisering av:
    - Systemstatus och modulhälsa
    - Dataflöde genom alla moduler
    - RL-analys (reward flow, parameter evolution)
    - Agentutveckling (versioner, evolution, performance)
    - Simulering (priser, indikatorer, beslut)
    - Debug/loggning (events, feedback, system health)

Funktioner:
    - Realtidsvisualisering av systemstatus
    - Simulerad eller live datamatning
    - RL-analys och parameter tracking
    - Agentevolution och performance
    - Feedback flow och kommunikation
    - System health monitoring
    - Resource allocation och team dynamics (Sprint 7)

Användning:
    python analyzer_debug.py
    
    Öppna sedan webbläsaren på http://localhost:8050
    Stoppa med Ctrl+C
"""

import time
import random
import math
import json
from datetime import datetime
from typing import Dict, Any, List
import threading

# Dash och Plotly för visualisering
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go

# WebSocket för Finnhub
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
from modules.vote_engine import VoteEngine
from modules.reward_tuner import RewardTunerAgent
from modules.decision_simulator import DecisionSimulator
from modules.consensus_engine import ConsensusEngine
from modules.timespan_tracker import TimespanTracker
from modules.action_chain_engine import ActionChainEngine
from modules.system_monitor import SystemMonitor
from modules.dqn_controller import DQNController
from modules.gan_evolution_engine import GANEvolutionEngine
from modules.gnn_timespan_analyzer import GNNTimespanAnalyzer


class AnalyzerDebugDashboard:
    """Comprehensive debug dashboard för hela NextGen systemet."""
    
    def __init__(self):
        """Initialiserar dashboarden."""
        # Finnhub API key - must be set before setup_modules
        self.api_key = "d3in10hr01qmn7fkr2a0d3in10hr01qmn7fkr2ag"
        
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
        
        # Dashboard state
        self.running = False
        self.live_mode = False
        self.iteration_count = 0
        self.simulation_thread = None
        self.ws = None
        self.ws_thread = None
        
        # Setup Dash app
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_modules(self) -> None:
        """Initialiserar alla Sprint 1-7 moduler."""
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
        
        # Sprint 4.4 modul - RewardTunerAgent
        self.reward_tuner = RewardTunerAgent(
            message_bus=self.message_bus,
            reward_scaling_factor=1.0,
            volatility_penalty_weight=0.3,
            overfitting_detector_threshold=0.2
        )
        
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
        
        # Sprint 4.4: Register RewardTunerAgent callback
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
        # state_dim=4: matches sim_test.py state representation
        self.dqn_controller = DQNController(self.message_bus, state_dim=4, action_dim=3)
        self.gan_evolution = GANEvolutionEngine(self.message_bus, latent_dim=64, param_dim=16)
        self.gnn_analyzer = GNNTimespanAnalyzer(self.message_bus, input_dim=32, temporal_window=20)
        
        print("✅ Alla moduler initialiserade (inkl. Sprint 8: DQN, GAN, GNN)")
    
    def setup_layout(self) -> None:
        """Skapar dashboard layout."""
        self.app.layout = html.Div([
            html.H1("🔍 NextGen AI Trader - Analyzer Debug Dashboard", 
                    style={'textAlign': 'center', 'color': '#2c3e50'}),
            
            # Control panel
            html.Div([
                html.Button('Start Simulation', id='start-btn', n_clicks=0,
                           style={'margin': '10px', 'padding': '10px 20px', 
                                  'backgroundColor': '#27ae60', 'color': 'white', 
                                  'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
                html.Button('Stop Simulation', id='stop-btn', n_clicks=0,
                           style={'margin': '10px', 'padding': '10px 20px',
                                  'backgroundColor': '#e74c3c', 'color': 'white',
                                  'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
                html.Button('Live Data (Finnhub)', id='live-btn', n_clicks=0,
                           style={'margin': '10px', 'padding': '10px 20px',
                                  'backgroundColor': '#3498db', 'color': 'white',
                                  'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
                html.Span(id='status-indicator', 
                         style={'margin': '10px', 'padding': '10px', 'fontWeight': 'bold'}),
            ], style={'textAlign': 'center', 'padding': '20px', 
                     'backgroundColor': '#ecf0f1', 'borderRadius': '10px'}),
            
            # Auto-refresh interval
            dcc.Interval(id='interval-component', interval=2000, n_intervals=0),
            
            # Tab layout for different sections
            dcc.Tabs([
                # Tab 1: System Overview
                dcc.Tab(label='System Overview', children=[
                    html.Div([
                        html.H2("📊 System Status", style={'color': '#34495e'}),
                        dcc.Graph(id='system-health-graph'),
                        dcc.Graph(id='module-status-graph'),
                    ])
                ]),
                
                # Tab 2: Data Flow & Simulation
                dcc.Tab(label='Data Flow & Simulation', children=[
                    html.Div([
                        html.H2("💹 Market Simulation", style={'color': '#34495e'}),
                        dcc.Graph(id='price-trends-graph'),
                        dcc.Graph(id='indicator-graph'),
                        html.H2("🔄 Decision Flow", style={'color': '#34495e'}),
                        dcc.Graph(id='decision-flow-graph'),
                    ])
                ]),
                
                # Tab 3: RL Analysis
                dcc.Tab(label='RL Analysis', children=[
                    html.Div([
                        html.H2("🎯 Reward Flow (Sprint 4.4)", style={'color': '#34495e'}),
                        dcc.Graph(id='reward-flow-graph'),
                        html.H2("🔧 Parameter Evolution (Sprint 4.3)", style={'color': '#34495e'}),
                        dcc.Graph(id='parameter-evolution-graph'),
                        html.H2("🤖 Agent Performance", style={'color': '#34495e'}),
                        dcc.Graph(id='agent-performance-graph'),
                        
                        # Sprint 8: DQN, GAN, GNN Metrics
                        html.Hr(style={'margin': '30px 0', 'border': '2px solid #3498db'}),
                        html.H2("🆕 Sprint 8 - Advanced RL & Temporal Intelligence", 
                               style={'color': '#3498db', 'textAlign': 'center'}),
                        
                        html.H3("🎯 DQN Controller Metrics", style={'color': '#34495e'}),
                        dcc.Graph(id='dqn-metrics-graph'),
                        dcc.Graph(id='dqn-training-graph'),
                        
                        html.H3("🧬 GAN Evolution Engine Metrics", style={'color': '#34495e'}),
                        dcc.Graph(id='gan-metrics-graph'),
                        dcc.Graph(id='gan-training-graph'),
                        
                        html.H3("📊 GNN Timespan Analyzer Metrics", style={'color': '#34495e'}),
                        dcc.Graph(id='gnn-metrics-graph'),
                        dcc.Graph(id='gnn-patterns-graph'),
                    ])
                ]),
                
                # Tab 4: Agent Development
                dcc.Tab(label='Agent Development', children=[
                    html.Div([
                        html.H2("🧬 Agent Evolution", style={'color': '#34495e'}),
                        dcc.Graph(id='agent-evolution-graph'),
                        html.H2("📈 Agent Metrics", style={'color': '#34495e'}),
                        dcc.Graph(id='agent-metrics-graph'),
                    ])
                ]),
                
                # Tab 5: Debug & Logging
                dcc.Tab(label='Debug & Logging', children=[
                    html.Div([
                        html.H2("📝 Event Log", style={'color': '#34495e'}),
                        html.Div(id='event-log', style={
                            'height': '300px', 'overflowY': 'scroll',
                            'backgroundColor': '#2c3e50', 'color': '#ecf0f1',
                            'padding': '10px', 'fontFamily': 'monospace',
                            'borderRadius': '5px'
                        }),
                        html.H2("🔍 Feedback Flow", style={'color': '#34495e'}),
                        dcc.Graph(id='feedback-flow-graph'),
                        html.H2("⏰ Timeline Analysis (Sprint 6)", style={'color': '#34495e'}),
                        dcc.Graph(id='timeline-graph'),
                    ])
                ]),
                
                # Tab 6: Portfolio & Performance
                dcc.Tab(label='Portfolio', children=[
                    html.Div([
                        html.H2("💰 Portfolio Status", style={'color': '#34495e'}),
                        dcc.Graph(id='portfolio-value-graph'),
                        dcc.Graph(id='positions-graph'),
                        html.Div(id='portfolio-details', style={
                            'padding': '20px', 'backgroundColor': '#ecf0f1',
                            'borderRadius': '10px', 'margin': '10px'
                        }),
                    ])
                ]),
            ]),
        ], style={'fontFamily': 'Arial, sans-serif', 'padding': '20px'})
    
    def setup_callbacks(self) -> None:
        """Setup Dash callbacks för realtidsuppdatering."""
        
        @self.app.callback(
            Output('status-indicator', 'children'),
            [Input('start-btn', 'n_clicks'),
             Input('stop-btn', 'n_clicks'),
             Input('live-btn', 'n_clicks')]
        )
        def update_simulation_status(start_clicks, stop_clicks, live_clicks):
            """Startar/stoppar simulation eller live data."""
            # Use Dash callback context to determine which button was clicked
            ctx = dash.callback_context
            if not ctx.triggered:
                return "⚪ Ready"
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            if button_id == 'live-btn':
                if not self.live_mode:
                    self.start_live_data()
                return "🔴 Live Data Running (Finnhub)"
            elif button_id == 'start-btn':
                if self.live_mode:
                    self.stop_live_data()
                if not self.running:
                    self.start_simulation()
                return "🟢 Simulation Running"
            elif button_id == 'stop-btn':
                if self.live_mode:
                    self.stop_live_data()
                if self.running:
                    self.stop_simulation()
                return "🔴 Stopped"
            return "⚪ Ready"
        
        @self.app.callback(
            [Output('system-health-graph', 'figure'),
             Output('module-status-graph', 'figure'),
             Output('price-trends-graph', 'figure'),
             Output('indicator-graph', 'figure'),
             Output('decision-flow-graph', 'figure'),
             Output('reward-flow-graph', 'figure'),
             Output('parameter-evolution-graph', 'figure'),
             Output('agent-performance-graph', 'figure'),
             Output('agent-evolution-graph', 'figure'),
             Output('agent-metrics-graph', 'figure'),
             Output('event-log', 'children'),
             Output('feedback-flow-graph', 'figure'),
             Output('timeline-graph', 'figure'),
             Output('portfolio-value-graph', 'figure'),
             Output('positions-graph', 'figure'),
             Output('portfolio-details', 'children'),
             # Sprint 8 graphs
             Output('dqn-metrics-graph', 'figure'),
             Output('dqn-training-graph', 'figure'),
             Output('gan-metrics-graph', 'figure'),
             Output('gan-training-graph', 'figure'),
             Output('gnn-metrics-graph', 'figure'),
             Output('gnn-patterns-graph', 'figure')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_all_graphs(n):
            """Uppdaterar alla grafer."""
            return (
                self.create_system_health_graph(),
                self.create_module_status_graph(),
                self.create_price_trends_graph(),
                self.create_indicator_graph(),
                self.create_decision_flow_graph(),
                self.create_reward_flow_graph(),
                self.create_parameter_evolution_graph(),
                self.create_agent_performance_graph(),
                self.create_agent_evolution_graph(),
                self.create_agent_metrics_graph(),
                self.create_event_log(),
                self.create_feedback_flow_graph(),
                self.create_timeline_graph(),
                self.create_portfolio_value_graph(),
                self.create_positions_graph(),
                self.create_portfolio_details(),
                # Sprint 8 graphs
                self.create_dqn_metrics_graph(),
                self.create_dqn_training_graph(),
                self.create_gan_metrics_graph(),
                self.create_gan_training_graph(),
                self.create_gnn_metrics_graph(),
                self.create_gnn_patterns_graph()
            )
    
    def start_simulation(self) -> None:
        """Startar simuleringstråden."""
        if not self.running:
            self.running = True
            self.simulation_thread = threading.Thread(target=self.run_simulation, daemon=True)
            self.simulation_thread.start()
            print("✅ Simulation started")
    
    def stop_simulation(self) -> None:
        """Stoppar simuleringstråden."""
        if self.running:
            self.running = False
            print("⏹️  Simulation stopped")
    
    def run_simulation(self) -> None:
        """Kör simulering i bakgrunden."""
        while self.running:
            self.simulate_iteration()
            time.sleep(0.5)  # 2 iterationer per sekund
    
    def simulate_iteration(self) -> None:
        """Simulerar en iteration (kopierat från sim_test.py)."""
        self.iteration_count += 1
        
        # Process varje symbol
        for symbol in self.symbols:
            # Generera prisrörelse
            new_price = self.generate_price_movement(symbol)
            
            # Publicera market data
            self.message_bus.publish('market_data', {
                'symbol': symbol,
                'price': new_price,
                'volume': random.randint(100, 1000),
                'timestamp': int(time.time() * 1000)
            })
            
            # Generera och publicera indikatorer
            indicators = self.generate_indicators(symbol, self.iteration_count)
            self.message_bus.publish('indicator_data', indicators)
            
            # Uppdatera modulernas indikatordata
            self.strategy_engine.current_indicators[symbol] = indicators
            self.risk_manager.current_indicators[symbol] = indicators
            
            # Fatta handelsbeslut
            self.make_trading_decision(symbol, new_price)
    
    def generate_price_movement(self, symbol: str) -> float:
        """Genererar prisrörelser."""
        # Ändra trend ibland
        if random.random() < 0.1:
            self.price_trends[symbol] = random.choice([-1, 1]) * random.uniform(0.02, 0.05)
        
        # Lägg till noise
        noise = random.uniform(-0.02, 0.02)
        
        # Uppdatera pris
        price_change_pct = self.price_trends[symbol] + noise
        new_price = self.current_prices[symbol] * (1 + price_change_pct)
        
        # Håll inom gränser
        min_price = self.base_prices[symbol] * 0.5
        max_price = self.base_prices[symbol] * 1.5
        new_price = max(min_price, min(max_price, new_price))
        
        self.current_prices[symbol] = new_price
        return new_price
    
    def generate_indicators(self, symbol: str, iteration: int) -> Dict[str, Any]:
        """Genererar indikatorer."""
        rsi_base = 50 + 40 * math.sin(iteration * 0.3 + hash(symbol) % 10)
        rsi = max(10, min(90, rsi_base))
        
        macd_hist = 3 * math.sin(iteration * 0.2 + hash(symbol) % 5)
        
        price_volatility = abs(self.price_trends.get(symbol, 0))
        atr = 2.0 + price_volatility * 50
        
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
    
    def make_trading_decision(self, symbol: str, price: float) -> None:
        """Fattar handelsbeslut."""
        # Strategy engine genererar förslag
        proposal = self.strategy_engine.generate_proposal(symbol)
        self.strategy_engine.publish_proposal(proposal)
        
        # Risk manager bedömer risk
        risk_profile = self.risk_manager.assess_risk(symbol)
        self.risk_manager.publish_risk_profile(risk_profile)
        
        # Decision engine fattar beslut
        decision = self.decision_engine.make_decision(symbol, current_price=price)
        
        # Publicera beslut för vote_engine och consensus_engine
        self.decision_engine.publish_decision(decision)
        
        if decision and decision.get('action') != 'HOLD':
            decision['current_price'] = price
            
            # Execution engine exekverar
            execution_result = self.execution_engine.execute_trade(decision)
            
            # Portfolio manager uppdaterar
            self.portfolio_manager.update_portfolio(execution_result)
            
            # Logga till strategic memory
            self.strategic_memory.log_decision({
                'symbol': symbol,
                'action': decision.get('action'),
                'price': price,
                'execution_result': execution_result
            })
            
            # Publicera reward
            portfolio = self.portfolio_manager.get_status(self.current_prices)
            portfolio_value = portfolio.get('total_value', 1000)
            reward_value = (portfolio_value - 1000) / 1000
            
            self.message_bus.publish('reward', {
                'value': reward_value,
                'portfolio_value': portfolio_value,
                'timestamp': time.time()
            })
    
    # Graph creation methods
    def create_system_health_graph(self):
        """Skapar system health graf."""
        system_health = self.system_monitor.get_system_health()
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=system_health['health_score'] * 100,
            title={'text': "System Health Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=300)
        return fig
    
    def create_module_status_graph(self):
        """Skapar modulstatus graf."""
        system_health = self.system_monitor.get_system_health()
        
        # Get both active and stale modules, build union
        active_modules = set(system_health.get('active_modules', []))
        stale_modules = set(system_health.get('stale_modules', []))
        all_modules = list(active_modules | stale_modules)
        status = ['Active' if m in active_modules else 'Stale' for m in all_modules]
        
        fig = go.Figure(data=[
            go.Bar(x=all_modules, y=[1]*len(all_modules), 
                   marker_color=['green' if s == 'Active' else 'red' for s in status])
        ])
        
        fig.update_layout(
            title="Module Status",
            xaxis_title="Module",
            yaxis_title="Status",
            showlegend=False,
            height=300
        )
        
        return fig
    
    def create_price_trends_graph(self):
        """Skapar prisutvecklingsgraf."""
        fig = go.Figure()
        
        for symbol in self.symbols:
            change_pct = ((self.current_prices[symbol] - self.base_prices[symbol]) / 
                         self.base_prices[symbol]) * 100
            fig.add_trace(go.Bar(
                name=symbol,
                x=[symbol],
                y=[change_pct],
                text=[f"${self.current_prices[symbol]:.2f}"],
                textposition='auto',
            ))
        
        fig.update_layout(
            title="Price Changes (%)",
            xaxis_title="Symbol",
            yaxis_title="Change %",
            height=400
        )
        
        return fig
    
    def create_indicator_graph(self):
        """Skapar indikatorgraf."""
        # Get latest indicators
        indicators_data = {'RSI': [], 'Symbols': []}
        
        for symbol in self.symbols:
            if symbol in self.strategy_engine.current_indicators:
                ind = self.strategy_engine.current_indicators[symbol]
                rsi = ind.get('technical', {}).get('RSI', 50)
                indicators_data['RSI'].append(rsi)
                indicators_data['Symbols'].append(symbol)
        
        fig = go.Figure()
        
        if indicators_data['RSI']:
            fig.add_trace(go.Bar(
                x=indicators_data['Symbols'],
                y=indicators_data['RSI'],
                name='RSI',
                marker_color=['red' if rsi > 70 else 'green' if rsi < 30 else 'blue' 
                             for rsi in indicators_data['RSI']]
            ))
            
            # Add overbought/oversold lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
            fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        
        fig.update_layout(
            title="Technical Indicators (RSI)",
            xaxis_title="Symbol",
            yaxis_title="RSI Value",
            height=400
        )
        
        return fig
    
    def create_decision_flow_graph(self):
        """Skapar beslutsflödesgraf."""
        insights = self.strategic_memory.generate_insights()
        
        categories = ['Total Decisions', 'Executions', 'Success Rate (%)']
        values = [
            insights['total_decisions'],
            insights['total_executions'],
            insights['success_rate'] * 100 if insights['total_executions'] > 0 else 0
        ]
        
        fig = go.Figure(data=[
            go.Bar(x=categories, y=values, marker_color=['blue', 'green', 'orange'])
        ])
        
        fig.update_layout(
            title="Decision Flow Metrics",
            xaxis_title="Metric",
            yaxis_title="Value",
            height=400
        )
        
        return fig
    
    def create_reward_flow_graph(self):
        """Skapar reward flow graf (Sprint 4.4)."""
        reward_metrics = self.reward_tuner.get_reward_metrics()
        
        base_rewards = reward_metrics['base_reward_history']
        tuned_rewards = reward_metrics['tuned_reward_history']
        
        fig = go.Figure()
        
        if base_rewards and tuned_rewards:
            x = list(range(len(base_rewards)))
            fig.add_trace(go.Scatter(x=x, y=base_rewards, name='Base Reward', 
                                    mode='lines', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=x, y=tuned_rewards, name='Tuned Reward',
                                    mode='lines', line=dict(color='green')))
        
        fig.update_layout(
            title="Reward Flow: Base vs Tuned (Sprint 4.4)",
            xaxis_title="Episode",
            yaxis_title="Reward Value",
            height=400
        )
        
        return fig
    
    def create_parameter_evolution_graph(self):
        """Skapar parameterutvecklingsgraf (Sprint 4.3)."""
        param_history = self.introspection_panel.parameter_adjustments
        
        fig = go.Figure()
        
        if len(param_history) > 1:
            # Extract a few key parameters to visualize
            params_to_show = ['signal_threshold', 'risk_tolerance', 'consensus_threshold']
            
            for param_name in params_to_show:
                values = []
                for entry in param_history:
                    params = entry.get('parameters', entry.get('adjusted_parameters', {}))
                    if param_name in params:
                        values.append(params[param_name])
                
                if values:
                    fig.add_trace(go.Scatter(
                        x=list(range(len(values))),
                        y=values,
                        name=param_name,
                        mode='lines+markers'
                    ))
        
        fig.update_layout(
            title="Adaptive Parameter Evolution (Sprint 4.3)",
            xaxis_title="Adjustment Event",
            yaxis_title="Parameter Value",
            height=400
        )
        
        return fig
    
    def create_agent_performance_graph(self):
        """Skapar agentperformance graf."""
        agent_status = self.introspection_panel.agent_status_history
        
        fig = go.Figure()
        
        if agent_status:
            # Group by agent and show loss trends
            agents = {}
            for status in agent_status:
                agent_id = status.get('agent_id', 'unknown')
                if agent_id not in agents:
                    agents[agent_id] = []
                agents[agent_id].append(status.get('loss', 0))
            
            for agent_id, losses in agents.items():
                if losses:
                    fig.add_trace(go.Scatter(
                        x=list(range(len(losses))),
                        y=losses,
                        name=agent_id,
                        mode='lines'
                    ))
        
        fig.update_layout(
            title="Agent Training Loss",
            xaxis_title="Training Step",
            yaxis_title="Loss",
            height=400
        )
        
        return fig
    
    def create_agent_evolution_graph(self):
        """Skapar agentutvecklingsgraf."""
        profiles = self.agent_manager.get_all_profiles()
        
        agents = []
        versions = []
        
        for agent_id, profile in profiles.items():
            agents.append(agent_id)
            versions.append(profile['version'])
        
        fig = go.Figure(data=[
            go.Bar(x=agents, y=[1]*len(agents), text=versions, textposition='auto')
        ])
        
        fig.update_layout(
            title="Agent Versions",
            xaxis_title="Agent",
            yaxis_title="Active",
            height=300
        )
        
        return fig
    
    def create_agent_metrics_graph(self):
        """Skapar agentmetrikergraf."""
        # Get agent profiles to show basic metrics
        profiles = self.agent_manager.get_all_profiles()
        
        agents = list(profiles.keys())
        versions = [profiles[a]['version'] for a in agents] if agents else []
        
        # Convert version strings to numbers for plotting (e.g., "1.0.0" -> 1.0)
        version_numbers = []
        for v in versions:
            try:
                parts = v.split('.')
                version_numbers.append(float(f"{parts[0]}.{parts[1]}"))
            except (ValueError, IndexError, AttributeError):
                version_numbers.append(1.0)
        
        fig = go.Figure(data=[
            go.Bar(x=agents, y=version_numbers, marker_color='lightblue',
                   text=versions, textposition='auto')
        ])
        
        fig.update_layout(
            title="Agent Metrics",
            xaxis_title="Agent",
            yaxis_title="Version",
            height=400
        )
        
        return fig
    
    def create_event_log(self):
        """Skapar event log."""
        feedback_events = self.introspection_panel.feedback_events[-20:]
        
        log_entries = []
        for event in feedback_events:
            timestamp = datetime.fromtimestamp(
                event.get('timestamp', event.get('route_timestamp', time.time()))
            ).strftime('%H:%M:%S')
            source = event.get('source', 'unknown')
            priority = event.get('priority', 'medium')
            message = event.get('message', str(event.get('feedback_type', 'event')))
            
            log_entries.append(
                html.Div(f"[{timestamp}] [{priority.upper()}] {source}: {message}",
                        style={'padding': '2px', 'borderBottom': '1px solid #34495e'})
            )
        
        return log_entries if log_entries else [html.Div("No events yet")]
    
    def create_feedback_flow_graph(self):
        """Skapar feedback flow graf."""
        feedback_metrics = self.introspection_panel._calculate_feedback_metrics()
        
        sources = list(feedback_metrics['by_source'].keys())
        counts = list(feedback_metrics['by_source'].values())
        
        fig = go.Figure(data=[
            go.Pie(labels=sources, values=counts, hole=.3)
        ])
        
        fig.update_layout(
            title="Feedback Flow by Source",
            height=400
        )
        
        return fig
    
    def create_timeline_graph(self):
        """Skapar timeline graf (Sprint 6)."""
        timeline_summary = self.timespan_tracker.get_timeline_summary()
        
        categories = ['Decision Events', 'Final Decisions', 'Symbols Tracked']
        values = [
            timeline_summary['decision_events'],
            timeline_summary['final_decisions'],
            len(timeline_summary['symbols_tracked'])
        ]
        
        fig = go.Figure(data=[
            go.Bar(x=categories, y=values, marker_color=['purple', 'orange', 'cyan'])
        ])
        
        fig.update_layout(
            title="Timeline Analysis (Sprint 6)",
            xaxis_title="Metric",
            yaxis_title="Count",
            height=400
        )
        
        return fig
    
    def create_portfolio_value_graph(self):
        """Skapar portföljvärdesgraf."""
        portfolio = self.portfolio_manager.get_status(self.current_prices)
        
        categories = ['Cash', 'Holdings Value', 'Total Value']
        values = [
            portfolio.get('cash', 0),
            portfolio.get('total_value', 0) - portfolio.get('cash', 0),
            portfolio.get('total_value', 0)
        ]
        
        colors = ['green', 'blue', 'purple']
        
        fig = go.Figure(data=[
            go.Bar(x=categories, y=values, marker_color=colors)
        ])
        
        fig.update_layout(
            title="Portfolio Breakdown",
            xaxis_title="Category",
            yaxis_title="Value ($)",
            height=400
        )
        
        return fig
    
    def create_positions_graph(self):
        """Skapar positionsgraf."""
        portfolio = self.portfolio_manager.get_status(self.current_prices)
        positions = portfolio.get('positions', {})
        
        if not positions:
            fig = go.Figure()
            fig.add_annotation(text="No positions yet", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            fig.update_layout(title="Current Positions", height=400)
            return fig
        
        symbols = list(positions.keys())
        quantities = [pos['quantity'] for pos in positions.values()]
        values = [pos['quantity'] * self.current_prices.get(sym, pos['avg_price']) 
                 for sym, pos in positions.items()]
        
        fig = go.Figure(data=[
            go.Bar(name='Quantity', x=symbols, y=quantities, yaxis='y', offsetgroup=1),
            go.Bar(name='Value ($)', x=symbols, y=values, yaxis='y2', offsetgroup=2)
        ])
        
        fig.update_layout(
            title="Current Positions",
            xaxis_title="Symbol",
            yaxis=dict(title="Quantity"),
            yaxis2=dict(title="Value ($)", overlaying='y', side='right'),
            barmode='group',
            height=400
        )
        
        return fig
    
    def create_portfolio_details(self):
        """Skapar portfolio detaljer."""
        portfolio = self.portfolio_manager.get_status(self.current_prices)
        
        total_value = portfolio.get('total_value', 0)
        pnl = total_value - 1000
        roi = (pnl / 1000) * 100 if pnl != 0 else 0
        
        details = html.Div([
            html.H3("Portfolio Summary"),
            html.P(f"💰 Total Value: ${total_value:.2f}"),
            html.P(f"📈 P&L: ${pnl:.2f} ({roi:+.2f}%)"),
            html.P(f"💵 Cash: ${portfolio.get('cash', 0):.2f}"),
            html.P(f"📊 Positions: {len(portfolio.get('positions', {}))}")
        ])
        
        return details
    
    # Sprint 8: DQN, GAN, GNN Graph Creation Methods
    
    def create_dqn_metrics_graph(self):
        """Create DQN controller metrics graph showing epsilon, buffer, and training progress."""
        dqn_metrics = self.dqn_controller.get_metrics()
        
        fig = go.Figure()
        
        # Add epsilon decay trace
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=dqn_metrics['epsilon'],
            title={'text': f"Epsilon (Exploration)<br><sub>Training Steps: {dqn_metrics['training_steps']}</sub>"},
            delta={'reference': 1.0, 'decreasing': {'color': "green"}},
            gauge={
                'axis': {'range': [None, 1.0]},
                'bar': {'color': "#3498db"},
                'steps': [
                    {'range': [0, 0.1], 'color': "#e74c3c"},
                    {'range': [0.1, 0.5], 'color': "#f39c12"},
                    {'range': [0.5, 1.0], 'color': "#2ecc71"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': dqn_metrics.get('epsilon_min', 0.01)
                }
            },
            domain={'x': [0, 0.45], 'y': [0, 1]}
        ))
        
        # Add buffer usage gauge
        buffer_pct = (dqn_metrics['buffer_size'] / self.dqn_controller.replay_buffer.buffer.maxlen) * 100
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=buffer_pct,
            title={'text': f"Replay Buffer<br><sub>{dqn_metrics['buffer_size']}/{self.dqn_controller.replay_buffer.buffer.maxlen}</sub>"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#9b59b6"},
                'steps': [
                    {'range': [0, 50], 'color': "#ecf0f1"},
                    {'range': [50, 80], 'color': "#bdc3c7"},
                    {'range': [80, 100], 'color': "#95a5a6"}
                ]
            },
            domain={'x': [0.55, 1], 'y': [0, 1]}
        ))
        
        fig.update_layout(
            title="DQN Controller - Core Metrics",
            height=300,
            showlegend=False
        )
        
        return fig
    
    def create_dqn_training_graph(self):
        """Create DQN training progress graph showing loss and Q-values over time."""
        dqn_metrics = self.dqn_controller.get_metrics()
        
        fig = go.Figure()
        
        # Training loss over time (if we have history)
        if hasattr(self.dqn_controller, 'losses') and self.dqn_controller.losses:
            loss_history = self.dqn_controller.losses[-50:]  # Last 50 training steps
            fig.add_trace(go.Scatter(
                y=loss_history,
                mode='lines+markers',
                name='Training Loss',
                line=dict(color='#e74c3c', width=2),
                marker=dict(size=4)
            ))
        else:
            # Show current average loss
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[dqn_metrics['avg_loss'], dqn_metrics['avg_loss']],
                mode='lines',
                name='Avg Loss',
                line=dict(color='#e74c3c', width=3, dash='dash')
            ))
        
        fig.update_layout(
            title=f"DQN Training Loss (Avg: {dqn_metrics['avg_loss']:.4f})",
            xaxis_title="Training Step",
            yaxis_title="Loss",
            height=350,
            hovermode='x unified'
        )
        
        return fig
    
    def create_gan_metrics_graph(self):
        """Create GAN evolution engine metrics showing generator/discriminator performance."""
        gan_metrics = self.gan_evolution.get_metrics()
        
        fig = go.Figure()
        
        # Generator and Discriminator losses
        fig.add_trace(go.Bar(
            name='Generator Loss',
            x=['Generator'],
            y=[gan_metrics['g_loss']],
            marker_color='#3498db',
            text=[f"{gan_metrics['g_loss']:.4f}"],
            textposition='auto'
        ))
        
        fig.add_trace(go.Bar(
            name='Discriminator Loss',
            x=['Discriminator'],
            y=[gan_metrics['d_loss']],
            marker_color='#e74c3c',
            text=[f"{gan_metrics['d_loss']:.4f}"],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="GAN Evolution Engine - Adversarial Loss",
            yaxis_title="Loss",
            height=300,
            showlegend=True
        )
        
        return fig
    
    def create_gan_training_graph(self):
        """Create GAN candidate generation and acceptance metrics."""
        gan_metrics = self.gan_evolution.get_metrics()
        
        fig = go.Figure()
        
        # Acceptance rate gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=gan_metrics['acceptance_rate'] * 100,
            title={'text': f"Candidate Acceptance Rate<br><sub>{gan_metrics['candidates_accepted']}/{gan_metrics['candidates_generated']} accepted</sub>"},
            delta={'reference': 70, 'suffix': '%'},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#27ae60"},
                'steps': [
                    {'range': [0, 50], 'color': "#e74c3c"},
                    {'range': [50, 70], 'color': "#f39c12"},
                    {'range': [70, 100], 'color': "#2ecc71"}
                ],
                'threshold': {
                    'line': {'color': "blue", 'width': 4},
                    'thickness': 0.75,
                    'value': 60
                }
            }
        ))
        
        fig.update_layout(
            title="GAN Candidate Generation Performance",
            height=350
        )
        
        return fig
    
    def create_gnn_metrics_graph(self):
        """Create GNN timespan analyzer metrics showing graph size and history."""
        gnn_metrics = self.gnn_analyzer.get_metrics()
        
        fig = go.Figure()
        
        # History sizes as grouped bar chart
        categories = ['Decisions', 'Indicators', 'Outcomes']
        values = [
            gnn_metrics['decision_history_size'],
            gnn_metrics['indicator_history_size'],
            gnn_metrics['outcome_history_size']
        ]
        
        fig.add_trace(go.Bar(
            x=categories,
            y=values,
            marker_color=['#3498db', '#f39c12', '#27ae60'],
            text=values,
            textposition='auto'
        ))
        
        fig.update_layout(
            title=f"GNN Timespan Analyzer - Data History (Window: {gnn_metrics['temporal_window']})",
            yaxis_title="History Size",
            height=300
        )
        
        return fig
    
    def create_gnn_patterns_graph(self):
        """Create GNN pattern detection visualization."""
        gnn_metrics = self.gnn_analyzer.get_metrics()
        
        fig = go.Figure()
        
        # Pattern detection count
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=gnn_metrics['patterns_detected'],
            title={'text': "Patterns Detected"},
            delta={'reference': 0, 'increasing': {'color': "green"}},
            domain={'x': [0, 0.5], 'y': [0, 1]}
        ))
        
        # Temporal window gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=gnn_metrics['decision_history_size'],
            title={'text': f"Temporal Window Usage<br><sub>Max: {gnn_metrics['temporal_window']}</sub>"},
            gauge={
                'axis': {'range': [0, gnn_metrics['temporal_window']]},
                'bar': {'color': "#9b59b6"}
            },
            domain={'x': [0.5, 1], 'y': [0, 1]}
        ))
        
        fig.update_layout(
            title="GNN Pattern Detection Status",
            height=350
        )
        
        return fig
    
    # WebSocket methods for live data
    def start_live_data(self):
        """Startar live data från Finnhub WebSocket."""
        if self.live_mode:
            return
        
        print("📡 Starting Finnhub WebSocket connection...")
        self.live_mode = True
        
        # Start WebSocket in a separate thread
        self.ws_thread = threading.Thread(target=self.run_websocket, daemon=True)
        self.ws_thread.start()
    
    def stop_live_data(self):
        """Stoppar live data."""
        if not self.live_mode:
            return
        
        print("⏹️  Stopping Finnhub WebSocket connection...")
        self.live_mode = False
        
        if self.ws:
            self.ws.close()
            self.ws = None
    
    def run_websocket(self):
        """Kör WebSocket-anslutning."""
        websocket_url = f"wss://ws.finnhub.io?token={self.api_key}"
        
        self.ws = websocket.WebSocketApp(
            websocket_url,
            on_message=self.on_ws_message,
            on_error=self.on_ws_error,
            on_close=self.on_ws_close,
            on_open=self.on_ws_open
        )
        
        try:
            self.ws.run_forever()
        except Exception as e:
            print(f"❌ WebSocket error: {e}")
            self.live_mode = False
    
    def on_ws_open(self, ws):
        """Callback när WebSocket öppnas."""
        print(f"✅ WebSocket connected! Subscribing to {len(self.symbols)} symbols...")
        
        for symbol in self.symbols:
            subscribe_message = {
                'type': 'subscribe',
                'symbol': symbol
            }
            ws.send(json.dumps(subscribe_message))
            print(f"   ✓ {symbol}")
    
    def on_ws_message(self, ws, message):
        """Callback för WebSocket-meddelanden."""
        try:
            data = json.loads(message)
            
            if data.get('type') == 'trade':
                for trade in data.get('data', []):
                    self.process_live_trade(trade)
        except Exception as e:
            print(f"⚠️  Error processing WebSocket message: {e}")
    
    def on_ws_error(self, ws, error):
        """Callback för WebSocket-fel."""
        print(f"❌ WebSocket error: {error}")
    
    def on_ws_close(self, ws, close_status_code, close_msg):
        """Callback när WebSocket stängs."""
        print(f"🔌 WebSocket connection closed")
        self.live_mode = False
    
    def process_live_trade(self, trade: Dict[str, Any]):
        """Processar live trade från Finnhub."""
        symbol = trade.get('s')
        price = trade.get('p')
        volume = trade.get('v')
        timestamp = trade.get('t')
        
        if not all([symbol, price, volume, timestamp]):
            return
        
        # Update current price
        self.current_prices[symbol] = price
        
        # Publish market data
        self.message_bus.publish('market_data', {
            'symbol': symbol,
            'price': price,
            'volume': volume,
            'timestamp': timestamp
        })
        
        # Fetch indicators periodically
        self.iteration_count += 1
        if self.iteration_count % 5 == 0:
            indicators = self.indicator_registry.get_indicators(symbol)
            self.message_bus.publish('indicator_data', indicators)
            self.strategy_engine.current_indicators[symbol] = indicators
            self.risk_manager.current_indicators[symbol] = indicators
        
        # Make trading decision periodically
        if self.iteration_count % 10 == 0:
            self.make_trading_decision(symbol, price)
    
    def run(self, debug=True):
        """Kör dashboarden."""
        print("\n" + "="*70)
        print("🚀 NextGen AI Trader - Analyzer Debug Dashboard")
        print("="*70)
        print("\n📊 Dashboard starting...")
        print("🌐 Open browser at: http://localhost:8050")
        print("⏹️  Press Ctrl+C to stop\n")
        
        self.app.run(debug=debug, host='0.0.0.0', port=8050)


def main():
    """Huvudfunktion för analyzer debug dashboard."""
    dashboard = AnalyzerDebugDashboard()
    dashboard.run(debug=True)


if __name__ == "__main__":
    main()
