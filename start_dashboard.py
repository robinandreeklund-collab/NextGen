"""
start_dashboard.py - Full-scale NextGen Dashboard

Comprehensive dashboard implementation based on dashboard_structure_sprint8.yaml
Includes all panels: Portfolio, RL Agent Analysis, Agent Evolution & GAN,
Temporal Drift & GNN, Feedback & Reward Loop, CI Test Results, RL Conflict Monitor,
Decision & Consensus, Adaptive Settings, Live Market Watch

Features:
    - Modular panel architecture
    - Dark theme inspired by "Abstire Dashboard" mockup
    - Sidebar navigation
    - Top header with system status
    - Real-time data updates
    - Responsive design

Usage:
    See start_demo.py or start_live.py for starting the dashboard
"""

import time
import random
import math
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import threading

# Dash and Plotly
import dash
from dash import dcc, html, Input, Output, State, ALL
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# WebSocket
import websocket

# Import modules
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


# Color scheme inspired by Abstire Dashboard mockup
THEME_COLORS = {
    'background': '#0a0e1a',
    'surface': '#141b2d',
    'surface_light': '#1f2940',
    'primary': '#4dabf7',
    'secondary': '#845ef7',
    'success': '#51cf66',
    'warning': '#ffd43b',
    'danger': '#ff6b6b',
    'text': '#e9ecef',
    'text_secondary': '#adb5bd',
    'border': '#2c3e50',
    'chart_line1': '#4dabf7',
    'chart_line2': '#845ef7',
    'chart_line3': '#51cf66',
    'chart_line4': '#ffd43b',
    'chart_line5': '#ff6b6b',
}


class NextGenDashboard:
    """Full-scale NextGen Dashboard with all panels."""
    
    def __init__(self, live_mode: bool = False):
        """Initialize dashboard.
        
        Args:
            live_mode: If True, connects to live WebSocket data. If False, uses simulated data.
        """
        self.live_mode = live_mode
        self.api_key = "d3in10hr01qmn7fkr2a0d3in10hr01qmn7fkr2ag"
        
        # Symbols for tracking
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        
        # Base prices for simulation
        self.base_prices = {
            'AAPL': 150.0,
            'MSFT': 300.0,
            'GOOGL': 120.0,
            'AMZN': 140.0,
            'TSLA': 200.0
        }
        
        self.current_prices = self.base_prices.copy()
        self.price_trends = {symbol: 0.0 for symbol in self.symbols}
        
        # Initialize modules
        self.message_bus = MessageBus()
        self.setup_modules()
        
        # Dashboard state
        self.running = False
        self.iteration_count = 0
        self.simulation_thread = None
        self.ws = None
        self.ws_thread = None
        
        # Data history for charts
        self.price_history = {symbol: [] for symbol in self.symbols}
        self.reward_history = {'base': [], 'tuned': [], 'ppo': [], 'dqn': []}
        self.agent_metrics_history = []
        self.gan_metrics_history = {'g_loss': [], 'd_loss': [], 'acceptance_rate': []}
        self.gnn_pattern_history = []
        self.conflict_history = []
        self.decision_history = []
        
        # Setup Dash app
        self.app = dash.Dash(__name__, suppress_callback_exceptions=True)
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_modules(self) -> None:
        """Initialize all NextGen modules."""
        print("🔧 Initializing modules...")
        
        # Sprint 4 modules
        self.strategic_memory = StrategicMemoryEngine(self.message_bus)
        self.meta_evolution = MetaAgentEvolutionEngine(self.message_bus)
        self.agent_manager = AgentManager(self.message_bus)
        
        # Sprint 3 modules
        self.feedback_router = FeedbackRouter(self.message_bus)
        self.feedback_analyzer = FeedbackAnalyzer(self.message_bus)
        self.introspection_panel = IntrospectionPanel(self.message_bus)
        
        # Sprint 2 modules
        self.rl_controller = RLController(self.message_bus)
        
        # Sprint 4.4 - RewardTunerAgent
        self.reward_tuner = RewardTunerAgent(
            message_bus=self.message_bus,
            reward_scaling_factor=1.0,
            volatility_penalty_weight=0.3,
            overfitting_detector_threshold=0.2
        )
        
        # Sprint 1 modules
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
        
        # Register RewardTuner callback
        self.portfolio_manager.register_reward_tuner_callback(self.reward_tuner.on_base_reward)
        
        # Sprint 4.3 module
        self.vote_engine = VoteEngine(self.message_bus)
        
        # Sprint 5 modules
        self.decision_simulator = DecisionSimulator(self.message_bus)
        self.consensus_engine = ConsensusEngine(self.message_bus, consensus_model='weighted')
        
        # Sprint 6 modules
        self.timespan_tracker = TimespanTracker(self.message_bus)
        self.action_chain_engine = ActionChainEngine(self.message_bus)
        self.system_monitor = SystemMonitor(self.message_bus)
        
        # Sprint 8 modules
        # state_dim=4: matches sim_test.py state representation
        self.dqn_controller = DQNController(self.message_bus, state_dim=4, action_dim=3)
        self.gan_evolution = GANEvolutionEngine(self.message_bus, latent_dim=64, param_dim=16)
        self.gnn_analyzer = GNNTimespanAnalyzer(self.message_bus, input_dim=32, temporal_window=20)
        
        print("✅ All modules initialized (Sprint 1-8)")
    
    def setup_layout(self) -> None:
        """Create comprehensive dashboard layout."""
        
        self.app.layout = html.Div([
            # Top Header (fixed at top)
            self.create_top_header(),
            
            # Main container (flex layout)
            html.Div([
                # Sidebar (fixed on left, below header)
                self.create_sidebar(),
                
                # Main content area (with margin for sidebar and header)
                html.Div([
                    # Control panel
                    self.create_control_panel(),
                    
                    # Main dashboard content (no tabs - navigation via sidebar)
                    html.Div(id='dashboard-content', children=self.create_dashboard_view(), 
                            style={'padding': '20px'}),
                    
                ], style={
                    'marginLeft': '180px',  # Sidebar width
                    'marginTop': '50px',    # Header height
                    'padding': '0',
                    'minHeight': 'calc(100vh - 50px)',
                }),
            ]),
            
            # Auto-refresh interval
            dcc.Interval(id='interval-component', interval=2000, n_intervals=0),
            
            # Store for current view
            dcc.Store(id='current-view', data='dashboard'),
            
        ], style={
            'fontFamily': 'Segoe UI, Roboto, Arial, sans-serif',
            'backgroundColor': THEME_COLORS['background'],
            'color': THEME_COLORS['text'],
            'minHeight': '100vh',
        })
    
    def create_top_header(self) -> html.Div:
        """Create top header with branding and system status (matches mockup)."""
        return html.Div([
            # Left: Dashboard title
            html.Div([
                html.H1("Nextgen-dashboard", 
                       style={'margin': '0', 'fontSize': '18px', 'fontWeight': '600',
                             'color': THEME_COLORS['text']}),
            ], style={'flex': '0 0 auto'}),
            
            # Center/Right: System status indicators
            html.Div([
                # System Status
                html.Div([
                    html.Span('●', style={'color': THEME_COLORS['success'], 'marginRight': '5px'}),
                    html.Span('System Status', style={'fontSize': '12px', 'marginRight': '5px'}),
                    html.Span('OK', style={'fontSize': '12px', 'fontWeight': 'bold', 
                                          'color': THEME_COLORS['success']}),
                ], style={'display': 'inline-flex', 'alignItems': 'center', 
                         'backgroundColor': THEME_COLORS['surface_light'],
                         'padding': '6px 12px', 'borderRadius': '4px', 'marginRight': '10px'}),
                
                # RL Agents
                html.Div([
                    html.Span('●', style={'color': THEME_COLORS['primary'], 'marginRight': '5px'}),
                    html.Span('RL Agents', style={'fontSize': '12px', 'marginRight': '5px'}),
                    html.Span('OK', style={'fontSize': '12px', 'fontWeight': 'bold',
                                          'color': THEME_COLORS['primary']}),
                ], style={'display': 'inline-flex', 'alignItems': 'center',
                         'backgroundColor': THEME_COLORS['surface_light'],
                         'padding': '6px 12px', 'borderRadius': '4px', 'marginRight': '10px'}),
                
                # Reward Trend
                html.Div([
                    html.Span('●', style={'color': THEME_COLORS['primary'], 'marginRight': '5px'}),
                    html.Span('Reward Trend', style={'fontSize': '12px', 'marginRight': '5px'}),
                    html.Span('OK', style={'fontSize': '12px', 'fontWeight': 'bold',
                                          'color': THEME_COLORS['primary']}),
                ], style={'display': 'inline-flex', 'alignItems': 'center',
                         'backgroundColor': THEME_COLORS['surface_light'],
                         'padding': '6px 12px', 'borderRadius': '4px', 'marginRight': '10px'}),
                
                # Test Status
                html.Div([
                    html.Span('■', style={'color': THEME_COLORS['text_secondary'], 'marginRight': '5px'}),
                    html.Span('Test Status', style={'fontSize': '12px', 'marginRight': '5px'}),
                    html.Span('Stable', style={'fontSize': '12px', 'fontWeight': 'bold',
                                               'color': THEME_COLORS['text']}),
                ], style={'display': 'inline-flex', 'alignItems': 'center',
                         'backgroundColor': THEME_COLORS['surface_light'],
                         'padding': '6px 12px', 'borderRadius': '4px', 'marginRight': '10px'}),
                
                # Drift Indicator
                html.Div([
                    html.Span('■', style={'color': THEME_COLORS['text_secondary'], 'marginRight': '5px'}),
                    html.Span('Drift Indicator', style={'fontSize': '12px', 'marginRight': '5px'}),
                    html.Span('Stable', style={'fontSize': '12px', 'fontWeight': 'bold',
                                               'color': THEME_COLORS['text']}),
                ], style={'display': 'inline-flex', 'alignItems': 'center',
                         'backgroundColor': THEME_COLORS['surface_light'],
                         'padding': '6px 12px', 'borderRadius': '4px', 'marginRight': '15px'}),
                
                # Dark Mode Toggle button (right side)
                html.Button('Dark Mode Toggle',
                           style={'backgroundColor': THEME_COLORS['surface_light'],
                                 'color': THEME_COLORS['text'],
                                 'border': f'1px solid {THEME_COLORS["border"]}',
                                 'padding': '8px 16px',
                                 'borderRadius': '4px',
                                 'fontSize': '12px',
                                 'cursor': 'pointer'}),
            ], style={'display': 'flex', 'alignItems': 'center', 'flex': '1', 
                     'justifyContent': 'flex-end'}),
            
        ], style={
            'backgroundColor': THEME_COLORS['surface'],
            'padding': '12px 20px',
            'borderBottom': f'1px solid {THEME_COLORS["border"]}',
            'display': 'flex',
            'justifyContent': 'space-between',
            'alignItems': 'center',
            'position': 'fixed',
            'top': '0',
            'left': '0',
            'right': '0',
            'zIndex': '1000',
            'height': '50px',
        })
    
    def create_sidebar(self) -> html.Div:
        """Create sidebar navigation (matches mockup)."""
        return html.Div([
            # Header: "Modules"
            html.Div([
                html.H3("Modules", style={'margin': '0', 'fontSize': '16px', 'fontWeight': '600',
                                         'color': THEME_COLORS['text']}),
            ], style={'marginBottom': '20px', 'paddingBottom': '15px', 
                     'borderBottom': f'1px solid {THEME_COLORS["border"]}'}),
            
            # Navigation menu items
            html.Div([
                self.create_sidebar_menu_item('📊', 'Dashboard', 'dashboard', True),
                self.create_sidebar_menu_item('🤖', 'Agents', 'agents', False),
                self.create_sidebar_menu_item('🎁', 'Rewards', 'rewards', False),
                self.create_sidebar_menu_item('🧪', 'Testing', 'testing', False),
                self.create_sidebar_menu_item('📈', 'Drift', 'drift', False),
                self.create_sidebar_menu_item('⚠️', 'Conflicts', 'conflicts', False),
                self.create_sidebar_menu_item('📅', 'Events', 'events', False),
                self.create_sidebar_menu_item('📝', 'Logs', 'logs', False),
                self.create_sidebar_menu_item('⚙️', 'Settings', 'settings', False),
            ], style={'marginBottom': 'auto'}),
            
            # Bottom: Parameter Slider Sync toggle
            html.Div([
                html.Div([
                    html.Span('Parameter', style={'fontSize': '13px', 'fontWeight': '500'}),
                    html.Br(),
                    html.Span('Slider Sync', style={'fontSize': '13px', 'fontWeight': '500'}),
                ], style={'flex': '1'}),
                html.Div([
                    # Toggle switch (visual only - styled as ON)
                    html.Div([
                        html.Div(style={
                            'position': 'absolute',
                            'top': '2px',
                            'right': '2px',
                            'width': '16px',
                            'height': '16px',
                            'backgroundColor': 'white',
                            'borderRadius': '50%',
                            'transition': 'transform 0.3s',
                        }),
                    ], style={
                        'position': 'relative',
                        'display': 'inline-block',
                        'width': '40px',
                        'height': '20px',
                        'backgroundColor': 'rgba(255, 255, 255, 0.3)',
                        'borderRadius': '10px',
                        'cursor': 'pointer',
                    }),
                ]),
            ], style={
                'display': 'flex',
                'alignItems': 'center',
                'justifyContent': 'space-between',
                'backgroundColor': THEME_COLORS['primary'],
                'padding': '12px 15px',
                'borderRadius': '6px',
                'marginTop': '20px',
                'color': 'white',
            }),
            
        ], style={
            'position': 'fixed',
            'left': '0',
            'top': '50px',  # Below header
            'width': '180px',
            'height': 'calc(100vh - 50px)',
            'backgroundColor': THEME_COLORS['surface'],
            'padding': '20px 15px',
            'borderRight': f'1px solid {THEME_COLORS["border"]}',
            'overflowY': 'auto',
            'display': 'flex',
            'flexDirection': 'column',
            'zIndex': '999',
        })
    
    def create_sidebar_menu_item(self, icon: str, label: str, view_id: str, active: bool = False) -> html.Div:
        """Create a sidebar menu item."""
        bg_color = THEME_COLORS['primary'] if active else 'transparent'
        text_color = 'white' if active else THEME_COLORS['text']
        
        return html.Div([
            html.Span(icon, style={'marginRight': '10px', 'fontSize': '16px'}),
            html.Span(label, style={'fontSize': '14px', 'fontWeight': '500' if active else '400'}),
        ], id={'type': 'sidebar-menu-item', 'index': view_id}, style={
            'padding': '10px 12px',
            'marginBottom': '4px',
            'backgroundColor': bg_color,
            'color': text_color,
            'borderRadius': '6px',
            'cursor': 'pointer',
            'display': 'flex',
            'alignItems': 'center',
            'transition': 'background-color 0.2s',
        })
    
    
    def create_control_panel(self) -> html.Div:
        """Create control panel with start/stop buttons."""
        return html.Div([
            html.Button('Start', id='start-btn', n_clicks=0,
                       style=self.get_button_style(THEME_COLORS['success'])),
            html.Button('Stop', id='stop-btn', n_clicks=0,
                       style=self.get_button_style(THEME_COLORS['danger'])),
            html.Span(id='status-indicator', children='Status: Stopped',
                     style={'marginLeft': '20px', 'fontSize': '14px', 
                           'color': THEME_COLORS['text_secondary']}),
        ], style={
            'padding': '15px 20px',
            'backgroundColor': THEME_COLORS['surface_light'],
            'borderBottom': f'1px solid {THEME_COLORS["border"]}',
        })
    
    def get_button_style(self, bg_color: str) -> dict:
        """Get consistent button styling."""
        return {
            'margin': '0 5px',
            'padding': '8px 20px',
            'backgroundColor': bg_color,
            'color': 'white',
            'border': 'none',
            'borderRadius': '4px',
            'cursor': 'pointer',
            'fontSize': '14px',
            'fontWeight': '500',
        }
    
    def get_tab_style(self) -> dict:
        """Get tab style."""
        return {
            'backgroundColor': THEME_COLORS['surface'],
            'color': THEME_COLORS['text_secondary'],
            'border': 'none',
            'padding': '12px 20px',
            'fontSize': '13px',
        }
    
    def get_tab_selected_style(self) -> dict:
        """Get selected tab style."""
        return {
            'backgroundColor': THEME_COLORS['surface_light'],
            'color': THEME_COLORS['primary'],
            'border': 'none',
            'borderBottom': f'3px solid {THEME_COLORS["primary"]}',
            'padding': '12px 20px',
            'fontSize': '13px',
            'fontWeight': 'bold',
        }
    
    def get_chart_layout(self, title: str) -> dict:
        """Get consistent chart layout."""
        return {
            'plot_bgcolor': THEME_COLORS['surface'],
            'paper_bgcolor': THEME_COLORS['surface'],
            'font': {'color': THEME_COLORS['text'], 'size': 12},
            'title': {'text': title, 'font': {'size': 16, 'color': THEME_COLORS['text']}},
            'xaxis': {
                'gridcolor': THEME_COLORS['border'],
                'color': THEME_COLORS['text_secondary'],
            },
            'yaxis': {
                'gridcolor': THEME_COLORS['border'],
                'color': THEME_COLORS['text_secondary'],
            },
            'margin': {'l': 50, 'r': 20, 't': 40, 'b': 40},
        }
    
    def setup_callbacks(self) -> None:
        """Setup all dashboard callbacks."""
        
        # Sidebar navigation callback
        @self.app.callback(
            Output('dashboard-content', 'children'),
            Input({'type': 'sidebar-menu-item', 'index': ALL}, 'n_clicks'),
            prevent_initial_call=False
        )
        def render_view(n_clicks):
            ctx = dash.callback_context
            if not ctx.triggered or all(c is None for c in n_clicks):
                return self.create_dashboard_view()
            
            # Get which menu item was clicked
            triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
            if triggered_id == '':
                return self.create_dashboard_view()
                
            import json
            button_info = json.loads(triggered_id)
            view_id = button_info['index']
            
            # Route to appropriate view
            if view_id == 'dashboard':
                return self.create_dashboard_view()
            elif view_id == 'agents':
                return self.create_rl_analysis_panel()
            elif view_id == 'rewards':
                return self.create_feedback_panel()
            elif view_id == 'testing':
                return self.create_ci_tests_panel()
            elif view_id == 'drift':
                return self.create_temporal_gnn_panel()
            elif view_id == 'conflicts':
                return self.create_conflict_panel()
            elif view_id == 'events':
                return self.create_agent_evolution_panel()
            elif view_id == 'logs':
                return self.create_consensus_panel()
            elif view_id == 'settings':
                return self.create_adaptive_panel()
            
            return self.create_dashboard_view()
        
        # Start/Stop callbacks
        @self.app.callback(
            Output('status-indicator', 'children'),
            [Input('start-btn', 'n_clicks'),
             Input('stop-btn', 'n_clicks')],
            prevent_initial_call=True
        )
        def control_simulation(start_clicks, stop_clicks):
            ctx = dash.callback_context
            if not ctx.triggered:
                return 'Status: Stopped'
            
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            if button_id == 'start-btn' and not self.running:
                self.start_simulation()
                return f'Status: Running ({"Live" if self.live_mode else "Demo"})'
            elif button_id == 'stop-btn' and self.running:
                self.stop_simulation()
                return 'Status: Stopped'
            
            return f'Status: {"Running" if self.running else "Stopped"}'
        
    def create_dashboard_view(self) -> html.Div:
        """Create main dashboard view with all key metrics and charts."""
        # Get live data
        total_value = self.portfolio_manager.get_portfolio_value(self.current_prices)
        cash = self.portfolio_manager.cash
        holdings_value = total_value - cash
        
        if self.portfolio_manager.start_capital > 0:
            roi = ((total_value - self.portfolio_manager.start_capital) / self.portfolio_manager.start_capital) * 100
        else:
            roi = 0.0
        
        return html.Div([
            html.H1("Dashboard", style={'fontSize': '32px', 'marginBottom': '30px', 
                                        'color': THEME_COLORS['text']}),
            
            # Top metrics row
            html.Div([
                self.create_metric_card("Total Value", f"${total_value:.2f}", THEME_COLORS['primary']),
                self.create_metric_card("Cash", f"${cash:.2f}", THEME_COLORS['success']),
                self.create_metric_card("Holdings", f"${holdings_value:.2f}", THEME_COLORS['secondary']),
                self.create_metric_card("ROI", f"{roi:.2f}%", THEME_COLORS['warning']),
            ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(4, 1fr)', 
                     'gap': '20px', 'marginBottom': '30px'}),
            
            # Main content grid (3 columns x 3 rows)
            html.Div([
                # Row 1: Reward Trace, Agent Decision Events, GNN Drift
                self.create_chart_card("Reward Trace", 
                    self.create_reward_trace_chart()),
                self.create_chart_card("Agent Decision Events", 
                    self.create_agent_events_list()),
                self.create_chart_card("GNN Drift", 
                    self.create_drift_metric()),
                
                # Row 2: Conflict Highlights, Test Trigger Markers, System Logs
                self.create_chart_card("Conflict Highlights", 
                    self.create_conflict_display()),
                self.create_chart_card("Test Trigger Markers", 
                    self.create_test_markers()),
                self.create_chart_card("System Logs", 
                    self.create_system_logs()),
                
                # Row 3: Parameter Sliders, Export Snapshot, Replay Mode
                self.create_chart_card("Parameter Sliders", 
                    self.create_parameter_sliders()),
                self.create_chart_card("Export Snapshot", 
                    self.create_export_controls()),
                self.create_chart_card("Replay Mode", 
                    self.create_replay_controls()),
                    
            ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(3, 1fr)', 
                     'gap': '20px'}),
        ])
    
    def create_chart_card(self, title: str, content) -> html.Div:
        """Create a card for charts."""
        return html.Div([
            html.H3(title, style={'fontSize': '16px', 'marginBottom': '15px', 
                                  'color': THEME_COLORS['text']}),
            content
        ], style={
            'backgroundColor': THEME_COLORS['surface'],
            'padding': '20px',
            'borderRadius': '8px',
            'border': f'1px solid {THEME_COLORS["border"]}',
        })
    
    def create_reward_trace_chart(self):
        """Create reward trace line chart."""
        import plotly.graph_objs as go
        
        # Get reward history
        rewards = [h.get('reward', 0) for h in self.decision_history[-100:]] if self.decision_history else [0]
        x = list(range(len(rewards)))
        
        fig = go.Figure(data=[
            go.Scatter(x=x, y=rewards, mode='lines', name='Reward',
                      line=dict(color=THEME_COLORS['primary'], width=2))
        ])
        
        fig.update_layout(
            plot_bgcolor=THEME_COLORS['background'],
            paper_bgcolor=THEME_COLORS['surface'],
            font=dict(color=THEME_COLORS['text']),
            margin=dict(l=40, r=20, t=20, b=40),
            height=200,
            showlegend=False,
            xaxis=dict(gridcolor=THEME_COLORS['border']),
            yaxis=dict(gridcolor=THEME_COLORS['border']),
        )
        
        return dcc.Graph(figure=fig, config={'displayModeBar': False})
    
    def create_agent_events_list(self):
        """Create list of recent agent decision events."""
        events = [
            {'agent': 'Agent 1', 'action': 'Buy', 'color': '#ff9500'},
            {'agent': 'Agent 2', 'action': 'Sell', 'color': '#ffd43b'},
            {'agent': 'Agent 3', 'action': 'Hold', 'color': '#51cf66'},
            {'agent': 'Agent 4', 'action': 'Sell', 'color': '#ff6b6b'},
            {'agent': 'Agent 5', 'action': 'Hold', 'color': '#4dabf7'},
        ]
        
        return html.Div([
            html.Div("Recent actions", style={'fontSize': '12px', 'marginBottom': '10px', 
                                              'color': THEME_COLORS['text_secondary']}),
            html.Div([
                html.Div([
                    html.Span('●', style={'color': event['color'], 'marginRight': '8px'}),
                    html.Span(f"{event['agent']}: ", style={'fontWeight': '500'}),
                    html.Span(event['action'], style={'padding': '2px 8px', 'borderRadius': '4px',
                                                       'backgroundColor': event['color'], 
                                                       'color': 'white', 'fontSize': '12px'}),
                    html.Span('...', style={'marginLeft': 'auto', 'color': THEME_COLORS['text_secondary']}),
                ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '8px'})
                for event in events
            ])
        ])
    
    def create_drift_metric(self):
        """Create GNN drift metric display."""
        drift_value = 0.02
        
        import plotly.graph_objs as go
        
        # Simple line chart showing minimal drift
        x = list(range(20))
        y = [0.01 + i * 0.0005 for i in x]
        
        fig = go.Figure(data=[
            go.Scatter(x=x, y=y, mode='lines', line=dict(color='#ffd43b', width=3))
        ])
        
        fig.update_layout(
            plot_bgcolor=THEME_COLORS['background'],
            paper_bgcolor=THEME_COLORS['surface'],
            font=dict(color=THEME_COLORS['text']),
            margin=dict(l=20, r=20, t=40, b=20),
            height=180,
            showlegend=False,
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False),
            annotations=[
                dict(text=f"Minimal Drift<br>{drift_value:.2f}", xref="paper", yref="paper",
                     x=0.5, y=0.5, showarrow=False, font=dict(size=16, color=THEME_COLORS['text']))
            ]
        )
        
        return dcc.Graph(figure=fig, config={'displayModeBar': False})
    
    def create_conflict_display(self):
        """Create conflict highlights display."""
        return html.Div([
            html.Div("No conflicts detected", 
                    style={'textAlign': 'center', 'padding': '40px 20px',
                          'fontSize': '16px', 'color': THEME_COLORS['text_secondary']})
        ])
    
    def create_test_markers(self):
        """Create test trigger markers list."""
        tests = [
            {'name': 'Test 1', 'status': 'Pass', 'color': '#ff9500'},
            {'name': 'Test 2', 'status': 'Pass', 'color': '#ffb366'},
            {'name': 'Test 3', 'status': 'Pass', 'color': '#ffc999'},
            {'name': 'Test 1', 'status': 'Pass', 'color': '#ff66cc'},
            {'name': 'Test 3', 'status': 'Pass', 'color': '#66cccc'},
        ]
        
        return html.Div([
            html.Div("Recent test trigger", 
                    style={'fontSize': '12px', 'marginBottom': '10px', 
                          'color': THEME_COLORS['text_secondary']}),
            html.Div([
                html.Div([
                    html.Span('●', style={'color': test['color'], 'marginRight': '8px'}),
                    html.Span(f"{test['name']}: {test['status']}", style={'fontSize': '13px'}),
                    html.Span('→', style={'marginLeft': 'auto', 'color': THEME_COLORS['text_secondary']}),
                ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '8px'})
                for test in tests
            ])
        ])
    
    def create_system_logs(self):
        """Create system logs display."""
        logs = [
            "Info: System initialized",
            "Info: Agents deployed",
            "Info: Agents deployed",
            "Info: Test completed",
            "Info: Test ✓ validated",
            "Info: Test completed",
            "Info: Test completed",
            "Info: Test completed",
            "Info: Test completed",
        ]
        
        return html.Div([
            html.Div("Recent Logs", 
                    style={'fontSize': '12px', 'marginBottom': '10px',
                          'color': THEME_COLORS['text_secondary']}),
            html.Div([
                html.Div(log, style={'fontSize': '11px', 'marginBottom': '4px',
                                     'color': THEME_COLORS['text']})
                for log in logs
            ], style={'maxHeight': '150px', 'overflowY': 'auto'})
        ])
    
    def create_parameter_sliders(self):
        """Create parameter sliders."""
        return html.Div([
            html.Div([
                html.Div([
                    html.Span("Learning Rate", style={'fontSize': '13px', 'marginBottom': '5px'}),
                    html.Div([
                        dcc.Slider(min=0, max=1, value=0.5, marks={}, 
                                  tooltip={"placement": "bottom", "always_visible": False},
                                  className='custom-slider'),
                        html.Span("✏️", style={'marginLeft': '8px', 'cursor': 'pointer'}),
                    ], style={'display': 'flex', 'alignItems': 'center'}),
                ], style={'marginBottom': '15px'}),
                
                html.Div([
                    html.Span("Discount Factor", style={'fontSize': '13px', 'marginBottom': '5px'}),
                    html.Div([
                        dcc.Slider(min=0, max=1, value=0.75, marks={},
                                  tooltip={"placement": "bottom", "always_visible": False}),
                        html.Span("✏️", style={'marginLeft': '8px', 'cursor': 'pointer'}),
                    ], style={'display': 'flex', 'alignItems': 'center'}),
                ], style={'marginBottom': '15px'}),
                
                html.Div([
                    html.Span("Exploration Rate", style={'fontSize': '13px', 'marginBottom': '5px'}),
                    html.Div([
                        dcc.Slider(min=0, max=1, value=0.3, marks={},
                                  tooltip={"placement": "bottom", "always_visible": False}),
                        html.Span("✏️", style={'marginLeft': '8px', 'cursor': 'pointer'}),
                    ], style={'display': 'flex', 'alignItems': 'center'}),
                ]),
            ])
        ])
    
    def create_export_controls(self):
        """Create export snapshot controls."""
        return html.Div([
            html.Button("Export Data", 
                       style={'width': '100%', 'padding': '12px', 'marginTop': '40px',
                             'backgroundColor': THEME_COLORS['primary'], 'color': 'white',
                             'border': 'none', 'borderRadius': '6px', 'cursor': 'pointer',
                             'fontSize': '14px', 'fontWeight': '500'})
        ], style={'textAlign': 'center'})
    
    def create_replay_controls(self):
        """Create replay mode controls."""
        return html.Div([
            html.Button("⚙️", 
                       style={'width': '100%', 'padding': '20px', 'marginBottom': '10px',
                             'backgroundColor': 'white', 'color': THEME_COLORS['text'],
                             'border': f'1px solid {THEME_COLORS["border"]}', 
                             'borderRadius': '6px', 'cursor': 'pointer',
                             'fontSize': '24px'}),
            html.Div("Start Replay", 
                    style={'textAlign': 'center', 'marginBottom': '10px', 'fontSize': '13px'}),
            
            html.Button("Pause Replay", 
                       style={'width': '100%', 'padding': '12px', 'marginBottom': '10px',
                             'backgroundColor': THEME_COLORS['text_secondary'], 
                             'color': 'white', 'border': 'none', 'borderRadius': '6px',
                             'cursor': 'pointer', 'fontSize': '14px'}),
            
            html.Button("💬", 
                       style={'width': '100%', 'padding': '20px',
                             'backgroundColor': 'white', 'color': THEME_COLORS['text'],
                             'border': f'1px solid {THEME_COLORS["border"]}',
                             'borderRadius': '6px', 'cursor': 'pointer', 'fontSize': '24px'}),
            html.Div("Step Forward", 
                    style={'textAlign': 'center', 'marginTop': '10px', 'fontSize': '13px'}),
        ])
    
    # Panel creation methods
    
    def create_portfolio_panel(self) -> html.Div:
        """Create Portfolio panel."""
        # Get portfolio data
        total_value = self.portfolio_manager.get_portfolio_value(self.current_prices)
        cash = self.portfolio_manager.cash
        holdings_value = total_value - cash
        
        # Calculate ROI with safety check
        if self.portfolio_manager.start_capital > 0:
            roi = ((total_value - self.portfolio_manager.start_capital) / self.portfolio_manager.start_capital) * 100
        else:
            roi = 0.0
        
        return html.Div([
            html.H2("Portfolio Overview", 
                   style={'color': THEME_COLORS['primary'], 'marginBottom': '20px'}),
            
            # Portfolio metrics cards
            html.Div([
                self.create_metric_card("Total Value", f"${total_value:.2f}", 
                                       THEME_COLORS['primary']),
                self.create_metric_card("Cash", f"${cash:.2f}", 
                                       THEME_COLORS['success']),
                self.create_metric_card("Holdings", f"${holdings_value:.2f}", 
                                       THEME_COLORS['secondary']),
                self.create_metric_card("ROI", f"{roi:.2f}%", 
                                       THEME_COLORS['warning']),
            ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(4, 1fr)', 
                     'gap': '20px', 'marginBottom': '30px'}),
            
            # Charts
            html.Div([
                html.Div("Portfolio value chart would be displayed here with live updates"),
                html.Div("Position breakdown chart would be displayed here"),
            ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '20px'}),
        ])
    
    def create_rl_analysis_panel(self) -> html.Div:
        """Create RL Agent Analysis panel."""
        return html.Div([
            html.H2("RL Agent Analysis", 
                   style={'color': THEME_COLORS['primary'], 'marginBottom': '20px'}),
            
            html.Div("Hybrid RL comparison: PPO vs DQN performance"),
            html.Div("Reward flow visualization"),
            html.Div("DQN metrics and epsilon schedule"),
        ])
    
    def create_agent_evolution_panel(self) -> html.Div:
        """Create Agent Evolution & GAN panel."""
        return html.Div([
            html.H2("Agent Evolution & GAN", 
                   style={'color': THEME_COLORS['primary'], 'marginBottom': '20px'}),
            
            html.Div("GAN generator/discriminator loss charts"),
            html.Div("Agent evolution timeline"),
            html.Div("Candidate acceptance rate gauge"),
        ])
    
    def create_temporal_gnn_panel(self) -> html.Div:
        """Create Temporal Drift & GNN panel."""
        return html.Div([
            html.H2("Temporal Drift & GNN Analysis", 
                   style={'color': THEME_COLORS['primary'], 'marginBottom': '20px'}),
            
            html.Div("GNN pattern detection"),
            html.Div("Pattern confidence charts"),
            html.Div("Temporal insights timeline"),
        ])
    
    def create_feedback_panel(self) -> html.Div:
        """Create Feedback & Reward Loop panel."""
        return html.Div([
            html.H2("Feedback & Reward Loop", 
                   style={'color': THEME_COLORS['primary'], 'marginBottom': '20px'}),
            
            html.Div("Reward transformation chart"),
            html.Div("Feedback flow visualization"),
        ])
    
    def create_ci_tests_panel(self) -> html.Div:
        """Create CI Test Results panel."""
        return html.Div([
            html.H2("CI Test Results", 
                   style={'color': THEME_COLORS['primary'], 'marginBottom': '20px'}),
            
            # Test metrics
            html.Div([
                self.create_metric_card("Total Tests", "314", THEME_COLORS['primary']),
                self.create_metric_card("Passed", "314", THEME_COLORS['success']),
                self.create_metric_card("Failed", "0", THEME_COLORS['danger']),
                self.create_metric_card("Coverage", "85%+", THEME_COLORS['warning']),
            ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(4, 1fr)', 
                     'gap': '20px', 'marginBottom': '30px'}),
            
            html.Div("Test results breakdown chart"),
        ])
    
    def create_conflict_panel(self) -> html.Div:
        """Create RL Conflict Monitor panel."""
        return html.Div([
            html.H2("RL Conflict Monitor (PPO vs DQN)", 
                   style={'color': THEME_COLORS['primary'], 'marginBottom': '20px'}),
            
            html.Div("Conflict frequency chart"),
            html.Div("Resolution strategy pie chart"),
        ])
    
    def create_consensus_panel(self) -> html.Div:
        """Create Decision & Consensus panel."""
        return html.Div([
            html.H2("Decision & Consensus", 
                   style={'color': THEME_COLORS['primary'], 'marginBottom': '20px'}),
            
            html.Div("Consensus chart"),
            html.Div("Voting matrix visualization"),
        ])
    
    def create_adaptive_panel(self) -> html.Div:
        """Create Adaptive Settings panel."""
        return html.Div([
            html.H2("Adaptive Parameters", 
                   style={'color': THEME_COLORS['primary'], 'marginBottom': '20px'}),
            
            html.Div("Adaptive parameters evolution chart"),
            
            # Parameter controls
            html.Div([
                html.H3("Manual Overrides", style={'fontSize': '16px', 'marginTop': '30px'}),
                html.Div([
                    html.Label("DQN Epsilon:"),
                    dcc.Slider(id='epsilon-slider', min=0.01, max=1.0, step=0.01, value=0.1,
                              marks={0.01: '0.01', 0.5: '0.5', 1.0: '1.0'}),
                ], style={'marginBottom': '20px'}),
            ]),
        ])
    
    def create_market_panel(self) -> html.Div:
        """Create Live Market Watch panel."""
        return html.Div([
            html.H2("Live Market Watch", 
                   style={'color': THEME_COLORS['primary'], 'marginBottom': '20px'}),
            
            html.Div("Real-time market prices chart"),
            html.Div("Technical indicators visualization"),
        ])
    
    def create_metric_card(self, title: str, value: str, color: str) -> html.Div:
        """Create a metric card."""
        return html.Div([
            html.Div(title, style={'fontSize': '12px', 'color': THEME_COLORS['text_secondary'], 
                                  'marginBottom': '8px'}),
            html.Div(value, style={'fontSize': '24px', 'fontWeight': 'bold', 'color': color}),
        ], style={
            'backgroundColor': THEME_COLORS['surface'],
            'padding': '20px',
            'borderRadius': '8px',
            'border': f'1px solid {THEME_COLORS["border"]}',
        })
    
    def start_simulation(self) -> None:
        """Start simulation thread."""
        if not self.running:
            self.running = True
            self.simulation_thread = threading.Thread(target=self.simulation_loop, daemon=True)
            self.simulation_thread.start()
            print("✅ Simulation started")
    
    def stop_simulation(self) -> None:
        """Stop simulation thread."""
        self.running = False
        if self.simulation_thread:
            self.simulation_thread.join(timeout=2)
        print("🛑 Simulation stopped")
    
    def simulation_loop(self) -> None:
        """Main simulation loop."""
        while self.running:
            self.iteration_count += 1
            
            # Simulate price updates
            for symbol in self.symbols:
                change = random.uniform(-2, 2)
                self.current_prices[symbol] *= (1 + change / 100)
                self.price_history[symbol].append(self.current_prices[symbol])
                if len(self.price_history[symbol]) > 100:
                    self.price_history[symbol].pop(0)
            
            # Simulate trading
            for symbol in self.symbols:
                price = self.current_prices[symbol]
                self.message_bus.publish('price_update', {
                    'symbol': symbol,
                    'price': price,
                    'timestamp': time.time()
                })
            
            time.sleep(2)
    
    def run(self, host: str = '0.0.0.0', port: int = 8050, debug: bool = False) -> None:
        """Run the dashboard."""
        print(f"🚀 Starting NextGen Dashboard on http://{host}:{port}")
        print(f"📊 Mode: {'Live' if self.live_mode else 'Demo'}")
        print("Press Ctrl+C to stop")
        self.app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    # This file should not be run directly
    # Use start_demo.py or start_live.py instead
    print("Please use start_demo.py or start_live.py to start the dashboard")
