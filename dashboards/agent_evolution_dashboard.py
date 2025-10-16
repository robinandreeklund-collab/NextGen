"""
agent_evolution_dashboard.py - Dash Dashboard för Agent Evolution Tracking

Beskrivning:
    Interaktiv dashboard för att visualisera och övervaka agent evolution.
    Visar performance trends, evolution history och versionsträd.

Features:
    - Real-time performance tracking
    - Evolution event timeline
    - Agent version history
    - Interactive evolution tree visualization
    - Performance degradation alerts

Användning:
    python dashboards/agent_evolution_dashboard.py
    
    Öppna browser: http://localhost:8050
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
import json

# Import modules (när dashboard körs standalone)
try:
    import sys
    import os
    sys.path.insert(0, os.path.abspath('..'))
    from modules.message_bus import message_bus
    from modules.strategic_memory_engine import StrategicMemoryEngine
    from modules.meta_agent_evolution_engine import MetaAgentEvolutionEngine
    from modules.agent_manager import AgentManager
except ImportError:
    print("⚠️  Kunde inte importera modules - kör från root directory")


# Initiera Dash app
app = dash.Dash(__name__, update_title='NextGen Agent Evolution...')
app.title = 'NextGen Agent Evolution Dashboard'

# Global state (för demo)
memory_engine = None
evolution_engine = None
agent_manager = None


def init_engines():
    """Initialiserar engines för dashboard."""
    global memory_engine, evolution_engine, agent_manager
    
    if memory_engine is None:
        memory_engine = StrategicMemoryEngine(message_bus)
        evolution_engine = MetaAgentEvolutionEngine(message_bus)
        agent_manager = AgentManager(message_bus)
        
        # Simulera lite data för demo
        _populate_demo_data()


def _populate_demo_data():
    """Skapar demo-data för visualisering."""
    import random
    
    # Simulera performance history
    for i in range(30):
        for agent_id in ['strategy_agent', 'risk_agent', 'decision_agent']:
            # Performance som gradvis försämras
            base_performance = 0.8
            degradation = (i / 30) * 0.2
            noise = random.uniform(-0.05, 0.05)
            
            performance = max(0.4, base_performance - degradation + noise)
            
            status = {
                'agent_id': agent_id,
                'performance_score': performance,
                'total_trades': i + 1
            }
            
            message_bus.publish('agent_status', status)
    
    # Simulera några decisions i strategic memory
    for i in range(20):
        decision = {
            'symbol': random.choice(['AAPL', 'TSLA', 'NVDA']),
            'action': random.choice(['BUY', 'SELL']),
            'quantity': random.randint(1, 3),
            'price': random.uniform(100, 200),
            'indicators': {'RSI': random.uniform(20, 80)},
            'confidence': random.uniform(0.5, 0.9)
        }
        
        execution = {
            'success': random.choice([True, True, True, False]),
            'executed_price': decision['price'] * random.uniform(0.99, 1.01),
            'quantity': decision['quantity']
        }
        
        memory_engine.log_decision(decision, execution)


# Layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1('🧬 NextGen Agent Evolution Dashboard', 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 10}),
        html.P('Real-time tracking av agent performance och evolution',
               style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': 30})
    ]),
    
    # Status cards
    html.Div([
        html.Div([
            html.H4('Total Agents', style={'color': '#3498db'}),
            html.H2(id='total-agents', children='4', style={'color': '#2c3e50'})
        ], className='status-card'),
        
        html.Div([
            html.H4('Evolution Events', style={'color': '#e74c3c'}),
            html.H2(id='evolution-events', children='0', style={'color': '#2c3e50'})
        ], className='status-card'),
        
        html.Div([
            html.H4('Avg Performance', style={'color': '#2ecc71'}),
            html.H2(id='avg-performance', children='0.00', style={'color': '#2c3e50'})
        ], className='status-card'),
        
        html.Div([
            html.H4('Degradation Alert', style={'color': '#f39c12'}),
            html.H2(id='degradation-alert', children='OK', style={'color': '#2c3e50'})
        ], className='status-card'),
    ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': 30}),
    
    # Main content
    html.Div([
        # Left column - Performance tracking
        html.Div([
            html.H3('📈 Agent Performance Over Time'),
            dcc.Graph(id='performance-graph'),
            
            html.H3('🌳 Evolution Tree', style={'marginTop': 30}),
            dcc.Graph(id='evolution-tree'),
        ], style={'width': '65%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        
        # Right column - Details
        html.Div([
            html.H3('🤖 Agent Details'),
            dcc.Dropdown(
                id='agent-selector',
                options=[
                    {'label': '🎯 Strategy Agent', 'value': 'strategy_agent'},
                    {'label': '⚠️ Risk Agent', 'value': 'risk_agent'},
                    {'label': '⚖️ Decision Agent', 'value': 'decision_agent'},
                    {'label': '✅ Execution Agent', 'value': 'execution_agent'}
                ],
                value='strategy_agent',
                style={'marginBottom': 20}
            ),
            
            html.Div(id='agent-info', style={'backgroundColor': '#ecf0f1', 'padding': 20, 'borderRadius': 5}),
            
            html.H3('📊 Strategic Memory Insights', style={'marginTop': 30}),
            html.Div(id='memory-insights', style={'backgroundColor': '#ecf0f1', 'padding': 20, 'borderRadius': 5}),
            
            html.H3('🔄 Recent Evolution Events', style={'marginTop': 30}),
            html.Div(id='evolution-events-list', style={'backgroundColor': '#ecf0f1', 'padding': 20, 'borderRadius': 5, 'maxHeight': 300, 'overflowY': 'auto'}),
        ], style={'width': '33%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingLeft': 20}),
    ]),
    
    # Auto-refresh
    dcc.Interval(
        id='interval-component',
        interval=5*1000,  # 5 sekunder
        n_intervals=0
    ),
    
    # CSS
    html.Style('''
        .status-card {
            backgroundColor: white;
            padding: 20px;
            borderRadius: 5px;
            boxShadow: 0 2px 4px rgba(0,0,0,0.1);
            textAlign: center;
            width: 22%;
        }
        body {
            fontFamily: 'Arial', sans-serif;
            backgroundColor: #f5f6fa;
            padding: 20px;
        }
    ''')
])


@app.callback(
    [Output('total-agents', 'children'),
     Output('evolution-events', 'children'),
     Output('avg-performance', 'children'),
     Output('degradation-alert', 'children'),
     Output('performance-graph', 'figure'),
     Output('evolution-tree', 'figure'),
     Output('agent-info', 'children'),
     Output('memory-insights', 'children'),
     Output('evolution-events-list', 'children')],
    [Input('interval-component', 'n_intervals'),
     Input('agent-selector', 'value')]
)
def update_dashboard(n, selected_agent):
    """Uppdaterar hela dashboard."""
    init_engines()
    
    # Status cards
    tree = agent_manager.get_evolution_tree()
    total_agents = tree['total_agents']
    
    evolution_tree_data = evolution_engine.generate_evolution_tree()
    evolution_events = evolution_tree_data['total_evolution_events']
    
    # Beräkna average performance
    avg_perf = 0.0
    perf_count = 0
    for agent_id, history in evolution_engine.agent_performance_history.items():
        if history:
            avg_perf += history[-1].get('performance_score', 0)
            perf_count += 1
    
    avg_performance = f"{(avg_perf / perf_count):.2f}" if perf_count > 0 else "0.00"
    
    # Degradation alert
    alert = "OK"
    for agent_id in ['strategy_agent', 'risk_agent', 'decision_agent']:
        trend = evolution_engine.get_agent_performance_trend(agent_id)
        if trend['trend'] == 'declining':
            alert = f"⚠️ {agent_id}"
            break
    
    # Performance graph
    perf_fig = create_performance_figure()
    
    # Evolution tree
    tree_fig = create_evolution_tree_figure()
    
    # Agent info
    agent_info_div = create_agent_info(selected_agent)
    
    # Memory insights
    memory_div = create_memory_insights()
    
    # Evolution events list
    events_div = create_evolution_events_list()
    
    return (total_agents, evolution_events, avg_performance, alert,
            perf_fig, tree_fig, agent_info_div, memory_div, events_div)


def create_performance_figure():
    """Skapar performance tracking figure."""
    fig = make_subplots(rows=1, cols=1)
    
    colors = {
        'strategy_agent': '#3498db',
        'risk_agent': '#e74c3c',
        'decision_agent': '#2ecc71',
        'execution_agent': '#f39c12'
    }
    
    for agent_id, history in evolution_engine.agent_performance_history.items():
        if history:
            x = list(range(len(history)))
            y = [h.get('performance_score', 0) for h in history]
            
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='lines+markers',
                name=agent_id.replace('_', ' ').title(),
                line=dict(color=colors.get(agent_id, '#95a5a6'), width=2),
                marker=dict(size=6)
            ))
    
    # Add threshold line
    if evolution_engine.agent_performance_history:
        max_len = max(len(h) for h in evolution_engine.agent_performance_history.values())
        fig.add_hline(y=0.65, line_dash="dash", line_color="red", 
                      annotation_text="Degradation Threshold (25%)")
    
    fig.update_layout(
        title='Agent Performance Tracking',
        xaxis_title='Samples',
        yaxis_title='Performance Score',
        hovermode='x unified',
        height=400
    )
    
    return fig


def create_evolution_tree_figure():
    """Skapar evolution tree visualization."""
    tree_data = agent_manager.get_evolution_tree()
    
    # Skapa sunburst diagram
    labels = ['All Agents']
    parents = ['']
    values = [1]
    colors_list = []
    
    for agent in tree_data['agents']:
        agent_name = agent['agent_name']
        labels.append(agent_name)
        parents.append('All Agents')
        values.append(len(agent['versions']))
        colors_list.append(len(agent['versions']))
        
        # Add versions
        for version_info in agent['versions'][-5:]:  # Last 5 versions
            labels.append(f"v{version_info['version']}")
            parents.append(agent_name)
            values.append(1)
            colors_list.append(1)
    
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        marker=dict(
            colorscale='Blues',
            cmid=2
        )
    ))
    
    fig.update_layout(
        title='Agent Evolution Tree (Sunburst)',
        height=400
    )
    
    return fig


def create_agent_info(agent_id):
    """Skapar agent info display."""
    profile = agent_manager.get_agent_profile(agent_id)
    trend = evolution_engine.get_agent_performance_trend(agent_id)
    
    info = [
        html.H4(f"🤖 {agent_id.replace('_', ' ').title()}"),
        html.P(f"Version: v{profile['version']}", style={'fontWeight': 'bold'}),
        html.P(f"Total Versions: {len(profile['versions'])}"),
        html.P(f"Created: {profile['created_at']}"),
        html.Hr(),
        html.P(f"Performance Trend: {trend['trend'].upper()}", 
               style={'color': 'green' if trend['trend'] == 'improving' else 'red' if trend['trend'] == 'declining' else 'orange'}),
        html.P(f"Degradation: {trend['degradation_percent']:.1f}%"),
        html.Hr(),
        html.P("Latest Versions:", style={'fontWeight': 'bold'}),
    ]
    
    for v in profile['versions'][-3:]:
        info.append(html.P(f"v{v['version']}: {v['description']}", style={'fontSize': '0.9em', 'margin': '5px 0'}))
    
    return info


def create_memory_insights():
    """Skapar strategic memory insights display."""
    insights = memory_engine.generate_insights()
    
    content = [
        html.P(f"📊 Total Decisions: {insights['total_decisions']}", style={'fontWeight': 'bold'}),
        html.P(f"✅ Success Rate: {insights['success_rate']:.1%}"),
        html.P(f"💰 Avg Profit: ${insights['avg_profit']:.2f}"),
        html.Hr(),
    ]
    
    if insights.get('best_indicators'):
        content.append(html.P("🏆 Best Indicators:", style={'fontWeight': 'bold'}))
        for ind in insights['best_indicators'][:3]:
            content.append(html.P(f"  • {ind}", style={'fontSize': '0.9em', 'margin': '3px 0'}))
    
    if insights.get('recommendations'):
        content.append(html.Hr())
        content.append(html.P("💡 Recommendations:", style={'fontWeight': 'bold'}))
        for rec in insights['recommendations'][:2]:
            content.append(html.P(f"  • {rec}", style={'fontSize': '0.9em', 'margin': '3px 0'}))
    
    return content


def create_evolution_events_list():
    """Skapar lista av evolution events."""
    events = evolution_engine.evolution_history[-10:]  # Senaste 10
    
    if not events:
        return html.P("Inga evolution events ännu", style={'fontStyle': 'italic', 'color': '#7f8c8d'})
    
    content = []
    for event in reversed(events):
        content.append(html.Div([
            html.P(f"🧬 {event['agent_id']}", style={'fontWeight': 'bold', 'marginBottom': 5}),
            html.P(f"Reason: {event['reason']}", style={'fontSize': '0.85em', 'marginBottom': 3}),
            html.P(f"Severity: {event.get('severity', 'N/A')}", style={'fontSize': '0.85em', 'marginBottom': 3}),
            html.P(f"Time: {event['logged_at']}", style={'fontSize': '0.8em', 'color': '#7f8c8d'}),
            html.Hr(style={'margin': '10px 0'})
        ]))
    
    return content


if __name__ == '__main__':
    print("🚀 Startar Agent Evolution Dashboard...")
    print("📊 Öppna http://localhost:8050 i din browser")
    print("⏸️  Tryck Ctrl+C för att stoppa")
    
    app.run_server(debug=True, port=8050)
