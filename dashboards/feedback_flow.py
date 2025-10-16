"""
feedback_flow.py - Dash Dashboard för Feedbackflöde

Beskrivning:
    Visualiserar feedbackflöde mellan moduler i realtid.
    Visar feedback sources, routing, prioriteringar och trends.

Roll:
    - Realtidsvisualisering av feedback events
    - Nätverksdiagram av modul-kommunikation
    - Prioritetsfördelning och metrics
    - Trend-analys över tid

Används i Sprint: 3
"""

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from typing import Dict, Any, List


def create_feedback_flow_dashboard(introspection_panel) -> dash.Dash:
    """
    Skapar Dash-applikation för feedbackflöde.
    
    Args:
        introspection_panel: Referens till IntrospectionPanel för data
        
    Returns:
        Dash app instance
    """
    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    
    app.layout = html.Div([
        html.H1("NextGen AI Trader - Feedbackflöde", 
                style={'textAlign': 'center', 'color': '#2c3e50'}),
        
        # Uppdateringsintervall
        dcc.Interval(
            id='interval-component',
            interval=5*1000,  # Uppdatera var 5:e sekund
            n_intervals=0
        ),
        
        # Dashboard layout
        html.Div([
            # Vänster kolumn - Network graph
            html.Div([
                html.H3("Modul-kommunikation"),
                dcc.Graph(id='network-graph')
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            
            # Höger kolumn - Metrics
            html.Div([
                html.H3("Feedback Metrics"),
                html.Div(id='feedback-metrics'),
                html.Br(),
                html.H3("Prioritetsfördelning"),
                dcc.Graph(id='priority-distribution')
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '4%'})
        ]),
        
        html.Br(),
        
        # Trend-analys
        html.Div([
            html.H3("Feedback Trends över tid"),
            dcc.Graph(id='feedback-timeline')
        ]),
        
        html.Br(),
        
        # Senaste events
        html.Div([
            html.H3("Senaste Feedback Events"),
            html.Div(id='recent-events')
        ])
    ], style={'padding': '20px'})
    
    @app.callback(
        [Output('network-graph', 'figure'),
         Output('feedback-metrics', 'children'),
         Output('priority-distribution', 'figure'),
         Output('feedback-timeline', 'figure'),
         Output('recent-events', 'children')],
        [Input('interval-component', 'n_intervals')]
    )
    def update_dashboard(n):
        """Uppdaterar alla dashboard-komponenter."""
        dashboard_data = introspection_panel.render_dashboard()
        
        # Network graph
        network_fig = create_network_graph(dashboard_data.get('module_connections', []))
        
        # Metrics cards
        metrics = dashboard_data.get('feedback_metrics', {})
        metrics_html = create_metrics_cards(metrics)
        
        # Priority distribution
        priority_fig = create_priority_chart(metrics.get('by_priority', {}))
        
        # Timeline
        timeline_fig = create_timeline(dashboard_data.get('feedback_flow', []))
        
        # Recent events table
        events_html = create_events_table(dashboard_data.get('feedback_flow', [])[-10:])
        
        return network_fig, metrics_html, priority_fig, timeline_fig, events_html
    
    return app


def create_network_graph(connections: List[Dict[str, Any]]) -> go.Figure:
    """Skapar nätverksdiagram av modulkommunikation."""
    if not connections:
        return go.Figure().add_annotation(text="Ingen data ännu", showarrow=False)
    
    # Skapa nodes och edges
    nodes = set()
    for conn in connections:
        nodes.add(conn['source'])
        nodes.add(conn['target'])
    
    node_list = list(nodes)
    node_indices = {node: i for i, node in enumerate(node_list)}
    
    # Edge trace
    edge_x = []
    edge_y = []
    edge_text = []
    
    for conn in connections:
        x0, y0 = get_node_position(node_indices[conn['source']], len(node_list))
        x1, y1 = get_node_position(node_indices[conn['target']], len(node_list))
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_text.append(f"{conn['source']} → {conn['target']}: {conn['count']} events")
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='text',
        text=edge_text,
        mode='lines'
    )
    
    # Node trace
    node_x = []
    node_y = []
    node_text = []
    
    for node in node_list:
        x, y = get_node_position(node_indices[node], len(node_list))
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            size=20,
            color='#3498db',
            line=dict(width=2, color='white')
        )
    )
    
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=0),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=400
                    ))
    
    return fig


def get_node_position(index: int, total: int) -> tuple:
    """Beräknar position för node i cirkel."""
    import math
    angle = 2 * math.pi * index / total
    return (math.cos(angle), math.sin(angle))


def create_metrics_cards(metrics: Dict[str, Any]) -> html.Div:
    """Skapar metrics cards."""
    return html.Div([
        html.Div([
            html.H4(f"{metrics.get('total_events', 0)}", style={'margin': '0'}),
            html.P("Totalt Events", style={'margin': '0', 'color': '#7f8c8d'})
        ], style={'padding': '10px', 'backgroundColor': '#ecf0f1', 'borderRadius': '5px', 'marginBottom': '10px'}),
        
        html.Div([
            html.H4(f"{metrics.get('avg_per_minute', 0):.2f}", style={'margin': '0'}),
            html.P("Events/min", style={'margin': '0', 'color': '#7f8c8d'})
        ], style={'padding': '10px', 'backgroundColor': '#ecf0f1', 'borderRadius': '5px', 'marginBottom': '10px'}),
        
        html.Div([
            html.P("Events per källa:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
            *[html.P(f"{source}: {count}", style={'margin': '2px 0'}) 
              for source, count in metrics.get('by_source', {}).items()]
        ], style={'padding': '10px', 'backgroundColor': '#ecf0f1', 'borderRadius': '5px'})
    ])


def create_priority_chart(priority_data: Dict[str, int]) -> go.Figure:
    """Skapar prioritetsfördelning chart."""
    if not priority_data:
        return go.Figure().add_annotation(text="Ingen data", showarrow=False)
    
    fig = go.Figure(data=[go.Pie(
        labels=list(priority_data.keys()),
        values=list(priority_data.values()),
        marker=dict(colors=['#e74c3c', '#e67e22', '#f39c12', '#95a5a6'])
    )])
    
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
    return fig


def create_timeline(events: List[Dict[str, Any]]) -> go.Figure:
    """Skapar timeline över feedback events."""
    if not events:
        return go.Figure().add_annotation(text="Ingen data", showarrow=False)
    
    # Gruppera events per source över tid
    sources = {}
    for i, event in enumerate(events):
        source = event.get('source', 'unknown')
        if source not in sources:
            sources[source] = {'x': [], 'y': []}
        sources[source]['x'].append(i)
        sources[source]['y'].append(1)
    
    fig = go.Figure()
    for source, data in sources.items():
        fig.add_trace(go.Scatter(
            x=data['x'],
            y=data['y'],
            mode='markers',
            name=source,
            marker=dict(size=10)
        ))
    
    fig.update_layout(
        xaxis_title="Event Index",
        yaxis_title="",
        height=300,
        margin=dict(l=50, r=20, t=30, b=50),
        yaxis=dict(showticklabels=False)
    )
    
    return fig


def create_events_table(events: List[Dict[str, Any]]) -> html.Table:
    """Skapar tabell med senaste events."""
    if not events:
        return html.P("Inga events ännu")
    
    return html.Table([
        html.Thead(html.Tr([
            html.Th("Källa"),
            html.Th("Triggers"),
            html.Th("Prioritet"),
            html.Th("Timestamp")
        ])),
        html.Tbody([
            html.Tr([
                html.Td(event.get('source', 'unknown')),
                html.Td(', '.join(event.get('triggers', []))),
                html.Td(event.get('priority', 'medium')),
                html.Td(str(event.get('timestamp', event.get('route_timestamp', 'N/A')))[:10])
            ]) for event in events
        ])
    ], style={'width': '100%', 'borderCollapse': 'collapse', 'border': '1px solid #ddd'})


if __name__ == '__main__':
    # Demo mode - kör med mock data
    import sys
    import os
    import time
    
    # Lägg till projektroot till path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    from modules.introspection_panel import IntrospectionPanel
    from modules.message_bus import message_bus
    
    # Skapa panel
    panel = IntrospectionPanel(message_bus)
    
    # Populera med demo-data för att visa funktionalitet
    print("Genererar demo-data för dashboard...")
    
    # Simulera agent status updates
    for i in range(15):
        status = {
            'module': ['strategy_engine', 'risk_manager', 'decision_engine'][i % 3],
            'performance': 0.5 + (i * 0.03),
            'reward': 5.0 + i * 2.0,
            'timestamp': time.time() + i
        }
        message_bus.publish('agent_status', status)
    
    # Simulera feedback events
    sources = ['execution_engine', 'portfolio_manager', 'execution_engine', 'strategic_memory_engine']
    priorities = ['high', 'medium', 'critical', 'medium', 'high']
    for i in range(20):
        feedback = {
            'source': sources[i % len(sources)],
            'triggers': [['trade_result'], ['capital_change'], ['slippage'], ['decision_outcome']][i % 4],
            'priority': priorities[i % len(priorities)],
            'timestamp': time.time() + i * 10,
            'data': {
                'success': i % 3 != 0,
                'slippage': 0.002 + (i % 5) * 0.001,
                'performance': 0.7 - (i * 0.01) if i < 10 else 0.6
            }
        }
        message_bus.publish('feedback_event', feedback)
    
    # Simulera indicator data
    for i in range(5):
        indicators = {
            'symbol': ['AAPL', 'TSLA', 'MSFT'][i % 3],
            'technical': {
                'RSI': 50 + i * 5,
                'MACD': {'histogram': 0.3 - i * 0.1}
            }
        }
        message_bus.publish('indicator_data', indicators)
    
    print(f"✓ Genererade {len(panel.agent_status_history)} agent status updates")
    print(f"✓ Genererade {len(panel.feedback_events)} feedback events")
    print(f"✓ Genererade {len(panel.indicator_snapshots)} indicator snapshots")
    print()
    print("Dashboard startar på http://localhost:8050")
    print("Tryck Ctrl+C för att stoppa")
    print()
    
    app = create_feedback_flow_dashboard(panel)
    app.run(debug=True, port=8050)

