"""
Tests for the full-scale NextGen Dashboard.

Verifies that:
- Dashboard initializes correctly in both demo and live modes
- All panels can be created
- Modules are properly initialized
- Theme colors are correctly defined
"""

import sys
import os
import pytest

from start_dashboard import NextGenDashboard, THEME_COLORS


class TestNextGenDashboard:
    """Tests for NextGenDashboard."""
    
    def test_dashboard_initialization_demo_mode(self):
        """Test dashboard initialization in demo mode."""
        dashboard = NextGenDashboard(live_mode=False)
        
        assert dashboard.live_mode is False
        assert dashboard.symbols == ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        assert dashboard.running is True
        assert dashboard.iteration_count == 0
        assert dashboard.app is not None
    
    def test_dashboard_initialization_live_mode(self):
        """Test dashboard initialization in live mode."""
        dashboard = NextGenDashboard(live_mode=True)
        
        assert dashboard.live_mode is True
        assert dashboard.symbols == ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        assert dashboard.running is False
        assert dashboard.app is not None
    
    def test_modules_initialized(self):
        """Test that all required modules are initialized."""
        dashboard = NextGenDashboard(live_mode=False)
        
        # Sprint 1 modules
        assert hasattr(dashboard, 'portfolio_manager')
        assert hasattr(dashboard, 'strategy_engine')
        assert hasattr(dashboard, 'risk_manager')
        assert hasattr(dashboard, 'decision_engine')
        assert hasattr(dashboard, 'execution_engine')
        
        # Sprint 2-7 modules
        assert hasattr(dashboard, 'rl_controller')
        assert hasattr(dashboard, 'reward_tuner')
        assert hasattr(dashboard, 'vote_engine')
        assert hasattr(dashboard, 'consensus_engine')
        assert hasattr(dashboard, 'timespan_tracker')
        
        # Sprint 8 modules
        assert hasattr(dashboard, 'dqn_controller')
        assert hasattr(dashboard, 'gan_evolution')
        assert hasattr(dashboard, 'gnn_analyzer')
    
    def test_theme_colors_defined(self):
        """Test that theme colors are properly defined."""
        assert 'background' in THEME_COLORS
        assert 'surface' in THEME_COLORS
        assert 'primary' in THEME_COLORS
        assert 'success' in THEME_COLORS
        assert 'warning' in THEME_COLORS
        assert 'danger' in THEME_COLORS
        
        # Verify they are valid hex colors
        assert THEME_COLORS['background'].startswith('#')
        assert len(THEME_COLORS['background']) == 7
    
    def test_panel_creation_methods_exist(self):
        """Test that all panel creation methods exist."""
        dashboard = NextGenDashboard(live_mode=False)
        
        assert hasattr(dashboard, 'create_portfolio_panel')
        assert hasattr(dashboard, 'create_rl_analysis_panel')
        assert hasattr(dashboard, 'create_agent_evolution_panel')
        assert hasattr(dashboard, 'create_temporal_gnn_panel')
        assert hasattr(dashboard, 'create_feedback_panel')
        assert hasattr(dashboard, 'create_ci_tests_panel')
        assert hasattr(dashboard, 'create_conflict_panel')
        assert hasattr(dashboard, 'create_consensus_panel')
        assert hasattr(dashboard, 'create_adaptive_panel')
        assert hasattr(dashboard, 'create_market_panel')
    
    def test_portfolio_panel_creation(self):
        """Test portfolio panel can be created."""
        dashboard = NextGenDashboard(live_mode=False)
        panel = dashboard.create_portfolio_panel()
        
        assert panel is not None
        # Panel should be a Div component
        assert hasattr(panel, 'children')
    
    def test_rl_analysis_panel_creation(self):
        """Test RL analysis panel can be created."""
        dashboard = NextGenDashboard(live_mode=False)
        panel = dashboard.create_rl_analysis_panel()
        
        assert panel is not None
        assert hasattr(panel, 'children')
    
    def test_ci_tests_panel_creation(self):
        """Test CI tests panel can be created."""
        dashboard = NextGenDashboard(live_mode=False)
        panel = dashboard.create_ci_tests_panel()
        
        assert panel is not None
        assert hasattr(panel, 'children')
    
    def test_data_history_initialization(self):
        """Test that data history structures are initialized."""
        dashboard = NextGenDashboard(live_mode=False)
        
        assert isinstance(dashboard.price_history, dict)
        assert isinstance(dashboard.reward_history, dict)
        assert isinstance(dashboard.agent_metrics_history, list)
        assert isinstance(dashboard.gan_metrics_history, dict)
        assert isinstance(dashboard.gnn_pattern_history, list)
        assert isinstance(dashboard.conflict_history, list)
        assert isinstance(dashboard.decision_history, list)
    
    def test_base_prices_defined(self):
        """Test that base prices are defined for all symbols."""
        dashboard = NextGenDashboard(live_mode=False)
        
        for symbol in dashboard.symbols:
            assert symbol in dashboard.base_prices
            assert dashboard.base_prices[symbol] > 0
    
    def test_simulation_control_methods(self):
        """Test simulation control methods exist."""
        dashboard = NextGenDashboard(live_mode=False)
        
        assert hasattr(dashboard, 'start_simulation')
        assert hasattr(dashboard, 'stop_simulation')
        assert hasattr(dashboard, 'simulation_loop')
    
    def test_chart_layout_method(self):
        """Test chart layout method returns correct structure."""
        dashboard = NextGenDashboard(live_mode=False)
        layout = dashboard.get_chart_layout("Test Chart")
        
        assert 'plot_bgcolor' in layout
        assert 'paper_bgcolor' in layout
        assert 'title' in layout
        assert layout['title']['text'] == "Test Chart"
        assert layout['plot_bgcolor'] == THEME_COLORS['surface']
    
    def test_metric_card_creation(self):
        """Test metric card creation."""
        dashboard = NextGenDashboard(live_mode=False)
        card = dashboard.create_metric_card("Test", "100", THEME_COLORS['primary'])
        
        assert card is not None
        assert hasattr(card, 'children')
    
    def test_button_style_method(self):
        """Test button style method returns correct structure."""
        dashboard = NextGenDashboard(live_mode=False)
        style = dashboard.get_button_style(THEME_COLORS['success'])
        
        assert 'backgroundColor' in style
        assert style['backgroundColor'] == THEME_COLORS['success']
        assert 'cursor' in style
        assert style['cursor'] == 'pointer'
    
    def test_tab_styles(self):
        """Test tab style methods."""
        dashboard = NextGenDashboard(live_mode=False)
        
        tab_style = dashboard.get_tab_style()
        selected_style = dashboard.get_tab_selected_style()
        
        assert 'backgroundColor' in tab_style
        assert 'color' in tab_style
        assert 'backgroundColor' in selected_style
        assert 'color' in selected_style
        assert selected_style['color'] == THEME_COLORS['primary']


def test_theme_color_scheme():
    """Test the overall theme color scheme."""
    # Dark theme check
    assert THEME_COLORS['background'] == '#0a0e1a'
    assert THEME_COLORS['surface'] == '#141b2d'
    
    # Primary colors check
    assert THEME_COLORS['primary'] == '#4dabf7'
    assert THEME_COLORS['secondary'] == '#845ef7'
    
    # Status colors check
    assert THEME_COLORS['success'] == '#51cf66'
    assert THEME_COLORS['warning'] == '#ffd43b'
    assert THEME_COLORS['danger'] == '#ff6b6b'


def test_import_start_demo():
    """Test that start_demo.py can be imported."""
    import start_demo
    assert hasattr(start_demo, 'main')


def test_import_start_live():
    """Test that start_live.py can be imported."""
    import start_live
    assert hasattr(start_live, 'main')
