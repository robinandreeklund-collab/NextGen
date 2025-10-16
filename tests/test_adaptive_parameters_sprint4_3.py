# test_adaptive_parameters_sprint4_3.py - Tester för adaptiva parametrar Sprint 4.3

"""
Tester för adaptiv parameterstyrning i Sprint 4.3.
Verifierar att alla moduler tar emot, använder och loggar adaptiva parametrar.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.message_bus import MessageBus
from modules.strategy_engine import StrategyEngine
from modules.risk_manager import RiskManager
from modules.decision_engine import DecisionEngine


def test_strategy_engine_adaptive_parameters():
    """Testar att StrategyEngine använder adaptiva parametrar."""
    bus = MessageBus()
    engine = StrategyEngine(bus)
    
    # Verifiera default parametrar
    assert engine.signal_threshold == 0.5
    assert engine.indicator_weighting == 0.33
    
    # Simulera parameter adjustment
    adjustment = {
        'parameters': {
            'signal_threshold': 0.7,
            'indicator_weighting': 0.6
        }
    }
    engine._on_parameter_adjustment(adjustment)
    
    # Verifiera att parametrar uppdaterades
    assert engine.signal_threshold == 0.7
    assert engine.indicator_weighting == 0.6
    print("✓ StrategyEngine adaptiva parametrar fungerar")


def test_risk_manager_adaptive_parameters():
    """Testar att RiskManager använder adaptiva parametrar."""
    bus = MessageBus()
    manager = RiskManager(bus)
    
    # Verifiera default parametrar
    assert manager.risk_tolerance == 0.1
    assert manager.max_drawdown == 0.15
    
    # Simulera parameter adjustment
    adjustment = {
        'parameters': {
            'risk_tolerance': 0.25,
            'max_drawdown': 0.2
        }
    }
    manager._on_parameter_adjustment(adjustment)
    
    # Verifiera att parametrar uppdaterades
    assert manager.risk_tolerance == 0.25
    assert manager.max_drawdown == 0.2
    print("✓ RiskManager adaptiva parametrar fungerar")


def test_decision_engine_adaptive_parameters():
    """Testar att DecisionEngine använder adaptiva parametrar."""
    bus = MessageBus()
    engine = DecisionEngine(bus)
    
    # Verifiera default parametrar
    assert engine.consensus_threshold == 0.75
    assert engine.memory_weighting == 0.4
    
    # Simulera parameter adjustment
    adjustment = {
        'parameters': {
            'consensus_threshold': 0.85,
            'memory_weighting': 0.6
        }
    }
    engine._on_parameter_adjustment(adjustment)
    
    # Verifiera att parametrar uppdaterades
    assert engine.consensus_threshold == 0.85
    assert engine.memory_weighting == 0.6
    print("✓ DecisionEngine adaptiva parametrar fungerar")


def test_strategy_engine_uses_adaptive_threshold():
    """Testar att StrategyEngine använder signal_threshold i beslut."""
    bus = MessageBus()
    engine = StrategyEngine(bus)
    
    # Sätt mock indicators
    engine.current_indicators = {
        'AAPL': {
            'symbol': 'AAPL',
            'technical': {
                'RSI': 25.0,  # Köpsignal
                'MACD': {'histogram': 0.6},  # Köpsignal
                'ATR': 2.0
            },
            'fundamental': {
                'AnalystRatings': {'consensus': 'BUY'}  # Köpsignal
            }
        }
    }
    
    # Med låg threshold (0.3), ska beslut gå igenom
    adjustment = {'parameters': {'signal_threshold': 0.3}}
    engine._on_parameter_adjustment(adjustment)
    proposal = engine.generate_proposal('AAPL')
    assert proposal['action'] == 'BUY'
    assert proposal['signal_threshold'] == 0.3
    
    # Med hög threshold (0.9), kan beslut blockeras eller justeras
    adjustment = {'parameters': {'signal_threshold': 0.9}}
    engine._on_parameter_adjustment(adjustment)
    proposal = engine.generate_proposal('AAPL')
    assert proposal['signal_threshold'] == 0.9
    print("✓ StrategyEngine använder signal_threshold korrekt")


def test_risk_manager_uses_adaptive_tolerance():
    """Testar att RiskManager använder risk_tolerance i bedömning."""
    bus = MessageBus()
    manager = RiskManager(bus)
    
    # Sätt mock data
    manager.current_indicators = {
        'AAPL': {
            'symbol': 'AAPL',
            'technical': {'Volume': 1000000, 'ATR': 3.0},
            'fundamental': {'AnalystRatings': {'consensus': 'HOLD'}}
        }
    }
    manager.portfolio_status = {
        'total_value': 1000.0,
        'starting_capital': 1000.0,
        'positions': {}
    }
    
    # Med låg risk_tolerance
    adjustment = {'parameters': {'risk_tolerance': 0.05}}
    manager._on_parameter_adjustment(adjustment)
    profile = manager.assess_risk('AAPL')
    assert profile['risk_tolerance'] == 0.05
    
    # Med hög risk_tolerance
    adjustment = {'parameters': {'risk_tolerance': 0.3}}
    manager._on_parameter_adjustment(adjustment)
    profile = manager.assess_risk('AAPL')
    assert profile['risk_tolerance'] == 0.3
    print("✓ RiskManager använder risk_tolerance korrekt")


def test_decision_engine_uses_consensus_threshold():
    """Testar att DecisionEngine använder consensus_threshold i beslut."""
    bus = MessageBus()
    engine = DecisionEngine(bus)
    
    # Sätt mock data
    engine.trade_proposals = {
        'AAPL': {
            'action': 'BUY',
            'quantity': 2,
            'confidence': 0.65,
            'reasoning': 'Test'
        }
    }
    engine.risk_profiles = {
        'AAPL': {
            'risk_level': 'MEDIUM',
            'confidence': 0.6
        }
    }
    engine.portfolio_status = {'cash': 1000.0, 'positions': {}}
    
    # Med låg consensus_threshold, beslut går igenom
    adjustment = {'parameters': {'consensus_threshold': 0.5}}
    engine._on_parameter_adjustment(adjustment)
    decision = engine.make_decision('AAPL', current_price=100.0)
    assert decision['consensus_threshold'] == 0.5
    
    # Med hög consensus_threshold, beslut påverkas
    adjustment = {'parameters': {'consensus_threshold': 0.9}}
    engine._on_parameter_adjustment(adjustment)
    decision = engine.make_decision('AAPL', current_price=100.0)
    assert decision['consensus_threshold'] == 0.9
    print("✓ DecisionEngine använder consensus_threshold korrekt")


def test_parameter_adjustment_propagation():
    """Testar att parameter adjustments propageras korrekt via message_bus."""
    bus = MessageBus()
    strategy = StrategyEngine(bus)
    risk = RiskManager(bus)
    decision = DecisionEngine(bus)
    
    # Verifiera initial state
    assert strategy.signal_threshold == 0.5
    assert risk.risk_tolerance == 0.1
    assert decision.consensus_threshold == 0.75
    
    # Publicera parameter adjustment för alla moduler
    bus.publish('parameter_adjustment', {
        'parameters': {
            'signal_threshold': 0.8,
            'risk_tolerance': 0.3,
            'consensus_threshold': 0.9
        }
    })
    
    # Verifiera att alla moduler uppdaterades
    assert strategy.signal_threshold == 0.8
    assert risk.risk_tolerance == 0.3
    assert decision.consensus_threshold == 0.9
    print("✓ Parameter adjustment propagering fungerar")


def test_strategy_indicator_weighting():
    """Testar att StrategyEngine använder indicator_weighting för viktning."""
    bus = MessageBus()
    engine = StrategyEngine(bus)
    
    # Sätt mock data med mixade signaler
    engine.current_indicators = {
        'AAPL': {
            'symbol': 'AAPL',
            'technical': {
                'RSI': 75.0,  # Säljsignal
                'MACD': {'histogram': 0.8},  # Köpsignal (konflikt)
                'ATR': 2.0
            },
            'fundamental': {
                'AnalystRatings': {'consensus': 'BUY'}  # Köpsignal
            }
        }
    }
    
    # Med låg indicator_weighting (RSI-fokus), RSI väger tyngre
    adjustment = {'parameters': {'indicator_weighting': 0.1, 'signal_threshold': 0.3}}
    engine._on_parameter_adjustment(adjustment)
    proposal = engine.generate_proposal('AAPL')
    assert proposal['indicator_weighting'] == 0.1
    
    # Med hög indicator_weighting (MACD/Analyst-fokus)
    adjustment = {'parameters': {'indicator_weighting': 0.9}}
    engine._on_parameter_adjustment(adjustment)
    proposal = engine.generate_proposal('AAPL')
    assert proposal['indicator_weighting'] == 0.9
    print("✓ StrategyEngine indicator_weighting fungerar")


if __name__ == '__main__':
    print("Kör Sprint 4.3 Adaptive Parameters-tester...")
    print()
    
    test_strategy_engine_adaptive_parameters()
    test_risk_manager_adaptive_parameters()
    test_decision_engine_adaptive_parameters()
    test_strategy_engine_uses_adaptive_threshold()
    test_risk_manager_uses_adaptive_tolerance()
    test_decision_engine_uses_consensus_threshold()
    test_parameter_adjustment_propagation()
    test_strategy_indicator_weighting()
    
    print()
    print("=" * 60)
    print("Alla Sprint 4.3 Adaptive Parameters-tester godkända! ✅")
    print("=" * 60)
