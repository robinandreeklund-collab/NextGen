# test_strategy_engine.py - Tester för strategimotor

"""
Tester för StrategyEngine med RL-integration och Sprint 2 indikatorer.
Verifierar att tradeförslag genereras korrekt med MACD, ATR, och Analyst Ratings.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.message_bus import MessageBus
from modules.strategy_engine import StrategyEngine


def test_strategy_engine_initialization():
    """Testar att StrategyEngine initialiseras korrekt."""
    bus = MessageBus()
    strategy = StrategyEngine(bus)
    
    assert strategy.message_bus is not None
    assert strategy.rl_enabled == False
    assert len(strategy.current_indicators) == 0
    print("✓ StrategyEngine initialisering fungerar")


def test_strategy_engine_rsi_signals():
    """Testar att StrategyEngine genererar korrekta RSI-signaler."""
    bus = MessageBus()
    strategy = StrategyEngine(bus)
    
    # Simulera översåld RSI (BUY-signal)
    indicators = {
        'symbol': 'TEST',
        'technical': {
            'RSI': 25.0,
            'MACD': {'histogram': -0.3},
            'ATR': 2.0
        },
        'fundamental': {
            'AnalystRatings': {'consensus': 'HOLD'}
        }
    }
    strategy.current_indicators['TEST'] = indicators
    
    proposal = strategy.generate_proposal('TEST')
    
    assert proposal['action'] == 'BUY'
    assert proposal['quantity'] > 0
    assert 'RSI' in proposal['reasoning']
    print("✓ StrategyEngine RSI-signaler fungerar")


def test_strategy_engine_macd_signals():
    """Testar att StrategyEngine använder MACD korrekt."""
    bus = MessageBus()
    strategy = StrategyEngine(bus)
    
    # Simulera stark MACD köpsignal med översåld RSI för tillräcklig signalstyrka
    indicators = {
        'symbol': 'TEST',
        'technical': {
            'RSI': 28.0,  # Översåld för att ge tillräcklig signalstyrka tillsammans med MACD
            'MACD': {'histogram': 0.8},  # Stark positiv MACD
            'ATR': 2.0
        },
        'fundamental': {
            'AnalystRatings': {'consensus': 'BUY'}
        }
    }
    strategy.current_indicators['TEST'] = indicators
    
    proposal = strategy.generate_proposal('TEST')
    
    assert proposal['action'] == 'BUY'
    assert 'MACD' in proposal['reasoning']
    print("✓ StrategyEngine MACD-signaler fungerar")


def test_strategy_engine_atr_adjustment():
    """Testar att StrategyEngine justerar position baserat på ATR."""
    bus = MessageBus()
    strategy = StrategyEngine(bus)
    
    # Simulera hög volatilitet (ATR > 5.0)
    indicators = {
        'symbol': 'TEST',
        'technical': {
            'RSI': 25.0,
            'MACD': {'histogram': -0.3},
            'ATR': 8.5  # Hög volatilitet
        },
        'fundamental': {
            'AnalystRatings': {'consensus': 'HOLD'}
        }
    }
    strategy.current_indicators['TEST'] = indicators
    
    proposal = strategy.generate_proposal('TEST')
    
    assert proposal['action'] == 'BUY'
    assert proposal['quantity'] < 10  # Reducerad från 10
    assert 'volatilitet' in proposal['reasoning'].lower()
    print("✓ StrategyEngine ATR-justering fungerar")


def test_strategy_engine_analyst_ratings():
    """Testar att StrategyEngine använder Analyst Ratings."""
    bus = MessageBus()
    strategy = StrategyEngine(bus)
    
    # Simulera med stark analytisk konsensus OCH starkt översåld RSI för stark BUY-signal
    indicators = {
        'symbol': 'TEST',
        'technical': {
            'RSI': 22.0,  # Starkt översåld för stark signal
            'MACD': {'histogram': 0.2},
            'ATR': 2.0
        },
        'fundamental': {
            'AnalystRatings': {'consensus': 'STRONG_BUY'}
        }
    }
    strategy.current_indicators['TEST'] = indicators
    
    proposal = strategy.generate_proposal('TEST')
    
    assert proposal['action'] == 'BUY'
    # Använd case-insensitive check för båda
    reasoning_lower = proposal['reasoning'].lower()
    assert 'analyst' in reasoning_lower or 'consensus' in reasoning_lower
    print("✓ StrategyEngine Analyst Ratings fungerar")


def test_strategy_engine_rl_integration():
    """Testar att StrategyEngine kan ta emot RL-updates."""
    bus = MessageBus()
    strategy = StrategyEngine(bus)
    
    # Simulera RL agent update
    update = {
        'module': 'strategy_engine',
        'policy_updated': True,
        'metrics': {'average_reward': 0.5}
    }
    strategy._on_agent_update(update)
    
    assert strategy.rl_enabled == True
    assert strategy.agent_performance == 0.5
    print("✓ StrategyEngine RL-integration fungerar")


if __name__ == '__main__':
    print("Kör StrategyEngine-tester...")
    print()
    
    test_strategy_engine_initialization()
    test_strategy_engine_rsi_signals()
    test_strategy_engine_macd_signals()
    test_strategy_engine_atr_adjustment()
    test_strategy_engine_analyst_ratings()
    test_strategy_engine_rl_integration()
    
    print()
    print("=" * 50)
    print("Alla StrategyEngine-tester godkända! ✅")
    print("=" * 50)

