#!/usr/bin/env python
"""
Quick test to verify Sprint 6 modules are working correctly with the message bus.
"""

import time
from modules.message_bus import MessageBus
from modules.timespan_tracker import TimespanTracker
from modules.system_monitor import SystemMonitor
from modules.action_chain_engine import ActionChainEngine

def test_sprint6_integration():
    """Test that Sprint 6 modules work correctly together."""
    print("Testing Sprint 6 Integration...")
    print("="*60)
    
    # Initialize message bus and modules
    bus = MessageBus()
    tracker = TimespanTracker(bus)
    monitor = SystemMonitor(bus)
    chains = ActionChainEngine(bus)
    
    # Test 1: Decision tracking (should track decision_vote)
    print("\n1. Testing decision tracking (decision_vote)...")
    bus.publish('decision_vote', {'symbol': 'AAPL', 'action': 'BUY'})
    time.sleep(0.05)
    
    summary = tracker.get_timeline_summary()
    print(f"   Timeline events: {summary['total_events']}")
    print(f"   Decision events: {summary['decision_events']}")
    assert summary['decision_events'] > 0, "Decision events should be tracked"
    print("   ✅ Decision tracking works!")
    
    # Test 2: Final decision tracking
    print("\n2. Testing final decision tracking...")
    bus.publish('final_decision', {'symbol': 'MSFT', 'action': 'SELL'})
    time.sleep(0.05)
    
    summary = tracker.get_timeline_summary()
    print(f"   Timeline events: {summary['total_events']}")
    print(f"   Final decisions: {summary['final_decisions']}")
    assert summary['final_decisions'] > 0, "Final decisions should be tracked"
    print("   ✅ Final decision tracking works!")
    
    # Test 3: System monitor module tracking
    print("\n3. Testing system monitor module tracking...")
    
    # Simulate module activity
    bus.publish('decision_vote', {'test': 1})  # decision_engine
    bus.publish('final_decision', {'test': 2})  # consensus_engine
    bus.publish('execution_result', {'test': 3})  # execution_engine
    bus.publish('decision_proposal', {'test': 4})  # strategy_engine
    bus.publish('portfolio_status', {'portfolio_value': 1100})  # portfolio_manager
    bus.publish('agent_status', {'agent': 'test', 'performance': 0.8})  # rl_controller
    bus.publish('tuned_reward', {'value': 0.5})  # reward_tuner
    time.sleep(0.05)
    
    health = monitor.get_system_health()
    print(f"   Health score: {health['health_score']:.2f}")
    print(f"   Status: {health['status']}")
    print(f"   Active modules: {len(health['active_modules'])}/{health['total_modules']}")
    print(f"   Modules tracked: {', '.join(health['active_modules'])}")
    
    assert health['health_score'] > 0, "Health score should be > 0"
    assert len(health['active_modules']) >= 5, f"Should track at least 5 active modules, got {len(health['active_modules'])}"
    print("   ✅ System monitor tracking works!")
    
    # Test 4: Action chain execution
    print("\n4. Testing action chain execution...")
    chains.execute_chain('standard_trade', {'symbol': 'GOOGL'})
    time.sleep(0.05)
    
    chain_stats = chains.get_chain_statistics()
    print(f"   Total executions: {chain_stats['total_executions']}")
    print(f"   Templates available: {', '.join(chain_stats['available_templates'])}")
    assert chain_stats['total_executions'] > 0, "Should have executed at least 1 chain"
    print("   ✅ Action chain execution works!")
    
    # Test 5: Verify system monitor tracked chain execution
    print("\n5. Testing system monitor tracked chain execution...")
    health = monitor.get_system_health()
    assert 'action_chain_engine' in health['active_modules'], "action_chain_engine should be tracked"
    print(f"   action_chain_engine tracked: {health['active_modules'].count('action_chain_engine') > 0}")
    print("   ✅ System monitor tracks action chains!")
    
    print("\n" + "="*60)
    print("✅ All Sprint 6 integration tests passed!")
    print(f"\nFinal Summary:")
    print(f"  - Timeline events tracked: {tracker.get_timeline_summary()['total_events']}")
    print(f"  - System health score: {monitor.get_system_health()['health_score']:.2f}")
    print(f"  - Active modules: {len(monitor.get_system_health()['active_modules'])}")
    print(f"  - Chain executions: {chains.get_chain_statistics()['total_executions']}")
    print("\n✅ Sprint 6 modules are working correctly!")

if __name__ == '__main__':
    test_sprint6_integration()
