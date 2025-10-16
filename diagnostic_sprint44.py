#!/usr/bin/env python3
"""
Diagnostic script for Sprint 4.4 RewardTunerAgent
Run this to verify your environment is using the latest code.
"""

import sys
import os

print("="*80)
print("🔍 Sprint 4.4 RewardTunerAgent Diagnostic")
print("="*80)

# Check Python version
print(f"\n1. Python version: {sys.version}")

# Check current directory
print(f"\n2. Current directory: {os.getcwd()}")

# Check if modules can be imported
print("\n3. Testing module imports...")
try:
    from modules.message_bus import MessageBus
    print("   ✅ message_bus imported")
except Exception as e:
    print(f"   ❌ message_bus import failed: {e}")
    sys.exit(1)

try:
    from modules.portfolio_manager import PortfolioManager
    print("   ✅ portfolio_manager imported")
except Exception as e:
    print(f"   ❌ portfolio_manager import failed: {e}")
    sys.exit(1)

try:
    from modules.reward_tuner import RewardTunerAgent
    print("   ✅ reward_tuner imported")
except Exception as e:
    print(f"   ❌ reward_tuner import failed: {e}")
    sys.exit(1)

# Check if debug code is present
print("\n4. Checking for debug code...")
import inspect

# Check portfolio_manager
pm_source = inspect.getsource(PortfolioManager.calculate_and_publish_reward)
if "[PortfolioManager] Publishing base_reward" in pm_source:
    print("   ✅ Debug code found in portfolio_manager.calculate_and_publish_reward()")
else:
    print("   ❌ Debug code NOT found in portfolio_manager.calculate_and_publish_reward()")
    print("   ⚠️  You are running OLD CODE! Run: git pull origin copilot/add-rewardtuner-agent-integration")

if "publish('base_reward'" in pm_source:
    print("   ✅ portfolio_manager publishes to 'base_reward'")
else:
    print("   ❌ portfolio_manager does NOT publish to 'base_reward'")
    print("   ⚠️  You are running OLD CODE!")

if "publish('reward'" in pm_source:
    print("   ❌ portfolio_manager still publishes to old 'reward' topic")
    print("   ⚠️  You are running OLD CODE! The backward compat code should be removed.")
else:
    print("   ✅ portfolio_manager does NOT publish to old 'reward' (correct)")

# Check reward_tuner
rt_source = inspect.getsource(RewardTunerAgent.__init__)
if "[RewardTuner] Initialized" in rt_source:
    print("   ✅ Debug code found in reward_tuner.__init__()")
else:
    print("   ❌ Debug code NOT found in reward_tuner.__init__()")
    print("   ⚠️  You are running OLD CODE!")

# Test actual flow
print("\n5. Testing actual message flow...")
message_bus = MessageBus()

# Track messages
messages = {'base_reward': [], 'reward': [], 'tuned_reward': []}

def track(topic):
    def handler(data):
        messages[topic].append(data)
    return handler

message_bus.subscribe('base_reward', track('base_reward'))
message_bus.subscribe('reward', track('reward'))
message_bus.subscribe('tuned_reward', track('tuned_reward'))

print("   Creating reward_tuner (should see init message)...")
reward_tuner = RewardTunerAgent(message_bus=message_bus)

print("   Creating portfolio_manager...")
portfolio_manager = PortfolioManager(message_bus=message_bus, start_capital=1000.0)

print("   Simulating execution...")
execution = {
    'success': True,
    'symbol': 'TEST',
    'action': 'BUY',
    'quantity': 10,
    'executed_price': 100.0
}
message_bus.publish('execution_result', execution)

print(f"\n6. Results:")
print(f"   base_reward messages: {len(messages['base_reward'])}")
print(f"   tuned_reward messages: {len(messages['tuned_reward'])}")
print(f"   reward messages: {len(messages['reward'])}")

# Check reward_tuner state
metrics = reward_tuner.get_reward_metrics()
print(f"\n7. RewardTuner internal state:")
print(f"   base_reward_history: {len(metrics['base_reward_history'])}")
print(f"   tuned_reward_history: {len(metrics['tuned_reward_history'])}")

print("\n" + "="*80)
if len(metrics['base_reward_history']) > 0:
    print("✅ SUCCESS! Sprint 4.4 code is working correctly!")
    print("\nIf sim_test.py still shows 0 rewards, the issue is:")
    print("1. sim_test.py is importing from wrong location")
    print("2. Python is using cached .pyc files from __pycache__/")
    print("\nSolution:")
    print("   cd /home/runner/work/NextGen/NextGen")
    print("   find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null")
    print("   find . -name '*.pyc' -delete")
    print("   python sim_test.py")
else:
    print("❌ FAILURE! Code is not working as expected")
    print("\nYour environment is loading OLD CODE from somewhere.")
    print("\nSteps to fix:")
    print("1. Run: git status")
    print("2. Run: git pull origin copilot/add-rewardtuner-agent-integration")
    print("3. Delete ALL cache: find . -type d -name __pycache__ -exec rm -rf {} +")
    print("4. Restart Python completely")
print("="*80)
