#!/usr/bin/env python3
"""
Quick verification script for RewardTunerAgent reward flow.
Tests that RewardTunerAgent receives base_reward from portfolio_manager.
"""

import sys
from modules.message_bus import MessageBus
from modules.portfolio_manager import PortfolioManager
from modules.reward_tuner import RewardTunerAgent

print("=" * 80)
print("🔍 VERIFYING REWARD FLOW")
print("=" * 80)

# Create message bus
message_bus = MessageBus()

# Test 1: Create reward_tuner AFTER portfolio_manager (WRONG ORDER - will fail)
print("\n❌ TEST 1: Creating reward_tuner AFTER portfolio_manager (should fail)")
print("-" * 80)

pm1 = PortfolioManager(message_bus=message_bus, start_capital=1000.0)
rt1 = RewardTunerAgent(message_bus=message_bus)

# Simulate portfolio value change to trigger reward
pm1._prev_portfolio_value = 1000.0
pm1.calculate_and_publish_reward()

# Wait a moment for message processing
import time
time.sleep(0.1)

print(f"\nResult: Reward tuner received {len(rt1.base_reward_history)} base_rewards")
if len(rt1.base_reward_history) == 0:
    print("✅ Expected failure confirmed: reward_tuner created too late!")
else:
    print("❌ UNEXPECTED: reward_tuner should not have received anything!")

# Test 2: Create reward_tuner BEFORE portfolio_manager (CORRECT ORDER - will work)
print("\n\n✅ TEST 2: Creating reward_tuner BEFORE portfolio_manager (should work)")
print("-" * 80)

# Create new message bus for clean test
message_bus2 = MessageBus()

# CORRECT ORDER: reward_tuner first, then portfolio_manager
rt2 = RewardTunerAgent(message_bus=message_bus2)
pm2 = PortfolioManager(message_bus=message_bus2, start_capital=1000.0)

# Simulate portfolio value change to trigger reward
pm2.previous_portfolio_value = 1000.0
pm2.calculate_and_publish_reward()

# Wait a moment for message processing
time.sleep(0.1)

print(f"\nResult: Reward tuner received {len(rt2.base_reward_history)} base_rewards")
print(f"Result: Reward tuner generated {len(rt2.tuned_reward_history)} tuned_rewards")

if len(rt2.base_reward_history) > 0:
    print(f"✅ SUCCESS: reward_tuner received base_reward: {rt2.base_reward_history[0]:.4f}")
    print(f"✅ SUCCESS: reward_tuner generated tuned_reward: {rt2.tuned_reward_history[0]:.4f}")
    print(f"✅ Transformation ratio: {rt2.transformation_ratios[0]:.4f}" if rt2.transformation_ratios else "")
else:
    print("❌ FAILURE: reward_tuner should have received base_reward!")

print("\n" + "=" * 80)
print("🎯 CONCLUSION")
print("=" * 80)
print("The fix is to create RewardTunerAgent BEFORE PortfolioManager in sim_test.py")
print("This ensures the reward_tuner subscription is registered before portfolio_manager")
print("starts publishing base_reward events.")
print("=" * 80)
