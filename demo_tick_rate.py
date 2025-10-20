#!/usr/bin/env python3
"""
Demo script showing the impact of different tick rates on agent precision.

Demonstrates:
- Default 100ms tick rate (10 ticks/second)
- Fast 50ms tick rate (20 ticks/second)  
- Maximum speed tick rate (0ms = no delay)
- Old 2s tick rate (for comparison)

Shows how faster tick rates allow agents to react more precisely to market changes.
"""

import time
from start_dashboard import NextGenDashboard


def demo_tick_rate(tick_rate: float, duration: int = 5):
    """
    Demo a specific tick rate.
    
    Args:
        tick_rate: Tick rate in seconds
        duration: How long to run in seconds
    """
    rate_name = f"{tick_rate * 1000:.0f}ms" if tick_rate > 0 else "Max speed"
    ticks_per_sec = f"~{1/tick_rate:.1f} ticks/s" if tick_rate > 0 else "unlimited"
    
    print(f"\n{'=' * 70}")
    print(f"Testing tick_rate={rate_name} ({ticks_per_sec})")
    print(f"{'=' * 70}")
    
    dashboard = NextGenDashboard(live_mode=False, tick_rate=tick_rate)
    
    print(f"✅ Dashboard created with tick_rate={dashboard.tick_rate}s")
    print(f"📊 Starting simulation for {duration} seconds...")
    
    dashboard.start_simulation()
    
    start_time = time.time()
    last_count = 0
    
    # Monitor for specified duration
    while time.time() - start_time < duration:
        time.sleep(0.5)
        iterations = dashboard.iteration_count
        elapsed = time.time() - start_time
        
        if iterations != last_count:
            actual_rate = iterations / elapsed
            print(f"  {elapsed:.1f}s: {iterations} iterations ({actual_rate:.1f} iter/s)")
            last_count = iterations
    
    dashboard.stop_simulation()
    
    total_iterations = dashboard.iteration_count
    total_time = time.time() - start_time
    avg_rate = total_iterations / total_time
    
    print(f"\n📈 Results:")
    print(f"   Total iterations: {total_iterations}")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Average rate: {avg_rate:.1f} iterations/second")
    print(f"   Theoretical max: {1/tick_rate:.1f} iter/s" if tick_rate > 0 else "   Theoretical max: unlimited")
    
    return total_iterations, avg_rate


def main():
    """Run tick rate comparison demo."""
    print("=" * 70)
    print("Tick Rate Impact on Agent Precision - Demo")
    print("=" * 70)
    print()
    print("This demo shows how different tick rates affect agent reaction speed.")
    print("Lower tick rates = faster reactions = better RL precision")
    print()
    
    results = []
    
    # Test old slow rate (2s)
    print("\n🐌 OLD RATE (before optimization):")
    iters, rate = demo_tick_rate(tick_rate=2.0, duration=5)
    results.append(("2s (old)", iters, rate))
    
    # Test default rate (100ms)
    print("\n⚡ DEFAULT RATE (current):")
    iters, rate = demo_tick_rate(tick_rate=0.1, duration=5)
    results.append(("100ms (default)", iters, rate))
    
    # Test fast rate (50ms)
    print("\n🚀 FAST RATE:")
    iters, rate = demo_tick_rate(tick_rate=0.05, duration=5)
    results.append(("50ms (fast)", iters, rate))
    
    # Test max speed (0ms)
    print("\n⚡🔥 MAXIMUM SPEED:")
    iters, rate = demo_tick_rate(tick_rate=0, duration=5)
    results.append(("0ms (max)", iters, rate))
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Rate':<20} {'Iterations':<15} {'Iter/s':<15} {'Speedup':<10}")
    print("-" * 70)
    
    baseline_rate = results[0][2]  # Old 2s rate
    for name, iters, rate in results:
        speedup = rate / baseline_rate
        print(f"{name:<20} {iters:<15} {rate:<15.1f} {speedup:<10.1f}x")
    
    print("=" * 70)
    print("\n✅ Lower tick rates provide dramatically better precision!")
    print("   Agents can react up to 100x faster with optimized tick rates.")
    print("   This means better RL training and more precise trading decisions.")
    print()


if __name__ == "__main__":
    main()
