"""
Test script to verify Sprint 8 modules are visible and functional.
"""

from modules.message_bus import MessageBus
from modules.system_monitor import SystemMonitor
from modules.dqn_controller import DQNController
from modules.gan_evolution_engine import GANEvolutionEngine
from modules.gnn_timespan_analyzer import GNNTimespanAnalyzer
import time

def test_sprint8_visibility():
    """Test that Sprint 8 modules are tracked by system monitor."""
    print("="*80)
    print("Testing Sprint 8 Module Visibility")
    print("="*80)
    
    # Initialize message bus
    message_bus = MessageBus()
    
    # Initialize system monitor
    system_monitor = SystemMonitor(message_bus)
    
    # Initialize Sprint 8 modules
    print("\n✅ Initializing Sprint 8 modules...")
    dqn = DQNController(message_bus, state_dim=10, action_dim=3)
    print(f"   DQN Controller: {dqn.state_dim} states, {dqn.action_dim} actions")
    
    gan = GANEvolutionEngine(message_bus, latent_dim=64, param_dim=16)
    print(f"   GAN Evolution: latent_dim={gan.latent_dim}, param_dim={gan.param_dim}")
    
    gnn = GNNTimespanAnalyzer(message_bus, input_dim=32, temporal_window=20)
    print(f"   GNN Analyzer: input_dim={gnn.input_dim}, temporal_window={gnn.temporal_window}")
    
    # Wait a moment for initialization
    time.sleep(0.5)
    
    # Check initial system health
    print("\n📊 Initial System Health:")
    health = system_monitor.get_system_health()
    print(f"   Health Score: {health['health_score']*100:.1f}%")
    print(f"   Status: {health['status']}")
    print(f"   Active Modules: {len(health['active_modules'])}/{health['total_modules']}")
    print(f"   Modules tracked: {', '.join(sorted(health['active_modules'][:10]))}")
    
    # Trigger some activity from Sprint 8 modules
    print("\n🔄 Triggering Sprint 8 module activity...")
    
    # DQN publishes metrics
    dqn_metrics = dqn.get_metrics()
    message_bus.publish('dqn_metrics', dqn_metrics)
    print(f"   ✓ DQN metrics published: epsilon={dqn_metrics['epsilon']:.4f}")
    
    # GAN publishes metrics
    gan_metrics = gan.get_metrics()
    message_bus.publish('gan_metrics', gan_metrics)
    print(f"   ✓ GAN metrics published: g_loss={gan_metrics['g_loss']:.4f}")
    
    # GNN publishes analysis
    gnn_insights = gnn.get_temporal_insights()
    message_bus.publish('gnn_analysis_response', {'insights': gnn_insights})
    print(f"   ✓ GNN analysis published")
    
    # Wait for messages to be processed
    time.sleep(0.2)
    
    # Check updated system health
    print("\n📊 Updated System Health:")
    health = system_monitor.get_system_health()
    print(f"   Health Score: {health['health_score']*100:.1f}%")
    print(f"   Status: {health['status']}")
    print(f"   Active Modules: {len(health['active_modules'])}/{health['total_modules']}")
    
    # Show Sprint 8 module status
    print("\n🆕 Sprint 8 Module Status:")
    module_status = system_monitor.get_module_status()
    
    sprint8_modules = ['dqn_controller', 'gan_evolution', 'gnn_analyzer']
    for module_name in sprint8_modules:
        if module_name in module_status:
            status = module_status[module_name]
            age = time.time() - status.get('last_update', 0)
            updates = status.get('update_count', 0)
            state = "🟢 ACTIVE" if age < 60 else "🔴 STALE"
            print(f"   {state} {module_name:20} - {updates} updates, last: {age:.1f}s ago")
        else:
            print(f"   ⚠️  {module_name:20} - NOT TRACKED")
    
    print("\n" + "="*80)
    print("✅ Sprint 8 modules are visible and tracked!")
    print("="*80)
    
    # Return summary
    return {
        'health_score': health['health_score'],
        'sprint8_tracked': all(m in module_status for m in sprint8_modules),
        'all_active': all(
            time.time() - module_status[m].get('last_update', 0) < 60 
            for m in sprint8_modules if m in module_status
        )
    }

if __name__ == "__main__":
    result = test_sprint8_visibility()
    
    print("\n📋 Test Summary:")
    print(f"   Health Score: {result['health_score']*100:.1f}%")
    print(f"   Sprint 8 Tracked: {'✅' if result['sprint8_tracked'] else '❌'}")
    print(f"   All Active: {'✅' if result['all_active'] else '❌'}")
    
    if result['sprint8_tracked'] and result['all_active']:
        print("\n🎉 SUCCESS: Sprint 8 modules are fully integrated!")
        exit(0)
    else:
        print("\n❌ FAILURE: Sprint 8 modules not properly tracked")
        exit(1)
