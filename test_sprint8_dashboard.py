"""
Test script to verify analyzer_debug.py Sprint 8 graphs are functional.
"""

from modules.message_bus import MessageBus
from modules.dqn_controller import DQNController
from modules.gan_evolution_engine import GANEvolutionEngine
from modules.gnn_timespan_analyzer import GNNTimespanAnalyzer
import sys

def test_sprint8_dashboard_graphs():
    """Test that Sprint 8 dashboard graph methods work correctly."""
    print("="*80)
    print("Testing Sprint 8 Dashboard Graph Generation")
    print("="*80)
    
    # We'll import the dashboard class without starting it
    sys.path.insert(0, '/home/runner/work/NextGen/NextGen')
    from analyzer_debug import AnalyzerDebugDashboard
    
    print("\n✅ Creating dashboard instance...")
    dashboard = AnalyzerDebugDashboard()
    
    # Test DQN graph creation
    print("\n🎯 Testing DQN graphs...")
    try:
        dqn_metrics_fig = dashboard.create_dqn_metrics_graph()
        print(f"   ✓ DQN Metrics Graph: {len(dqn_metrics_fig.data)} traces")
        
        dqn_training_fig = dashboard.create_dqn_training_graph()
        print(f"   ✓ DQN Training Graph: {len(dqn_training_fig.data)} traces")
    except Exception as e:
        print(f"   ❌ DQN graphs failed: {e}")
        return False
    
    # Test GAN graph creation
    print("\n🧬 Testing GAN graphs...")
    try:
        gan_metrics_fig = dashboard.create_gan_metrics_graph()
        print(f"   ✓ GAN Metrics Graph: {len(gan_metrics_fig.data)} traces")
        
        gan_training_fig = dashboard.create_gan_training_graph()
        print(f"   ✓ GAN Training Graph: {len(gan_training_fig.data)} traces")
    except Exception as e:
        print(f"   ❌ GAN graphs failed: {e}")
        return False
    
    # Test GNN graph creation
    print("\n📊 Testing GNN graphs...")
    try:
        gnn_metrics_fig = dashboard.create_gnn_metrics_graph()
        print(f"   ✓ GNN Metrics Graph: {len(gnn_metrics_fig.data)} traces")
        
        gnn_patterns_fig = dashboard.create_gnn_patterns_graph()
        print(f"   ✓ GNN Patterns Graph: {len(gnn_patterns_fig.data)} traces")
    except Exception as e:
        print(f"   ❌ GNN graphs failed: {e}")
        return False
    
    print("\n" + "="*80)
    print("✅ All Sprint 8 dashboard graphs generated successfully!")
    print("="*80)
    
    # Display graph details
    print("\n📊 Graph Details:")
    print(f"   DQN Metrics: {dqn_metrics_fig.layout.title.text}")
    print(f"   DQN Training: {dqn_training_fig.layout.title.text}")
    print(f"   GAN Metrics: {gan_metrics_fig.layout.title.text}")
    print(f"   GAN Training: {gan_training_fig.layout.title.text}")
    print(f"   GNN Metrics: {gnn_metrics_fig.layout.title.text}")
    print(f"   GNN Patterns: {gnn_patterns_fig.layout.title.text}")
    
    return True

if __name__ == "__main__":
    try:
        success = test_sprint8_dashboard_graphs()
        
        print("\n📋 Test Summary:")
        if success:
            print("   ✅ All Sprint 8 dashboard graphs functional")
            print("\n🎉 SUCCESS: Sprint 8 RL Analysis tab ready!")
            print("\nRun 'python analyzer_debug.py' and navigate to 'RL Analysis' tab")
            print("to see DQN, GAN, and GNN metrics visualization.")
            exit(0)
        else:
            print("   ❌ Some graphs failed")
            exit(1)
    except Exception as e:
        print(f"\n❌ FAILURE: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
