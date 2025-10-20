"""
finnhub_orchestrator.py - Central Orchestrator for Finnhub Data Flow

Description:
    Central orchestration module for managing REST/WebSocket data flow from Finnhub.
    Coordinates symbol rotation, indicator synthesis, stream management, and RL-driven
    optimization for market data ingestion and distribution.

Features:
    - RL-driven symbol prioritization and rotation
    - Adaptive batching and resource allocation
    - Stream replay for simulation and testing
    - Rate limiting and failover mechanisms
    - Comprehensive audit logging
    - Plug-n-play architecture with dynamic configuration

Submodules:
    - indicator_synth_engine: Synthesizes indicator combinations
    - symbol_rotation_engine: Manages symbol rotation
    - rotation_strategy_engine: Determines rotation strategies
    - stream_strategy_agent: RL agent for stream optimization
    - stream_replay_engine: Replays historical data
    - stream_ontology_mapper: Normalizes data formats
    - rl_engine_integration: RL controller integration

Integration:
    Publishes to message_bus:
        - orchestrator_status: Health and metrics
        - symbol_rotation: Rotation events
        - stream_metrics: Performance data
        - rl_scores: Symbol priorities
        
    Subscribes to:
        - rl_feedback: Feedback from RL controllers
        - market_conditions: Market state changes
        - module_requests: Data requests from modules
"""

import time
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
import threading
import random
import yaml
import os

# Import submodules
try:
    from modules.indicator_synth_engine import IndicatorSynthEngine
    from modules.symbol_rotation_engine import SymbolRotationEngine
    from modules.rotation_strategy_engine import RotationStrategyEngine
    from modules.stream_strategy_agent import StreamStrategyAgent
    from modules.stream_replay_engine import StreamReplayEngine
    from modules.stream_ontology_mapper import StreamOntologyMapper
except ImportError:
    from .indicator_synth_engine import IndicatorSynthEngine
    from .symbol_rotation_engine import SymbolRotationEngine
    from .rotation_strategy_engine import RotationStrategyEngine
    from .stream_strategy_agent import StreamStrategyAgent
    from .stream_replay_engine import StreamReplayEngine
    from .stream_ontology_mapper import StreamOntologyMapper


class FinnhubOrchestrator:
    """
    Central orchestrator for Finnhub data flow management.
    
    Coordinates all aspects of data ingestion, symbol rotation, indicator synthesis,
    and RL-driven optimization.
    """
    
    def __init__(
        self,
        api_key: str,
        message_bus,
        config: Optional[Dict[str, Any]] = None,
        live_mode: bool = False
    ):
        """
        Initialize the Finnhub orchestrator.
        
        Args:
            api_key: Finnhub API key
            message_bus: Reference to message bus for pub/sub
            config: Optional configuration dictionary
            live_mode: If True, connects to live data; else uses simulation
        """
        self.api_key = api_key
        self.message_bus = message_bus
        self.live_mode = live_mode
        
        # Load configuration
        self.config = self._load_default_config()
        if config:
            self.config.update(config)
        
        # State tracking
        self.active_symbols: List[str] = []
        self.symbol_priorities: Dict[str, float] = {}
        self.stream_metrics: Dict[str, Dict[str, Any]] = {}
        self.rotation_history: deque = deque(maxlen=100)
        self.is_running = False
        self.last_rotation_time = None
        
        # Portfolio protection: track symbols we must keep (stocks in portfolio)
        self.protected_symbols: set = set()  # Symbols that can't be dropped (in portfolio)
        
        # Detailed metrics tracking
        self.detailed_metrics = {
            'active_subscriptions': 0,
            'historical_symbols_used': set(),
            'websocket_usage_pct': 0.0,
            'total_data_points': 0,
            'submodule_details': {}
        }
        
        # Rate limiting
        self.rate_limiter = {
            'requests': deque(maxlen=self.config['rate_limit']['requests_per_second']),
            'last_reset': time.time()
        }
        
        # Setup logging first
        self._setup_logging()
        
        # Initialize submodules
        self._initialize_submodules()
        
        # Subscribe to message bus topics
        self._setup_subscriptions()
        
        self.logger.info(f"FinnhubOrchestrator initialized (live_mode={live_mode})")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        # Try to load NASDAQ 100 symbols
        default_symbols = self._load_nasdaq_symbols()
        if not default_symbols:
            default_symbols = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "NVDA", "META"]
        
        return {
            'default_symbols': default_symbols,
            'rotation_interval': 300,  # seconds
            'max_concurrent_streams': 50,  # Finnhub WebSocket limit
            'websocket_limit': 50,  # Hard limit for WebSocket connections
            'test_slot_count': 1,  # Reserved slots for testing new symbols
            'rest_batch_size': 12,  # Batch size for REST API calls (10-15)
            'buffer_size': 1000,
            'priority_update_interval': 60,
            'adaptive_params': {
                'rotation_threshold': 0.5,
                'batch_size': 12,  # 10-15 symbols per REST call
                'priority_weight': 0.7,
                'replay_speed': 1.0
            },
            'rate_limit': {
                'requests_per_second': 10,
                'burst_size': 20,
                'backoff_strategy': 'exponential'
            },
            'failover': {
                'retry_attempts': 3,
                'retry_delay': 2,
                'fallback_mode': 'cached_data'
            },
            'audit_logging': {
                'enabled': True,
                'log_level': 'INFO',
                'log_file': 'logs/orchestrator_audit.json'
            }
        }
    
    def _load_nasdaq_symbols(self) -> List[str]:
        """Load NASDAQ 100 symbols from YAML file."""
        try:
            # Try multiple paths
            paths = [
                'config/nasdaq_100_symbols.yaml',
                '../config/nasdaq_100_symbols.yaml',
                os.path.join(os.path.dirname(__file__), '..', 'config', 'nasdaq_100_symbols.yaml')
            ]
            
            for path in paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        data = yaml.safe_load(f)
                        if data and 'symbols' in data:
                            return data['symbols']
            
            return []
        except Exception as e:
            print(f"Warning: Could not load NASDAQ symbols: {e}")
            return []
    
    def _initialize_submodules(self):
        """Initialize all submodules."""
        self.indicator_synth = IndicatorSynthEngine(self.message_bus)
        self.symbol_rotation = SymbolRotationEngine(
            self.message_bus,
            self.config['rotation_interval']
        )
        self.rotation_strategy = RotationStrategyEngine(self.message_bus)
        self.stream_strategy = StreamStrategyAgent(self.message_bus)
        self.replay_engine = StreamReplayEngine(self.message_bus)
        self.ontology_mapper = StreamOntologyMapper(self.message_bus)
        
        self.logger.info("All submodules initialized")
    
    def _setup_logging(self):
        """Setup audit logging."""
        self.logger = logging.getLogger('FinnhubOrchestrator')
        self.logger.setLevel(getattr(logging, self.config['audit_logging']['log_level']))
        
        # File handler for audit log
        if self.config['audit_logging']['enabled']:
            log_file = self.config['audit_logging']['log_file']
            # Ensure the log directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def _setup_subscriptions(self):
        """Subscribe to message bus topics."""
        self.message_bus.subscribe('rl_feedback', self._handle_rl_feedback)
        self.message_bus.subscribe('market_conditions', self._handle_market_conditions)
        self.message_bus.subscribe('portfolio_status', self._handle_portfolio_status)
        self.message_bus.subscribe('module_requests', self._handle_module_request)
    
    def start(self):
        """Start the orchestrator."""
        if self.is_running:
            self.logger.warning("Orchestrator already running")
            return
        
        self.is_running = True
        self.last_rotation_time = time.time()
        
        # Initialize with limited symbols based on WebSocket limit
        # Reserve slots: (websocket_limit - test_slot_count) for active trading
        available_slots = self.config['websocket_limit'] - self.config['test_slot_count']
        all_symbols = self.config['default_symbols']
        
        # Start with top priority symbols (or first N if no priorities yet)
        if len(all_symbols) > available_slots:
            self.active_symbols = all_symbols[:available_slots]
        else:
            self.active_symbols = all_symbols.copy()
        
        self._initialize_symbol_priorities()
        
        self.logger.info(f"Starting with {len(self.active_symbols)} active symbols (limit: {self.config['websocket_limit']})")
        
        # Start orchestration loop in background thread
        self.orchestration_thread = threading.Thread(
            target=self._orchestration_loop,
            daemon=True
        )
        self.orchestration_thread.start()
        
        # Publish initial status
        self._publish_status()
        
        self.logger.info("Orchestrator started")
    
    def stop(self):
        """Stop the orchestrator."""
        self.is_running = False
        if hasattr(self, 'orchestration_thread'):
            self.orchestration_thread.join(timeout=5)
        
        self.logger.info("Orchestrator stopped")
    
    def _initialize_symbol_priorities(self):
        """Initialize priorities for active symbols."""
        for symbol in self.active_symbols:
            self.symbol_priorities[symbol] = 1.0 / len(self.active_symbols)
        
        self.logger.info(f"Initialized priorities for {len(self.active_symbols)} symbols")
    
    def _orchestration_loop(self):
        """Main orchestration loop."""
        while self.is_running:
            try:
                # Check if rotation is needed
                if self._should_rotate():
                    self._perform_rotation()
                
                # Update RL-driven priorities
                self._update_priorities()
                
                # Synthesize indicators
                self._synthesize_indicators()
                
                # Update stream strategies
                self._update_stream_strategies()
                
                # Collect and publish metrics
                self._collect_metrics()
                
                # Publish status
                self._publish_status()
                
                # Sleep for a short interval
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in orchestration loop: {e}", exc_info=True)
                time.sleep(5)  # Back off on error
    
    def _should_rotate(self) -> bool:
        """Determine if symbol rotation should occur."""
        if not self.last_rotation_time:
            return True
        
        time_since_rotation = time.time() - self.last_rotation_time
        rotation_interval = self.config['rotation_interval']
        
        # Time-based trigger
        if time_since_rotation >= rotation_interval:
            return True
        
        # RL-recommendation trigger (check strategy engine)
        strategy = self.rotation_strategy.get_current_strategy()
        if strategy.get('recommend_rotation', False):
            return True
        
        return False
    
    def _perform_rotation(self):
        """Perform symbol rotation based on priorities and strategy."""
        self.logger.info("Performing symbol rotation")
        
        # Get rotation strategy
        strategy = self.rotation_strategy.compute_rotation_strategy(
            self.symbol_priorities,
            self.stream_metrics,
            self.active_symbols
        )
        
        # Calculate max symbols (WebSocket limit - test slots)
        websocket_limit = self.config.get('websocket_limit', 50)
        test_slots = self.config.get('test_slot_count', 1)
        max_active_symbols = websocket_limit - test_slots
        
        # Execute rotation with proper limit
        new_symbols = self.symbol_rotation.rotate_symbols(
            current_symbols=self.active_symbols,
            priorities=self.symbol_priorities,
            strategy=strategy,
            max_symbols=max_active_symbols
        )
        
        # CRITICAL: Ensure all protected symbols (portfolio holdings) are included
        # We must maintain WebSocket connections for stocks we own
        for protected_symbol in self.protected_symbols:
            if protected_symbol not in new_symbols:
                self.logger.warning(f"Rotation tried to drop {protected_symbol} which is in portfolio - forcing inclusion")
                new_symbols.append(protected_symbol)
        
        # If we exceeded limit due to protected symbols, remove lowest priority non-protected symbols
        if len(new_symbols) > max_active_symbols:
            # Split into protected and non-protected
            protected_in_new = [s for s in new_symbols if s in self.protected_symbols]
            non_protected_in_new = [s for s in new_symbols if s not in self.protected_symbols]
            
            # Keep all protected, trim non-protected
            slots_remaining = max_active_symbols - len(protected_in_new)
            if slots_remaining > 0:
                new_symbols = protected_in_new + non_protected_in_new[:slots_remaining]
            else:
                # All slots taken by protected symbols (edge case)
                new_symbols = protected_in_new[:max_active_symbols]
                if len(protected_in_new) > max_active_symbols:
                    self.logger.error(f"Portfolio has {len(protected_in_new)} positions but only {max_active_symbols} WebSocket slots!")
        
        # Ensure we don't exceed the limit
        if len(new_symbols) > max_active_symbols:
            new_symbols = new_symbols[:max_active_symbols]
        
        # Record rotation event
        rotation_event = {
            'timestamp': datetime.now().isoformat(),
            'old_symbols': self.active_symbols.copy(),
            'new_symbols': new_symbols,
            'strategy': strategy,
            'trigger': 'time_based',  # Could be enhanced with actual trigger
            'websocket_limit': websocket_limit,
            'test_slots_reserved': test_slots
        }
        
        self.rotation_history.append(rotation_event)
        self.active_symbols = new_symbols
        self.last_rotation_time = time.time()
        
        # Publish rotation event
        self.message_bus.publish('symbol_rotation', rotation_event)
        
        self.logger.info(f"Rotation completed: {len(new_symbols)} active symbols")
    
    def _update_priorities(self):
        """Update symbol priorities based on RL feedback and performance."""
        # Get RL scores from stream strategy agent
        rl_scores = self.stream_strategy.get_symbol_scores(
            self.active_symbols,
            self.stream_metrics
        )
        
        # Update priorities with RL guidance
        priority_weight = self.config['adaptive_params']['priority_weight']
        
        for symbol in self.active_symbols:
            old_priority = self.symbol_priorities.get(symbol, 0.5)
            rl_score = rl_scores.get(symbol, 0.5)
            
            # Blend old priority with new RL score
            new_priority = (priority_weight * rl_score + 
                           (1 - priority_weight) * old_priority)
            
            self.symbol_priorities[symbol] = new_priority
        
        # Normalize priorities
        total = sum(self.symbol_priorities.values())
        if total > 0:
            for symbol in self.symbol_priorities:
                self.symbol_priorities[symbol] /= total
        
        # Publish RL scores
        self.message_bus.publish('rl_scores', {
            'timestamp': datetime.now().isoformat(),
            'scores': self.symbol_priorities.copy()
        })
    
    def _synthesize_indicators(self):
        """Synthesize indicator combinations."""
        # Get current market data
        synthetic_indicators = self.indicator_synth.synthesize(
            symbols=self.active_symbols,
            priorities=self.symbol_priorities
        )
        
        # Publish synthetic indicators
        if synthetic_indicators:
            self.message_bus.publish('indicator_synth_data', synthetic_indicators)
    
    def _update_stream_strategies(self):
        """Update streaming strategies based on performance."""
        strategy_update = self.stream_strategy.update_strategy(
            metrics=self.stream_metrics,
            priorities=self.symbol_priorities
        )
        
        if strategy_update:
            self.message_bus.publish('strategy_decision', strategy_update)
    
    def _collect_metrics(self):
        """Collect metrics from all streams and submodules."""
        # Update detailed metrics with proper limits
        websocket_limit = self.config.get('websocket_limit', 50)
        self.detailed_metrics['active_subscriptions'] = min(len(self.active_symbols), websocket_limit)
        self.detailed_metrics['historical_symbols_used'].update(self.active_symbols)
        
        # Calculate WebSocket usage percentage (capped at 100%)
        usage_pct = (len(self.active_symbols) / websocket_limit * 100) if websocket_limit > 0 else 0
        self.detailed_metrics['websocket_usage_pct'] = min(usage_pct, 100.0)
        
        self.detailed_metrics['total_data_points'] += len(self.active_symbols)
        
        # Get submodule details
        self.detailed_metrics['submodule_details'] = {
            'indicator_synth': {
                'status': 'active',
                'recipes_count': len(self.indicator_synth.recipes),
                'cached_symbols': len(self.indicator_synth.indicator_cache)
            },
            'symbol_rotation': {
                'status': 'active',
                'rotation_count': self.symbol_rotation.rotation_count,
                'symbol_pool_size': len(self.symbol_rotation.symbol_pool),
                'websocket_limit': websocket_limit,
                'test_slots': self.config.get('test_slot_count', 1)
            },
            'rotation_strategy': {
                'status': 'active',
                'current_strategy': self.rotation_strategy.current_strategy.get('type', 'unknown'),
                'feedback_buffer_size': len(self.rotation_strategy.feedback_buffer)
            },
            'stream_strategy': {
                'status': 'active',
                'experience_buffer_size': len(self.stream_strategy.experience_buffer),
                'current_batch_size': self.stream_strategy.current_strategy.get('batch_size', 12),
                'rest_batch_size': self.config.get('rest_batch_size', 12)
            },
            'replay_engine': {
                'status': 'active',
                'is_replaying': self.replay_engine.is_replaying,
                'replay_mode': self.replay_engine.replay_mode,
                'replay_position': self.replay_engine.replay_position
            },
            'ontology_mapper': {
                'status': 'active',
                'supported_sources': len(self.ontology_mapper.mapping_rules)
            }
        }
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'active_symbols': len(self.active_symbols),
            'total_symbols_tracked': len(self.symbol_priorities),
            'avg_priority': sum(self.symbol_priorities.values()) / len(self.symbol_priorities) if self.symbol_priorities else 0,
            'rotations_count': len(self.rotation_history),
            'stream_health': self._calculate_stream_health(),
            'submodule_status': self._get_submodule_status(),
            'detailed_metrics': self.detailed_metrics.copy()
        }
        
        # Convert set to list for JSON serialization
        if 'historical_symbols_used' in metrics['detailed_metrics']:
            metrics['detailed_metrics']['historical_symbols_used'] = list(
                metrics['detailed_metrics']['historical_symbols_used']
            )
        
        self.stream_metrics['orchestrator'] = metrics
        
        # Publish metrics
        self.message_bus.publish('stream_metrics', metrics)
    
    def _calculate_stream_health(self) -> float:
        """Calculate overall stream health score."""
        # Placeholder: could be enhanced with actual health metrics
        return 0.95
    
    def _get_submodule_status(self) -> Dict[str, str]:
        """Get status of all submodules."""
        return {
            'indicator_synth': 'active',
            'symbol_rotation': 'active',
            'rotation_strategy': 'active',
            'stream_strategy': 'active',
            'replay_engine': 'active',
            'ontology_mapper': 'active'
        }
    
    def _publish_status(self):
        """Publish orchestrator status to message bus."""
        status = {
            'timestamp': datetime.now().isoformat(),
            'is_running': self.is_running,
            'live_mode': self.live_mode,
            'active_symbols': self.active_symbols,
            'priorities': self.symbol_priorities,
            'config': self.config,
            'metrics': self.stream_metrics.get('orchestrator', {}),
            'last_rotation': self.last_rotation_time
        }
        
        self.message_bus.publish('orchestrator_status', status)
    
    def _handle_rl_feedback(self, feedback: Dict[str, Any]):
        """Handle feedback from RL controllers."""
        self.logger.debug(f"Received RL feedback: {feedback}")
        
        # Pass to stream strategy agent
        self.stream_strategy.process_feedback(feedback)
        
        # Update rotation strategy
        self.rotation_strategy.process_feedback(feedback)
    
    def _handle_market_conditions(self, conditions: Dict[str, Any]):
        """Handle market condition changes."""
        self.logger.debug(f"Market conditions updated: {conditions}")
        
        # Check if rotation is triggered by market regime change
        if conditions.get('regime_change', False):
            if self._should_rotate():
                self._perform_rotation()
    
    def _handle_portfolio_status(self, status: Dict[str, Any]):
        """
        Handle portfolio status updates to protect positions.
        
        Updates the list of protected symbols (stocks in portfolio) that
        must maintain WebSocket connections even during rotation.
        """
        positions = status.get('positions', {})
        # Extract symbols from portfolio positions
        portfolio_symbols = set(positions.keys())
        
        # Update protected symbols
        old_protected = self.protected_symbols.copy()
        self.protected_symbols = portfolio_symbols
        
        # Log if protection changed
        if old_protected != self.protected_symbols:
            added = self.protected_symbols - old_protected
            removed = old_protected - self.protected_symbols
            if added:
                self.logger.info(f"Protected symbols added (in portfolio): {added}")
            if removed:
                self.logger.info(f"Protected symbols removed (sold from portfolio): {removed}")
            
            # If we have protected symbols not in active_symbols, we need to add them
            missing_symbols = self.protected_symbols - set(self.active_symbols)
            if missing_symbols:
                self.logger.warning(f"Portfolio contains symbols not in active WebSocket: {missing_symbols}")
                self.logger.info("Adding portfolio symbols to active_symbols to maintain price tracking")
                # Add to active symbols (will be included in next update)
                for symbol in missing_symbols:
                    if symbol not in self.active_symbols:
                        self.active_symbols.append(symbol)
    
    def _handle_module_request(self, request: Dict[str, Any]):
        """Handle data requests from downstream modules."""
        request_type = request.get('type')
        
        if request_type == 'replay':
            # Start replay mode
            self.replay_engine.start_replay(request.get('config', {}))
        elif request_type == 'priority_update':
            # Force priority update
            self._update_priorities()
        elif request_type == 'rotation':
            # Force rotation
            self._perform_rotation()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current orchestrator status."""
        return {
            'is_running': self.is_running,
            'live_mode': self.live_mode,
            'active_symbols': self.active_symbols,
            'priorities': self.symbol_priorities,
            'rotation_count': len(self.rotation_history),
            'last_rotation': self.last_rotation_time
        }
    
    def update_config(self, new_config: Dict[str, Any]):
        """
        Dynamically update orchestrator configuration.
        
        Args:
            new_config: New configuration values to apply
        """
        self.config.update(new_config)
        self.logger.info(f"Configuration updated: {new_config}")
        
        # Propagate config changes to submodules
        if 'rotation_interval' in new_config:
            self.symbol_rotation.update_interval(new_config['rotation_interval'])
    
    def enable_replay_mode(self, replay_config: Optional[Dict[str, Any]] = None):
        """
        Enable replay mode for simulation.
        
        Args:
            replay_config: Configuration for replay engine
        """
        self.replay_engine.start_replay(replay_config or {})
        self.logger.info("Replay mode enabled")
    
    def disable_replay_mode(self):
        """Disable replay mode."""
        self.replay_engine.stop_replay()
        self.logger.info("Replay mode disabled")
