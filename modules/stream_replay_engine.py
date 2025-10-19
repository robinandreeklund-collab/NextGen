"""
stream_replay_engine.py - Stream Replay Engine

Description:
    Replays historical data for simulation and testing.
    Supports multiple replay modes: historical, synthetic, and hybrid.

Features:
    - Historical data replay
    - Synthetic data generation
    - Hybrid mode (mix of real and synthetic)
    - Configurable replay speed
    - Time-travel capabilities for testing
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
import threading
import time
import random


class StreamReplayEngine:
    """
    Engine for replaying historical or synthetic market data.
    """
    
    def __init__(self, message_bus):
        """
        Initialize the stream replay engine.
        
        Args:
            message_bus: Reference to message bus
        """
        self.message_bus = message_bus
        self.logger = logging.getLogger('StreamReplayEngine')
        
        # Replay state
        self.is_replaying = False
        self.replay_thread = None
        self.replay_mode = 'historical'  # 'historical', 'synthetic', 'hybrid'
        self.replay_speed = 1.0
        
        # Data storage
        self.historical_data = {}
        self.replay_position = 0
        
        # Configuration
        self.config = {
            'mode': 'historical',
            'speed': 1.0,
            'symbols': [],
            'start_time': None,
            'end_time': None
        }
    
    def start_replay(self, config: Optional[Dict[str, Any]] = None):
        """
        Start replay mode.
        
        Args:
            config: Replay configuration
        """
        if self.is_replaying:
            self.logger.warning("Replay already running")
            return
        
        # Update configuration
        if config:
            self.config.update(config)
        
        self.replay_mode = self.config.get('mode', 'historical')
        self.replay_speed = self.config.get('speed', 1.0)
        
        # Load historical data if needed
        if self.replay_mode in ['historical', 'hybrid']:
            self._load_historical_data()
        
        self.is_replaying = True
        self.replay_position = 0
        
        # Start replay thread
        self.replay_thread = threading.Thread(
            target=self._replay_loop,
            daemon=True
        )
        self.replay_thread.start()
        
        self.logger.info(f"Replay started: mode={self.replay_mode}, speed={self.replay_speed}")
        
        # Publish replay event
        self.message_bus.publish('replay_event', {
            'type': 'started',
            'mode': self.replay_mode,
            'speed': self.replay_speed,
            'timestamp': datetime.now().isoformat()
        })
    
    def stop_replay(self):
        """Stop replay mode."""
        if not self.is_replaying:
            return
        
        self.is_replaying = False
        
        if self.replay_thread:
            self.replay_thread.join(timeout=5)
        
        self.logger.info("Replay stopped")
        
        # Publish replay event
        self.message_bus.publish('replay_event', {
            'type': 'stopped',
            'timestamp': datetime.now().isoformat()
        })
    
    def _load_historical_data(self):
        """Load historical market data."""
        # Placeholder: In real implementation, load from database or files
        symbols = self.config.get('symbols', ['AAPL', 'TSLA', 'MSFT'])
        
        for symbol in symbols:
            # Generate mock historical data
            data_points = []
            base_price = 100.0 + random.random() * 100
            
            for i in range(1000):
                timestamp = datetime.now() - timedelta(minutes=1000-i)
                price = base_price + random.gauss(0, 5)
                volume = random.randint(1000000, 10000000)
                
                data_points.append({
                    'timestamp': timestamp.isoformat(),
                    'symbol': symbol,
                    'price': price,
                    'volume': volume,
                    'high': price + random.random() * 2,
                    'low': price - random.random() * 2,
                    'open': price + random.gauss(0, 1),
                    'close': price
                })
            
            self.historical_data[symbol] = data_points
        
        self.logger.info(f"Loaded historical data for {len(symbols)} symbols")
    
    def _replay_loop(self):
        """Main replay loop."""
        while self.is_replaying:
            try:
                # Generate or retrieve next data point
                if self.replay_mode == 'historical':
                    data = self._get_next_historical_data()
                elif self.replay_mode == 'synthetic':
                    data = self._generate_synthetic_data()
                elif self.replay_mode == 'hybrid':
                    # Mix of historical and synthetic
                    if random.random() < 0.7:
                        data = self._get_next_historical_data()
                    else:
                        data = self._generate_synthetic_data()
                else:
                    data = None
                
                if data:
                    # Publish replay data
                    self.message_bus.publish('replay_data', data)
                
                # Sleep based on replay speed
                sleep_time = 1.0 / self.replay_speed
                time.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Error in replay loop: {e}", exc_info=True)
                time.sleep(1)
    
    def _get_next_historical_data(self) -> Optional[Dict[str, Any]]:
        """Get next historical data point."""
        if not self.historical_data:
            return None
        
        # Round-robin through symbols
        symbols = list(self.historical_data.keys())
        if not symbols:
            return None
        
        symbol_index = self.replay_position % len(symbols)
        symbol = symbols[symbol_index]
        
        data_list = self.historical_data[symbol]
        data_index = (self.replay_position // len(symbols)) % len(data_list)
        
        self.replay_position += 1
        
        return data_list[data_index]
    
    def _generate_synthetic_data(self) -> Dict[str, Any]:
        """Generate synthetic market data."""
        symbols = self.config.get('symbols', ['AAPL', 'TSLA', 'MSFT'])
        symbol = random.choice(symbols)
        
        base_price = 100.0 + random.random() * 100
        
        return {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'price': base_price + random.gauss(0, 5),
            'volume': random.randint(1000000, 10000000),
            'high': base_price + random.random() * 5,
            'low': base_price - random.random() * 5,
            'open': base_price + random.gauss(0, 2),
            'close': base_price,
            'synthetic': True
        }
    
    def set_replay_speed(self, speed: float):
        """
        Set replay speed multiplier.
        
        Args:
            speed: Speed multiplier (e.g., 2.0 = 2x speed)
        """
        self.replay_speed = max(0.1, min(10.0, speed))
        self.config['speed'] = self.replay_speed
        self.logger.info(f"Replay speed set to {self.replay_speed}x")
    
    def jump_to_position(self, position: int):
        """
        Jump to specific position in replay.
        
        Args:
            position: Position index to jump to
        """
        self.replay_position = position
        self.logger.info(f"Jumped to replay position {position}")
    
    def get_replay_status(self) -> Dict[str, Any]:
        """Get current replay status."""
        return {
            'is_replaying': self.is_replaying,
            'mode': self.replay_mode,
            'speed': self.replay_speed,
            'position': self.replay_position,
            'config': self.config
        }
