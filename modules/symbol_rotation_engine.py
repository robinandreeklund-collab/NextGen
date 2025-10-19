"""
symbol_rotation_engine.py - Symbol Rotation Engine

Description:
    Manages symbol rotation based on RL priorities, market conditions, and
    performance metrics. Implements various rotation strategies.

Features:
    - Time-based rotation
    - Performance-based rotation
    - RL-driven rotation
    - Adaptive rotation intervals
    - Loads NASDAQ 100 symbols from configuration
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import random
import yaml
import os


class SymbolRotationEngine:
    """
    Engine for managing symbol rotation in the orchestrator.
    """
    
    def __init__(self, message_bus, rotation_interval: int = 300):
        """
        Initialize the symbol rotation engine.
        
        Args:
            message_bus: Reference to message bus
            rotation_interval: Default rotation interval in seconds
        """
        self.message_bus = message_bus
        self.rotation_interval = rotation_interval
        self.logger = logging.getLogger('SymbolRotationEngine')
        
        # Available symbol pool - load from NASDAQ 100 if available
        self.symbol_pool = self._load_symbol_pool()
        
        # Rotation history
        self.rotation_count = 0
    
    def _load_symbol_pool(self) -> List[str]:
        """Load symbol pool from NASDAQ 100 YAML or use default."""
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
                            self.logger.info(f"Loaded {len(data['symbols'])} symbols from NASDAQ 100")
                            return data['symbols']
            
            # Fallback to default pool
            self.logger.warning("Using default symbol pool")
            return [
                "AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "NVDA", "META",
                "NFLX", "AMD", "INTC", "BABA", "DIS", "BA", "GE",
                "JPM", "BAC", "WMT", "JNJ", "V", "MA"
            ]
        except Exception as e:
            self.logger.error(f"Error loading symbol pool: {e}")
            return [
                "AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "NVDA", "META",
                "NFLX", "AMD", "INTC", "BABA", "DIS", "BA", "GE",
                "JPM", "BAC", "WMT", "JNJ", "V", "MA"
            ]
    
    def rotate_symbols(
        self,
        current_symbols: List[str],
        priorities: Dict[str, float],
        strategy: Dict[str, Any],
        max_symbols: int = 10
    ) -> List[str]:
        """
        Rotate symbols based on priorities and strategy.
        
        Args:
            current_symbols: Currently active symbols
            priorities: Symbol priority scores
            strategy: Rotation strategy configuration
            max_symbols: Maximum number of symbols to maintain
            
        Returns:
            New list of active symbols
        """
        strategy_type = strategy.get('type', 'top_priority')
        rotation_rate = strategy.get('rotation_rate', 0.3)
        
        # Determine how many symbols to rotate out
        num_to_rotate = max(1, int(len(current_symbols) * rotation_rate))
        
        if strategy_type == 'top_priority':
            # Keep top priority symbols, rotate out lowest
            new_symbols = self._rotate_by_priority(
                current_symbols,
                priorities,
                num_to_rotate,
                max_symbols
            )
        
        elif strategy_type == 'random':
            # Random rotation for exploration
            new_symbols = self._rotate_random(
                current_symbols,
                num_to_rotate,
                max_symbols
            )
        
        elif strategy_type == 'hybrid':
            # Mix of priority-based and random
            new_symbols = self._rotate_hybrid(
                current_symbols,
                priorities,
                num_to_rotate,
                max_symbols
            )
        
        else:
            # Default: keep current
            new_symbols = current_symbols
        
        self.rotation_count += 1
        self.logger.info(
            f"Rotation #{self.rotation_count}: "
            f"{len(current_symbols)} -> {len(new_symbols)} symbols"
        )
        
        # Publish rotation event
        self.message_bus.publish('symbol_rotation_event', {
            'timestamp': datetime.now().isoformat(),
            'old_symbols': current_symbols,
            'new_symbols': new_symbols,
            'strategy': strategy_type,
            'rotation_count': self.rotation_count
        })
        
        return new_symbols
    
    def _rotate_by_priority(
        self,
        current: List[str],
        priorities: Dict[str, float],
        num_to_rotate: int,
        max_symbols: int
    ) -> List[str]:
        """Rotate based on priority scores."""
        # Sort current symbols by priority (descending)
        sorted_current = sorted(
            current,
            key=lambda s: priorities.get(s, 0),
            reverse=True
        )
        
        # Keep top symbols
        keep_symbols = sorted_current[:-num_to_rotate] if num_to_rotate < len(sorted_current) else []
        
        # Find new symbols from pool
        available = [s for s in self.symbol_pool if s not in keep_symbols]
        
        # Add new symbols to fill up to max_symbols
        num_to_add = min(num_to_rotate, max_symbols - len(keep_symbols))
        
        # Prioritize symbols not recently active
        new_additions = random.sample(available, min(num_to_add, len(available)))
        
        return keep_symbols + new_additions
    
    def _rotate_random(
        self,
        current: List[str],
        num_to_rotate: int,
        max_symbols: int
    ) -> List[str]:
        """Random rotation for exploration."""
        # Randomly select symbols to keep
        num_to_keep = len(current) - num_to_rotate
        keep_symbols = random.sample(current, max(0, num_to_keep))
        
        # Add random new symbols
        available = [s for s in self.symbol_pool if s not in keep_symbols]
        num_to_add = min(num_to_rotate, max_symbols - len(keep_symbols))
        new_additions = random.sample(available, min(num_to_add, len(available)))
        
        return keep_symbols + new_additions
    
    def _rotate_hybrid(
        self,
        current: List[str],
        priorities: Dict[str, float],
        num_to_rotate: int,
        max_symbols: int
    ) -> List[str]:
        """Hybrid rotation: 70% priority-based, 30% random."""
        priority_rotate = int(num_to_rotate * 0.7)
        random_rotate = num_to_rotate - priority_rotate
        
        # Priority-based rotation
        sorted_current = sorted(
            current,
            key=lambda s: priorities.get(s, 0),
            reverse=True
        )
        
        keep_symbols = sorted_current[:-priority_rotate] if priority_rotate < len(sorted_current) else sorted_current
        
        # Remove some randomly as well
        if random_rotate > 0 and len(keep_symbols) > random_rotate:
            to_remove = set(random.sample(keep_symbols, random_rotate))
            keep_symbols = [s for s in keep_symbols if s not in to_remove]
        
        # Add new symbols
        available = [s for s in self.symbol_pool if s not in keep_symbols]
        num_to_add = min(max_symbols - len(keep_symbols), len(available))
        new_additions = random.sample(available, num_to_add)
        
        return keep_symbols + new_additions
    
    def update_interval(self, new_interval: int):
        """Update rotation interval."""
        self.rotation_interval = new_interval
        self.logger.info(f"Rotation interval updated to {new_interval}s")
    
    def add_symbols_to_pool(self, symbols: List[str]):
        """Add symbols to the available pool."""
        for symbol in symbols:
            if symbol not in self.symbol_pool:
                self.symbol_pool.append(symbol)
        
        self.logger.info(f"Added {len(symbols)} symbols to pool")
    
    def get_symbol_pool(self) -> List[str]:
        """Get the current symbol pool."""
        return self.symbol_pool.copy()
