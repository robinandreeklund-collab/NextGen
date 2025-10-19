"""
indicator_synth_engine.py - Indicator Synthesis Engine

Description:
    Synthesizes indicator combinations and derived metrics from raw indicators.
    Creates composite indicators and advanced technical analysis metrics.

Features:
    - Combines multiple indicators into synthetic metrics
    - Calculates derived indicators (e.g., divergence, convergence)
    - Supports custom indicator recipes
    - Publishes synthetic indicators to message bus
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import logging


class IndicatorSynthEngine:
    """
    Engine for synthesizing indicator combinations and derived metrics.
    """
    
    def __init__(self, message_bus):
        """
        Initialize the indicator synthesis engine.
        
        Args:
            message_bus: Reference to message bus
        """
        self.message_bus = message_bus
        self.logger = logging.getLogger('IndicatorSynthEngine')
        
        # Synthesis recipes
        self.recipes = self._load_default_recipes()
        
        # Cache for indicator data
        self.indicator_cache: Dict[str, Dict[str, Any]] = {}
    
    def _load_default_recipes(self) -> Dict[str, Dict[str, Any]]:
        """Load default synthesis recipes."""
        return {
            'momentum_composite': {
                'inputs': ['RSI', 'MACD', 'Stochastic'],
                'weights': [0.4, 0.4, 0.2],
                'operation': 'weighted_average'
            },
            'volatility_composite': {
                'inputs': ['ATR', 'Bollinger_Width'],
                'weights': [0.6, 0.4],
                'operation': 'weighted_average'
            },
            'trend_strength': {
                'inputs': ['ADX', 'MACD_histogram'],
                'weights': [0.7, 0.3],
                'operation': 'weighted_average'
            },
            'price_momentum_divergence': {
                'inputs': ['price_change', 'RSI_change'],
                'operation': 'divergence_detection'
            }
        }
    
    def synthesize(
        self,
        symbols: List[str],
        priorities: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Synthesize indicators for given symbols.
        
        Args:
            symbols: List of symbols to synthesize indicators for
            priorities: Optional priority weights for symbols
            
        Returns:
            Dictionary of synthetic indicators
        """
        synthetic_indicators = {
            'timestamp': datetime.now().isoformat(),
            'symbols': {}
        }
        
        for symbol in symbols:
            # Get cached indicators for symbol
            cached_data = self.indicator_cache.get(symbol, {})
            
            # Apply synthesis recipes
            symbol_synthetics = {}
            for recipe_name, recipe in self.recipes.items():
                result = self._apply_recipe(recipe, cached_data)
                if result is not None:
                    symbol_synthetics[recipe_name] = result
            
            synthetic_indicators['symbols'][symbol] = symbol_synthetics
        
        return synthetic_indicators
    
    def _apply_recipe(
        self,
        recipe: Dict[str, Any],
        indicator_data: Dict[str, Any]
    ) -> Optional[float]:
        """
        Apply a synthesis recipe to indicator data.
        
        Args:
            recipe: Recipe configuration
            indicator_data: Raw indicator data
            
        Returns:
            Synthesized value or None if inputs missing
        """
        operation = recipe.get('operation')
        inputs = recipe.get('inputs', [])
        
        # Check if all inputs are available
        values = []
        for input_name in inputs:
            if input_name in indicator_data:
                values.append(indicator_data[input_name])
            else:
                return None  # Missing input
        
        if operation == 'weighted_average':
            weights = recipe.get('weights', [1.0] * len(values))
            return sum(v * w for v, w in zip(values, weights)) / sum(weights)
        
        elif operation == 'divergence_detection':
            if len(values) >= 2:
                # Simple divergence: opposite signs
                return 1.0 if values[0] * values[1] < 0 else 0.0
        
        return None
    
    def update_indicator_cache(self, symbol: str, indicators: Dict[str, Any]):
        """
        Update cached indicator data for a symbol.
        
        Args:
            symbol: Symbol to update
            indicators: New indicator values
        """
        self.indicator_cache[symbol] = indicators
    
    def add_recipe(self, name: str, recipe: Dict[str, Any]):
        """
        Add a custom synthesis recipe.
        
        Args:
            name: Recipe name
            recipe: Recipe configuration
        """
        self.recipes[name] = recipe
        self.logger.info(f"Added synthesis recipe: {name}")
    
    def get_available_recipes(self) -> List[str]:
        """Get list of available synthesis recipes."""
        return list(self.recipes.keys())
