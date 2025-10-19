"""
stream_ontology_mapper.py - Stream Ontology Mapper

Description:
    Maps and normalizes data from different sources and formats.
    Provides unified data schema for downstream consumption.

Features:
    - Multi-source data mapping
    - Schema normalization
    - Data validation
    - Format conversion
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import logging


class StreamOntologyMapper:
    """
    Mapper for normalizing data from different sources and formats.
    """
    
    def __init__(self, message_bus):
        """
        Initialize the stream ontology mapper.
        
        Args:
            message_bus: Reference to message bus
        """
        self.message_bus = message_bus
        self.logger = logging.getLogger('StreamOntologyMapper')
        
        # Define standard schema
        self.standard_schema = {
            'timestamp': str,
            'symbol': str,
            'price': float,
            'volume': int,
            'high': float,
            'low': float,
            'open': float,
            'close': float,
            'source': str
        }
        
        # Mapping rules for different sources
        self.mapping_rules = self._load_mapping_rules()
    
    def _load_mapping_rules(self) -> Dict[str, Dict[str, str]]:
        """Load mapping rules for different data sources."""
        return {
            'finnhub': {
                'p': 'price',
                's': 'symbol',
                't': 'timestamp',
                'v': 'volume'
            },
            'yahoo': {
                'Close': 'close',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Volume': 'volume',
                'Symbol': 'symbol'
            },
            'alpha_vantage': {
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. volume': 'volume'
            }
        }
    
    def map_data(
        self,
        data: Dict[str, Any],
        source: str = 'finnhub'
    ) -> Optional[Dict[str, Any]]:
        """
        Map data to standard schema.
        
        Args:
            data: Raw data from source
            source: Data source identifier
            
        Returns:
            Normalized data or None if mapping fails
        """
        if source not in self.mapping_rules:
            self.logger.warning(f"Unknown source: {source}")
            return None
        
        rules = self.mapping_rules[source]
        mapped_data = {'source': source}
        
        # Apply mapping rules
        for source_key, target_key in rules.items():
            if source_key in data:
                mapped_data[target_key] = data[source_key]
        
        # Fill in missing fields with defaults or derived values
        mapped_data = self._complete_data(mapped_data)
        
        # Validate mapped data
        if self._validate_data(mapped_data):
            # Publish mapped data
            self.message_bus.publish('mapped_data', mapped_data)
            return mapped_data
        else:
            self.logger.warning(f"Data validation failed for {data}")
            return None
    
    def _complete_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Complete missing fields in mapped data.
        
        Args:
            data: Partially mapped data
            
        Returns:
            Completed data
        """
        # Ensure timestamp
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().isoformat()
        elif isinstance(data['timestamp'], (int, float)):
            # Convert unix timestamp to ISO format
            data['timestamp'] = datetime.fromtimestamp(
                data['timestamp'] / 1000 if data['timestamp'] > 1e12 else data['timestamp']
            ).isoformat()
        
        # Derive OHLC from price if missing
        if 'price' in data:
            price = data['price']
            data.setdefault('open', price)
            data.setdefault('high', price)
            data.setdefault('low', price)
            data.setdefault('close', price)
        
        # Default volume
        data.setdefault('volume', 0)
        
        return data
    
    def _validate_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate mapped data against schema.
        
        Args:
            data: Mapped data
            
        Returns:
            True if valid, False otherwise
        """
        # Check required fields
        required_fields = ['timestamp', 'symbol', 'price']
        
        for field in required_fields:
            if field not in data:
                self.logger.debug(f"Missing required field: {field}")
                return False
        
        # Type checking
        try:
            float(data['price'])
            if 'volume' in data:
                int(data['volume'])
            return True
        except (ValueError, TypeError):
            return False
    
    def batch_map(
        self,
        data_list: List[Dict[str, Any]],
        source: str = 'finnhub'
    ) -> List[Dict[str, Any]]:
        """
        Map multiple data points in batch.
        
        Args:
            data_list: List of raw data
            source: Data source identifier
            
        Returns:
            List of normalized data
        """
        mapped_list = []
        
        for data in data_list:
            mapped = self.map_data(data, source)
            if mapped:
                mapped_list.append(mapped)
        
        self.logger.debug(f"Batch mapped {len(mapped_list)}/{len(data_list)} items")
        
        return mapped_list
    
    def add_mapping_rule(
        self,
        source: str,
        rules: Dict[str, str]
    ):
        """
        Add or update mapping rules for a source.
        
        Args:
            source: Source identifier
            rules: Mapping rules (source_key -> target_key)
        """
        self.mapping_rules[source] = rules
        self.logger.info(f"Added mapping rules for source: {source}")
    
    def get_supported_sources(self) -> List[str]:
        """Get list of supported data sources."""
        return list(self.mapping_rules.keys())
    
    def get_standard_schema(self) -> Dict[str, type]:
        """Get the standard data schema."""
        return self.standard_schema.copy()
