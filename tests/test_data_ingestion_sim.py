"""
Tests for data_ingestion_sim.py

Verifies that simulated market data is generated correctly and published via message_bus.
"""

import pytest
import time
from modules.data_ingestion_sim import DataIngestionSim
from modules.message_bus import MessageBus


class TestDataIngestionSim:
    """Test suite for DataIngestionSim module."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.message_bus = MessageBus()
        self.symbols = ['AAPL', 'MSFT', 'GOOGL']
        self.data_ingestion_sim = DataIngestionSim(self.message_bus, self.symbols)
        
        # Track published messages
        self.published_messages = []
        
        def capture_message(msg):
            self.published_messages.append(msg)
        
        self.message_bus.subscribe('market_data', capture_message)
    
    def test_initialization(self):
        """Test that DataIngestionSim initializes correctly."""
        assert self.data_ingestion_sim.message_bus is not None
        assert len(self.data_ingestion_sim.symbols) == 3
        assert 'AAPL' in self.data_ingestion_sim.symbols
        assert len(self.data_ingestion_sim.current_prices) == 3
        assert self.data_ingestion_sim.iteration == 0
    
    def test_simulate_market_tick_publishes_data(self):
        """Test that simulate_market_tick publishes market data."""
        # Clear any previous messages
        self.published_messages.clear()
        
        # Simulate one tick
        self.data_ingestion_sim.simulate_market_tick()
        
        # Should have published data for all symbols
        assert len(self.published_messages) == len(self.symbols)
        
        # Check structure of published messages
        for msg in self.published_messages:
            assert 'symbol' in msg
            assert 'data' in msg
            assert 'source' in msg
            assert msg['source'] == 'simulation'
            
            data = msg['data']
            assert 'price' in data
            assert 'volume' in data
            assert 'timestamp' in data
            assert 'change' in data
            assert 'iteration' in data
    
    def test_price_changes_over_time(self):
        """Test that prices change realistically over multiple ticks."""
        initial_prices = self.data_ingestion_sim.get_current_prices()
        
        # Simulate multiple ticks
        for _ in range(10):
            self.data_ingestion_sim.simulate_market_tick()
        
        final_prices = self.data_ingestion_sim.get_current_prices()
        
        # Prices should have changed
        for symbol in self.symbols:
            assert initial_prices[symbol] != final_prices[symbol]
            
            # Prices should be positive
            assert final_prices[symbol] > 0
            
            # Price change should be reasonable (not > 50% in 10 ticks)
            change_pct = abs(final_prices[symbol] - initial_prices[symbol]) / initial_prices[symbol]
            assert change_pct < 0.5, f"Price changed too much: {change_pct*100:.2f}%"
    
    def test_volume_generation(self):
        """Test that volume is generated within reasonable bounds."""
        self.published_messages.clear()
        self.data_ingestion_sim.simulate_market_tick()
        
        for msg in self.published_messages:
            data = msg['data']
            volume = data['volume']
            
            # Volume should be positive
            assert volume > 0
            
            # Volume should be reasonable (not zero, not astronomical)
            assert volume > 1_000_000  # At least 1M
            assert volume < 1_000_000_000  # Less than 1B
    
    def test_iteration_counter(self):
        """Test that iteration counter increments correctly."""
        initial_iteration = self.data_ingestion_sim.iteration
        assert initial_iteration == 0
        
        self.data_ingestion_sim.simulate_market_tick()
        assert self.data_ingestion_sim.iteration == 1
        
        self.data_ingestion_sim.simulate_market_tick()
        assert self.data_ingestion_sim.iteration == 2
    
    def test_get_current_prices(self):
        """Test get_current_prices returns correct data."""
        prices = self.data_ingestion_sim.get_current_prices()
        
        assert isinstance(prices, dict)
        assert len(prices) == len(self.symbols)
        
        for symbol in self.symbols:
            assert symbol in prices
            assert prices[symbol] > 0
    
    def test_get_symbols(self):
        """Test get_symbols returns correct symbol list."""
        symbols = self.data_ingestion_sim.get_symbols()
        
        assert isinstance(symbols, list)
        assert len(symbols) == len(self.symbols)
        assert symbols == self.symbols
    
    def test_reset_prices(self):
        """Test that reset_prices resets to base values."""
        # Change prices by simulating ticks
        for _ in range(5):
            self.data_ingestion_sim.simulate_market_tick()
        
        # Prices should have changed
        current_prices = self.data_ingestion_sim.get_current_prices()
        base_prices = self.data_ingestion_sim.base_prices
        
        # Reset prices
        self.data_ingestion_sim.reset_prices()
        
        # Check that prices are back to base
        reset_prices = self.data_ingestion_sim.get_current_prices()
        for symbol in self.symbols:
            assert reset_prices[symbol] == base_prices[symbol]
        
        # Iteration should be reset
        assert self.data_ingestion_sim.iteration == 0
    
    def test_price_realism(self):
        """Test that price movements are realistic (bounded volatility)."""
        # Run many ticks and track price changes
        price_changes = []
        
        for _ in range(100):
            old_prices = self.data_ingestion_sim.get_current_prices()
            self.data_ingestion_sim.simulate_market_tick()
            new_prices = self.data_ingestion_sim.get_current_prices()
            
            for symbol in self.symbols:
                change_pct = (new_prices[symbol] - old_prices[symbol]) / old_prices[symbol]
                price_changes.append(change_pct)
        
        # Check that changes are mostly reasonable (within ±10% per tick)
        extreme_changes = [c for c in price_changes if abs(c) > 0.10]
        
        # Allow some extreme changes (news events) but not too many
        assert len(extreme_changes) < len(price_changes) * 0.05, \
            f"Too many extreme price changes: {len(extreme_changes)}/{len(price_changes)}"
    
    def test_default_symbols(self):
        """Test that default symbols are used when none provided."""
        sim = DataIngestionSim(self.message_bus)
        
        symbols = sim.get_symbols()
        assert len(symbols) == 5
        assert 'AAPL' in symbols
        assert 'MSFT' in symbols
        assert 'GOOGL' in symbols
        assert 'AMZN' in symbols
        assert 'TSLA' in symbols
    
    def test_publish_market_data_format(self):
        """Test that published data has correct format."""
        self.published_messages.clear()
        
        # Manually publish test data
        test_data = {
            'price': 150.0,
            'volume': 50000000,
            'timestamp': time.time(),
            'change': 2.5
        }
        self.data_ingestion_sim.publish_market_data('TEST', test_data)
        
        # Check published message
        assert len(self.published_messages) == 1
        msg = self.published_messages[0]
        
        assert msg['symbol'] == 'TEST'
        assert msg['data'] == test_data
        assert msg['source'] == 'simulation'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
