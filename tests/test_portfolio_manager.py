"""
test_portfolio_manager.py - Tests for portfolio management and reward calculation

Tests the correct behavior of reward calculation for BUY and SELL actions:
- BUY actions should always give reward = 0.0
- SELL actions should give reward = actual P&L (including all fees)
- HOLD/no-action should give reward = 0.0
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.portfolio_manager import PortfolioManager
from modules.message_bus import MessageBus


class TestPortfolioManagerRewards:
    """Test suite for reward calculation in PortfolioManager"""
    
    def setup_method(self):
        """Setup for each test."""
        self.message_bus = MessageBus()
        self.pm = PortfolioManager(
            self.message_bus,
            start_capital=10000.0,
            transaction_fee=0.0025
        )
        
        # Track rewards
        self.rewards = []
        
        def on_base_reward(data):
            self.rewards.append(data.get('reward', 0.0))
        
        self.message_bus.subscribe('base_reward', on_base_reward)
    
    def test_buy_always_zero_reward(self):
        """Test that BUY actions always give reward = 0.0"""
        # Test multiple BUY actions
        test_cases = [
            ('AAPL', 10, 150.0),
            ('MSFT', 5, 300.0),
            ('GOOGL', 20, 120.0),
        ]
        
        for symbol, quantity, price in test_cases:
            buy_result = {
                'success': True,
                'symbol': symbol,
                'action': 'BUY',
                'quantity': quantity,
                'executed_price': price,
                'market_price': price,
                'timestamp': 1234567890
            }
            self.pm.update_portfolio(buy_result)
            self.pm.set_current_prices({symbol: price})
            self.pm.calculate_and_publish_reward()
            
            assert self.rewards[-1] == 0.0, f"BUY {symbol} should give reward 0.0, got {self.rewards[-1]}"
    
    def test_sell_profit_correct_pnl(self):
        """Test that SELL with profit gives correct P&L as reward"""
        # BUY 10 shares @ $100
        buy_result = {
            'success': True,
            'symbol': 'TEST',
            'action': 'BUY',
            'quantity': 10,
            'executed_price': 100.0,
            'market_price': 100.0,
            'timestamp': 1234567890
        }
        self.pm.update_portfolio(buy_result)
        self.pm.set_current_prices({'TEST': 100.0})
        self.pm.calculate_and_publish_reward()
        
        assert self.rewards[-1] == 0.0, "BUY should give reward 0.0"
        
        # SELL 10 shares @ $120 (20% profit)
        sell_result = {
            'success': True,
            'symbol': 'TEST',
            'action': 'SELL',
            'quantity': 10,
            'executed_price': 120.0,
            'market_price': 120.0,
            'timestamp': 1234567891
        }
        self.pm.update_portfolio(sell_result)
        self.pm.set_current_prices({'TEST': 120.0})
        self.pm.calculate_and_publish_reward()
        
        # Calculate expected P&L
        buy_cost = 100.0 * 10 * (1 + 0.0025)  # $1002.50
        sell_revenue = 120.0 * 10 * (1 - 0.0025)  # $1197.00
        expected_pnl = sell_revenue - buy_cost  # $194.50
        
        assert abs(self.rewards[-1] - expected_pnl) < 0.01, \
            f"SELL reward should be {expected_pnl:.4f}, got {self.rewards[-1]:.4f}"
    
    def test_sell_loss_correct_negative_pnl(self):
        """Test that SELL with loss gives correct negative P&L as reward"""
        # BUY 10 shares @ $100
        buy_result = {
            'success': True,
            'symbol': 'TEST',
            'action': 'BUY',
            'quantity': 10,
            'executed_price': 100.0,
            'market_price': 100.0,
            'timestamp': 1234567890
        }
        self.pm.update_portfolio(buy_result)
        self.pm.set_current_prices({'TEST': 100.0})
        self.pm.calculate_and_publish_reward()
        
        # SELL 10 shares @ $80 (20% loss)
        sell_result = {
            'success': True,
            'symbol': 'TEST',
            'action': 'SELL',
            'quantity': 10,
            'executed_price': 80.0,
            'market_price': 80.0,
            'timestamp': 1234567891
        }
        self.pm.update_portfolio(sell_result)
        self.pm.set_current_prices({'TEST': 80.0})
        self.pm.calculate_and_publish_reward()
        
        # Calculate expected P&L
        buy_cost = 100.0 * 10 * (1 + 0.0025)  # $1002.50
        sell_revenue = 80.0 * 10 * (1 - 0.0025)  # $798.00
        expected_pnl = sell_revenue - buy_cost  # -$204.50
        
        assert abs(self.rewards[-1] - expected_pnl) < 0.01, \
            f"SELL reward should be {expected_pnl:.4f}, got {self.rewards[-1]:.4f}"
    
    def test_partial_sell_correct_pnl(self):
        """Test partial sell calculates correct P&L"""
        # BUY 20 shares @ $100
        buy_result = {
            'success': True,
            'symbol': 'TEST',
            'action': 'BUY',
            'quantity': 20,
            'executed_price': 100.0,
            'market_price': 100.0,
            'timestamp': 1234567890
        }
        self.pm.update_portfolio(buy_result)
        self.pm.set_current_prices({'TEST': 100.0})
        self.pm.calculate_and_publish_reward()
        
        # SELL 10 shares @ $110 (partial sell)
        sell_result = {
            'success': True,
            'symbol': 'TEST',
            'action': 'SELL',
            'quantity': 10,
            'executed_price': 110.0,
            'market_price': 110.0,
            'timestamp': 1234567891
        }
        self.pm.update_portfolio(sell_result)
        self.pm.set_current_prices({'TEST': 110.0})
        self.pm.calculate_and_publish_reward()
        
        # Calculate expected P&L for 10 shares
        avg_price_with_fee = 100.0 * (1 + 0.0025)  # $100.25 per share
        buy_cost = avg_price_with_fee * 10  # $1002.50
        sell_revenue = 110.0 * 10 * (1 - 0.0025)  # $1097.25
        expected_pnl = sell_revenue - buy_cost  # $94.75
        
        assert abs(self.rewards[-1] - expected_pnl) < 0.01, \
            f"SELL reward should be {expected_pnl:.4f}, got {self.rewards[-1]:.4f}"
        assert self.pm.positions['TEST']['quantity'] == 10, "Should have 10 shares remaining"
    
    def test_multiple_buys_average_cost_basis(self):
        """Test FIFO with multiple buys followed by sell uses average cost basis"""
        # BUY 10 shares @ $100
        buy_result1 = {
            'success': True,
            'symbol': 'TEST',
            'action': 'BUY',
            'quantity': 10,
            'executed_price': 100.0,
            'market_price': 100.0,
            'timestamp': 1234567890
        }
        self.pm.update_portfolio(buy_result1)
        self.pm.set_current_prices({'TEST': 100.0})
        self.pm.calculate_and_publish_reward()
        
        # BUY 10 more shares @ $120
        buy_result2 = {
            'success': True,
            'symbol': 'TEST',
            'action': 'BUY',
            'quantity': 10,
            'executed_price': 120.0,
            'market_price': 120.0,
            'timestamp': 1234567891
        }
        self.pm.update_portfolio(buy_result2)
        self.pm.set_current_prices({'TEST': 120.0})
        self.pm.calculate_and_publish_reward()
        
        # SELL 20 shares @ $130
        sell_result = {
            'success': True,
            'symbol': 'TEST',
            'action': 'SELL',
            'quantity': 20,
            'executed_price': 130.0,
            'market_price': 130.0,
            'timestamp': 1234567892
        }
        self.pm.update_portfolio(sell_result)
        self.pm.set_current_prices({'TEST': 130.0})
        self.pm.calculate_and_publish_reward()
        
        # Calculate expected P&L based on average cost
        cost1 = 100.0 * 10 * (1 + 0.0025)  # $1002.50
        cost2 = 120.0 * 10 * (1 + 0.0025)  # $1203.00
        total_cost = cost1 + cost2  # $2205.50
        sell_revenue = 130.0 * 20 * (1 - 0.0025)  # $2593.50
        expected_pnl = sell_revenue - total_cost  # $388.00
        
        assert abs(self.rewards[-1] - expected_pnl) < 0.01, \
            f"SELL reward should be {expected_pnl:.4f}, got {self.rewards[-1]:.4f}"
    
    def test_failed_trade_no_reward(self):
        """Test that failed trades don't generate rewards"""
        # Failed BUY (insufficient funds)
        buy_result = {
            'success': False,
            'symbol': 'TEST',
            'action': 'BUY',
            'quantity': 1000,
            'executed_price': 100.0,
            'market_price': 100.0,
            'timestamp': 1234567890
        }
        self.pm.update_portfolio(buy_result)
        self.pm.calculate_and_publish_reward()
        
        assert self.rewards[-1] == 0.0, "Failed BUY should give reward 0.0"
        assert len(self.pm.positions) == 0, "No position should be created"
    
    def test_avg_price_includes_buy_fee(self):
        """Test that average price includes the buy transaction fee"""
        # BUY 10 shares @ $100
        buy_result = {
            'success': True,
            'symbol': 'TEST',
            'action': 'BUY',
            'quantity': 10,
            'executed_price': 100.0,
            'market_price': 100.0,
            'timestamp': 1234567890
        }
        self.pm.update_portfolio(buy_result)
        
        # Average price should include the transaction fee
        expected_avg = 100.0 * (1 + 0.0025)  # $100.25
        actual_avg = self.pm.positions['TEST']['avg_price']
        
        assert abs(actual_avg - expected_avg) < 0.01, \
            f"Average price should be {expected_avg:.4f}, got {actual_avg:.4f}"
    
    def test_sold_history_tracks_pnl(self):
        """Test that sold_history correctly tracks P&L"""
        # BUY and SELL
        buy_result = {
            'success': True,
            'symbol': 'TEST',
            'action': 'BUY',
            'quantity': 10,
            'executed_price': 100.0,
            'market_price': 100.0,
            'timestamp': 1234567890
        }
        self.pm.update_portfolio(buy_result)
        self.pm.set_current_prices({'TEST': 100.0})
        
        sell_result = {
            'success': True,
            'symbol': 'TEST',
            'action': 'SELL',
            'quantity': 10,
            'executed_price': 120.0,
            'market_price': 120.0,
            'timestamp': 1234567891
        }
        self.pm.update_portfolio(sell_result)
        
        # Check sold_history
        sold_history = self.pm.get_sold_history(limit=1)
        assert len(sold_history) == 1, "Should have one sold record"
        
        sold_record = sold_history[0]
        expected_pnl = 120.0 * 10 * (1 - 0.0025) - 100.0 * 10 * (1 + 0.0025)
        
        assert abs(sold_record['net_profit'] - expected_pnl) < 0.01, \
            f"Sold history P&L should be {expected_pnl:.4f}, got {sold_record['net_profit']:.4f}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
