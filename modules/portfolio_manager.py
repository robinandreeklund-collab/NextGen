"""
portfolio_manager.py - Portföljhantering

Beskrivning:
    Hanterar demoportfölj med startkapital, uppdaterar positioner och beräknar reward.
    Central för kapitaltillstånd och performance tracking.

Roll:
    - Håller koll på kapital och positioner
    - Uppdaterar portfölj baserat på execution_result
    - Beräknar transaktionsavgifter (0.25% från sprint_plan.yaml)
    - Genererar reward för RL-controller
    - Publicerar portfolio_status för andra moduler
    - Genererar feedback om capital changes

Inputs:
    - execution_result: Dict - Resultat av exekverad trade från execution_engine

Outputs:
    - portfolio_status: Dict - Aktuell portföljstatus (kapital, positioner, värde)
    - reward: float - Belöning för RL-controller

Publicerar till message_bus:
    - portfolio_status: Uppdaterad portföljstatus
    - reward: RL-belöning

Prenumererar på (från functions.yaml):
    - execution_result (från execution_engine)

Använder RL: Nej (från functions.yaml)
Tar emot feedback: Ja (från risk_manager, strategic_memory)

Anslutningar (från flowchart.yaml - portfolio_flow):
    Från: execution_engine (execution_result)
    Till:
    - rl_controller (reward)
    - introspection_panel (portfolio_status)
    - strategy_engine (portfolio_status, via prenumeration)

Feedback-generering (från feedback_loop.yaml):
    Triggers:
    - capital_change: När totalt kapital ändras
    - transaction_cost: Kostnad för varje trade
    
    Reward sources (för rl_controller):
    - Portfolio value change
    - Trade profitability
    - Risk-adjusted returns

Startkapital: 1000 USD (från sprint_plan.yaml)
Transaktionsavgift: 0.25% (från sprint_plan.yaml)

Indikatorer från indicator_map.yaml:
    Använder:
    - VWAP: price fairness, execution quality
    - Dividend Yield: income generation and valuation

Används i Sprint: 1, 2, 3
"""

from typing import Dict, Any, List
import time


class PortfolioManager:
    """Hanterar demoportfölj, kapital och beräknar RL-reward."""
    
    def __init__(self, message_bus, start_capital: float = 1000.0, transaction_fee: float = 0.0025):
        """
        Initialiserar portföljhanteraren.
        
        Args:
            message_bus: Referens till central message_bus
            start_capital: Startkapital i USD (default 1000)
            transaction_fee: Transaktionsavgift som decimal (0.0025 = 0.25%)
        """
        self.reward_tuner_callback = None  # Sprint 4.4: Direct callback for RewardTunerAgent
        self.message_bus = message_bus
        self.start_capital = start_capital
        self.cash = start_capital
        self.transaction_fee = transaction_fee
        self.positions: Dict[str, Dict[str, Any]] = {}  # symbol -> {quantity, avg_price}
        self.trade_history: List[Dict[str, Any]] = []
        self.sold_history: List[Dict[str, Any]] = []  # Track sold stocks with P/L
        self.previous_portfolio_value = start_capital
        self.last_action = None  # Track last action for reward calculation (BUY/SELL/HOLD)
        
        # Prenumerera på execution_result
        self.message_bus.subscribe('execution_result', self._on_execution_result)
    
    def _on_execution_result(self, result: Dict[str, Any]) -> None:
        """
        Callback för execution result från execution_engine.
        
        Args:
            result: Resultat av exekverad trade
        """
        self.update_portfolio(result)
        self.publish_status()
        self.generate_feedback(result)
        self.calculate_and_publish_reward()
    
    def update_portfolio(self, execution_result: Dict[str, Any]) -> None:
        """
        Uppdaterar portföljen baserat på execution result.
        
        Args:
            execution_result: Trade execution result
        """
        # Ignorera misslyckade trades
        if not execution_result.get('success', False):
            self.last_action = None  # No action taken on failed trades
            return
            
        symbol = execution_result['symbol']
        action = execution_result['action']
        quantity = execution_result['quantity']
        executed_price = execution_result['executed_price']
        
        # Track the action for reward calculation
        self.last_action = action
        self.last_sell_pnl = 0.0  # Reset for each update
        
        # Beräkna transaktionskostnad
        trade_value = executed_price * quantity
        fee = trade_value * self.transaction_fee
        
        if action == 'BUY':
            # Köp aktier
            total_cost = trade_value + fee
            if total_cost <= self.cash:
                self.cash -= total_cost
                
                # Uppdatera eller skapa position
                # Include fee in average price calculation for accurate P&L on sell
                avg_price_with_fee = (trade_value + fee) / quantity
                
                if symbol in self.positions:
                    # Uppdatera genomsnittspris
                    current_qty = self.positions[symbol]['quantity']
                    current_avg = self.positions[symbol]['avg_price']
                    new_qty = current_qty + quantity
                    # Weight both positions by their total cost
                    new_avg = ((current_avg * current_qty) + (avg_price_with_fee * quantity)) / new_qty
                    self.positions[symbol] = {
                        'quantity': new_qty,
                        'avg_price': new_avg
                    }
                else:
                    self.positions[symbol] = {
                        'quantity': quantity,
                        'avg_price': avg_price_with_fee
                    }
            else:
                # Otillräckligt kapital - trade misslyckades
                self.last_action = None  # No action taken on failed trades
                return
        
        elif action == 'SELL':
            # Sälj aktier
            if symbol in self.positions and self.positions[symbol]['quantity'] >= quantity:
                revenue = trade_value - fee
                avg_buy_price = self.positions[symbol]['avg_price']
                
                # Calculate P/L and return %
                cost_basis = avg_buy_price * quantity
                gross_profit = trade_value - cost_basis
                net_profit = revenue - cost_basis  # After fees
                return_pct = (net_profit / cost_basis) * 100 if cost_basis > 0 else 0.0
                
                self.cash += revenue
                
                # Store P&L for reward calculation
                self.last_sell_pnl = net_profit
                
                # Track sold stock with details
                sold_record = {
                    'symbol': symbol,
                    'quantity': quantity,
                    'avg_buy_price': avg_buy_price,
                    'sell_price': executed_price,
                    'cost_basis': cost_basis,
                    'revenue': revenue,
                    'gross_profit': gross_profit,
                    'net_profit': net_profit,
                    'return_pct': return_pct,
                    'fee': fee,
                    'timestamp': execution_result.get('timestamp', time.time()),
                    'agent_decision': execution_result.get('agent', 'unknown')
                }
                self.sold_history.append(sold_record)
                
                # Keep only last 100 sold records
                if len(self.sold_history) > 100:
                    self.sold_history = self.sold_history[-100:]
                
                # Uppdatera position
                self.positions[symbol]['quantity'] -= quantity
                if self.positions[symbol]['quantity'] == 0:
                    del self.positions[symbol]
            else:
                # Otillräckligt innehav - trade misslyckades
                self.last_action = None  # No action taken on failed trades
                return
        
        # Spara current prices för portfolio value calculation
        if not hasattr(self, 'current_prices'):
            self.current_prices = {}
        self.current_prices[symbol] = execution_result.get('market_price', executed_price)
        
        # Logga trade
        self.trade_history.append({
            **execution_result,
            'fee': fee,
            'portfolio_value_after': self.get_portfolio_value(self.current_prices)
        })
        
        # Limit trade history to prevent memory leak (keep last 10000)
        if len(self.trade_history) > 10000:
            self.trade_history = self.trade_history[-10000:]
    
    def get_portfolio_value(self, current_prices: Dict[str, float] = None) -> float:
        """
        Beräknar totalt portföljvärde (cash + positions).
        
        Args:
            current_prices: Dict med aktuella priser för positioner (optional)
            
        Returns:
            Totalt portföljvärde i USD
        """
        total_value = self.cash
        
        # Använd sparade current prices om inte andra anges
        if current_prices is None and hasattr(self, 'current_prices'):
            current_prices = self.current_prices
        
        # Lägg till värdet av alla positioner
        for symbol, position in self.positions.items():
            if current_prices and symbol in current_prices:
                price = current_prices[symbol]
            else:
                # Använd average price om inget annat finns
                price = position['avg_price']
            
            total_value += position['quantity'] * price
        
        return total_value
    
    def set_current_prices(self, prices: Dict[str, float]) -> None:
        """
        Uppdaterar aktuella marknadspriser för positioner.
        
        Args:
            prices: Dict med symbol -> pris
        """
        if not hasattr(self, 'current_prices'):
            self.current_prices = {}
        self.current_prices.update(prices)
    
    def get_status(self, current_prices: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Hämtar aktuell portföljstatus.
        
        Args:
            current_prices: Dict med aktuella priser (optional)
            
        Returns:
            Dict med portfolio_status
        """
        # Använd current_prices om de finns
        if current_prices is None and hasattr(self, 'current_prices'):
            current_prices = self.current_prices
            
        total_value = self.get_portfolio_value(current_prices)
        
        return {
            'cash': self.cash,
            'positions': self.positions,
            'total_value': total_value,
            'start_capital': self.start_capital,
            'pnl': total_value - self.start_capital,
            'pnl_pct': ((total_value - self.start_capital) / self.start_capital) * 100,
            'num_trades': len(self.trade_history)
        }
    
    def publish_status(self) -> None:
        """Publicerar portföljstatus till message_bus."""
        status = self.get_status()
        self.message_bus.publish('portfolio_status', status)
    
    def get_sold_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Hämtar historik över sålda aktier.
        
        Args:
            limit: Max antal resultat att returnera (default 20)
            
        Returns:
            Lista med sålda aktier (senaste först)
        """
        if not hasattr(self, 'sold_history'):
            self.sold_history = []
        
        # Return most recent sells first
        return list(reversed(self.sold_history[-limit:]))
        
    def register_reward_tuner_callback(self, callback):
        """
        Register a direct callback for RewardTunerAgent to receive rewards.
        Sprint 4.4: Provides guaranteed delivery using callback pattern.
        
        Args:
            callback: Function that accepts base_reward data dict
        """
        self.reward_tuner_callback = callback
        print("[PortfolioManager] RewardTuner callback registered ✅")

    def calculate_and_publish_reward(self) -> None:
        """
        Beräknar reward för RL-controller och publicerar.
        
        Reward calculation based on action type:
        - BUY: reward = 0.0 (outcome unknown until position closed)
        - SELL: reward = actual P&L (profit/loss - fees)
        - HOLD or None: reward = 0.0 (no action taken)
        
        Sprint 4.4: Publicerar base_reward istället för reward (går via RewardTunerAgent)
        """
        # Calculate reward based on last action
        if self.last_action == 'BUY':
            # BUY always gives 0.0 reward (outcome unknown until position closed)
            reward = 0.0
        elif self.last_action == 'SELL':
            # SELL gives the actual P&L from the sale
            reward = self.last_sell_pnl if hasattr(self, 'last_sell_pnl') else 0.0
        else:
            # HOLD or no action gives 0.0 reward
            reward = 0.0
        
        # Reset last_action for next iteration
        self.last_action = None
        
        reward_data = {
            'reward': reward,
            'source': 'portfolio_manager',
            'portfolio_value': self.get_portfolio_value(),
            'num_trades': len(self.trade_history),
            'timestamp': time.time()
        }
        
        # Sprint 4.4: Use BOTH callback AND message bus for guaranteed delivery
        if len(self.trade_history) <= 3:
            print(f"[PortfolioManager] Publishing base_reward #{len(self.trade_history)}: {reward:.4f}")
        
        # 1. Direct callback (guaranteed delivery if registered)
        if self.reward_tuner_callback:
            self.reward_tuner_callback(reward_data)
        
        # 2. Message bus (for backward compatibility)
        self.message_bus.publish('base_reward', reward_data)
    
    def generate_feedback(self, execution_result: Dict[str, Any]) -> None:
        """
        Genererar feedback om capital changes (från feedback_loop.yaml).
        
        Args:
            execution_result: Trade execution result
        """
        feedback = {
            'source': 'portfolio_manager',
            'triggers': [],
            'data': {}
        }
        
        # Capital change feedback
        current_value = self.get_portfolio_value()
        capital_change = current_value - self.previous_portfolio_value
        if abs(capital_change) > 0:
            feedback['triggers'].append('capital_change')
            feedback['data']['capital_change'] = capital_change
            feedback['data']['portfolio_value'] = current_value
        
        # Transaction cost feedback
        trade_value = execution_result['executed_price'] * execution_result['quantity']
        fee = trade_value * self.transaction_fee
        feedback['triggers'].append('transaction_cost')
        feedback['data']['transaction_cost'] = fee
        
        # Publicera feedback
        self.message_bus.publish('feedback_event', feedback)
    
    def receive_feedback(self, feedback: Dict[str, Any]) -> None:
        """
        Tar emot feedback från risk_manager och strategic_memory.
        
        Args:
            feedback: Feedback om portfolio performance
        """
        # Stub: Implementeras fullt ut i Sprint 3
        pass

