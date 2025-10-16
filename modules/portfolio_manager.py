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
        self.message_bus = message_bus
        self.start_capital = start_capital
        self.cash = start_capital
        self.transaction_fee = transaction_fee
        self.positions: Dict[str, Dict[str, Any]] = {}  # symbol -> {quantity, avg_price}
        self.trade_history: List[Dict[str, Any]] = []
        self.previous_portfolio_value = start_capital
        
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
            return
            
        symbol = execution_result['symbol']
        action = execution_result['action']
        quantity = execution_result['quantity']
        executed_price = execution_result['executed_price']
        
        # Beräkna transaktionskostnad
        trade_value = executed_price * quantity
        fee = trade_value * self.transaction_fee
        
        if action == 'BUY':
            # Köp aktier
            total_cost = trade_value + fee
            if total_cost <= self.cash:
                self.cash -= total_cost
                
                # Uppdatera eller skapa position
                if symbol in self.positions:
                    # Uppdatera genomsnittspris
                    current_qty = self.positions[symbol]['quantity']
                    current_avg = self.positions[symbol]['avg_price']
                    new_qty = current_qty + quantity
                    new_avg = ((current_avg * current_qty) + (executed_price * quantity)) / new_qty
                    self.positions[symbol] = {
                        'quantity': new_qty,
                        'avg_price': new_avg
                    }
                else:
                    self.positions[symbol] = {
                        'quantity': quantity,
                        'avg_price': executed_price
                    }
            else:
                # Otillräckligt kapital - trade misslyckades
                return
        
        elif action == 'SELL':
            # Sälj aktier
            if symbol in self.positions and self.positions[symbol]['quantity'] >= quantity:
                revenue = trade_value - fee
                self.cash += revenue
                
                # Uppdatera position
                self.positions[symbol]['quantity'] -= quantity
                if self.positions[symbol]['quantity'] == 0:
                    del self.positions[symbol]
            else:
                # Otillräckligt innehav - trade misslyckades
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
    
    def calculate_and_publish_reward(self) -> None:
        """
        Beräknar reward för RL-controller och publicerar.
        Reward baserat på portfolio value change.
        Sprint 4.4: Publicerar base_reward istället för reward (går via RewardTunerAgent)
        """
        current_value = self.get_portfolio_value()
        reward = current_value - self.previous_portfolio_value
        self.previous_portfolio_value = current_value
        
        # Sprint 4.4: Publicera base_reward till reward_tuner (istället för direkt till rl_controller)
        # Debug: Print first 3 rewards to verify this code path is executed
        if len(self.trade_history) <= 3:
            print(f"[PortfolioManager] Publishing base_reward #{len(self.trade_history)}: {reward:.4f}")
        
        self.message_bus.publish('base_reward', {
            'reward': reward,
            'source': 'portfolio_manager',
            'portfolio_value': current_value,
            'num_trades': len(self.trade_history)
        })
    
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

