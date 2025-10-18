"""
data_ingestion_sim.py - Simulated Data Ingestion for Demo Mode

Beskrivning:
    Simulerar realtidsmarknadsdata och publicerar via message_bus.
    Används i demo-läge istället för live WebSocket-anslutningar.
    Genererar realistiska prisrörelser med trend, volatilitet och mean reversion.

Roll:
    - Simulerar marknadsdata för demo-läge
    - Publicerar market_data till message_bus
    - Emulerar samma API som data_ingestion.py

Inputs:
    - symbols: List[str] - Symboler att simulera
    - message_bus: MessageBus - Central kommunikation

Outputs:
    - market_data: Dict - Simulerad marknadsdata (price, volume, timestamp)

Publicerar till message_bus:
    - market_data: Dict med symbol, timestamp, price, volume, change

Prenumererar på:
    - Ingen (entry point för demo-läge)

Använder RL: Nej
Tar emot feedback: Nej

Anslutningar:
    Från: Intern simulering
    Till: indicator_registry (via market_data)
    
Används i: Demo-läge (start_demo.py)
"""

import time
import random
from typing import Dict, List, Any
from datetime import datetime


class DataIngestionSim:
    """Simulerar datainsamling för demo-läge med realistiska marknadsrörelser."""
    
    def __init__(self, message_bus, symbols: List[str] = None):
        """
        Initialiserar simulerad datainsamling.
        
        Args:
            message_bus: Referens till central message_bus
            symbols: Lista med symboler att simulera (default: AAPL, MSFT, GOOGL, AMZN, TSLA)
        """
        self.message_bus = message_bus
        self.symbols = symbols or ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        
        # Base prices for realistic starting points
        self.base_prices = {
            'AAPL': 150.0,
            'MSFT': 300.0,
            'GOOGL': 120.0,
            'AMZN': 140.0,
            'TSLA': 200.0
        }
        
        # Current simulated prices
        self.current_prices = self.base_prices.copy()
        
        # Price trends (momentum)
        self.price_trends = {symbol: 0.0 for symbol in self.symbols}
        
        # Volume simulation
        self.base_volumes = {
            'AAPL': 50000000,
            'MSFT': 30000000,
            'GOOGL': 25000000,
            'AMZN': 40000000,
            'TSLA': 80000000
        }
        
        # Iteration counter
        self.iteration = 0
        
        print(f"📊 Simulated data ingestion initialized for symbols: {', '.join(self.symbols)}")
    
    def simulate_market_tick(self) -> None:
        """
        Simulerar ett market tick (prisuppdatering) för alla symboler.
        Publicerar uppdaterad data till message_bus.
        
        Använder realistisk prismodell med:
        - Trend/momentum
        - Mean reversion
        - Random volatility
        - Volume variations
        """
        self.iteration += 1
        
        for symbol in self.symbols:
            # Get current state
            current_price = self.current_prices[symbol]
            base_price = self.base_prices[symbol]
            trend = self.price_trends[symbol]
            
            # Market dynamics
            # 1. Trend/momentum (0.3 weight)
            momentum = trend * 0.3
            
            # 2. Mean reversion (pull towards base price, 0.01 weight)
            mean_reversion = (base_price - current_price) / base_price * 0.01
            
            # 3. Random volatility (1.5% standard deviation)
            volatility = random.gauss(0, 0.015)
            
            # 4. Occasional price jumps (news events, etc.)
            if random.random() < 0.02:  # 2% chance of jump
                volatility += random.choice([-0.03, 0.03])  # ±3% jump
            
            # Combine all factors
            price_change_pct = momentum + mean_reversion + volatility
            
            # Update price
            new_price = current_price * (1 + price_change_pct)
            self.current_prices[symbol] = new_price
            
            # Update trend (random walk)
            self.price_trends[symbol] += random.gauss(0, 0.1)
            # Limit trend to avoid runaway prices
            self.price_trends[symbol] = max(min(self.price_trends[symbol], 0.02), -0.02)
            
            # Simulate volume with variability
            base_volume = self.base_volumes.get(symbol, 10000000)
            volume = int(base_volume * (1 + random.gauss(0, 0.3)))
            volume = max(volume, int(base_volume * 0.5))  # Minimum 50% of base
            
            # Calculate price change percentage
            price_change = ((new_price - base_price) / base_price) * 100
            
            # Publish market data to message bus
            self.publish_market_data(symbol, {
                'price': new_price,
                'volume': volume,
                'timestamp': time.time(),
                'change': price_change,
                'high': new_price * 1.005,  # Simulated high
                'low': new_price * 0.995,   # Simulated low
                'open': current_price,      # Previous price as "open"
                'iteration': self.iteration
            })
    
    def publish_market_data(self, symbol: str, data: Dict[str, Any]) -> None:
        """
        Publicerar simulerad marknadsdata till message_bus.
        
        Args:
            symbol: Aktiesymbol
            data: Marknadsdata (price, volume, timestamp, etc.)
        """
        self.message_bus.publish('market_data', {
            'symbol': symbol,
            'data': data,
            'source': 'simulation'
        })
    
    def get_current_prices(self) -> Dict[str, float]:
        """
        Returnerar nuvarande priser för alla symboler.
        
        Returns:
            Dict mapping symbol -> current price
        """
        return self.current_prices.copy()
    
    def get_symbols(self) -> List[str]:
        """
        Returnerar lista med aktiva symboler.
        
        Returns:
            Lista med symboler som simuleras
        """
        return self.symbols.copy()
    
    def reset_prices(self) -> None:
        """Återställer priser till base values."""
        self.current_prices = self.base_prices.copy()
        self.price_trends = {symbol: 0.0 for symbol in self.symbols}
        self.iteration = 0
        print("🔄 Market prices reset to base values")
