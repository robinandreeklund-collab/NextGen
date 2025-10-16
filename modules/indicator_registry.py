"""
indicator_registry.py - Indikatorregister från Finnhub

Beskrivning:
    Hämtar och distribuerar tekniska, fundamentala och alternativa indikatorer från Finnhub.
    Centralt nav för all indikatordata i systemet med 5 minuters uppdateringsintervall.

Roll:
    - Hämtar tekniska indikatorer (OHLC, RSI, MACD, SMA, etc.)
    - Hämtar fundamentala indikatorer (EPS, ROE, ROA, etc.)
    - Hämtar alternativa indikatorer (News Sentiment, ESG, etc.)
    - Distribuerar indicator_data till relevanta moduler via message_bus

Inputs:
    - symbol: str - Aktiesymbol att hämta indikatorer för
    - timeframe: str - Tidsram för tekniska indikatorer (ex: '1D', '1H')

Outputs:
    - indicators: Dict - Samlad indikatordata för symbol

Publicerar till message_bus:
    - indicator_data: Dict med alla indikatorer för en symbol

Prenumererar på:
    - Ingen (entry point från flowchart.yaml)

Använder RL: Nej
Tar emot feedback: Nej

Anslutningar (från flowchart.yaml - indicator_flow):
    Från: Finnhub API
    Till: 
    - strategy_engine
    - risk_manager
    - decision_engine
    - strategic_memory_engine
    - introspection_panel

Indikatorer från indicator_map.yaml för Sprint 1:
    Tekniska:
    - OHLC: price analysis, entry/exit signals
    - Volume: liquidity assessment, volatility detection
    - SMA: trend detection, smoothing
    - RSI: overbought/oversold detection

Uppdateringsintervall: 5 min (från indicator_map.yaml)

Används i Sprint: 1, 7
"""

from typing import Dict, Any


class IndicatorRegistry:
    """Hanterar hämtning och distribution av indikatorer från Finnhub."""
    
    def __init__(self, api_key: str, message_bus):
        """
        Initialiserar indikatorregistret.
        
        Args:
            api_key: Finnhub API-nyckel
            message_bus: Referens till central message_bus
        """
        self.api_key = api_key
        self.message_bus = message_bus
        self.update_interval = 300  # 5 minuter i sekunder
        self.cached_indicators: Dict[str, Dict[str, Any]] = {}
    
    def fetch_technical_indicators(self, symbol: str, timeframe: str = '1D') -> Dict[str, Any]:
        """
        Hämtar tekniska indikatorer från Finnhub.
        
        Args:
            symbol: Aktiesymbol
            timeframe: Tidsram för indikatorer
            
        Returns:
            Dict med tekniska indikatorer (OHLC, Volume, SMA, RSI, etc.)
        """
        # Stub: Skulle hämta från Finnhub API
        # För demo, variera RSI baserat på symbol för att visa olika scenarier
        rsi_values = {
            'AAPL': 65.0,  # Neutral
            'TSLA': 25.0,  # Översåld - köpsignal
            'MSFT': 75.0   # Överköpt - säljsignal
        }
        
        return {
            'OHLC': {'open': 150.0, 'high': 152.0, 'low': 149.0, 'close': 151.0},
            'Volume': 1000000,
            'SMA': {'SMA_20': 150.5, 'SMA_50': 148.0},
            'RSI': rsi_values.get(symbol, 50.0)
        }
    
    def fetch_fundamental_indicators(self, symbol: str) -> Dict[str, Any]:
        """
        Hämtar fundamentala indikatorer från Finnhub.
        
        Args:
            symbol: Aktiesymbol
            
        Returns:
            Dict med fundamentala indikatorer (EPS, ROE, ROA, etc.)
        """
        # Stub: Skulle hämta från Finnhub API
        return {
            'EPS': 5.2,
            'ROE': 0.18,
            'ROA': 0.12,
            'ProfitMargin': 0.25
        }
    
    def fetch_alternative_indicators(self, symbol: str) -> Dict[str, Any]:
        """
        Hämtar alternativa indikatorer från Finnhub.
        
        Args:
            symbol: Aktiesymbol
            
        Returns:
            Dict med alternativa indikatorer (News Sentiment, ESG, etc.)
        """
        # Stub: Skulle hämta från Finnhub API
        return {
            'NewsSentiment': 0.65,
            'InsiderSentiment': 0.55,
            'ESG': {'total': 75, 'environmental': 80, 'social': 70, 'governance': 75}
        }
    
    def get_indicators(self, symbol: str, timeframe: str = '1D') -> Dict[str, Any]:
        """
        Hämtar alla indikatorer för en symbol och publicerar till message_bus.
        
        Args:
            symbol: Aktiesymbol
            timeframe: Tidsram för tekniska indikatorer
            
        Returns:
            Dict med alla indikatorer kombinerade
        """
        indicators = {
            'symbol': symbol,
            'timeframe': timeframe,
            'technical': self.fetch_technical_indicators(symbol, timeframe),
            'fundamental': self.fetch_fundamental_indicators(symbol),
            'alternative': self.fetch_alternative_indicators(symbol)
        }
        
        # Cacha indikatorer
        self.cached_indicators[symbol] = indicators
        
        # Publicera till message_bus
        self.message_bus.publish('indicator_data', indicators)
        
        return indicators
    
    def get_cached_indicators(self, symbol: str) -> Dict[str, Any]:
        """
        Hämtar cachade indikatorer för en symbol.
        
        Args:
            symbol: Aktiesymbol
            
        Returns:
            Cachad indikatordata eller None
        """
        return self.cached_indicators.get(symbol)

