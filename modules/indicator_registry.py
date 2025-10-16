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

Indikatorer från indicator_map.yaml för Sprint 2:
    Tekniska:
    - MACD: momentum and trend strength
    - ATR: volatility-based risk adjustment
    Fundamentala:
    - Analyst Ratings: external confidence and sentiment

Uppdateringsintervall: 5 min (från indicator_map.yaml)

Används i Sprint: 1, 2, 7
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
            Dict med tekniska indikatorer (OHLC, Volume, SMA, RSI, MACD, ATR, etc.)
        """
        # Stub: Skulle hämta från Finnhub API
        # För demo, variera RSI baserat på symbol för att visa olika scenarier
        rsi_values = {
            'AAPL': 65.0,  # Neutral
            'TSLA': 25.0,  # Översåld - köpsignal
            'MSFT': 75.0   # Överköpt - säljsignal
        }
        
        # Sprint 2: Lägg till MACD och ATR
        macd_values = {
            'AAPL': {'macd': 1.5, 'signal': 1.2, 'histogram': 0.3},
            'TSLA': {'macd': -2.1, 'signal': -1.5, 'histogram': -0.6},
            'MSFT': {'macd': 2.8, 'signal': 2.5, 'histogram': 0.3}
        }
        
        atr_values = {
            'AAPL': 2.5,   # Medelhög volatilitet
            'TSLA': 8.5,   # Hög volatilitet
            'MSFT': 1.8    # Låg volatilitet
        }
        
        return {
            'OHLC': {'open': 150.0, 'high': 152.0, 'low': 149.0, 'close': 151.0},
            'Volume': 1000000,
            'SMA': {'SMA_20': 150.5, 'SMA_50': 148.0},
            'RSI': rsi_values.get(symbol, 50.0),
            'MACD': macd_values.get(symbol, {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0}),
            'ATR': atr_values.get(symbol, 2.0)
        }
    
    def fetch_fundamental_indicators(self, symbol: str) -> Dict[str, Any]:
        """
        Hämtar fundamentala indikatorer från Finnhub.
        
        Args:
            symbol: Aktiesymbol
            
        Returns:
            Dict med fundamentala indikatorer (EPS, ROE, ROA, Analyst Ratings, etc.)
        """
        # Stub: Skulle hämta från Finnhub API
        
        # Sprint 2: Lägg till Analyst Ratings
        analyst_ratings = {
            'AAPL': {'buy': 25, 'hold': 10, 'sell': 2, 'consensus': 'BUY', 'target_price': 180.0},
            'TSLA': {'buy': 15, 'hold': 15, 'sell': 8, 'consensus': 'HOLD', 'target_price': 250.0},
            'MSFT': {'buy': 30, 'hold': 5, 'sell': 1, 'consensus': 'STRONG_BUY', 'target_price': 350.0}
        }
        
        return {
            'EPS': 5.2,
            'ROE': 0.18,
            'ROA': 0.12,
            'ProfitMargin': 0.25,
            'AnalystRatings': analyst_ratings.get(symbol, {
                'buy': 10, 'hold': 10, 'sell': 5, 'consensus': 'HOLD', 'target_price': 150.0
            })
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

