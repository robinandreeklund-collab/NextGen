"""
data_ingestion.py - Datainsamling från Finnhub

Beskrivning:
    Hämtar trending symboler och öppnar WebSocket-strömmar från Finnhub för realtidsdata.
    Första modulen i dataflödet som förser systemet med marknadsdata.

Roll:
    - Hämtar trending symboler från Finnhub API
    - Öppnar och hanterar WebSocket-anslutningar för realtidsdata
    - Publicerar market_data till message_bus

Inputs:
    - api_key: str - Finnhub API-nyckel
    - symbol_filters: List[str] - Filter för vilka symboler som ska hämtas

Outputs:
    - streamed_data: Dict - Strömmar marknadsdata (OHLC, volym, etc.)

Publicerar till message_bus:
    - market_data: Dict med symbol, timestamp, price, volume

Prenumererar på:
    - Ingen (entry point från flowchart.yaml)

Använder RL: Nej
Tar emot feedback: Nej

Anslutningar (från flowchart.yaml):
    Från: Finnhub (extern källa)
    Till: indicator_registry (via market_data)

Indikatorer från indicator_map.yaml:
    Tillhandahåller rådata för:
    - OHLC (Open, High, Low, Close)
    - Volume

Används i Sprint: 1
"""

from typing import Dict, List, Any


class DataIngestion:
    """Hanterar datainsamling från Finnhub API och WebSocket."""
    
    def __init__(self, api_key: str, message_bus):
        """
        Initialiserar datainsamlingsmodulen.
        
        Args:
            api_key: Finnhub API-nyckel
            message_bus: Referens till central message_bus
        """
        self.api_key = api_key
        self.message_bus = message_bus
        self.active_streams: Dict[str, Any] = {}
    
    def fetch_trending_symbols(self, filters: List[str] = None) -> List[str]:
        """
        Hämtar trending symboler från Finnhub.
        
        Args:
            filters: Valfria filter för symboler
            
        Returns:
            Lista med symboler att handla
        """
        # Stub: Returnerar exempel-symboler för Sprint 1
        return ["AAPL", "TSLA", "MSFT"]
    
    def open_websocket(self, symbol: str) -> None:
        """
        Öppnar WebSocket-ström för en symbol.
        
        Args:
            symbol: Aktiesymbol att starta ström för
        """
        # Stub: Skulle öppna WebSocket och publicera market_data
        pass
    
    def publish_market_data(self, symbol: str, data: Dict[str, Any]) -> None:
        """
        Publicerar marknadsdata till message_bus.
        
        Args:
            symbol: Aktiesymbol
            data: Marknadsdata (price, volume, timestamp)
        """
        self.message_bus.publish('market_data', {
            'symbol': symbol,
            'data': data
        })
    
    def close_streams(self) -> None:
        """Stänger alla aktiva WebSocket-strömmar."""
        # Stub: Skulle stänga alla WebSocket-anslutningar
        self.active_streams.clear()

