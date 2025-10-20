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
import websocket
import json
import threading
import time


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
        self.ws = None
        self.ws_thread = None
        self.subscribed_symbols = []
        self.running = False
    
    def start_websocket(self, symbols: List[str]) -> None:
        """
        Startar WebSocket-anslutning och prenumererar på symboler.
        
        Args:
            symbols: Lista med symboler att prenumerera på
        """
        if self.running:
            print("⚠️  WebSocket already running")
            return
        
        self.subscribed_symbols = symbols
        self.running = True
        
        # Start WebSocket in background thread
        self.ws_thread = threading.Thread(target=self._run_websocket, daemon=True)
        self.ws_thread.start()
        print(f"📡 Started Finnhub WebSocket for {len(symbols)} symbols")
    
    def _run_websocket(self):
        """Run WebSocket connection in background."""
        websocket_url = f"wss://ws.finnhub.io?token={self.api_key}"
        
        def on_message(ws, message):
            """Handle incoming WebSocket messages."""
            try:
                data = json.loads(message)
                if data.get('type') == 'trade':
                    # Finnhub sends trade data
                    for trade in data.get('data', []):
                        symbol = trade.get('s')
                        price = trade.get('p')
                        volume = trade.get('v', 0)
                        timestamp = trade.get('t', time.time())
                        
                        if symbol and price:
                            # Publish to message bus
                            self.message_bus.publish('market_data', {
                                'symbol': symbol,
                                'price': price,
                                'volume': volume,
                                'timestamp': timestamp
                            })
            except Exception as e:
                print(f"Error processing WebSocket message: {e}")
        
        def on_error(ws, error):
            """Handle WebSocket errors."""
            print(f"WebSocket error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            """Handle WebSocket close."""
            print(f"WebSocket closed: {close_status_code} - {close_msg}")
            self.running = False
        
        def on_open(ws):
            """Handle WebSocket open - subscribe to symbols."""
            print(f"✅ WebSocket connected, subscribing to {len(self.subscribed_symbols)} symbols...")
            for symbol in self.subscribed_symbols:
                subscribe_message = {"type": "subscribe", "symbol": symbol}
                ws.send(json.dumps(subscribe_message))
                time.sleep(0.01)  # Small delay to avoid overwhelming the server
            print(f"📡 Subscribed to {len(self.subscribed_symbols)} symbols")
        
        try:
            self.ws = websocket.WebSocketApp(
                websocket_url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open
            )
            self.ws.run_forever()
        except Exception as e:
            print(f"WebSocket connection error: {e}")
            self.running = False
    
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
        # This is now handled by start_websocket
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
        self.running = False
        if self.ws:
            self.ws.close()
        self.active_streams.clear()
        print("🔌 WebSocket streams closed")

