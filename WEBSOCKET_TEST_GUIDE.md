# WebSocket Test - Live Trading System Test

## Översikt

`websocket_test.py` är ett live-test av hela NextGen AI Trader-systemet som använder realtidsdata från Finnhub WebSocket API.

## Funktioner

- ✅ **Live WebSocket-anslutning** till Finnhub
- ✅ **10 fasta NASDAQ 100-aktier** (AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, NFLX, AMD, INTC)
- ✅ **Sprint 1-4 moduler** fullständigt integrerade
- ✅ **Realtidsanalys** av strategiskt minne, agent evolution och beslut
- ✅ **Live statistik** uppdateras var 10:e meddelande
- ✅ **Graceful shutdown** med komplett sammanfattning

## Installation

Kör från projektets root-katalog:

```bash
# Installera beroenden (om inte redan gjort)
pip install -r requirements.txt

# Kör websocket-testet
python websocket_test.py
```

## Användning

### Starta testet

```bash
python websocket_test.py
```

### Exempel på output

```
================================================================================
🚀 NextGen AI Trader - Live WebSocket Test
================================================================================

📡 Ansluter till Finnhub WebSocket...
🎯 Symboler: AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, NFLX, AMD, INTC
💰 Start kapital: $1000.00
💵 Transaktionsavgift: 0.25%

✅ WebSocket-anslutning etablerad!
📡 Prenumererar på 10 symboler...
   ✓ AAPL
   ✓ MSFT
   ... (och så vidare)

🚀 Live trading-systemet körs nu!
⏹️  Tryck Ctrl+C för att stoppa och visa sammanfattning

================================================================================
⏱️  Runtime: 1m 23s
📨 Meddelanden: 450
💹 Trades processade: 124
🎯 Beslut fattade: 12
🧬 Evolution events: 2

💰 Portfolio:
   Kapital: $985.50
   Värde: $1015.75
   P&L: $15.75

📊 Mest aktiva symboler:
   AAPL: 35 trades
   TSLA: 28 trades
   MSFT: 22 trades

🧠 Strategic Memory:
   Beslut: 12
   Success rate: 66.7%
   Bästa indikator: RSI (75.0%)

🤖 Agenter:
   Strategy Agent: v1.0.1
   Risk Management Agent: v1.0.0
================================================================================
```

### Stoppa testet

Tryck `Ctrl+C` för att avsluta gracefully. Systemet visar då en komplett sammanfattning:

```
📊 SLUTLIG SAMMANFATTNING
================================================================================

⏱️  Total körtid: 15m 42s
📨 Totalt meddelanden: 8,542
💹 Totalt trades: 2,156
🎯 Totalt beslut: 215
🧬 Evolution events: 18

💰 PORTFOLIO RESULTAT:
   Start kapital: $1000.00
   Slutligt värde: $1,127.50
   Profit/Loss: $127.50
   ROI: 12.75%

🧠 STRATEGIC MEMORY:
   Totalt beslut: 215
   Success rate: 64.2%
   Genomsnittlig profit: $0.59

   📈 Bästa indikatorer:
      1. RSI: 72.3% success, $0.85 avg profit
      2. MACD: 68.5% success, $0.72 avg profit
      3. ATR: 61.2% success, $0.45 avg profit

   💡 Rekommendationer:
      - Fokusera på indikatorer: RSI, MACD, ATR
      - Minska vikt på indikatorer: Volume, SMA

🧬 AGENT EVOLUTION:
   Total evolution events: 18
   strategy_agent: 5 evolutioner
   risk_agent: 3 evolutioner
   decision_agent: 4 evolutioner

🤖 AGENT VERSIONER:
   Strategy Agent: v1.0.5 (6 versioner)
   Risk Management Agent: v1.0.3 (4 versioner)
   Decision Agent: v1.0.4 (5 versioner)
   Execution Agent: v1.0.0 (1 versioner)

📊 TRADES PER SYMBOL:
   AAPL: 285 trades (13.2%)
   TSLA: 265 trades (12.3%)
   NVDA: 242 trades (11.2%)
   ... (och så vidare)
```

## Vad testas?

### Sprint 1 - Kärnsystem
- ✅ Market data ingestion från WebSocket
- ✅ Indicator registry (ROE, ROA, ESG, Earnings Calendar)
- ✅ Strategy engine med RL-agenter
- ✅ Risk manager med volatilitetsanalys
- ✅ Decision engine för slutgiltiga beslut
- ✅ Execution engine med slippage simulation
- ✅ Portfolio manager med transaktionsavgifter

### Sprint 2 - RL och belöningsflöde
- ✅ RL controller med PPO-agenter
- ✅ Agent träning baserat på portfolio performance
- ✅ Reward distribution till alla RL-moduler

### Sprint 3 - Feedbackloopar
- ✅ Feedback router med prioritering
- ✅ Feedback analyzer med mönsterdetektering
- ✅ Introspection panel för dashboard-data

### Sprint 4 - Strategiskt minne och evolution
- ✅ **Strategic Memory Engine** - Beslutshistorik och korrelationsanalys
- ✅ **Meta Agent Evolution Engine** - Performance degradation detection
- ✅ **Agent Manager** - Versionshantering och profiler
- ✅ **Evolution Tree** - Spårning av agent-utveckling
- ✅ **Best/Worst Indicators** - Data-driven insights

## API-begränsningar

- **REST API**: Max 60 calls/min (används inte av websocket_test.py)
- **WebSocket**: Max 50 symboler samtidigt (vi använder 10)
- **Rate limiting**: Inget för WebSocket-meddelanden

## Statistik som spåras

| Metrik | Beskrivning |
|--------|-------------|
| `messages_received` | Totalt antal WebSocket-meddelanden |
| `trades_processed` | Antal trades processade genom systemet |
| `decisions_made` | Antal handelsbeslut fattade |
| `evolution_events` | Antal agent evolution-händelser |
| `symbol_counts` | Trades per symbol |
| `runtime` | Total körtid |

## Moduler som testas

Alla dessa moduler körs live under testet:

1. **MessageBus** - Central pub/sub-kommunikation
2. **StrategicMemoryEngine** - Loggar och analyserar beslut ⭐ Sprint 4
3. **MetaAgentEvolutionEngine** - Evolutionär agentutveckling ⭐ Sprint 4
4. **AgentManager** - Versionshantering av agenter ⭐ Sprint 4
5. **FeedbackRouter** - Intelligent feedback-routing
6. **FeedbackAnalyzer** - Mönsteranalys
7. **IntrospectionPanel** - Dashboard-data
8. **RLController** - PPO agent träning
9. **IndicatorRegistry** - Tekniska och fundamentala indikatorer
10. **StrategyEngine** - Tradeförslag
11. **RiskManager** - Riskbedömning
12. **DecisionEngine** - Slutgiltiga beslut
13. **ExecutionEngine** - Trade execution
14. **PortfolioManager** - Kapitalhantering

## Felsökning

### WebSocket-anslutning misslyckas

```bash
# Verifiera API-nyckel
echo "API Key: d3in10hr01qmn7fkr2a0d3in10hr01qmn7fkr2ag"

# Testa anslutning manuellt
wscat -c "wss://ws.finnhub.io?token=d3in10hr01qmn7fkr2a0d3in10hr01qmn7fkr2ag"
```

### Inga trades mottas

- Kontrollera att marknaden är öppen (US market hours: 09:30-16:00 EST)
- NASDAQ 100-aktier har högre aktivitet under handelstider
- Testa med en bredare lista av symboler om nödvändigt

### Import-fel

```bash
# Installera alla beroenden
pip install -r requirements.txt

# Verifiera installation
python -c "import websocket_test; print('OK')"
```

## Framtida förbättringar

- [ ] Dash dashboard för live visualisering
- [ ] Loggning till fil för analys
- [ ] Konfigurerbar symbol-lista
- [ ] REST API integration för historisk data
- [ ] Performance benchmarking
- [ ] Multi-threading för högre throughput

## Relaterade filer

- `modules/data_ingestion.py` - WebSocket grund-struktur
- `dashboards/feedback_flow.py` - Dashboard-inspiration
- `demo_sprint3.py` - Sprint 3 demo
- `tests/test_strategic_memory_engine.py` - Strategic memory tests

## Support

För frågor eller problem, kontakta utvecklingsteamet eller öppna en issue i GitHub-repot.

---

**Skapad**: 2025-10-16  
**Sprint**: 4 - Strategiskt minne och agentutveckling  
**Version**: 1.0.0
