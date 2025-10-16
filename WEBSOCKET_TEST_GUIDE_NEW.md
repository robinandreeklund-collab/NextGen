# NextGen WebSocket Live Test Guide - UPDATED

## Översikt

`websocket_test.py` testar hela NextGen AI Trader-systemet med live data från Finnhub WebSocket. Inkluderar alla Sprint 1-4 moduler med omfattande debug-diagnostik, live priskalkylering och realistisk portfolio management.

## Nya Funktioner (v2.0)

### Portfolio Management
- ✅ **Live price tracking**: Portfolio value beräknas med aktuella WebSocket-priser
- ✅ **Insufficient funds check**: BUY-beslut justeras automatiskt om otillräckligt kapital
- ✅ **Insufficient holdings check**: SELL-beslut justeras baserat på faktiska innehav  
- ✅ **Realistiska volymer**: Köp/sälj 1-3 aktier per trade (reducerat från 10)
- ✅ **Dynamic valuation**: Total portfolio value uppdateras med live market data

### Beslutsdistribution
Visar nu procentuell fördelning av BUY/SELL/HOLD beslut:
- 🟢 BUY: X%
- 🔴 SELL: X%
- ⚪ HOLD: X%

### Förbättrad Debug Output
- Visa ALLA positioner (inte bara top 3)
- P&L per position med live priser
- Current price vs average price comparison
- Totalt antal positioner

## Beslutskriterier

### BUY Signals (kräver minst 2 signaler)
- **RSI < 30** (stark): Översåld (2 poäng)
- **MACD > 0.5**: Positiv momentum (1 poäng)
- **Analyst Rating BUY/STRONG_BUY**: Positiv konsensus (1 poäng)

**Volume justeras baserat på ATR (volatilitet):**
- ATR > 5.0: 1 aktie (hög volatilitet)
- ATR > 3.0: 2 aktier (medium volatilitet)
- ATR ≤ 3.0: 3 aktier (låg volatilitet)

**Insufficient funds protection:**
- Om estimated cost > tillgängligt kapital: Justera quantity
- Om kan inte köpa ens 1 aktie: HOLD istället

### SELL Signals (kräver minst 2 signaler)
- **RSI > 70** (stark): Överköpt (2 poäng)
- **MACD < -0.5**: Negativ momentum (1 poäng)
- **Analyst Rating SELL**: Negativ konsensus (1 poäng)

**Volume:** Sälj upp till 3 aktier åt gången

**Insufficient holdings protection:**
- Om quantity > ägda aktier: Justera till allt vi äger
- Om inga aktier att sälja: HOLD istället

### HOLD (Default)
- Inte tillräckligt starka signaler (< 2 signaler)
- Risk level för hög (HIGH + risk confidence > 0.7)
- Otillräckligt kapital för BUY
- Inga aktier att sälja för SELL

## Användning

### Starta testet
```bash
python websocket_test.py
```

### Stoppa testet
Tryck `Ctrl+C` för graceful shutdown och komplett sammanfattning.

## Debug Output Exempel

```
📊 Trade #1: AAPL @ $150.25 (vol: 100)
   📈 Indikatorer hämtade för AAPL
      RSI: 28.5
      MACD: 0.8

🤔 Beslut #1 för AAPL:
   💡 Strategy proposal: BUY (confidence: 0.70)
      Reasoning: RSI översåld (28.5), MACD positiv (0.80), Analystconsensus: BUY
   ⚠️  Risk assessment: LOW (score: 0.30)
   ⚖️  Final decision: BUY (confidence: 0.84)

   🔨 EXECUTION #1:
      BUY 3 AAPL
      @ $150.32 (market: $150.25)
      Cost: $450.96
      Slippage: 0.047%
   💰 Portfolio: $549.04 cash, $1000.00 total
   📊 Innehav (1 positioner):
      AAPL: 3 @ avg $150.32 (nuv: $150.25, P&L: $-0.21)
```

## Slutlig Sammanfattning

Vid avslut visas:

### 📊 Beslutsdistribution (NYT!)
- Antal och procent BUY/SELL/HOLD beslut
- Visar hur aggressiv trading-strategin är

### 💰 Portfolio Resultat (FÖRBÄTTRAT!)
- Start kapital och slutligt värde (med live prices)
- Profit/Loss och ROI %
- **Slutliga innehav**: Alla positioner med:
  - Quantity och average price
  - Current price (från WebSocket)
  - P&L per position med faktiska prisrörelser

### 🧠 Strategic Memory
- Totalt beslut loggade
- Success rate %
- Genomsnittlig profit
- Bästa indikatorer med success rate
- Rekommendationer baserat på historisk data

### 🧬 Agent Evolution
- Total evolution events
- Evolution count per agent

### 🤖 Agent Versioner
- Nuvarande version för varje agent
- Antal versioner i historik

### 📊 Trades per Symbol
- Distribution av trades mellan symboler

## Troubleshooting

### "Inga beslut fattade!"
**Normal situation** - Systemet är selektivt och väntar på starka signaler.

**Kontrollera:**
1. Debug output visar beslutprocess - är alla HOLD pga dåliga signaler?
2. Portfolio cash - om $0, kan vi inte köpa mer (insufficient funds)
3. Innehav - om inga aktier, kan vi inte sälja

**Lösning:**
- Låt testet köra längre (5+ minuter)
- Kolla beslutsdistribution för att se HOLD%

### "Portfolio värde oförändrat"
**FIXAT!** Portfolio värde beräknas nu med live WebSocket-priser.

Tidigare problem: Placeholder-priser istället för aktuella.
Nuvarande: Alla trades sparar current market price, portfolio value uppdateras live.

### "För många trades, inga executions"
**FIXAT!** Volymer reducerade till 1-3 aktier.

Tidigare problem: Försökte köpa 10 aktier per trade.
Nuvarande: 
- BUY: 1-3 aktier beroende på volatilitet
- SELL: Max 3 aktier
- Insufficient funds check innan execution

## Tips

- **Långsiktig körning**: Kör minst 5-10 minuter för mönster
- **Marknadstider**: Testa under NYSE handelstider (09:30-16:00 ET)
- **Portfolio Tracking**: Följ cash level - när den är låg blir fler HOLD
- **Beslutsdistribution**: Kolla BUY/SELL/HOLD % för att förstå trading-stil
- **Live Prices**: Portfolio P&L ändras baserat på faktiska prisrörelser

## Stub Data - BORTTAGET

Alla stub-värden har tagits bort från systemet:
- ✅ Execution Engine: Använder timestamp istället för 'timestamp_placeholder'
- ✅ Portfolio Manager: Använder live prices, inte placeholder
- ✅ Strategy Engine: Dynamiska volymer baserat på volatilitet
- ✅ Decision Engine: Insufficient funds/holdings checks
- ✅ Vote Engine: Använder datetime för timestamps
- ✅ RL Controller: Kommentarer uppdaterade
- ✅ Consensus Engine: Kommentarer uppdaterade

Alla moduler använder nu dynamisk data!