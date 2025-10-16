# Sprint-dokumentation

Detaljerad beskrivning av alla sprintar i NextGen AI Trader-projektet.

---

## Sprint 1: Kärnsystem och demoportfölj ✅

**Status:** FÄRDIG

**Mål:** Bygg ett fungerande end-to-end-flöde med verkliga data, strategi, beslut, exekvering och portfölj.

### Implementerade moduler:
- ✅ `data_ingestion.py` - Hämtar trending symboler och öppnar WebSocket
- ✅ `indicator_registry.py` - Distribuerar OHLC, Volume, SMA, RSI
- ✅ `strategy_engine.py` - RSI-baserade tradeförslag
- ✅ `decision_engine.py` - Beslutsfattande med strategi och risk
- ✅ `execution_engine.py` - Simulerad execution med slippage (0-0.5%)
- ✅ `portfolio_manager.py` - 1000 USD startkapital, 0.25% transaktionsavgift
- ✅ `message_bus.py` - Central pub/sub-kommunikation

### Indikatorer i Sprint 1:
- OHLC (Open, High, Low, Close) - Prisanalys
- Volume - Likviditetsbedömning
- SMA (Simple Moving Average) - Trenddetektering
- RSI (Relative Strength Index) - Överköpt/översålt analys

### Testresultat:
- ✅ Simulerad handel fungerar korrekt
- ✅ Portföljstatus uppdateras efter trades
- ✅ Indikatorflöde från Finnhub fungerar
- ✅ Message bus hanterar 12+ meddelanden över 9 topics
- ✅ RSI-signaler genererar korrekta BUY/SELL-förslag

### Summering:
Sprint 1 levererade ett komplett grundsystem med alla kärnmoduler implementerade. Systemet kan:
- Hämta marknadsdata och indikatorer
- Generera tradeförslag baserat på tekniska indikatorer
- Fatta beslut och exekvera trades i simuleringsläge
- Hantera portfölj med korrekt kapitaltillstånd och avgifter
- Kommunicera mellan moduler via message_bus

Alla 21 moduler är skapade med svenska docstrings och följer projekt-strukturen. Feedbackloopar och RL-grund är på plats för Sprint 2.

---

## Sprint 2: RL och belöningsflöde (PÅGÅR)

**Status:** PÅGÅR

**Mål:** Inför PPO-agenter i strategi, risk och beslut. Belöning via portfölj.

### Moduler i fokus:
- `rl_controller.py` - PPO-agentträning och distribution
- `strategy_engine.py` - RL-optimerad strategigenerering
- `risk_manager.py` - RL-baserad riskbedömning
- `decision_engine.py` - RL-förbättrat beslutsfattande
- `portfolio_manager.py` - Reward-generering för RL

### Nya indikatorer i Sprint 2:
- MACD (Moving Average Convergence Divergence) - Momentum och trendstyrka
- ATR (Average True Range) - Volatilitetsbaserad riskjustering
- Analyst Ratings - Extern confidence och sentiment

### Testbara mål:
- [ ] RL-belöning beräknas korrekt från portfolio
- [ ] PPO-agenter tränas i rl_controller
- [ ] Agentuppdateringar distribueras till moduler
- [ ] Reward-trender loggas och visualiseras
- [ ] Strategier förbättras över tid med RL

---
