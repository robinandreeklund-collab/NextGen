# 🚀 NextGen AI Trader

Ett självreflekterande, modulärt och RL-drivet handelssystem byggt för transparens, agentutveckling och realtidsanalys. Systemet simulerar handel med verkliga data, strategier, feedbackloopar och belöningsbaserad inlärning.

---

## 📍 Sprintstatus

**Sprint 1 pågår** – Bygger kärnsystem och demoportfölj

### Sprintplan - Sprint 1: Kärnsystem och demoportfölj

**Mål:** Bygg ett fungerande end-to-end-flöde med verkliga data, strategi, beslut, exekvering och portfölj.

**Moduler i fokus:**
- `data_ingestion` - Hämtar trending symboler och öppnar WebSocket
- `strategy_engine` - Genererar tradeförslag baserat på indikatorer
- `decision_engine` - Samlar insikter och fattar beslut
- `execution_engine` - Simulerar eller exekverar trades
- `portfolio_manager` - Hanterar demoportfölj med startkapital (1000 USD) och avgifter (0.25%)
- `indicator_registry` - Hämtar och distribuerar indikatorer från Finnhub

**Indikatorer som används:**
- OHLC (Open, High, Low, Close)
- Volume (Volym)
- SMA (Simple Moving Average)
- RSI (Relative Strength Index)

**Testbara mål:**
- ✅ Simulerad handel fungerar
- ✅ Portföljstatus uppdateras korrekt
- ✅ Indikatorflöde från Finnhub fungerar

**Startkapital:** 1000 USD  
**Transaktionsavgift:** 0.25%

---

## 🔄 Sprint 1: Systemflöde och Arkitektur

### Dataflöde och Modulanslutningar

Sprint 1 implementerar ett komplett end-to-end handelssystem med följande flöde:

```
┌─────────────────┐
│    Finnhub      │
│   (Data källa)  │
└────────┬────────┘
         │
         ├──────────────────┐
         │                  │
         ▼                  ▼
┌──────────────────┐  ┌──────────────────┐
│ data_ingestion   │  │indicator_registry│
│  (Market data)   │  │  (Indikatorer)   │
└────────┬─────────┘  └────────┬─────────┘
         │                     │
         │                     └──────┐
         │                            ▼
         │                   ┌──────────────────┐
         │                   │ strategy_engine  │
         │                   │ (Tradeförslag)   │
         │                   └────────┬─────────┘
         │                            │
         │                            ▼
         │                   ┌──────────────────┐
         │                   │ decision_engine  │
         │                   │ (Slutgiltigt     │
         │                   │  beslut)         │
         │                   └────────┬─────────┘
         │                            │
         │                            ▼
         │                   ┌──────────────────┐
         │                   │ execution_engine │
         │                   │ (Exekvering)     │
         │                   └────────┬─────────┘
         │                            │
         │                            ▼
         │                   ┌──────────────────┐
         └──────────────────▶│portfolio_manager │
                             │ (Portföljstatus) │
                             └──────────────────┘
                                      │
                                      ▼
                             ┌──────────────────┐
                             │  message_bus     │
                             │  (Pub/Sub)       │
                             └──────────────────┘
```

### Modulbeskrivningar och Kopplingar

#### 1. **data_ingestion** (Entry Point)
- **Roll:** Hämtar marknadsdata från Finnhub via WebSocket
- **Publicerar:** `market_data` till message_bus
- **Används av:** Alla moduler som behöver realtidsdata

#### 2. **indicator_registry** (Entry Point)
- **Roll:** Hämtar och distribuerar tekniska indikatorer från Finnhub
- **Publicerar:** `indicator_data` till message_bus
- **Uppdateringsintervall:** 5 minuter
- **Indikatorer:** OHLC, Volume, SMA, RSI (Sprint 1)
- **Prenumeranter:** strategy_engine, decision_engine

#### 3. **strategy_engine**
- **Roll:** Genererar tradeförslag baserat på tekniska indikatorer
- **Prenumererar på:** `indicator_data`, `portfolio_status`
- **Publicerar:** `decision_proposal` till decision_engine
- **Indikatoranvändning:**
  - OHLC: Entry/exit signals
  - Volume: Liquidity assessment
  - SMA: Trend detection
  - RSI: Overbought/oversold (< 30 = köp, > 70 = sälj)

#### 4. **decision_engine**
- **Roll:** Fattar slutgiltiga handelsbeslut
- **Prenumererar på:** `decision_proposal`, `risk_profile`, `memory_insights`
- **Publicerar:** `final_decision` till execution_engine
- **Logik:** Kombinerar strategi med risk (Sprint 1: enkel logik, Sprint 2: RL)

#### 5. **execution_engine**
- **Roll:** Simulerar trade-exekvering med slippage
- **Prenumererar på:** `final_decision`
- **Publicerar:** `execution_result`, `trade_log`, `feedback_event`
- **Simulering:**
  - Slippage: 0-0.5%
  - Latency tracking
  - Execution quality feedback

#### 6. **portfolio_manager**
- **Roll:** Hanterar portfölj och beräknar reward
- **Prenumererar på:** `execution_result`
- **Publicerar:** `portfolio_status`, `reward`, `feedback_event`
- **Parametrar:**
  - Startkapital: 1000 USD
  - Transaktionsavgift: 0.25%
  - Tracking: P&L, positioner, trade history

### Feedbackloop-koncept (Sprint 1 grund, fullt i Sprint 3)

Sprint 1 lägger grunden för feedback-systemet som används i kommande sprintar:

#### Feedback-källor (enligt feedback_loop.yaml):

**1. execution_engine feedback:**
- **Triggers:**
  - `trade_result`: Lyckad/misslyckad trade
  - `slippage`: Skillnad mellan förväntat och verkligt pris (>0.2% loggas)
  - `latency`: Exekveringstid
- **Emitterar:** `feedback_event` till message_bus

**2. portfolio_manager feedback:**
- **Triggers:**
  - `capital_change`: Ändring i totalt portföljvärde
  - `transaction_cost`: Kostnad för varje trade
- **Emitterar:** `feedback_event` och `reward` till message_bus

**3. Feedback Routing (Sprint 3):**
```
feedback_event → feedback_router → 
  ├─ rl_controller (för agentträning)
  ├─ feedback_analyzer (mönsteridentifiering)
  └─ strategic_memory_engine (loggning)
```

**4. RL Response (Sprint 2):**
- `rl_controller` tar emot reward från portfolio_manager
- Uppdaterar RL-agenter i strategy_engine, decision_engine, execution_engine
- Belöning baserad på:
  - Portfolio value change
  - Trade profitability
  - Execution quality

### Indikatoranvändning (från indicator_map.yaml)

| Indikator | Typ       | Används av        | Syfte                           |
|-----------|-----------|-------------------|---------------------------------|
| OHLC      | Technical | strategy, execution | Price analysis, entry/exit    |
| Volume    | Technical | strategy          | Liquidity assessment            |
| SMA       | Technical | strategy          | Trend detection, smoothing      |
| RSI       | Technical | strategy, decision | Overbought/oversold detection  |

**Kommande indikatorer (Sprint 2-7):**
- Sprint 2: MACD, ATR, Analyst Ratings
- Sprint 3: News Sentiment, Insider Sentiment
- Sprint 4: ROE, ROA, ESG, Earnings Calendar
- Sprint 5: Bollinger Bands, ADX, Stochastic Oscillator

### Message Bus - Central Kommunikation

Alla moduler kommunicerar via `message_bus.py` med pub/sub-mönster:

**Topics i Sprint 1:**
- `market_data`: Från data_ingestion
- `indicator_data`: Från indicator_registry
- `decision_proposal`: Från strategy_engine
- `final_decision`: Från decision_engine
- `execution_result`: Från execution_engine
- `portfolio_status`: Från portfolio_manager
- `reward`: Från portfolio_manager
- `feedback_event`: Från execution_engine och portfolio_manager

**Fördelar:**
- Lös koppling mellan moduler
- Enkel att lägga till nya prenumeranter
- Meddelandelogg för debugging och introspektion

---

## 🧠 Arkitekturöversikt

Systemet består av fristående moduler som kommunicerar via en central `message_bus`. Varje modul kan:
- Skicka och ta emot feedback
- Tränas med PPO-agenter via `rl_controller`
- Visualiseras via introspektionspaneler
- Använda indikatorer från Finnhub via `indicator_registry`

---

## 📦 Modulöversikt

| Modul                      | Syfte                                                                 |
|---------------------------|------------------------------------------------------------------------|
| `data_ingestion.py`       | Hämtar trending symboler och öppnar WebSocket                         |
| `strategy_engine.py`      | Genererar tradeförslag baserat på indikatorer och RL                  |
| `risk_manager.py`         | Bedömer risk och justerar strategi                                    |
| `decision_engine.py`      | Samlar insikter och fattar beslut                                     |
| `execution_engine.py`     | Simulerar eller exekverar trades                                      |
| `portfolio_manager.py`    | Hanterar demoportfölj med startkapital och avgifter                   |
| `indicator_registry.py`   | Hämtar och distribuerar indikatorer från Finnhub                      |
| `rl_controller.py`        | Tränar PPO-agenter och samlar belöning                                |
| `feedback_router.py`      | Skickar feedback mellan moduler                                       |
| `feedback_analyzer.py`    | Identifierar mönster i feedbackflöden                                 |
| `strategic_memory_engine.py` | Loggar beslut, röster och utfall                                     |
| `meta_agent_evolution_engine.py` | Utvärderar och utvecklar agentlogik                          |
| `agent_manager.py`        | Hanterar agentprofiler och versioner                                  |
| `vote_engine.py`          | Genomför röstning mellan agenter                                     |
| `consensus_engine.py`     | Väljer konsensusmodell och löser konflikter                           |
| `decision_simulator.py`   | Testar alternativa beslut i sandbox                                   |
| `timespan_tracker.py`     | Synkroniserar beslut över tid                                         |
| `action_chain_engine.py`  | Definierar återanvändbara beslutskedjor                               |
| `introspection_panel.py`  | Visualiserar modulstatus och RL-performance                           |
| `system_monitor.py`       | Visar systemöversikt, indikatortrender och agentrespons               |

---


## 📊 Indikatorer från Finnhub

Systemet använder tekniska, fundamentala och alternativa indikatorer:

- **Tekniska:** OHLC, RSI, MACD, Bollinger Bands, ATR, VWAP, ADX
- **Fundamentala:** EPS, ROE, ROA, margin, analyst ratings, dividend yield
- **Alternativa:** News sentiment, insider sentiment, ESG, social media

Alla indikatorer hämtas via `indicator_registry.py` och distribueras via `message_bus`.

---


## 🏁 Sprintstruktur

Projektet är uppdelat i 7 sprintar. Se `sprint_plan.yaml` för detaljer.

| Sprint | Fokus                                |
|--------|--------------------------------------|
| 1      | Kärnsystem och demoportfölj          |
| 2      | RL och belöningsflöde                |
| 3      | Feedbackloopar och introspektion     |
| 4      | Strategiskt minne och agentutveckling|
| 5      | Simulering och konsensus             |
| 6      | Tidsanalys och action chains         |
| 7      | Indikatorvisualisering och översikt  |

Se `README_sprints.md` för detaljerad beskrivning av varje sprint.

---

## 🧪 Teststruktur

Alla moduler har motsvarande testfiler i `tests/`. Testerna är uppdelade i:
- Modulfunktionalitet
- RL-belöning och agentrespons
- Feedbackflöde
- Indikatorintegration

---

## 🧩 Onboardingtips

- Alla moduler kommunicerar via `message_bus.py`
- RL-belöning hanteras centralt via `rl_controller.py`
- Feedback skickas via `feedback_router.py`
- Indikatorer hämtas via `indicator_registry.py`
- Varje modul har introspektionspanel för transparens

---


NextGenAITrader/
├── main.py                      # Startpunkt för systemet
├── requirements.txt             # Pythonberoenden

├── modules/                     # Alla kärnmoduler
│   ├── data_ingestion.py
│   ├── strategy_engine.py
│   ├── risk_manager.py
│   ├── decision_engine.py
│   ├── vote_engine.py
│   ├── consensus_engine.py
│   ├── execution_engine.py
│   ├── portfolio_manager.py
│   ├── indicator_registry.py
│   ├── rl_controller.py
│   ├── feedback_router.py
│   ├── feedback_analyzer.py
│   ├── strategic_memory_engine.py
│   ├── meta_agent_evolution_engine.py
│   ├── agent_manager.py
│   ├── decision_simulator.py
│   ├── timespan_tracker.py
│   ├── action_chain_engine.py
│   ├── introspection_panel.py
│   └── system_monitor.py

├── tests/                       # Testfiler per modul
│   ├── test_data_ingestion.py
│   ├── test_strategy_engine.py
│   ├── test_risk_manager.py
│   ├── test_decision_engine.py
│   ├── test_vote_engine.py
│   ├── test_consensus_engine.py
│   ├── test_execution_engine.py
│   ├── test_portfolio_manager.py
│   ├── test_indicator_registry.py
│   ├── test_rl_controller.py
│   ├── test_feedback_router.py
│   ├── test_feedback_analyzer.py
│   ├── test_strategic_memory_engine.py
│   ├── test_meta_agent_evolution_engine.py
│   ├── test_agent_manager.py
│   ├── test_decision_simulator.py
│   ├── test_timespan_tracker.py
│   ├── test_action_chain_engine.py
│   ├── test_introspection_panel.py
│   └── test_system_monitor.py

├── dashboards/                  # Dash-paneler för visualisering
│   ├── portfolio_overview.py
│   ├── rl_metrics.py
│   ├── feedback_flow.py
│   ├── indicator_trends.py
│   ├── consensus_visualizer.py
│   ├── agent_evolution.py
│   └── system_status.py

├── docs/                        # Dokumentation och onboarding
│   ├── README.md
│   ├── README_sprints.md
│   ├── onboarding_guide.md
│   ├── sprint_plan.yaml
│   ├── structure.yaml
│   ├── functions.yaml
│   ├── indicator_map.yaml
│   ├── agent_profiles.yaml
│   ├── consensus_models.yaml
│   ├── action_chains.yaml
│   ├── test_map.yaml
│   └── introspection_config.yaml

├── config/                      # Inställningar och nycklar
│   ├── finnhub_keys.yaml
│   ├── agent_roles.yaml
│   ├── chain_templates.yaml
│   └── rl_parameters.yaml

├── logs/                        # Loggar och historik
│   ├── feedback_log.json
│   ├── decision_history.json
│   ├── agent_performance.json
│   └── trade_log.json

├── data/                        # Lokala datakällor och cache
│   ├── cached_indicators/
│   ├── simulation_results/
│   └── snapshots/
