# 🚀 NextGen AI Trader

Ett självreflekterande, modulärt och RL-drivet handelssystem byggt för transparens, agentutveckling och realtidsanalys. Systemet simulerar handel med verkliga data, strategier, feedbackloopar och belöningsbaserad inlärning.

---

## 📍 Sprintstatus

**Sprint 1 färdig ✅** – Kärnsystem och demoportfölj komplett
**Sprint 2 färdig ✅** – RL och belöningsflöde komplett
**Sprint 3 pågår 🔄** – Feedbackloopar och introspektion under utveckling

### Sprint 3: Feedbackloopar och introspektion (PÅGÅR)

**Mål:** Inför feedback mellan moduler och visualisera kommunikation.

**Moduler i fokus:**
- `message_bus` - Central pub/sub-kommunikation (förbättrad)
- `feedback_router` - Intelligent feedback-routing med prioritering
- `feedback_analyzer` - Avancerad mönsteranalys och detektering
- `introspection_panel` - Dashboard-data för Dash-visualisering

**Nya indikatorer i Sprint 3:**
- News Sentiment - Marknadssentiment från nyhetsflöden
- Insider Sentiment - Insiderhandel och confidence-signaler

**Implementerat:**
- ✅ Intelligent feedback-routing med prioritering (critical, high, medium, low)
- ✅ Performance pattern detection (slippage, success rate, capital changes)
- ✅ Indicator mismatch detection för korrelationsanalys
- ✅ Agent drift detection för performance degradation
- ✅ Dashboard-data med agent adaptation metrics
- ✅ Modul-kopplingar och kommunikationsflöden
- ✅ Dash-baserad feedback flow visualisering
- ✅ 23 tester för feedback-systemet (alla passerar)

**Testresultat:**
- ✅ Modulkommunikation fungerar via message_bus
- ✅ Feedbackflöde routas och loggas med prioriteter
- ✅ Mönsteranalys identifierar 3+ pattern-typer
- ✅ Dashboard genererar rik visualiseringsdata
- ✅ Agent adaptation tracking visar trends

### Sprint 2: RL och belöningsflöde ✅

**Mål:** Inför PPO-agenter i strategi, risk och beslut. Belöning via portfölj.

**Moduler i fokus:**
- `rl_controller` - PPO-agentträning och distribution
- `strategy_engine` - RL-förbättrade strategier med MACD
- `risk_manager` - RL-baserad riskbedömning med ATR
- `decision_engine` - RL-optimerade beslut
- `portfolio_manager` - Reward-generering för RL

**Nya indikatorer i Sprint 2:**
- MACD (Moving Average Convergence Divergence) - Momentum och trendstyrka
- ATR (Average True Range) - Volatilitetsbaserad riskjustering
- Analyst Ratings - Extern confidence och sentiment

**Testresultat:**
- ✅ RL-belöning beräknas från portfolio changes
- ✅ PPO-agenter tränas i rl_controller
- ✅ Agentuppdateringar distribueras till moduler (strategy, risk, decision, execution)
- ✅ 4 RL-agenter aktiva och tränas parallellt
- ✅ Feedback-flöde implementerat och loggas
- ✅ Strategier använder flera indikatorer kombinerat (RSI + MACD + Analyst Ratings)
- ✅ Riskbedömning anpassad efter volatilitet (ATR)

### Sprintplan - Sprint 1: Kärnsystem och demoportfölj ✅

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

## 🔄 Sprint 2: RL och Belöningsflöde

### RL-arkitektur och PPO-agenter

Sprint 2 introducerar reinforcement learning (RL) med PPO-agenter (Proximal Policy Optimization) för att optimera handelsbeslut baserat på portfolio performance.

```
┌─────────────────┐
│ portfolio_mgr   │
│ (Beräknar       │
│  reward)        │
└────────┬────────┘
         │ reward
         ▼
┌─────────────────┐     agent_update      ┌──────────────────┐
│ rl_controller   │────────────────────────▶│ strategy_engine  │
│ (PPO-träning)   │                        │ (RL-förstärkt)   │
└────────┬────────┘                        └──────────────────┘
         │ agent_update
         ├──────────────────────────────────▶┌──────────────────┐
         │                                   │ risk_manager     │
         │                                   │ (RL-förstärkt)   │
         │                                   └──────────────────┘
         │ agent_update
         ├──────────────────────────────────▶┌──────────────────┐
         │                                   │ decision_engine  │
         │                                   │ (RL-optimerat)   │
         │                                   └──────────────────┘
         │ agent_update
         └──────────────────────────────────▶┌──────────────────┐
                                             │ execution_engine │
                                             │ (RL-optimerat)   │
                                             └──────────────────┘
```

### RL-agenter och deras roller

**1. strategy_engine RL-agent:**
- State: 10 dimensioner (OHLC, Volume, SMA, RSI, MACD, portfolio info)
- Action: 3 möjligheter (BUY, SELL, HOLD)
- Syfte: Optimera tradeförslag baserat på indikator-kombinationer
- Förbättrar: Timing och kvantitet för trades

**2. risk_manager RL-agent:**
- State: 8 dimensioner (Volume, ATR, volatility, portfolio exposure)
- Action: 3 nivåer (LOW, MEDIUM, HIGH risk)
- Syfte: Justera riskbedömning baserat på historisk accuracy
- Förbättrar: Risk-adjusted returns

**3. decision_engine RL-agent:**
- State: 12 dimensioner (Strategy proposal, risk profile, memory insights)
- Action: 3 alternativ (ACCEPT, MODIFY, REJECT)
- Syfte: Optimera slutgiltiga beslut med balans mellan risk och reward
- Förbättrar: Confidence och beslutskvalitet

**4. execution_engine RL-agent:**
- State: 6 dimensioner (Price, volume, timing, slippage)
- Action: 2 alternativ (EXECUTE_NOW, WAIT)
- Syfte: Minimera slippage och förbättra execution quality
- Förbättrar: Execution timing

### Reward-beräkning och feedback

**Reward-källor:**
1. **Portfolio value change** (primär):
   - Positiv reward när portfolio värde ökar
   - Negativ reward när portfolio värde minskar
   
2. **Trade profitability**:
   - Belönar lönsamma trades
   - Straffar förlustbringande trades
   
3. **Risk-adjusted returns** (kommande):
   - Högre reward för vinster med låg risk
   - Lägre reward för vinster med hög risk

**Feedback-flöde:**
```
execution_engine ──┐
                   │
portfolio_manager ─┼──▶ feedback_event ──▶ feedback_router ──┬──▶ rl_controller
                   │                                           │
strategic_memory ──┘                                           ├──▶ feedback_analyzer
                                                               │
                                                               └──▶ strategic_memory
```

### Sprint 2 indikatorer och användning

| Indikator         | Modul            | Syfte                                    |
|-------------------|------------------|------------------------------------------|
| RSI               | strategy         | Overbought/oversold detection            |
| MACD              | strategy         | Momentum och trend strength              |
| ATR               | risk, strategy   | Volatility-based risk adjustment         |
| Analyst Ratings   | risk, decision   | External confidence och sentiment        |
| Volume            | strategy, risk   | Liquidity assessment                     |

**MACD-användning:**
- Histogram > 0.5: Köpsignal (bullish momentum)
- Histogram < -0.5: Säljsignal (bearish momentum)
- Kombineras med RSI för starkare signaler

**ATR-användning:**
- ATR > 5.0: Hög volatilitet → Reducera position size
- ATR < 2.0: Låg volatilitet → Normal position size
- Används för risk-adjusted quantity

**Analyst Ratings-användning:**
- BUY/STRONG_BUY: Ökar confidence, minskar risk
- SELL: Minskar confidence, ökar risk
- HOLD: Neutral påverkan

### RL-träningsprocess

1. **Trade execution** genererar portfolio change
2. **Portfolio manager** beräknar reward baserat på change
3. **RL controller** tar emot reward och tränar alla agenter
4. **Agent updates** distribueras till moduler
5. **Moduler** använder uppdaterade policies för nästa beslut
6. **Feedback** från execution och portfolio förbättrar reward shaping

**Träningsparametrar (config/rl_parameters.yaml):**
- Learning rate: 0.0003
- Gamma (discount factor): 0.99
- Update frequency: Var 10:e trade
- Batch size: 32

---

## 🔄 Sprint 3: Feedbackloopar och Introspektion

### Feedback-arkitektur

Sprint 3 introducerar ett omfattande feedback-system för att övervaka och förbättra systemets performance i realtid.

```
┌─────────────────┐
│ execution_engine│──┐
│ portfolio_mgr   │  │
│ strategic_mem   │  │ feedback_event
└─────────────────┘  │
                     ▼
            ┌────────────────┐
            │ feedback_router│
            │ (Prioritering) │
            └────────┬───────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
┌──────────────┐ ┌────────────┐ ┌──────────────┐
│ rl_controller│ │ feedback   │ │ strategic    │
│              │ │ analyzer   │ │ memory       │
└──────────────┘ └─────┬──────┘ └──────────────┘
                       │
                       ▼ feedback_insight
              ┌──────────────────┐
              │ meta_agent       │
              │ evolution_engine │
              └──────────────────┘
```

### Feedback-routing med intelligent prioritering

**FeedbackRouter** klassificerar feedback i fyra prioritetsnivåer:

| Prioritet | Trigger Exempel | Användning |
|-----------|-----------------|------------|
| **Critical** | Stora kapitalförluster (>$100) | Omedelbar åtgärd krävs |
| **High** | Hög slippage (>0.5%), misslyckade trades | Snabb respons önskvärd |
| **Medium** | Standard trade results, feedback | Normal processing |
| **Low** | Informativa events utan triggers | Loggning endast |

### Mönsteranalys i FeedbackAnalyzer

**FeedbackAnalyzer** identifierar tre huvudtyper av mönster:

#### 1. Performance Patterns
- **High Slippage**: Genomsnittlig slippage > 0.3%
- **Trade Success Rate**: Beräknar success rate över alla trades
- **Low Success Rate**: Varning när success rate < 50%
- **Capital Change Trends**: Genomsnittlig kapitalförändring över tid

#### 2. Indicator Mismatch
- Korrelerar indikator-signaler med trade outcomes
- Identifierar när indikatorer ger dåliga prediktioner (< 40% success)
- Föreslår strategi-justeringar baserat på korrelation

#### 3. Agent Drift
- Jämför agent performance över tid (första vs andra halvan av historik)
- Detekterar performance degradation > 15%
- Triggar reträning eller parameteråterställning

### Introspection Dashboard

**IntrospectionPanel** genererar rik data för visualisering:

**Dashboard Metrics:**
- Total feedback events och events/minut rate
- Events per källa (execution, portfolio, etc.)
- Events per prioritet (critical, high, medium, low)

**Agent Adaptation Tracking:**
- Adaptation rate: Hur snabbt agenter förbättras
- Performance trend: improving, stable, eller declining
- Learning progress: Aktuell performance-nivå
- Recent performances: Senaste 5 performance-värden

**Modul-kopplingar:**
- Nätverksanalys av kommunikation mellan moduler
- Connection strength baserat på antal events
- Visualisering av feedback flow-paths

### Sprint 3 Indikatorer

**News Sentiment (0.0 - 1.0)**
- Aggregerat sentiment från nyhetsartiklar
- 0.0 = Bearish, 0.5 = Neutral, 1.0 = Bullish
- Används av: strategy_engine, feedback_analyzer
- Syfte: Fånga marknadssentiment och reaktioner

**Insider Sentiment (0.0 - 1.0)**
- Baserat på insiderhandel och SEC-filings
- 0.0 = Insiders säljer, 0.5 = Neutral, 1.0 = Insiders köper
- Används av: strategy_engine, meta_agent_evolution_engine
- Syfte: Interna confidence-signaler från företagsledning

### Dash Visualisering

Sprint 3 inkluderar en komplett Dash-baserad dashboard (`dashboards/feedback_flow.py`):

**Komponenter:**
1. **Network Graph**: Visuell representation av modulkommunikation
2. **Metrics Cards**: Real-time feedback statistics
3. **Priority Distribution**: Pie chart över feedback-prioriteter
4. **Timeline**: Feedback events över tid per källa
5. **Recent Events Table**: Senaste feedback events med detaljer

**Kör dashboard:**
```bash
# Installera beroenden först (om inte redan gjort)
pip install -r requirements.txt

# Kör dashboard
python dashboards/feedback_flow.py
# Öppna http://localhost:8050 i webbläsare
```

**Obs:** Dashboard kräver att följande paket är installerade: `dash`, `plotly`, `numpy`. Installera alla beroenden med `pip install -r requirements.txt`.

### Demo och Testning

**Kör Sprint 3 Demo:**
```bash
python demo_sprint3.py
```

**Kör Sprint 3 Tester:**
```bash
pytest tests/test_feedback_analyzer.py -v
```

**Testresultat:** 23/23 tester passerar
- FeedbackAnalyzer: 7 tester
- FeedbackRouter: 6 tester
- IntrospectionPanel: 8 tester
- Integrerade system-tester: 2 tester

---

## 🔄 Sprint 2: RL och Belöningsflöde

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

| Sprint | Fokus                                | Status  |
|--------|--------------------------------------|---------|
| 1      | Kärnsystem och demoportfölj          | ✅ Färdig|
| 2      | RL och belöningsflöde                | ✅ Färdig|
| 3      | Feedbackloopar och introspektion     | 🔄 Pågår|
| 4      | Strategiskt minne och agentutveckling| ⏳ Planerad|
| 5      | Simulering och konsensus             | ⏳ Planerad|
| 6      | Tidsanalys och action chains         | ⏳ Planerad|
| 7      | Indikatorvisualisering och översikt  | ⏳ Planerad|

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
