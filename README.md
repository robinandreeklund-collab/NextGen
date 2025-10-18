# 🚀 NextGen AI Trader

Ett självreflekterande, modulärt och RL-drivet handelssystem byggt för transparens, agentutveckling och realtidsanalys. Systemet simulerar handel med verkliga data, strategier, feedbackloopar och belöningsbaserad inlärning.

## 🎯 Snabbstart

### 🆕 Kör Fullskalig Dashboard (Rekommenderat)
```bash
# Demo-läge (simulerad data)
python start_demo.py

# Live-läge (WebSocket streaming)
python start_live.py

# Öppna http://localhost:8050 i webbläsaren
```

**⚠️ VIKTIGT: Data Integrity Policy**

Dashboarden är designad för att visa **ENDAST verklig data från systemets moduler och message_bus**. 

**Förbjudet:**
- ❌ Hårdkodade värden i dashboard-paneler
- ❌ Mockup-data eller randomvärden i visualiseringar
- ❌ UI-genererade metrics (alla metrics ska komma från moduler)
- ❌ Statiska siffror i grafer eller tabeller

**Tillåtet:**
- ✅ Data från `data_ingestion_sim.py` (demo-läge)
- ✅ Data från `data_ingestion.py` (live-läge)
- ✅ Metrics från portfolio_manager, rl_controller, dqn_controller, etc.
- ✅ Real-time data via message_bus

**Dataflöde:**
```
Demo-läge:  data_ingestion_sim → message_bus → modules → dashboard
Live-läge:  data_ingestion (Finnhub) → message_bus → modules → dashboard
```

Alla paneler och visualiseringar måste reflektera systemets faktiska tillstånd!

### Kör Analyzer Debug Dashboard
```bash
python analyzer_debug.py
# Öppna http://localhost:8050 i webbläsaren
```

### Kör Simulering
```bash
python sim_test.py
```

### Kör Tester
```bash
pytest tests/ -v
```

---

## 📊 Fullskalig NextGen Dashboard

**start_demo.py / start_live.py** - Omfattande full-scale dashboard för hela NextGen projektet.

### Översikt

Fullskalig dashboard byggd enligt `docs/dashboard_structure_sprint8.yaml` med modern design inspirerad av "Abstire Dashboard". Innehåller alla 10 huvudpaneler för komplett systemövervakning och kontroll.

### Starta Dashboard

**Demo-läge (Simulerad data):**
```bash
python start_demo.py
# Startar med simulerad realtidsdata från data_ingestion_sim.py
# Genererar realistiska prisrörelser med trend, volatilitet och mean reversion
# Publicerar data via message_bus till alla moduler
# Ingen WebSocket-anslutning krävs
```

**Live-läge (Real-time data):**
```bash
python start_live.py
# Ansluter till Finnhub WebSocket för live marknadsdata
# Använder data_ingestion.py för att hämta data från Finnhub API
# Kräver giltig Finnhub API-nyckel
```

Öppna sedan `http://localhost:8050` i webbläsaren.

**Datakällor:**

Alla dashboardpaneler använder **ENDAST** data från:

1. **Market Data**: 
   - Demo: `data_ingestion_sim.py` → `market_data` topic
   - Live: `data_ingestion.py` → `market_data` topic

2. **Portfolio Metrics**:
   - `portfolio_manager.py` → portfolio value, cash, positions, ROI

3. **RL Agent Metrics**:
   - `rl_controller.py` → PPO rewards, agent actions
   - `dqn_controller.py` → DQN rewards, epsilon, actions

4. **Evolution Metrics**:
   - `gan_evolution_engine.py` → generator loss, discriminator loss, acceptance rate

5. **Pattern Detection**:
   - `gnn_timespan_analyzer.py` → pattern types, confidence levels

6. **Reward Flow**:
   - `reward_tuner.py` → base rewards, tuned rewards

7. **Conflicts**:
   - Decision history → PPO vs DQN conflicts

**Inga hårdkodade värden eller mockup-data används!**

### Dashboard-paneler

Dashboard innehåller 10 huvudpaneler baserade på YAML-specifikationerna:

#### 1. **Portfolio**
- Portföljöversikt med total värde, cash, holdings och ROI
- Realtidsuppdateringar av portföljvärde
- Positionsvisualisering
- P&L-tracking

#### 2. **RL Agent Analysis**
- Hybrid RL-jämförelse: PPO vs DQN prestanda
- Reward flow-visualisering (base → tuned → PPO/DQN)
- DQN-specifika metriker (epsilon, loss, buffer)
- Epsilon decay schedule
- Training progress för båda agentsystem

#### 3. **Agent Evolution & GAN**
- GAN generator/discriminator loss-kurvor
- Kandidatacceptans-gauge
- Agentevolutions-tidslinje
- Candidate distribution histogram
- Deployment timeline för nya agenter

#### 4. **Temporal Drift & GNN**
- GNN pattern detection (8 mönstertyper)
- Pattern confidence charts
- Temporal graph visualization
- Pattern timeline med Gantt-diagram
- Success rate per pattern
- Temporal insights och rekommendationer

#### 5. **Feedback & Reward Loop**
- Reward transformation visualization
- Base vs tuned reward comparison
- Volatility och overfitting tracking
- Feedback flow mellan moduler
- Transformation ratio över tid

#### 6. **CI Test Results**
- Test suite overview (314 tester)
- Pass/fail metrics
- Coverage tracking (85%+)
- Test results breakdown
- Sprint-specific test status

#### 7. **RL Conflict Monitor**
- PPO vs DQN konfliktfrekvens
- Resolution strategy breakdown
- Conflict details table
- Parameter conflicts vs decision conflicts
- Outcome tracking för olika resolutions

#### 8. **Decision & Consensus**
- Consensus model visualization
- Voting matrix heatmap
- Decision robustness metrics
- Agent agreement rates
- Consensus confidence över tid

#### 9. **Adaptive Settings**
- 16+ adaptiva parametrar live-tracking
- Parameter evolution över tid
- Manuella overrides (sliders)
- DQN epsilon control
- GAN threshold adjustment
- Parameter groups visualization

#### 10. **Live Market Watch**
- Real-time prisdiagram för alla symboler
- Tekniska indikatorer (RSI, MACD, ATR)
- Volume och trend analysis
- Market sentiment indicators

### Design och Tema

Dashboard använder ett modernt dark theme inspirerat av "Abstire Dashboard" mockup:

**Färgschema:**
- **Background:** `#0a0e1a` (djup mörk blå)
- **Surface:** `#141b2d` (mörk panel)
- **Primary:** `#4dabf7` (ljusblå för accenter)
- **Secondary:** `#845ef7` (lila för sekundära element)
- **Success:** `#51cf66` (grön för positiva värden)
- **Warning:** `#ffd43b` (gul för varningar)
- **Danger:** `#ff6b6b` (röd för fel/kritiska värden)

**Komponenter:**
- **Top Header:** Systemstatus, mode (Demo/Live), realtidsklocka
- **Sidebar:** Quick stats, module status, navigation, Start/Stop knappar, statusindikator
- **Tab Navigation:** 10 paneler med smooth övergångar
- **Responsive Design:** Funkar på desktop, tablet och mobile

### Funktioner

**Realtidsuppdateringar:**
- Auto-refresh var 2:a sekund
- WebSocket streaming i live-läge
- Smooth chart animations

**Interaktiva kontroller:**
- Start/stop simulation
- Parameter overrides via sliders
- Tab navigation mellan paneler
- Expandable charts

**Modulär arkitektur:**
- Separata paneler för varje systemdel
- Återanvändbara chart-komponenter
- Enhetlig styling via tema-system

### Arkitektur

Dashboard implementerar följande struktur:

```
start_demo.py / start_live.py
    ↓
NextGenDashboard (start_dashboard.py)
    ├── Module Initialization (Sprint 1-8)
    ├── Layout Creation
    │   ├── Top Header
    │   ├── Sidebar
    │   ├── Control Panel
    │   └── Tab Content (10 paneler)
    ├── Callbacks
    │   ├── Tab rendering
    │   ├── Start/stop control
    │   ├── Auto-refresh
    │   └── Sidebar updates
    └── Simulation Loop (demo) / WebSocket (live)
```

### YAML-referenser

Dashboard implementerar specifikationer från:
- `docs/dashboard_structure_sprint8.yaml` - Huvudstruktur och paneler
- `docs/adaptive_parameters_sprint8.yaml` - Adaptiva parametrar
- `docs/sprint_8.yaml` - Sprint 8 funktioner
- `docs/ci_pipeline_sprint8.yaml` - CI/CD integration

### Teknisk Stack

- **Backend:** Python 3.12+
- **Dashboard:** Dash 3.2+ och Plotly 6.3+
- **Styling:** Inline CSS med tema-system
- **Real-time:** WebSocket-client för live data
- **Threading:** Async simulation loop

### Användningsexempel

**Starta i demo-läge och övervaka systemet:**
```bash
# Terminal 1: Starta dashboard
python start_demo.py

# Terminal 2: Öppna browser
# Navigera till http://localhost:8050
# Simuleringen startar automatiskt i demo-läge
# Växla mellan paneler för att se olika aspekter
```

**Live trading med WebSocket:**
```bash
# Sätt API-nyckel (om inte redan i koden)
export FINNHUB_API_KEY="your_api_key_here"

# Starta live dashboard
python start_live.py

# Dashboard ansluter automatiskt till Finnhub WebSocket
# Real-time marknadsdata visas i Live Market Watch panel
```

### Troubleshooting

**Dashboard startar inte:**
```bash
# Kontrollera dependencies
pip install -r requirements.txt

# Testa import
python -c "from start_dashboard import NextGenDashboard; print('OK')"
```

**Inga data visas:**
- Klicka "Start" knappen i control panel
- Kontrollera att simulation loop körs (check console output)
- Verifiera att moduler är initialiserade korrekt

**WebSocket fel i live-läge:**
- Kontrollera Finnhub API-nyckel
- Verifiera internet-anslutning
- Kolla att Finnhub-tjänsten är tillgänglig

---

## 📍 Sprintstatus

| Sprint | Status | Beskrivning |
|--------|--------|-------------|
| Sprint 1 | ✅ Färdig | Kärnsystem och demoportfölj |
| Sprint 2 | ✅ Färdig | RL och belöningsflöde |
| Sprint 3 | ✅ Färdig | Feedbackloopar och introspektion |
| Sprint 4 | ✅ Färdig | Strategiskt minne och agentutveckling |
| Sprint 4.2 | ✅ Färdig | Adaptiv parameterstyrning via RL/PPO |
| Sprint 4.3 | ✅ Färdig | Full adaptiv parameterstyrning i alla moduler |
| Sprint 4.4 | ✅ Färdig | Meta-belöningsjustering via RewardTunerAgent |
| Sprint 5 | ✅ Färdig | Simulering och konsensus |
| Sprint 6 | ✅ Färdig | Tidsanalys och action chains |
| Sprint 7 | ✅ Färdig | Indikatorvisualisering och systemöversikt |
| Sprint 8 | ✅ Färdig | DQN, GAN, GNN – Hybridiserad RL & Temporal Intelligence |

**Testresultat:** ✅ 314/314 tester passerar (100%)

---

## 🔍 Analyzer Debug Dashboard

**analyzer_debug.py** - Omfattande debug- och analysdashboard för hela NextGen systemet.

### Funktioner

Dashboard med 6 huvudsektioner:

1. **System Overview**
   - Systemhälsa (health score 0-100%)
   - Modulstatus (aktiva/stale moduler)
   - Realtidsövervakning av alla komponenter

2. **Data Flow & Simulation**
   - Prisutveckling för alla symboler
   - Tekniska indikatorer (RSI, MACD, ATR)
   - Beslutsflöde (decisions, executions, success rate)

3. **RL Analysis**
   - Reward flow (base vs tuned rewards, Sprint 4.4)
   - Parameter evolution (adaptiva parametrar, Sprint 4.3)
   - Agent performance (training loss per agent)

4. **Agent Development**
   - Agent evolution (versioner över tid)
   - Agent metriker (performance per agent)
   - Evolutionshistorik

5. **Debug & Logging**
   - Event log (realtidslogg av alla events)
   - Feedback flow (visualisering av feedback)
   - Timeline analysis (Sprint 6)

6. **Portfolio**
   - Portfolio värde (cash, holdings, total)
   - Positioner (nuvarande innehav)
   - P&L och ROI

### Användning

```bash
# Starta dashboarden
python analyzer_debug.py

# Öppna webbläsaren på
http://localhost:8050
```

**Kontroller:**
- Start Simulation: Startar simulerad trading
- Stop Simulation: Stoppar simulering
- Auto-refresh: Uppdaterar var 2:a sekund

**Datakällor:**
- Återanvänder kod från sim_test.py för datainmatning
- Integrerar med alla moduler via message_bus
- Realtidsdata från introspection_panel, system_monitor, RL-controller

---

## 📦 Systemöversikt

### Kärnmoduler

| Modul | Beskrivning | Sprint |
|-------|-------------|--------|
| `analyzer_debug.py` | Debug dashboard med fullständig systemvisualisering | Sprint 7 |
| `data_ingestion.py` | WebSocket-dataflöde från Finnhub | Sprint 1 |
| `data_ingestion_sim.py` | Simulerad marknadsdata för demo-läge | Sprint 1 |
| `strategy_engine.py` | Genererar tradeförslag baserat på indikatorer | Sprint 1-2 |
| `risk_manager.py` | Riskbedömning och justering | Sprint 1-2 |
| `decision_engine.py` | Fattar handelsbeslut | Sprint 1-2 |
| `execution_engine.py` | Exekverar trades | Sprint 1 |
| `portfolio_manager.py` | Hanterar portfölj och genererar rewards | Sprint 1, 4.4 |
| `rl_controller.py` | PPO-agentträning och distribution | Sprint 2, 4.2 |
| `reward_tuner.py` | Meta-belöningsjustering | Sprint 4.4 |
| `vote_engine.py` | Agentröstning | Sprint 4.3, 5 |
| `consensus_engine.py` | Konsensusbeslut | Sprint 5 |
| `decision_simulator.py` | Beslutssimuleringar | Sprint 5 |
| `timespan_tracker.py` | Timeline-analys | Sprint 6 |
| `action_chain_engine.py` | Återanvändbara beslutskedjor | Sprint 6 |
| `system_monitor.py` | Systemhälsoövervakning | Sprint 6 |
| `dqn_controller.py` | DQN reinforcement learning med experience replay | Sprint 8 |
| `gan_evolution_engine.py` | GAN för agentevolution och kandidatgenerering | Sprint 8 |
| `gnn_timespan_analyzer.py` | Graph Neural Network för temporal analys | Sprint 8 |
| `strategic_memory_engine.py` | Beslutshistorik och analys | Sprint 4 |
| `meta_agent_evolution_engine.py` | Agentevolution | Sprint 4 |
| `agent_manager.py` | Agentprofiler och versioner | Sprint 4 |
| `feedback_router.py` | Intelligent feedback-routing | Sprint 3 |
| `feedback_analyzer.py` | Mönsteranalys i feedback | Sprint 3 |
| `introspection_panel.py` | Dashboard-data för visualisering | Sprint 3, 7 |

### Adaptiva Parametrar

Systemet har **16 adaptiva parametrar** som justeras automatiskt via RL/PPO:

**Sprint 4.4 (RewardTunerAgent):**
- reward_scaling_factor (0.5-2.0)
- volatility_penalty_weight (0.0-1.0)
- overfitting_detector_threshold (0.05-0.5)

**Sprint 4.2 (Meta-parametrar):**
- evolution_threshold (0.05-0.5)
- min_samples (5-50)
- update_frequency (1-100)
- agent_entropy_threshold (0.1-0.9)

**Sprint 4.3 (Modulparametrar):**
- signal_threshold (0.1-0.9) - Strategy Engine
- indicator_weighting (0.0-1.0) - Strategy Engine
- risk_tolerance (0.01-0.5) - Risk Manager
- max_drawdown (0.01-0.3) - Risk Manager
- consensus_threshold (0.5-1.0) - Decision Engine
- memory_weighting (0.0-1.0) - Decision Engine
- agent_vote_weight (0.1-2.0) - Vote Engine
- execution_delay (0-10) - Execution Engine
- slippage_tolerance (0.001-0.05) - Execution Engine

---

## 🏗️ Systemarkitektur

### Dataflöde

```
Market Data (Finnhub) → Data Ingestion → Indicators
                                              ↓
                                    Strategy Engine ← RL Controller
                                              ↓
                                      Risk Manager
                                              ↓
                                    Decision Engine ← Memory/Voting
                                              ↓
                                    Execution Engine
                                              ↓
                                   Portfolio Manager → Base Reward
                                              ↓
                                      Reward Tuner → Tuned Reward
                                              ↓
                                       RL Controller → Agent Updates
```

### Demo-läge Dataflöde (data_ingestion_sim.py)

```
data_ingestion_sim.simulate_market_tick()
    ↓ publicerar market_data
message_bus
    ↓ distribuerar till
[indicator_registry, strategy_engine, portfolio_manager, ...]
    ↓ processerar och publicerar resultat
message_bus
    ↓ distribuerar till
dashboard (visualisering)
```

**Viktig princip:** ALL data som visas i dashboarden kommer från message_bus topics som publicerats av modulerna. Ingen data genereras direkt i UI-lagret.

### Reward Flow (Sprint 4.4)

RewardTunerAgent transformerar volatila portfolio rewards till stabila RL-signaler:

1. **Portfolio Manager** → base_reward (rådata)
2. **Reward Tuner** → analys och transformation
   - Beräkna volatilitet
   - Detektera overfitting
   - Applicera penalties
   - Skala med reward_scaling_factor
3. **RL Controller** → tuned_reward (stabil signal)

**Transformation Ratio:** 0.67 genomsnitt (33% reduktion vid hög volatilitet)

## 🚀 Kom igång

### Installation

```bash
# Klona repositoryt
git clone https://github.com/robinandreeklund-collab/NextGen.git
cd NextGen

# Installera dependencies
pip install -r requirements.txt

# Kör tester
pytest tests/ -v
```

### Snabbstart

```bash
# 1. Starta debug-dashboarden (rekommenderat)
python analyzer_debug.py
# → Öppna http://localhost:8050

# 2. Kör simulering i terminal
python sim_test.py

# 3. Kör med live WebSocket-data (kräver Finnhub API-nyckel)
python websocket_test.py
```

---

## 🎨 Funktioner

### 🔍 Analyzer Debug Dashboard
- Realtidsvisualisering av hela systemet
- 6 sektioner: System, Data Flow, RL Analysis, Agent Development, Debug, Portfolio
- Start/stop kontroller för simulering
- Auto-refresh var 2:a sekund

### 🤖 Adaptiv RL-optimering
- **16 adaptiva parametrar** som justeras automatiskt via PPO
- Reward transformation för stabil träning (Sprint 4.4)
- Meta-parameterstyrning (Sprint 4.2, 4.3)
- Agent evolution och versionshantering (Sprint 4)

### 🗳️ Konsensusbeslutsfattande
- Simulering av alternativa beslut (Sprint 5)
- Röstmatris med viktning (Sprint 5)
- 4 konsensusmodeller: Majority, Weighted, Unanimous, Threshold

### ⏰ Timeline och Action Chains
- Tidsbaserad spårning av beslut (Sprint 6)
- Återanvändbara beslutskedjor (Sprint 6)
- Systemhälsoövervakning (Sprint 6)

### 📊 Omfattande indikatorstöd
- Tekniska: RSI, MACD, ATR, Bollinger Bands, ADX, Stochastic
- Fundamentala: EPS, ROE, ROA, Analyst Ratings, Earnings
- Alternativa: News Sentiment, Insider Sentiment, ESG Score

---

## 📁 Projektstruktur

```
NextGen/
├── start_dashboard.py           # 🆕 Fullskalig dashboard (main)
├── start_demo.py                # 🆕 Starta i demo-läge (använder data_ingestion_sim)
├── start_live.py                # 🆕 Starta i live-läge (använder data_ingestion)
├── analyzer_debug.py            # Debug dashboard (legacy)
├── sim_test.py                  # Simulerad trading
├── websocket_test.py            # Live trading med Finnhub
├── modules/                     # Alla kärnmoduler (30 stycken)
│   ├── data_ingestion.py        # Live WebSocket från Finnhub
│   ├── data_ingestion_sim.py    # 🆕 Simulerad marknadsdata för demo
│   ├── reward_tuner.py          # Sprint 4.4: Reward transformation
│   ├── rl_controller.py         # Sprint 2, 4.2: PPO-agenter
│   ├── dqn_controller.py        # Sprint 8: DQN RL
│   ├── gan_evolution_engine.py  # Sprint 8: GAN evolution
│   ├── gnn_timespan_analyzer.py # Sprint 8: GNN temporal analysis
│   ├── consensus_engine.py      # Sprint 5: Konsensusbeslut
│   ├── timespan_tracker.py      # Sprint 6: Timeline-analys
│   └── ...
├── dashboards/                  # Dash-visualiseringar (komponenter)
├── tests/                       # 332 tester (100% pass rate)
├── docs/                        # Dokumentation och YAML-specs
│   ├── dashboard_structure_sprint8.yaml  # Dashboard spec
│   ├── adaptive_parameters_sprint8.yaml  # Parameter spec
│   ├── sprint_8.yaml                     # Sprint 8 overview
│   └── ...
└── requirements.txt
```

---

## 🧪 Testning

```bash
# Kör alla tester
pytest tests/ -v

# Kör specifika test-suiter
pytest tests/test_reward_tuner.py -v              # Sprint 4.4
pytest tests/test_adaptive_parameters_sprint4_3.py -v  # Sprint 4.3
pytest tests/test_consensus_engine.py -v          # Sprint 5
pytest tests/test_timespan_tracker.py -v          # Sprint 6

# Med coverage
pytest tests/ --cov=modules --cov-report=html
```

**Testresultat:** ✅ 214/214 tester passerar

---

## 📈 Prestanda och Metriker

### Reward Transformation (Sprint 4.4)
- Base rewards: 50
- Tuned rewards: 50 (1:1 match)
- Volatilitet: 31.31 genomsnitt, 48.75 senaste
- Transformation ratio: 0.67 (33% reduktion vid hög volatilitet)
- **Resultat:** Stabilare RL-träning, bättre generalisering

### Konsensus (Sprint 5)
- 1000 beslut processade
- Robusthet: 0.88 genomsnitt
- Konsensus confidence: 0.19 (konservativ trading)
- **Resultat:** Risk-medvetet beslutsfattande

### System Health
- Module health score: 0.95+ (95%+)
- 214/214 tester passerar
- Alla 26 moduler aktiva

---

## 🔗 Integrationer och API

### Finnhub API
- WebSocket för realtidsdata
- REST API för indikatorer
- Kräver API-nyckel (gratis tier tillräcklig)

### Message Bus
- Central pub/sub-kommunikation
- 20+ topics för modul-kommunikation
- Event logging för debugging

---

## 📚 Detaljerad Dokumentation

För detaljerad information om sprintar, implementationer och arkitektur:

- **Sprint-dokumentation:** Se `README_detailed_backup.md`
- **YAML-specifikationer:** Se `docs/` för alla system-specs
- **Test-dokumentation:** Se `docs/reward_tuner_sprint4_4/rl_test_suite.yaml`

---

## 🤝 Sprint-sammanfattning

| Sprint | Fokus | Nyckelfunktioner |
|--------|-------|------------------|
| **1** | Kärnsystem | WebSocket, Strategy, Execution, Portfolio |
| **2** | RL | PPO-agenter, Reward flow, Agent training |
| **3** | Feedback | Feedback routing, Analyzer, Introspection |
| **4** | Memory & Evolution | Strategic memory, Agent evolution, Versions |
| **4.2** | Adaptiv RL | Meta-parametrar (4 stycken) |
| **4.3** | Modulparametrar | 9 adaptiva modulparametrar |
| **4.4** | Reward Tuner | Reward transformation, Volatility control |
| **5** | Konsensus | Decision simulator, Voting, 4 consensus models |
| **6** | Timeline | Timeline tracking, Action chains, System monitor |
| **7** | Visualisering | **analyzer_debug.py**, Resource planner, Teams |
| **8** | Hybrid RL & Deep Learning | DQN, GAN, GNN, Temporal intelligence |

---

## 🆕 Sprint 8: DQN, GAN, GNN – Hybridiserad RL & Temporal Intelligence

Sprint 8 integrerar avancerade deep learning-tekniker för förbättrad beslutsfattande och agentevolution:

### Nya Moduler

**1. DQN Controller (`dqn_controller.py`)**
- Deep Q-Network för reinforcement learning
- Experience replay buffer för stabil träning
- Target network för stable Q-value estimation
- Epsilon-greedy exploration strategy
- Körs parallellt med PPO för hybrid RL

**2. GAN Evolution Engine (`gan_evolution_engine.py`)**
- Generative Adversarial Network för agentevolution
- Generator: Skapar nya agentparameterkandidater
- Discriminator: Bedömer kvalitet mot historisk performance
- Integrerar med meta_agent_evolution_engine
- Evolution threshold för kvalitetskontroll

**3. GNN Timespan Analyzer (`gnn_timespan_analyzer.py`)**
- Graph Neural Network för temporal mönsteranalys
- Graph Attention Layer för viktad analys
- Identifierar 8 mönstertyper: uptrend, downtrend, reversal, consolidation, breakout, breakdown, divergence, convergence
- Ger djupare insikter än traditionell tidsserieanalys
- Integrerar med timespan_tracker

### Hybrid RL-Arkitektur

**PPO + DQN Parallell Exekvering:**
- PPO (från Sprint 2-7): Policy gradient-optimering
- DQN (Sprint 8): Q-value-optimering
- Båda får samma rewards från portfolio_manager och reward_tuner
- Koordinerad via message_bus
- Konfliktdetektion och resolution

**Fördelar:**
- PPO: Bra för kontinuerliga åtgärdsval
- DQN: Bra för diskreta beslutsrum
- Kombinerat: Robustare och mer stabilt

### GAN-driven Evolution

**Kandidatgenerering:**
- GAN tränas på historisk agentperformance
- Generator skapar nya parameterkonfigurationer
- Discriminator filtrerar ut låg-kvalitet kandidater
- Endast kandidater över evolution_threshold accepteras

**Integration:**
- GAN-kandidater skickas till meta_agent_evolution_engine
- Används för att skapa nya agentversioner
- Evolutionscykel: Performance → GAN → Kandidater → Evolution → Deployment

### GNN Temporal Intelligence

**Graph-baserad Analys:**
- Beslut, indikatorer och outcomes som noder
- Temporala relationer som edges
- Attention mechanism för viktad analys
- Identifierar komplexa mönster över tid

**Mönstertyper:**
1. **Uptrend**: Stigande prisrörelse
2. **Downtrend**: Fallande prisrörelse
3. **Reversal**: Trendvändning
4. **Consolidation**: Sidledes rörelse
5. **Breakout**: Brott uppåt genom motstånd
6. **Breakdown**: Brott nedåt genom support
7. **Divergence**: Pris och indikator divergerar
8. **Convergence**: Pris och indikator konvergerar

### Testresultat Sprint 8

**100 nya tester:**
- test_dqn_controller.py: 21 tester
- test_gan_evolution_engine.py: 24 tester
- test_gnn_timespan_analyzer.py: 27 tester
- test_hybrid_rl.py: 14 tester
- test_sprint8_integration.py: 14 tester

**Total testning:** 314/314 tester passerar (100%)

**Täckning:**
- DQN: Q-learning, replay buffer, target network, epsilon decay
- GAN: Generator, discriminator, adversarial training, kandidatfiltrering
- GNN: Graph construction, attention mechanism, pattern detection
- Hybrid RL: Parallel execution, conflict detection, reward distribution
- Integration: End-to-end scenario, regression testing

### Adaptive Parameters Sprint 8

**DQN-specifika:**
- learning_rate: 0.0001 - 0.01
- discount_factor: 0.9 - 0.999
- epsilon: 0.01 - 1.0
- epsilon_decay: 0.99 - 0.9999
- replay_buffer_size: 1000 - 100000
- batch_size: 16 - 256
- target_update_frequency: 10 - 1000

**GAN-specifika:**
- generator_lr: 0.0001 - 0.01
- discriminator_lr: 0.0001 - 0.01
- latent_dim: 16 - 256
- evolution_threshold: 0.6 - 0.95

**GNN-specifika:**
- num_layers: 2 - 5
- hidden_dim: 32 - 256
- attention_heads: 1 - 8
- temporal_window: 10 - 100

### Prestanda Sprint 8

**DQN Training:**
- Convergence: 50-100 episodes
- Loss reduction: 70-80% efter träning
- Epsilon decay: 1.0 → 0.01 över 1000 steps

**GAN Evolution:**
- Acceptance rate: 60-80%
- Discriminator accuracy: ~50% (balanserad)
- Kandidater per generation: 3-10

**GNN Analysis:**
- Graph construction: <100ms för 20 noder
- Pattern detection: 0.7-0.9 confidence
- Temporal window: 10-100 beslut

**System Overhead:**
- Memory: <2GB totalt
- CPU: <80% under träning
- Response time: <2s för beslut

---

## 🛠️ Utveckling

### Kodstil
- Python 3.8+
- Type hints för alla funktioner
- Docstrings för alla moduler och klasser
- PEP 8-kompatibel

### CI/CD
- Automatiska tester vid varje push
- 100% test pass rate krävs
- Coverage tracking

---

## 📚 YAML-specifikationer och Artefakter

Fullskalig dashboard implementerar följande YAML-specifikationer:

### Dashboard-struktur
- **`docs/dashboard_structure_sprint8.yaml`**
  - Definierar alla 10 paneler och deras komponenter
  - Layout-specifikationer (grid, columns, etc.)
  - Refresh rates och data sources
  - Responsive design guidelines
  - Accessibility features

### Adaptive Parameters
- **`docs/adaptive_parameters_sprint8.yaml`**
  - DQN-parametrar (learning_rate, epsilon, batch_size, etc.)
  - GAN-parametrar (generator_lr, discriminator_lr, latent_dim)
  - GNN-parametrar (num_layers, hidden_dim, attention_heads)
  - Hybrid RL-parametrar (weights, conflict resolution)

### Sprint Specifikationer
- **`docs/sprint_8.yaml`**
  - Sprint 8 goals och features
  - Hybrid RL architecture
  - GAN evolution strategy
  - GNN temporal analysis
  - Test requirements

### CI/CD Pipeline
- **`docs/ci_pipeline_sprint8.yaml`**
  - Test suite struktur (314 tester)
  - Stages: setup, lint, unit tests, integration, regression
  - Coverage requirements (85%+)
  - Performance metrics

### Övriga Referenser
- **`docs/consensus_models.yaml`** - Konsensusmodeller
- **`docs/feedback_loop_sprint1_7.yaml`** - Feedback flow
- **`docs/functions.yaml`** - Modul-funktioner
- **`docs/indicator_map.yaml`** - Tekniska indikatorer

### Mockup och Design

Dashboard designen är inspirerad av "Abstire Dashboard" mockup med:
- Modern dark theme (#0a0e1a background)
- Blå/lila färgschema (#4dabf7, #845ef7)
- Card-baserad layout
- Sidebar navigation
- Real-time updates
- Responsive grid system

---

## 📝 Licens och Credits

NextGen AI Trader är utvecklat som ett demonstrations- och utbildningssystem för:
- Reinforcement Learning i trading
- Modulär systemarkitektur
- Agent-baserat beslutsfattande
- Adaptiv parameterstyrning

**OBS:** Detta är ett simulerings- och utbildningssystem. Använd inte för verklig trading utan grundlig testning och riskbedömning.
