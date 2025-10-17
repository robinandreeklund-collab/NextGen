# 🚀 NextGen AI Trader

Ett självreflekterande, modulärt och RL-drivet handelssystem byggt för transparens, agentutveckling och realtidsanalys. Systemet simulerar handel med verkliga data, strategier, feedbackloopar och belöningsbaserad inlärning.

## 🎯 Snabbstart

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

**Testresultat:** ✅ 214/214 tester passerar (100%)

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
├── analyzer_debug.py           # 🆕 Debug dashboard
├── sim_test.py                 # Simulerad trading
├── websocket_test.py           # Live trading med Finnhub
├── modules/                    # Alla kärnmoduler (26 stycken)
│   ├── reward_tuner.py         # Sprint 4.4: Reward transformation
│   ├── rl_controller.py        # Sprint 2, 4.2: PPO-agenter
│   ├── consensus_engine.py     # Sprint 5: Konsensusbeslut
│   ├── timespan_tracker.py     # Sprint 6: Timeline-analys
│   └── ...
├── dashboards/                 # Dash-visualiseringar
├── tests/                      # 214 tester (100% pass rate)
├── docs/                       # Dokumentation och YAML-specs
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

## 📝 Licens och Credits

NextGen AI Trader är utvecklat som ett demonstrations- och utbildningssystem för:
- Reinforcement Learning i trading
- Modulär systemarkitektur
- Agent-baserat beslutsfattande
- Adaptiv parameterstyrning

**OBS:** Detta är ett simulerings- och utbildningssystem. Använd inte för verklig trading utan grundlig testning och riskbedömning.
