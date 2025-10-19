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

---

## 🎯 Finnhub Orchestrator

**Ny i Sprint 9** - Central orchestrator för koordination av dataflöden från Finnhub.

### Översikt

Finnhub Orchestrator är en central modul som koordinerar alla aspekter av datainsamling, symbolrotation, indikatorsyntes och RL-driven optimering. Designad för skalbarhet och flexibilitet med plug-n-play-arkitektur.

### Huvudfunktioner

1. **RL-Driven Symbolprioritering**
   - Automatisk prioritering av symboler baserat på RL-feedback
   - Dynamisk justering av symbolval för optimal portföljprestanda
   - Integration med PPO och DQN controllers

2. **Adaptiv Symbolrotation**
   - Tid-baserad rotation
   - Prestanda-baserad rotation
   - RL-rekommenderad rotation
   - Marknadsregim-driven rotation

3. **Indikatorsyntes**
   - Kombinerar flera indikatorer till syntetiska metrics
   - Momentum composite (RSI + MACD + Stochastic)
   - Volatilitet composite (ATR + Bollinger Width)
   - Trend strength (ADX + MACD histogram)
   - Divergens-detektering

4. **Stream Replay Engine**
   - Historisk datareprisering för test
   - Syntetisk datagenerering
   - Hybrid-läge (mix av verklig och syntetisk data)
   - Konfigurerbar replay-hastighet (0.1x - 10x)

5. **Stream Ontology Mapper**
   - Normaliserar data från olika källor (Finnhub, Yahoo, Alpha Vantage)
   - Enhetlig dataschema för downstream-moduler
   - Datavalidering och typkonvertering

6. **Infrastruktur**
   - Audit logging till `logs/orchestrator_audit.json`
   - Rate limiting (10 req/s med burst support)
   - Failover med retry-logik
   - Health monitoring

### Submoduler

#### 1. **IndicatorSynthEngine**
Synthesiserar indikatorkombinationer och härledda metrics.

```python
from modules.indicator_synth_engine import IndicatorSynthEngine

engine = IndicatorSynthEngine(message_bus)
synthetic_indicators = engine.synthesize(['AAPL', 'TSLA'])
```

#### 2. **SymbolRotationEngine**
Hanterar symbolrotation baserat på prioriteringar och strategier.

```python
from modules.symbol_rotation_engine import SymbolRotationEngine

rotation_engine = SymbolRotationEngine(message_bus, rotation_interval=300)
new_symbols = rotation_engine.rotate_symbols(
    current_symbols=['AAPL', 'TSLA'],
    priorities={'AAPL': 0.8, 'TSLA': 0.3},
    strategy={'type': 'top_priority', 'rotation_rate': 0.3},
    max_symbols=10
)
```

#### 3. **RotationStrategyEngine**
Bestämmer rotationsstrategier baserat på RL-feedback och prestanda.

```python
from modules.rotation_strategy_engine import RotationStrategyEngine

strategy_engine = RotationStrategyEngine(message_bus)
strategy = strategy_engine.compute_rotation_strategy(
    priorities={'AAPL': 0.6, 'TSLA': 0.4},
    metrics={},
    current_symbols=['AAPL', 'TSLA']
)
```

#### 4. **StreamStrategyAgent**
RL-agent som optimerar streaming-strategier och resursallokering.

```python
from modules.stream_strategy_agent import StreamStrategyAgent

agent = StreamStrategyAgent(message_bus)
scores = agent.get_symbol_scores(['AAPL', 'TSLA'], metrics={})
strategy = agent.update_strategy(metrics={}, priorities={'AAPL': 0.7})
```

#### 5. **StreamReplayEngine**
Repriserar historisk eller syntetisk data för simulering och testning.

```python
from modules.stream_replay_engine import StreamReplayEngine

replay_engine = StreamReplayEngine(message_bus)
replay_engine.start_replay({
    'mode': 'historical',  # eller 'synthetic', 'hybrid'
    'speed': 2.0,
    'symbols': ['AAPL', 'TSLA']
})
```

#### 6. **StreamOntologyMapper**
Mappar och normaliserar data från olika källor.

```python
from modules.stream_ontology_mapper import StreamOntologyMapper

mapper = StreamOntologyMapper(message_bus)
normalized_data = mapper.map_data(
    {'p': 150.5, 's': 'AAPL', 't': 1234567890000, 'v': 1000000},
    source='finnhub'
)
```

### Användning

#### Starta Orchestrator

```python
from modules.finnhub_orchestrator import FinnhubOrchestrator
from modules.message_bus import MessageBus

message_bus = MessageBus()
orchestrator = FinnhubOrchestrator(
    api_key='your_finnhub_api_key',
    message_bus=message_bus,
    live_mode=False  # False för demo, True för live
)

# Starta orchestrator
orchestrator.start()

# Orchestratorn kör nu i bakgrunden och:
# - Roterar symboler automatiskt
# - Uppdaterar RL-prioriteringar
# - Synthesiserar indikatorer
# - Publicerar metrics till message_bus
```

#### Dynamisk Konfiguration

```python
# Uppdatera konfiguration under körning
orchestrator.update_config({
    'rotation_interval': 600,  # 10 minuter
    'max_concurrent_streams': 15,
    'adaptive_params': {
        'rotation_threshold': 0.7,
        'batch_size': 20
    }
})
```

#### Replay-Läge

```python
# Aktivera replay för test/simulering
orchestrator.enable_replay_mode({
    'mode': 'synthetic',
    'speed': 5.0,  # 5x hastighet
    'symbols': ['AAPL', 'TSLA', 'GOOGL']
})

# Stäng av replay
orchestrator.disable_replay_mode()
```

### Dashboard-Integration

Orchestratorn har en dedikerad panel i dashboarden som visar:

1. **Status** - Körstatus, läge (Live/Demo), aktiva symboler
2. **Symbol Rotation Timeline** - Visualisering av rotationshändelser
3. **RL-Driven Priorities** - Bar chart med symbolprioriteter
4. **Stream Health** - Gauge som visar systemhälsa
5. **Replay Status** - Status för replay engine

**Navigera till Orchestrator-panelen:**
- Starta dashboarden: `python start_demo.py`
- Öppna webbläsaren: `http://localhost:8050`
- Klicka på "🎯 Orchestrator" i sidomenyn

### Dataflöde

```
Finnhub API/WebSocket
        ↓
StreamOntologyMapper (normalisering)
        ↓
FinnhubOrchestrator
        ├── IndicatorSynthEngine → Syntetiska indikatorer
        ├── SymbolRotationEngine → Symbolval
        ├── RotationStrategyEngine → Strategibeslut
        ├── StreamStrategyAgent → RL-optimering
        └── StreamReplayEngine → Test/Simulering
        ↓
message_bus (topics: orchestrator_status, symbol_rotation, rl_scores, etc.)
        ↓
Downstream Modules (indicator_registry, strategy_engine, etc.)
```

### Message Bus Topics

Orchestratorn publicerar till följande topics:

- **orchestrator_status** - Hälsa och metrics
- **symbol_rotation** - Rotationshändelser
- **stream_metrics** - Prestanda data
- **rl_scores** - Symbolprioriteter
- **replay_data** - Repriserad data
- **indicator_synth_data** - Syntetiska indikatorer
- **mapped_data** - Normaliserad data

Och prenumererar på:

- **rl_feedback** - Feedback från RL controllers
- **market_conditions** - Marknadslägesändringar
- **module_requests** - Förfrågningar från moduler

### Konfiguration

Standardkonfiguration finns i `docs/finnhub_orchestrator.yaml`. Viktiga parametrar:

```yaml
configuration:
  default_symbols: ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN"]
  rotation_interval: 300  # sekunder
  max_concurrent_streams: 10
  buffer_size: 1000
  priority_update_interval: 60

adaptive_parameters:
  rotation_threshold: 0.5  # (0.1 - 0.9)
  batch_size: 10  # (1 - 100)
  priority_weight: 0.7  # (0.0 - 1.0)
  replay_speed: 1.0  # (0.1 - 10.0)
```

### Testning

Omfattande testsvit med 25 tester täcker alla aspekter:

```bash
# Kör orchestrator-tester
pytest tests/test_finnhub_orchestrator.py -v

# Resultat: 25/25 tester passerar
```

Tester inkluderar:
- Initialisering av alla submoduler
- Symbolrotation (priority-based, random, hybrid)
- Indikatorsyntes
- RL-strategioptimering
- Replay engine (start/stop, speed adjustment)
- Data mapping och normalisering
- Message bus integration
- Dynamisk konfiguration

### Plug-n-Play Design

Orchestratorn är designad för att kunna köras både fristående och integrerat:

**Fristående:**
```python
# Kör orchestrator isolerat
orchestrator = FinnhubOrchestrator(api_key, message_bus, live_mode=False)
orchestrator.start()
# Orchestratorn kör sin egen loop och publicerar data
```

**Integrerat:**
```python
# Automatiskt integrerat i dashboard
# start_demo.py och start_live.py startar orchestratorn automatiskt
```

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
| Sprint 9 | ✅ Färdig | Finnhub Orchestrator – Central datakoordinering och RL-driven symbolrotation |
| Sprint 10 | ✅ Färdig | Decision Transformer – Sequence-based RL & 5-agent ensemble |

**Testresultat:** ✅ 396/396 tester passerar (100%)

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
| `finnhub_orchestrator.py` | Central orchestrator för datakoordinering och RL-driven rotation | Sprint 9 |
| `indicator_synth_engine.py` | Synthesiserar indikatorkombinationer | Sprint 9 |
| `symbol_rotation_engine.py` | Hanterar symbolrotation | Sprint 9 |
| `rotation_strategy_engine.py` | RL-driven rotationsstrategi | Sprint 9 |
| `stream_strategy_agent.py` | RL-agent för streaming-optimering | Sprint 9 |
| `stream_replay_engine.py` | Reprisering av historisk data | Sprint 9 |
| `stream_ontology_mapper.py` | Datanormalisering från olika källor | Sprint 9 |
| `decision_transformer_agent.py` | Decision Transformer för sequence-based RL | Sprint 10 |
| `ensemble_coordinator.py` | Koordinerar 5-agent ensemble (PPO, DQN, DT, GAN, GNN) | Sprint 10 |
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
| `agent_manager.py` | Agentprofiler och versioner | Sprint 4, 10 |
| `feedback_router.py` | Intelligent feedback-routing | Sprint 3 |
| `feedback_analyzer.py` | Mönsteranalys i feedback | Sprint 3 |
| `introspection_panel.py` | Dashboard-data för visualisering | Sprint 3, 7 |

### Adaptiva Parametrar

Systemet har **23+ adaptiva parametrar** som justeras automatiskt via RL/PPO:

**Sprint 10 (Decision Transformer):**
- dt_learning_rate (0.00001-0.001)
- dt_sequence_length (10-50)
- dt_num_layers (2-6)
- dt_target_return_weight (0.5-2.0)
- dt_embed_dim (64-256)
- dt_num_heads (2-8)
- dt_dropout (0.0-0.3)

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
Market Data (Finnhub API/WebSocket)
        ↓
StreamOntologyMapper (normalisering)
        ↓
FinnhubOrchestrator
        ├── SymbolRotationEngine (dynamisk symbolval)
        ├── IndicatorSynthEngine (syntetiska indikatorer)
        ├── RotationStrategyEngine (RL-driven strategi)
        └── StreamStrategyAgent (RL-optimering)
        ↓
message_bus → orchestrator_status, symbol_rotation, rl_scores
        ↓
Data Ingestion → Indicators
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
├── modules/                     # Alla kärnmoduler (37 stycken)
│   ├── finnhub_orchestrator.py       # 🆕 Sprint 9: Central orchestrator
│   ├── indicator_synth_engine.py     # 🆕 Sprint 9: Indikatorsyntes
│   ├── symbol_rotation_engine.py     # 🆕 Sprint 9: Symbolrotation
│   ├── rotation_strategy_engine.py   # 🆕 Sprint 9: Rotationsstrategi
│   ├── stream_strategy_agent.py      # 🆕 Sprint 9: RL stream-agent
│   ├── stream_replay_engine.py       # 🆕 Sprint 9: Replay engine
│   ├── stream_ontology_mapper.py     # 🆕 Sprint 9: Data mapping
│   ├── data_ingestion.py        # Live WebSocket från Finnhub
│   ├── data_ingestion_sim.py    # Simulerad marknadsdata för demo
│   ├── reward_tuner.py          # Sprint 4.4: Reward transformation
│   ├── rl_controller.py         # Sprint 2, 4.2: PPO-agenter
│   ├── dqn_controller.py        # Sprint 8: DQN RL
│   ├── gan_evolution_engine.py  # Sprint 8: GAN evolution
│   ├── gnn_timespan_analyzer.py # Sprint 8: GNN temporal analysis
│   ├── consensus_engine.py      # Sprint 5: Konsensusbeslut
│   ├── timespan_tracker.py      # Sprint 6: Timeline-analys
│   └── ...
├── dashboards/                  # Dash-visualiseringar (komponenter)
├── tests/                       # 368 tester (100% pass rate)
│   └── test_finnhub_orchestrator.py  # 🆕 25 orchestrator-tester
├── docs/                        # Dokumentation och YAML-specs
│   ├── finnhub_orchestrator.yaml     # 🆕 Orchestrator spec
│   ├── dashboard_structure_sprint8.yaml  # Dashboard spec
│   ├── adaptive_parameters_sprint8.yaml  # Parameter spec
│   ├── sprint_8.yaml                     # Sprint 8 overview
│   └── ...
├── logs/
│   └── orchestrator_audit.json       # 🆕 Orchestrator audit log
└── requirements.txt
```

---

## 🧪 Testning

```bash
# Kör alla tester
pytest tests/ -v

# Kör specifika test-suiter
pytest tests/test_finnhub_orchestrator.py -v    # Sprint 9: Orchestrator (25 tester)
pytest tests/test_reward_tuner.py -v              # Sprint 4.4
pytest tests/test_adaptive_parameters_sprint4_3.py -v  # Sprint 4.3
pytest tests/test_consensus_engine.py -v          # Sprint 5
pytest tests/test_timespan_tracker.py -v          # Sprint 6

# Med coverage
pytest tests/ --cov=modules --cov-report=html
```

**Testresultat:** ✅ 368/368 tester passerar

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
| **9** | Finnhub Orchestrator | RL-driven symbolrotation och datakoordinering |
| **10** | Decision Transformer | Sequence-based RL med transformer-arkitektur |

---

## 🆕 Sprint 10: Decision Transformer – Sequence-Based RL

Sprint 10 integrerar Decision Transformer (DT) för sekvensbaserad reinforcement learning:

### Nya Moduler

**1. Decision Transformer Agent (`decision_transformer_agent.py`)**
- Transformer-arkitektur för sekvensbaserad RL
- Multi-head attention mechanism för temporal dependencies
- Processar (state, action, reward, return-to-go) sekvenser
- Offline learning från historiska trajectories
- Integreras i 5-agent ensemble med PPO, DQN, GAN, GNN

**2. Ensemble Coordinator (`ensemble_coordinator.py`)**
- Koordinerar beslut från alla agenter (PPO, DQN, DT, GAN, GNN)
- Weighted voting för final decisions
- Konfliktdetektion och resolution
- Performance tracking per agent
- Adaptiva vikter baserat på agent performance

### Decision Transformer Funktioner

**Transformer Architecture:**
- State embeddings: Linear projection till embed_dim
- Action embeddings: Linear projection till embed_dim
- Return-to-go embeddings: Target return representation
- Positional encoding: För temporal order
- Multi-head attention: 4 heads för parallel attention
- 3 transformer layers: Deep sequence modeling
- Action prediction head: Final action från state embedding

**Sequence Processing:**
- Max sequence length: 20 timesteps
- Return-to-go calculation: Discounted cumulative rewards
- Causal masking: Prevent future information leakage
- Batch training: 32 sequences per batch
- Experience replay: 1000 sequence buffer

**Integration Points:**
- Strategic Memory Engine: Provides decision histories
- Reward Tuner: Supplies tuned rewards for training
- Message Bus: Publishes actions and metrics
- Ensemble Coordinator: Participates in voting

### 5-Agent Ensemble Architecture

**Agent Weights (default):**
- PPO: 30% - Policy gradient optimization
- DQN: 30% - Q-value optimization
- DT: 20% - Sequence-based decisions
- GAN: 10% - Evolution guidance
- GNN: 10% - Temporal pattern insights

**Conflict Resolution:**
- Weighted average voting
- Confidence-based weighting
- Agreement score calculation
- Entropy-based clarity measure
- Adaptive weight adjustment

**Performance Tracking:**
- Individual agent rewards
- Prediction accuracy per agent
- Conflict frequency monitoring
- Ensemble diversity metrics
- Agent contribution analysis

### Dashboard Integration

**DT Analysis Panel (`dt_analysis_panel.yaml`):**
1. **Action Predictions** - Current action, confidence, probability distribution
2. **Return-to-Go Tracking** - Target vs predicted RTG, achievement rate
3. **Attention Analysis** - Multi-head attention heatmap, per-layer distribution
4. **Training Progress** - Loss curves, training steps, buffer utilization
5. **Agent Comparison** - Performance comparison across PPO, DQN, DT, GAN, GNN
6. **Sequence Visualization** - Decision sequence timeline, state-action space

### Adaptive Parameters

**DT-specific (controlled by RL):**
- dt_learning_rate: 0.00001 - 0.001 (default 0.0001)
- dt_sequence_length: 10 - 50 (default 20)
- dt_num_layers: 2 - 6 (default 3)
- dt_target_return_weight: 0.5 - 2.0 (default 1.0)
- dt_embed_dim: 64 - 256 (default 128)
- dt_num_heads: 2 - 8 (default 4)
- dt_dropout: 0.0 - 0.3 (default 0.1)

**Ensemble weights:**
- ppo_weight, dqn_weight, dt_weight, gan_weight, gnn_weight
- Constraint: Sum to 1.0
- Adaptive adjustment based on performance

### Testning

**26 nya tester för Decision Transformer:**
- Transformer block tests (2)
- DT model tests (3)
- DT agent tests (19)
- Integration tests (2)

**Total testning:** 396/396 tester passerar (100%)

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

**PPO + DQN + DT Parallell Exekvering:**
- PPO (från Sprint 2-7): Policy gradient-optimering
- DQN (Sprint 8): Q-value-optimering
- DT (Sprint 10): Sequence-based transformer
- Alla får samma rewards från portfolio_manager och reward_tuner
- Koordinerad via ensemble_coordinator och message_bus
- Konfliktdetektion och resolution

**Fördelar:**
- PPO: Bra för kontinuerliga åtgärdsval
- DQN: Bra för diskreta beslutsrum
- DT: Utmärkt för sekventiellt beslutsfattande
- Kombinerat: Maximalt robust och stabilt

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
