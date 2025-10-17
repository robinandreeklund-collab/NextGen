# 🔍 Analyzer Debug Dashboard - Guide

## Översikt

`analyzer_debug.py` är en omfattande debug- och analysdashboard för NextGen AI Trader-systemet. Den ger realtidsvisualisering av alla systemkomponenter, dataflöden, RL-analys, agentutveckling och systemhälsa.

## 🚀 Snabbstart

```bash
# Starta dashboarden
python analyzer_debug.py

# Öppna webbläsaren
http://localhost:8050

# Verifiera installation
python test_analyzer_debug.py
```

## 📊 Dashboard-sektioner

### 1. System Overview
- **System Health Gauge** - Visar systemhälsa 0-100%
- **Module Status Chart** - Status för alla 26 moduler (aktiva/stale)

### 2. Data Flow & Simulation
- **Price Trends** - Prisutveckling för alla symboler (AAPL, MSFT, GOOGL, AMZN, TSLA)
- **Technical Indicators** - RSI med overbought (70) och oversold (30) linjer
- **Decision Flow** - Totalt beslut, executions, success rate

### 3. RL Analysis
- **Reward Flow** (Sprint 4.4) - Base rewards vs tuned rewards över tid
- **Parameter Evolution** (Sprint 4.3) - Adaptiva parametrar över tid
- **Agent Performance** - Training loss per agent

### 4. Agent Development
- **Agent Evolution** - Agent versioner över tid
- **Agent Metrics** - Performance metriker per agent

### 5. Debug & Logging
- **Event Log** - Realtidslogg av alla events (senaste 20)
- **Feedback Flow** - Pie chart över feedback per källa
- **Timeline Analysis** (Sprint 6) - Decision events, final decisions, symboler

### 6. Portfolio
- **Portfolio Value** - Cash, holdings, total value
- **Current Positions** - Nuvarande innehav med quantity och värde
- **Portfolio Details** - Sammanfattning av P&L, ROI, positions

## 🎮 Kontroller

- **Start Simulation** - Startar simulerad trading med syntetiska prisrörelser
- **Stop Simulation** - Stoppar aktiv simulering eller live data
- **Live Data (Finnhub)** - Ansluter till Finnhub WebSocket för realtidsdata från marknaden
- **Auto-refresh** - Uppdaterar automatiskt var 2:a sekund

## 🔧 Funktionalitet

### Simulering (Offline Mode)
Dashboard inkluderar inbyggd simulering baserad på `sim_test.py`:

- **Prisrörelser** - Simulerar aggressiva prisrörelser med trends
- **Indikatorer** - Genererar RSI, MACD, ATR, analyst ratings
- **Beslut** - Komplett beslutsflöde genom alla moduler
- **Execution** - Simulerad trade execution med portfolio updates

### Live Data (Finnhub WebSocket)
**NYT!** Dashboard kan nu ansluta till verklig marknadsdata:

- **WebSocket-anslutning** - Direktanslutning till Finnhub för realtidsdata
- **Live priser** - Faktiska marknadsdata från 5 symboler (AAPL, MSFT, GOOGL, AMZN, TSLA)
- **Realtidsindikatorer** - Hämtar faktiska tekniska indikatorer från Finnhub API
- **Verkliga beslut** - Fattar handelsbeslut baserat på live marknadsdata
- **Portfolio tracking** - Spårar portfolio med verkliga prisrörelser

**Användning:**
1. Klicka på "Live Data (Finnhub)" knappen
2. Vänta på WebSocket-anslutning (visas i konsolen)
3. Dashboard börjar ta emot och processa live trades
4. Beslut fattas var 10:e trade, indikatorer uppdateras var 5:e trade
5. Klicka "Stop Simulation" för att stoppa

**Obs:** Live data kräver giltig Finnhub API-nyckel. Ange API-nyckeln via en miljövariabel (`FINNHUB_API_KEY`) eller en säker konfigurationsfil – inkludera aldrig nyckeln direkt i koden.

### Datakällor
- **Message Bus** - Central pub/sub för all modulkommunikation
- **Introspection Panel** - Agent status, feedback events, indicators
- **System Monitor** - System health, module status
- **Portfolio Manager** - Portfolio status, positions, P&L
- **Reward Tuner** (Sprint 4.4) - Base/tuned rewards, volatility
- **RL Controller** (Sprint 4.2, 4.3) - Agent performance, parameter adjustments
- **Strategic Memory** (Sprint 4) - Decision history, insights
- **Timespan Tracker** (Sprint 6) - Timeline events

## 📈 Visualiseringsdetaljer

### System Health Gauge
- Range: 0-100%
- Färger: Grå (0-50), Ljusgrå (50-80), Blå (80-100)
- Tröskelvärde: 90% (röd linje)

### Price Trends
- Bar chart per symbol
- Visar % förändring från start
- Aktuellt pris visas på bar

### RSI Indicators
- Bar chart per symbol
- Färger:
  - Röd: Overbought (>70)
  - Grön: Oversold (<30)
  - Blå: Neutral (30-70)
- Horisontella linjer vid 70 och 30

### Reward Flow (Sprint 4.4)
- Line chart med två serier:
  - Base Reward (blå) - Rådata från portfolio
  - Tuned Reward (grön) - Transformerad av RewardTuner
- Visar reward transformation och volatility control

### Parameter Evolution (Sprint 4.3)
- Line chart med markers
- Visar 3 nyckelparametrar:
  - signal_threshold
  - risk_tolerance
  - consensus_threshold
- Spårar adaptiv evolution över tid

## 🧪 Testning

### Verification Test
```bash
python test_analyzer_debug.py
```

**Testar:**
1. Dashboard creation
2. Module initialization (20 moduler)
3. Simulation iteration
4. Graph creation (16 grafer)
5. Data flow through system
6. Price movement
7. Reward flow (Sprint 4.4)
8. Adaptive parameters (Sprint 4.3)

### Unit Tests
```bash
pytest tests/ -v
```

Alla 214 tester måste passera.

## 🔍 Felsökning

### Dashboard startar inte
```bash
# Kontrollera att alla dependencies är installerade
pip install -r requirements.txt

# Verifiera att moduler kan importeras
python -c "import analyzer_debug; print('OK')"
```

### Port redan i bruk
```python
# Ändra port i analyzer_debug.py, sista raden:
self.app.run(debug=True, host='0.0.0.0', port=8051)  # Ändra 8050 → 8051
```

### Grafer visar inga data
- Dashboard behöver köras en stund för att samla data
- Klicka "Start Simulation" för att börja generera data
- Vänta 10-20 sekunder för att data ska flöda genom systemet

### ModuleNotFoundError
```bash
# Installera saknade dependencies
pip install dash plotly numpy torch gymnasium

# Verifiera installation
python -c "import dash; import plotly; print('OK')"
```

## 📊 Metriker och Prestanda

### Uppdateringsfrekvens
- Refresh interval: 2 sekunder
- Simuleringstakt: 2 iterationer/sekund
- Graph rendering: <100ms per graf

### Minnesanvändning
- Dashboard: ~200MB
- Simulering: ~50MB
- History buffers: ~20MB (begränsade till senaste 50-100 entries)

### Nätverkstrafik
- Initial load: ~2MB
- Per update: ~50KB (endast data, inte assets)

## 🔗 Integration med Systemet

### Moduler som används (26 totalt)
1. message_bus - Central kommunikation
2. strategic_memory_engine - Beslutshistorik
3. meta_agent_evolution_engine - Agent evolution
4. agent_manager - Agent profiles
5. feedback_router - Feedback routing
6. feedback_analyzer - Mönsteranalys
7. introspection_panel - Dashboard data
8. rl_controller - RL training
9. reward_tuner - Reward transformation
10. strategy_engine - Trade förslag
11. risk_manager - Risk assessment
12. decision_engine - Beslut
13. execution_engine - Trade execution
14. portfolio_manager - Portfolio management
15. vote_engine - Voting
16. decision_simulator - Simulering
17. consensus_engine - Konsensus
18. timespan_tracker - Timeline
19. action_chain_engine - Action chains
20. system_monitor - System health

### Message Bus Topics
Dashboard prenumererar på:
- agent_status
- feedback_event
- indicator_data
- portfolio_status
- parameter_adjustment
- reward_metrics
- resource_allocation
- team_metrics

### Dataflöde
```
Market Data → Indicators → Strategy → Risk → Decision
                                                ↓
                                           Execution
                                                ↓
                                           Portfolio → Base Reward
                                                ↓
                                          Reward Tuner → Tuned Reward
                                                ↓
                                          RL Controller → Agent Updates
                                                ↓
                                          Dashboard (visualisering)
```

## 📚 Kodexempel

### Skapa egen dashboard-komponent
```python
from analyzer_debug import AnalyzerDebugDashboard

# Skapa dashboard
dashboard = AnalyzerDebugDashboard()

# Kör en simulering
dashboard.simulate_iteration()

# Hämta data
system_health = dashboard.system_monitor.get_system_health()
portfolio = dashboard.portfolio_manager.get_status(dashboard.current_prices)

# Skapa graf
fig = dashboard.create_price_trends_graph()
```

### Lägg till ny graf
```python
def create_my_custom_graph(self):
    """Skapar en custom graf."""
    # Hämta data från moduler
    data = self.my_module.get_data()
    
    # Skapa Plotly graf
    fig = go.Figure(data=[
        go.Scatter(x=data['x'], y=data['y'], mode='lines')
    ])
    
    fig.update_layout(
        title="My Custom Graph",
        xaxis_title="X Axis",
        yaxis_title="Y Axis",
        height=400
    )
    
    return fig
```

### Lägg till ny callback
```python
@self.app.callback(
    Output('my-graph-id', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_my_graph(n):
    return self.create_my_custom_graph()
```

## 🎯 Best Practices

### Användning
1. Starta dashboard först
2. Klicka "Start Simulation" 
3. Vänta 10-20 sekunder för data
4. Utforska olika tabs
5. Stoppa simulation när färdig

### Prestanda
- Starta inte flera dashboards samtidigt
- Använd "Stop Simulation" när du inte behöver data
- Stäng onödiga browser tabs

### Debugging
- Använd Event Log-sektionen för realtidslogg
- Kontrollera System Health för modulstatus
- Se Feedback Flow för kommunikation
- Reward Flow visar RL-training kvalitet

## 📝 Changelog

### Version 1.0 (2025-10-17)
- ✅ Initial release
- ✅ 6 huvudsektioner
- ✅ 16 visualiseringar
- ✅ Integration med alla 26 moduler
- ✅ Simulering från sim_test.py
- ✅ Realtidsuppdatering

## 🤝 Support

För frågor eller problem:
1. Kontrollera ANALYZER_DEBUG_GUIDE.md (denna fil)
2. Kör verification test: `python test_analyzer_debug.py`
3. Se README.md för systemöversikt
4. Se README_detailed_backup.md för detaljerad sprintinfo

## 📄 Licens

Del av NextGen AI Trader projektet.
För demonstration och utbildning.
