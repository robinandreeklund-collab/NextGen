# Dashboard Visuell Guide

## NextGen AI Trader - Feedback Flow Dashboard

När du kör `python dashboards/feedback_flow.py` och öppnar http://localhost:8050 i webbläsaren ser du:

## Layout

```
╔════════════════════════════════════════════════════════════════════╗
║         NextGen AI Trader - Feedbackflöde                          ║
╚════════════════════════════════════════════════════════════════════╝

┌────────────────────────────────┬───────────────────────────────────┐
│ Modul-kommunikation            │ Feedback Metrics                  │
│                                │                                   │
│  [Network Graph]               │  ┌─────────────────────────────┐ │
│   • execution_engine           │  │ Total Events: 20            │ │
│   • portfolio_manager          │  │ Events/min: 0.12            │ │
│   • rl_controller              │  └─────────────────────────────┘ │
│   • feedback_analyzer          │                                   │
│   • strategic_memory           │  Events per Källa:                │
│                                │  • execution_engine: 10           │
│  Visar kopplingar mellan       │  • portfolio_manager: 5           │
│  moduler som pilar             │  • strategic_memory: 5            │
│                                │                                   │
│                                │  Prioritetsfördelning             │
│                                │  [Pie Chart]                      │
│                                │  • High: 40%                      │
│                                │  • Medium: 40%                    │
│                                │  • Critical: 20%                  │
└────────────────────────────────┴───────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ Feedback Trends över tid                                          │
│                                                                    │
│  [Timeline Chart]                                                  │
│  Visar feedback events från olika källor över tid                 │
│  • execution_engine (blå punkter)                                 │
│  • portfolio_manager (orange punkter)                             │
│  • strategic_memory (grön punkter)                                │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ Senaste Feedback Events                                           │
│                                                                    │
│ Källa              │ Triggers        │ Prioritet │ Timestamp      │
│────────────────────┼─────────────────┼───────────┼────────────────│
│ execution_engine   │ trade_result    │ high      │ 1729082...     │
│ portfolio_manager  │ capital_change  │ medium    │ 1729082...     │
│ execution_engine   │ slippage        │ high      │ 1729082...     │
│ strategic_memory   │ decision_out... │ medium    │ 1729082...     │
│ ...                │ ...             │ ...       │ ...            │
└──────────────────────────────────────────────────────────────────┘
```

## Funktioner

### 1. Network Graph (Vänster övre)
- Visar modulkommunikation som nätverk
- Noder = moduler (execution_engine, portfolio_manager, etc.)
- Linjer = kommunikationsflöden mellan moduler
- Hover över noder för att se detaljer

### 2. Feedback Metrics (Höger övre)
- **Total Events**: Antal feedback events totalt
- **Events/min**: Genomsnittlig rate av events
- **Events per Källa**: Fördelning av events per modul
- **Prioritetsfördelning**: Pie chart av prioriteter (critical/high/medium/low)

### 3. Timeline (Mitten)
- Visar feedback events över tid
- Varje punkt = ett event
- Färger = olika källor
- Hover över punkter för detaljer

### 4. Recent Events Table (Botten)
- Lista över senaste 10 feedback events
- Kolumner: Källa, Triggers, Prioritet, Timestamp
- Uppdateras automatiskt var 5:e sekund

## Demo-data

När dashboarden körs standalone genereras automatiskt:
- **15 agent status updates** från strategy_engine, risk_manager, decision_engine
- **20 feedback events** med varierande prioriteter och triggers
- **5 indicator snapshots** med RSI, MACD data

Detta visar hur dashboarden ser ut med verkliga data.

## Auto-uppdatering

Dashboarden uppdateras automatiskt var 5:e sekund via Dash's Interval-komponent.

## Interaktivitet

- **Zoom**: Klicka och dra i grafer för att zooma
- **Pan**: Shift + klicka och dra för att panorera
- **Hover**: Håll musen över punkter för tooltips
- **Reset**: Dubbel-klicka för att återställa zoom

## Tips

1. **Öppna i större fönster** - Dashboard är optimerad för skärmar > 1200px bredd
2. **Använd Chrome/Firefox** - Bäst kompatibilitet med Plotly
3. **Refresh sidan** - Om data verkar hänga sig, refresha webbläsaren
4. **Kontrollera konsolen** - Vid problem, kolla browser developer console (F12)

## Integration med Live-system

För att se live data istället för demo-data, kör dashboarden som del av huvudapplikationen:

```python
# I din main.py eller liknande
from dashboards.feedback_flow import create_feedback_flow_dashboard
from modules.introspection_panel import IntrospectionPanel
from modules.message_bus import message_bus

# Skapa panel som lyssnar på systemets message_bus
panel = IntrospectionPanel(message_bus)

# Skapa och kör dashboard
app = create_feedback_flow_dashboard(panel)
app.run(debug=False, port=8050)
```

Nu kommer dashboarden visa data från det körande systemet i realtid!
