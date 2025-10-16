# NextGen AI Trader - Snabbstart

## Installation

### 1. Klona repository
```bash
git clone https://github.com/robinandreeklund-collab/NextGen.git
cd NextGen
```

### 2. Installera Python-beroenden
```bash
pip install -r requirements.txt
```

Detta installerar alla nödvändiga paket:
- `numpy` - Numeriska beräkningar
- `torch` - Deep learning (för RL-agenter)
- `gymnasium` - RL environments
- `dash` - Dashboard-visualisering
- `plotly` - Interaktiva grafer
- `pyyaml` - Konfigurationsfiler
- `pytest` - Testning
- `requests`, `websocket-client` - API-kommunikation

## Köra Demos

### Sprint 3 Demo - Feedbackloopar och Introspektion
```bash
python demo_sprint3.py
```

Detta visar:
- News & Insider Sentiment indikatorer
- Intelligent feedback-routing med prioritering
- Mönsteranalys (slippage, success rate, capital changes)
- Dashboard-data generering
- Agent adaptation tracking

### Sprint 3 Dashboard (Interaktiv Visualisering)
```bash
python dashboards/feedback_flow.py
```

Öppna sedan webbläsaren på: **http://localhost:8050**

Dashboard visar:
- Network graph av modulkommunikation
- Real-time feedback metrics
- Priority distribution
- Timeline av events
- Recent events table

**Obs:** Dashboard startar med demo-data när det körs standalone. För att se live data från systemet, integrera dashboarden med en körande instans av NextGen AI Trader.

**Obs:** Tryck `Ctrl+C` för att stoppa dashboard-servern.

## Köra Tester

### Alla Sprint 3 tester
```bash
pytest tests/test_feedback_analyzer.py -v
```

Förväntat resultat: **23/23 tester passerar**

### Specifika test-klasser
```bash
# Endast FeedbackAnalyzer tester
pytest tests/test_feedback_analyzer.py::TestFeedbackAnalyzer -v

# Endast FeedbackRouter tester
pytest tests/test_feedback_analyzer.py::TestFeedbackRouter -v

# Endast IntrospectionPanel tester
pytest tests/test_feedback_analyzer.py::TestIntrospectionPanel -v
```

## Felsökning

### Problem: "No module named 'dash'" eller "No module named 'plotly'"
**Lösning:** Kör `pip install -r requirements.txt`

### Problem: "No module named 'modules'"
**Lösning:** Kör scripten från projektets root-katalog (NextGen/), inte från dashboards/ eller modules/

### Problem: Dashboard startar inte
**Lösning:** 
1. Kontrollera att port 8050 inte används av annan applikation
2. Kör med annan port: Redigera `dashboards/feedback_flow.py` och ändra `port=8050` till `port=8051`

### Problem: "Plotly Express requires numpy"
**Lösning:** `pip install numpy plotly`

## Projektstruktur

```
NextGen/
├── modules/              # Alla kärnmoduler
│   ├── feedback_router.py
│   ├── feedback_analyzer.py
│   ├── introspection_panel.py
│   └── ...
├── dashboards/          # Dash-visualiseringar
│   └── feedback_flow.py
├── tests/               # Pytest tester
│   └── test_feedback_analyzer.py
├── demo_sprint3.py      # Sprint 3 demonstration
├── requirements.txt     # Python-beroenden
└── README.md           # Huvuddokumentation

```

## Nästa Steg

1. **Utforska Sprint 3 funktionalitet:**
   - Kör `python demo_sprint3.py`
   - Kör `python dashboards/feedback_flow.py` och öppna http://localhost:8050

2. **Kör tester:**
   - `pytest tests/test_feedback_analyzer.py -v`

3. **Läs dokumentation:**
   - `README.md` - Komplett systemöversikt
   - `docs/README_sprints.md` - Detaljerad sprint-dokumentation
   - `SPRINT3_SUMMARY.md` - Sprint 3 implementation summary

4. **Nästa Sprint:**
   - Sprint 4: Strategiskt minne och agentutveckling (kommande)

## Support

För frågor eller problem, skapa en issue på GitHub:
https://github.com/robinandreeklund-collab/NextGen/issues

## Licens

Se LICENSE-filen i repository.
