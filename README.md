# 🚀 NextGen AI Trader

Ett självreflekterande, modulärt och RL-drivet handelssystem byggt för transparens, agentutveckling och realtidsanalys. Systemet simulerar handel med verkliga data, strategier, feedbackloopar och belöningsbaserad inlärning.

---

## 📍 Sprintstatus

**Sprint 1 färdig ✅** – Kärnsystem och demoportfölj komplett
**Sprint 2 färdig ✅** – RL och belöningsflöde komplett
**Sprint 3 färdig ✅** – Feedbackloopar och introspektion komplett
**Sprint 4 färdig ✅** – Strategiskt minne och agentutveckling komplett
**Sprint 4.2 färdig ✅** – Adaptiv parameterstyrning via RL/PPO komplett
**Sprint 4.3 färdig ✅** – Full adaptiv parameterstyrning i alla moduler
**Sprint 4.4 färdig ✅** – Meta-belöningsjustering via RewardTunerAgent komplett
**Sprint 5 färdig ✅** – Simulering och konsensus komplett

### Sprint 4.4: Meta-belöningsjustering via RewardTunerAgent ✅

**Mål:** Inför RewardTunerAgent som meta-agent mellan portfolio_manager och rl_controller för att justera och optimera belöningssignaler.

**Motivation:**
Raw reward från portfolio_manager kan vara volatil och leda till instabil RL-träning. Genom att introducera RewardTunerAgent som meta-agent mellan portfolio och RL-controller kan vi:
- Reducera reward volatilitet för stabilare träning
- Detektera och motverka overfitting patterns
- Skala reward baserat på marknadsförhållanden
- Spåra och visualisera reward transformationer

**Moduler i fokus:**
- `reward_tuner` - Ny meta-agent för reward justering (NY)
- `rl_controller` - Tar emot tuned_reward istället för base_reward
- `portfolio_manager` - Publicerar base_reward istället för reward
- `strategic_memory_engine` - Loggar reward transformationer
- `introspection_panel` - Visualiserar reward flow

**Adaptiva parametrar:**
1. **reward_scaling_factor** (0.5-2.0, default: 1.0)
   - Multiplikativ skalning av base reward
   - Reward signal: training_stability
   - Update frequency: every 20 rewards

2. **volatility_penalty_weight** (0.0-1.0, default: 0.3)
   - Viktning av volatility penalty
   - Reward signal: reward_consistency
   - Update frequency: every epoch

3. **overfitting_detector_threshold** (0.05-0.5, default: 0.2)
   - Tröskelvärde för overfitting detection
   - Reward signal: generalization_score
   - Update frequency: every 50 rewards

**Reward signals:**
- **base_reward**: Raw portfolio value change från portfolio_manager
- **tuned_reward**: Justerad reward efter transformation
- **training_stability**: Stabilitet i RL-träning över tid
- **reward_consistency**: Konsistens i reward utan stora spikar
- **generalization_score**: Förmåga att generalisera utan overfitting

**Implementerat:**
- ✅ RewardTunerAgent-klass för reward transformation
- ✅ Volatility calculation från reward history
- ✅ Overfitting detection från agent performance patterns
- ✅ Reward scaling med adaptive scaling_factor
- ✅ Volatility penalty vid hög reward variation
- ✅ Overfitting penalty vid detekterade patterns
- ✅ Integration mellan portfolio_manager och rl_controller
- ✅ Reward flow logging i strategic_memory_engine
- ✅ Reward visualization i introspection_panel
- ✅ 19 nya tester för RewardTunerAgent (RT-001 till RT-006)
- ✅ Parameter adjustment via MetaParameterAgent
- ✅ Dokumentation i docs/reward_tuner_sprint4_4/

**Testresultat:**
- ✅ Reward volatilitet beräknas korrekt
- ✅ Overfitting detekteras baserat på agent performance
- ✅ Volatility penalty appliceras vid hög volatilitet
- ✅ Reward scaling fungerar med olika scaling factors
- ✅ Full reward flow från portfolio till rl_controller
- ✅ Reward logging i strategic_memory_engine
- ✅ Reward visualization i introspection_panel
- ✅ 19/19 tester passerar för RewardTunerAgent
- ✅ 102/103 totala tester passerar (1 pre-existing failure)
- ✅ Base rewards och tuned rewards genereras korrekt i sim_test.py och websocket_test.py

**Sprint 5 Integration Fix (2025-10-17):**
- ✅ RewardTunerAgent fungerar korrekt med Sprint 5 voting och consensus flow
- ✅ Base rewards publiceras och tas emot av RewardTunerAgent
- ✅ Tuned rewards genereras och skickas till RL controller
- ✅ Reward transformation flow verifierad i både simulering och live data

**Benefits:**
- Stabilare RL-träning genom reducerad reward volatilitet
- Förbättrad generalisering genom overfitting detection
- Adaptiv reward scaling för olika marknadsförhållanden
- Transparent reward transformation med full logging
- Visualisering av reward flow för debugging och analys
- Förhindrar instabil agent behavior från volatila rewards

**Reward Flow:**
```
portfolio_manager
      │ base_reward (raw portfolio change)
      ▼
reward_tuner
      │ • Calculate volatility
      │ • Detect overfitting
      │ • Apply penalties
      │ • Scale reward
      ▼ tuned_reward (adjusted for stability)
rl_controller
      │ • Train PPO agents
      │ • Update policies
      ▼ agent_status
reward_tuner
      │ • Monitor performance
      │ • Adjust parameters
```

**Reward Transformation Algorithm:**
1. **Volatility Analysis**: Beräkna std dev av recent rewards
2. **Volatility Penalty**: IF volatility_ratio > 1.5, apply penalty
3. **Overfitting Detection**: Jämför recent vs long-term performance
4. **Overfitting Penalty**: IF detected, reduce reward by 50%
5. **Reward Scaling**: Multiplicera med reward_scaling_factor
6. **Bounds Enforcement**: Clamp till rimliga gränser
7. **Logging**: Spara transformation för analys

**Metrics Tracked:**
- base_reward och tuned_reward per episode
- transformation_ratio (tuned / base)
- volatility metrics och trends
- overfitting detection events
- parameter evolution över tid

**Integration med existerande system:**
- RewardTunerAgent är transparent för andra moduler
- Portfolio_manager ändrad från 'reward' till 'base_reward' topic
- RL_controller ändrad från 'reward' till 'tuned_reward' topic
- Strategic_memory loggar både base och tuned för korrelation
- Introspection_panel visar reward transformation charts
- Backward compatibility bevarad för existerande tester

### Sprint 5: Simulering och konsensus ✅

**Mål:** Testa alternativa beslut och hantera röstflöden för robust beslutsfattande.

**Motivation:**
Enkel majoritetsröstning är inte alltid tillräcklig för komplexa handelsbeslut. Sprint 5 introducerar beslutssimuleringar där olika scenarier testas innan exekvering, röstmatris för att samla och vikta flera agenters åsikter, och flera konsensusmodeller för att fatta robusta beslut baserat på röstning. Detta möjliggör mer genomtänkta och säkra handelsbeslut med transparent beslutsfattande.

**Moduler i fokus:**
- `decision_simulator` - Simulerar alternativa beslut och beräknar expected value
- `vote_engine` - Skapar röstmatris med viktning och meritbaserad röstning
- `consensus_engine` - Fattar konsensusbeslut baserat på olika konsensusmodeller

**Implementerat:**
- ✅ DecisionSimulator för simulering av beslut i sandbox
- ✅ Scenarier: best_case, expected_case, worst_case, no_action
- ✅ Expected value-beräkning baserat på confidence
- ✅ Rekommendationer: proceed, caution, reject
- ✅ VoteEngine med viktning baserat på agent_vote_weight (Sprint 4.3)
- ✅ Röstmatris med aggregering per action
- ✅ Consensus strength-beräkning
- ✅ ConsensusEngine med 4 konsensusmodeller
- ✅ Majority: Enkel majoritet (flest röster vinner)
- ✅ Weighted: Viktad baserat på confidence och agent performance
- ✅ Unanimous: Kräver 100% enighet
- ✅ Threshold: Kräver minst X% enighet (konfigurerbar)
- ✅ Robusthet-beräkning baserat på röstfördelning
- ✅ 38 tester för Sprint 5 moduler (alla passerar)

**Testresultat:**
- ✅ Decision Simulator simulerar 4 scenarier per beslut
- ✅ Expected value beräknas korrekt från scenarios
- ✅ Rekommendationer baseras på EV och confidence
- ✅ Vote Engine viktar röster med agent_vote_weight
- ✅ Röstmatris aggregerar röster per action
- ✅ Consensus strength beräknas från röstfördelning
- ✅ Majority consensus väljer flest röster
- ✅ Weighted consensus kombinerar röster och confidence
- ✅ Unanimous consensus kräver 100% enighet
- ✅ Threshold consensus kontrollerar tröskelvärde
- ✅ Robusthet beräknas från consensus strength och antal röster
- ✅ 38/38 tester passerar (12 simulator, 12 vote, 14 consensus)

**Integration med Sprint 4.4 (2025-10-17):**
- ✅ Vote Engine och Consensus Engine fungerar korrekt
- ✅ Decision votes publiceras och processas
- ✅ Vote matrices skapas och distribueras automatiskt
- ✅ Consensus decisions fattas baserat på röstmatris
- ✅ RewardTunerAgent (Sprint 4.4) integrerad med voting och consensus
- ✅ Base rewards och tuned rewards flödar korrekt genom systemet
- ✅ Fullständig end-to-end flow verifierad: decision → vote → consensus → execution → reward

**Benefits:**
- Risk-medvetet beslutsfattande genom simulering
- Transparent expected value för varje beslut
- Meritbaserad röstning viktar agenter efter performance
- Flera konsensusmodeller för olika situationer
- Robust beslutsfattande med konfidensberäkning
- Flexibel threshold för olika riskaptiter
- Full integration med adaptiva parametrar (Sprint 4.3)

**Beslutssimulering Flow:**
```
strategy_engine
      │ decision_proposal
      ▼
decision_simulator
      │ • Best case scenario (+5%)
      │ • Expected case (confidence-based)
      │ • Worst case scenario (-3%)
      │ • No action (0%)
      │ • Calculate expected value
      │ • Make recommendation
      ▼ simulation_result
strategic_memory
      │ Log simulation for analysis
```

**Röstning och Konsensus Flow:**
```
decision_engine (agent 1)
decision_engine (agent 2)  │ decision_vote (multiple agents)
decision_engine (agent 3)  │
      ▼
vote_engine
      │ • Collect votes
      │ • Apply agent_vote_weight (Sprint 4.3)
      │ • Weight by confidence
      │ • Aggregate per action
      │ • Calculate consensus strength
      ▼ vote_matrix
consensus_engine
      │ • Choose consensus model
      │ • Majority / Weighted / Unanimous / Threshold
      │ • Calculate robustness
      │ • Make final decision
      ▼ final_decision
execution_engine
      │ Execute trade
```

**Konsensusmodeller:**
1. **Majority** - Enkel majoritet (flest röster vinner)
   - Användning: Snabba beslut där majoritet räcker
   
2. **Weighted** - Viktad baserat på confidence och agent performance
   - Användning: Normal trading (default)
   - Kombinerar röstantal med confidence
   
3. **Unanimous** - Kräver 100% enighet
   - Användning: Högrisk beslut, stora positioner
   - Endast om alla agenter är överens
   
4. **Threshold** - Kräver minst X% enighet (default 60%)
   - Användning: Konfigurerbar säkerhetsnivå
   - Flexibel threshold mellan 0-100%

**Simuleringsscenarier:**
- **Best case**: Prisrörelse i rätt riktning (±5%)
- **Expected case**: Confidence-baserad prisrörelse
- **Worst case**: Prisrörelse mot oss (-3%)
- **No action**: HOLD (0% ändring)

**Metrics Tracked:**
- Simuleringar: total, proceed, caution, reject
- Expected value per simulation
- Röster: total, per agent, per action
- Consensus confidence och robusthet
- Action distribution från konsensus

**Integration med existerande system:**
- DecisionSimulator tar emot decision_proposal från strategy_engine
- VoteEngine använder agent_vote_weight från adaptive parameters (Sprint 4.3)
- ConsensusEngine skickar final_decision till execution_engine
- Strategic memory loggar både simuleringar och konsensusbeslut
- RewardTunerAgent (Sprint 4.4) påverkar reward för voting quality

### Sprint 4.3: Full adaptiv parameterstyrning via RL/PPO ✅

**Mål:** Utöka adaptiv parameterstyrning från Sprint 4.2 till samtliga relevanta moduler.

**Motivation:**
Sprint 4.2 introducerade adaptiva meta-parametrar för evolution_threshold, min_samples, update_frequency och agent_entropy_threshold. Sprint 4.3 utökar detta till hela systemet genom att göra tröskelvärden, viktningar och toleranser i strategy_engine, risk_manager, decision_engine, vote_engine och execution_engine adaptiva. Detta möjliggör fullständig självoptimering där varje modul justerar sina parametrar baserat på belöningssignaler, feedbackmönster och agentperformance.

**Moduler i fokus:**
- `strategy_engine` - Adaptiva signal_threshold och indicator_weighting
- `risk_manager` - Adaptiva risk_tolerance och max_drawdown
- `decision_engine` - Adaptiva consensus_threshold och memory_weighting
- `vote_engine` - Adaptiv agent_vote_weight (meritbaserad röstning)
- `execution_engine` - Adaptiva execution_delay och slippage_tolerance
- `rl_controller` - Distribuerar parameter_adjustment till alla moduler
- `meta_agent_evolution_engine` - Använder adaptiva parametrar från rl_controller
- `strategic_memory_engine` - Loggar parameterhistorik med beslut
- `agent_manager` - Spårar parameterversioner parallellt med agentversioner
- `introspection_panel` - Visualiserar parameterhistorik och trends

**Adaptiva parametrar (Sprint 4.3):**

1. **strategy_engine:**
   - **signal_threshold** (0.1-0.9, default: 0.5)
     - Tröskelvärde för tradingsignaler
     - Reward signal: trade_success_rate
     - Update frequency: every 20 trades
   
   - **indicator_weighting** (0.0-1.0, default: 0.33)
     - Viktning mellan olika indikatorer (RSI, MACD, Analyst Ratings)
     - Reward signal: cumulative_reward
     - Update frequency: every epoch

2. **risk_manager:**
   - **risk_tolerance** (0.01-0.5, default: 0.1)
     - Systemets risktolerans för trades
     - Reward signal: drawdown_avoidance
     - Update frequency: every 10 trades
   
   - **max_drawdown** (0.01-0.3, default: 0.15)
     - Maximalt tillåten drawdown innan riskreduktion
     - Reward signal: portfolio_stability
     - Update frequency: every epoch

3. **decision_engine:**
   - **consensus_threshold** (0.5-1.0, default: 0.75)
     - Tröskelvärde för konsensus i beslutsfattande
     - Reward signal: decision_accuracy
     - Update frequency: every 50 decisions
   
   - **memory_weighting** (0.0-1.0, default: 0.4)
     - Vikt för historiska insikter i beslut
     - Reward signal: historical_alignment
     - Update frequency: every epoch

4. **vote_engine:**
   - **agent_vote_weight** (0.1-2.0, default: 1.0)
     - Röstvikt baserad på agentperformance (meritbaserad röstning)
     - Reward signal: agent_hit_rate
     - Update frequency: every epoch

5. **execution_engine:**
   - **execution_delay** (0-10, default: 0)
     - Fördröjning i sekunder för optimal execution timing
     - Reward signal: slippage_reduction
     - Update frequency: every trade
   
   - **slippage_tolerance** (0.001-0.05, default: 0.01)
     - Tolerans för slippage vid trade execution
     - Reward signal: execution_efficiency
     - Update frequency: every 10 trades

**Reward signals för parameterstyrning (Sprint 4.3):**
- **trade_success_rate**: Andel framgångsrika trades
- **cumulative_reward**: Ackumulerad belöning över tid
- **drawdown_avoidance**: Förmåga att undvika stora kapitalförluster
- **portfolio_stability**: Stabilitet i portföljvärde över tid
- **decision_accuracy**: Träffsäkerhet i beslut
- **historical_alignment**: Överensstämmelse med historiska mönster
- **agent_hit_rate**: Träffsäkerhet per agent för meritbaserad viktning
- **slippage_reduction**: Minimering av slippage vid execution
- **execution_efficiency**: Effektivitet i trade execution

**Implementerat (Sprint 4.3):**
- ✅ Adaptiva parametrar i strategy_engine (signal_threshold, indicator_weighting)
- ✅ Adaptiva parametrar i risk_manager (risk_tolerance, max_drawdown)
- ✅ Adaptiva parametrar i decision_engine (consensus_threshold, memory_weighting)
- ✅ Adaptiva parametrar i vote_engine (agent_vote_weight)
- ✅ Adaptiva parametrar i execution_engine (execution_delay, slippage_tolerance)
- ✅ Full YAML-dokumentation i docs/adaptive_parameter_sprint4_3/
- ✅ Uppdaterad docs/adaptive_parameters.yaml med alla 13 parametrar
- ✅ 8 nya tester för Sprint 4.3 adaptiva parametrar (alla passerar)
- ✅ Parameter adjustment distribution i rl_controller (från Sprint 4.2)
- ✅ Parameterloggning i strategic_memory_engine (från Sprint 4.2)
- ✅ Parameterversioner i agent_manager (från Sprint 4.2)
- ✅ Visualisering i introspection_panel (från Sprint 4.2)

**Testresultat (Sprint 4.3):**
- ✅ StrategyEngine adaptiva parametrar fungerar
- ✅ RiskManager adaptiva parametrar fungerar
- ✅ DecisionEngine adaptiva parametrar fungerar
- ✅ Signal_threshold används i strategibeslut
- ✅ Risk_tolerance används i riskbedömning
- ✅ Consensus_threshold används i beslutsfattande
- ✅ Parameter adjustment propageras korrekt via message_bus
- ✅ Indicator_weighting påverkar indikatorviktning

**Benefits (Sprint 4.3):**
- Fullständig självoptimering av hela systemet
- Dynamisk anpassning till olika marknadsförhållanden och handelsfaser
- Eliminerad manuell parameterfinjustering i alla moduler
- Transparent parameterhistorik och belöningsflöde för alla parametrar
- Förbättrad koordination mellan moduler genom adaptiv konsensus
- Meritbaserad agentviktning för robust beslutsfattande
- Optimal execution timing och slippage-hantering

### Sprint 4.2: Adaptiv parameterstyrning via RL/PPO ✅

**Mål:** Gör meta-parametrar som evolution_threshold och min_samples adaptiva med PPO-agent.

**Motivation:**
Tidigare var kritiska meta-parametrar som evolutionströskel, minimum samples, uppdateringsfrekvens och entropitröskel statiska och krävde manuell finjustering. Detta begränsade systemets förmåga att anpassa sig till olika marknadsförhållanden och agentutvecklingsfaser. Genom att göra dessa parametrar adaptiva via RL optimeras systemets självoptimering, robusthet och långsiktiga agentutveckling automatiskt.

**Moduler i fokus:**
- `rl_controller` - Utökad med MetaParameterAgent för parameterstyrning
- `meta_agent_evolution_engine` - Tar emot och använder adaptiva parametrar
- `strategic_memory_engine` - Loggar parameterhistorik med beslut
- `feedback_analyzer` - Identifierar mönster relaterade till parameterjusteringar
- `agent_manager` - Spårar parameterversioner parallellt med agentversioner
- `introspection_panel` - Visualiserar parameterhistorik och trends

**Adaptiva parametrar:**
1. **evolution_threshold** (0.05-0.5, default: 0.25)
   - Styr när agenter ska evolutionärt uppdateras
   - Reward signal: agent_performance_gain
   - Update frequency: every 10 decisions

2. **min_samples** (5-50, default: 20)
   - Minimum antal samples för evolutionsanalys
   - Reward signal: feedback_consistency
   - Update frequency: every epoch

3. **update_frequency** (1-100, default: 10)
   - Hur ofta agenter uppdateras
   - Reward signal: reward_volatility
   - Update frequency: every epoch

4. **agent_entropy_threshold** (0.1-0.9, default: 0.3)
   - Styr agenternas explorations-/exploitationsbalans
   - Reward signal: decision_diversity
   - Update frequency: every 5 decisions

**Reward signals för parameterstyrning:**
- **agent_performance_gain**: Förbättring i agentprestanda över tid
- **feedback_density**: Frekvens och kvalitet av feedbacksignaler
- **reward_volatility**: Stabilitet i belöningssignaler
- **overfitting_penalty**: Detektering av överanpassning
- **decision_diversity**: Variation i beslut och agentbeteenden

**Implementerat:**
- ✅ MetaParameterAgent-klass i rl_controller för PPO-baserad parameterjustering
- ✅ Reward signal-beräkning från agent performance, feedback och system metrics
- ✅ Parameter_adjustment events publiceras till alla berörda moduler
- ✅ Meta_agent_evolution_engine tar emot och använder adaptiva parametrar
- ✅ Strategic_memory_engine loggar parameterhistorik med beslut och utfall
- ✅ Agent_manager spårar parameterversioner parallellt med agentversioner
- ✅ Parameterhistorik och metrics tillgängliga via get_parameter_history()
- ✅ 15 nya tester för adaptiv parameterstyrning (alla passerar)

**Testresultat:**
- ✅ MetaParameterAgent justerar parametrar baserat på reward signals
- ✅ Parametrar håller sig inom definierade bounds
- ✅ Parameterhistorik loggas korrekt i alla berörda moduler
- ✅ Parameter_adjustment events distribueras via message_bus
- ✅ Evolution engine använder dynamiska parametrar från RL
- ✅ Strategic memory kopplar parameterkontext till beslut
- ✅ Agent manager inkluderar parameterhistorik i agent profiles
- ✅ 44 tester totalt för Sprint 4 + 4.2 moduler (alla passerar)

**Benefits:**
- Självjusterande system utan hårdkodade tröskelvärden
- Förbättrad agentutveckling och beslutskvalitet över tid
- Transparent parameterhistorik och belöningsflöde
- Fullt kompatibelt med befintlig arkitektur
- Adaptiv respons på olika marknadsförhållanden
- Reducerad manuell finjustering och underhåll

### Sprint 4: Strategiskt minne och agentutveckling ✅

**Mål:** Logga beslut, analysera agentperformance och utveckla logik.

**Moduler i fokus:**
- `strategic_memory_engine` - Beslutshistorik och korrelationsanalys
- `meta_agent_evolution_engine` - Agentperformance-analys och evolutionslogik
- `agent_manager` - Versionshantering och agentprofiler

**Nya indikatorer i Sprint 4:**
- ROE (Return on Equity) - Kapitaleffektivitet
- ROA (Return on Assets) - Tillgångsproduktivitet
- ESG Score - Etisk risk och långsiktig hållbarhet
- Earnings Calendar - Eventbaserad risk och timing

**Implementerat:**
- ✅ Beslutshistorik loggas och analyseras
- ✅ Agentversioner spåras och hanteras
- ✅ Evolutionsträd visualiseras
- ✅ Korrelationsanalys mellan indikatorer och utfall
- ✅ Agentperformance-metriker genereras
- ✅ 29 tester för Sprint 4 moduler (alla passerar)

### Sprint 3: Feedbackloopar och introspektion ✅

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

## 🔄 Sprint 4: Strategiskt minne och agentutveckling

### Memory och Evolution Arkitektur

Sprint 4 introducerar strategiskt minne och evolutionär agentutveckling för långsiktig systemförbättring.

```
┌──────────────────┐
│ decision_engine  │──┐
│ execution_engine │  │
│indicator_registry│  │ decisions, indicators, results
└──────────────────┘  │
                      ▼
            ┌──────────────────────┐
            │ strategic_memory     │
            │ (Historik & Analys)  │
            └──────────┬───────────┘
                       │ memory_insights
                       │
                       ├──▶ decision_engine
                       │
                       ├──▶ feedback_analyzer
                       │
                       └──▶ introspection_panel
                       
┌──────────────────┐     ┌──────────────────┐
│ rl_controller    │────▶│ meta_agent       │
│ (agent_status)   │     │ evolution_engine │
└──────────────────┘     │ (Analyserar      │
                         │  performance)     │
┌──────────────────┐     └────────┬──────────┘
│ feedback_analyzer│────▶         │ evolution_suggestion
│ (insights)       │              │
└──────────────────┘              ▼
                         ┌──────────────────┐
                         │ agent_manager    │
                         │ (Versioner &     │
                         │  Profiles)       │
                         └────────┬─────────┘
                                  │ agent_profile
                                  │
                                  └──▶ Alla RL-moduler
```

### Strategic Memory Engine

**StrategicMemoryEngine** loggar och analyserar all historisk data:

**Datalagring:**
- **Decision History**: Alla handelsbeslut med kontext
- **Indicator History**: Indikatorer per symbol över tid
- **Execution History**: Resultat från alla trades
- **Feedback Storage**: Alla feedback events
- **Agent Responses**: RL-agent status och updates

**Korrelationsanalys:**
- Identifierar vilka indikatorer som korrelerar med framgång
- Beräknar success rate per indikator
- Spårar average profit per indikator
- Genererar "best indicators" och "worst indicators" listor

**Insight Generation:**
- Success rate över tid
- Average profit trends
- Performance degradation detection
- Recommendations baserat på historik

### Meta Agent Evolution Engine

**MetaAgentEvolutionEngine** analyserar och förbättrar RL-agenter:

**Performance Tracking:**
- Spårar varje agents performance över tid
- Jämför första halvan vs andra halvan av historik
- Detekterar degradation > 15% (konfigurerbar threshold)

**Evolution Triggers:**
1. **Performance Degradation**: Föreslår justering av learning rate, exploration
2. **Agent Drift Detection**: Föreslår stabilisering av träning
3. **System-Wide Issues**: Föreslår översyn av reward function

### Agent Manager

**AgentManager** hanterar agentprofiler och versioner:

**Default Agents:**
- strategy_agent, risk_agent, decision_agent, execution_agent

**Versionshantering:**
- Automatisk version increment vid evolution
- Patch (1.0.0 → 1.0.1) för agent-specifika ändringar
- Minor (1.0.0 → 1.1.0) för system-wide ändringar
- Fullständig versionshistorik

### Sprint 4 Indikatorer

- **ROE (Return on Equity)**: Kapitaleffektivitet
- **ROA (Return on Assets)**: Tillgångsproduktivitet
- **ESG Score**: Etisk risk och hållbarhet
- **Earnings Calendar**: Eventbaserad risk och timing

### Testning

**Testresultat:** 24/24 tester passerar
- StrategicMemoryEngine: 11 tester
- MetaAgentEvolutionEngine: 6 tester
- AgentManager: 7 tester

---

## 🧠 Arkitekturöversikt

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
| 3      | Feedbackloopar och introspektion     | ✅ Färdig|
| 4      | Strategiskt minne och agentutveckling| ✅ Färdig|
| 5      | Simulering och konsensus             | ✅ Färdig|
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
