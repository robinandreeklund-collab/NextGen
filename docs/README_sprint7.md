# Sprint 7: Indikatorvisualisering och systemöversikt

## Översikt

Sprint 7 fokuserar på resurshantering, teamdynamik och visualisering för att optimera systemets prestanda och förstå agentbeteenden. Detta är den sista sprinten i NextGen AI Trader-projektet och introducerar omfattande resursallokering, teambaserat beslutsfattande, och förbättrad systemvisualisering.

## Mål

- Implementera resursallokering mellan moduler och agenter
- Skapa teambaserad koordinering för agenter
- Förbättra indikatorvisualisering
- Utöka systemöversikt och health monitoring
- Integrera med alla tidigare sprints (1-6)

## Nya Moduler

### ResourcePlanner

**Syfte:** Hanterar resursallokering mellan agenter och moduler

**Funktionalitet:**
- Tre resurstyper: compute (100 units), memory (100 units), training (100 units)
- Priority-based allocation: critical > high > medium > low
- Performance-weighted distribution
- Dynamic reallocation från låg till hög prioritet
- Resource efficiency tracking
- Allocation history logging

**Default Allocations (Compute):**
- strategy_agent: 25%
- risk_agent: 20%
- decision_agent: 20%
- execution_agent: 15%
- meta_parameter_agent: 10%
- reward_tuner_agent: 10%

**Default Allocations (Memory):**
- strategic_memory: 30%
- indicator_registry: 20%
- introspection_panel: 20%
- system_monitor: 15%
- resource_planner: 10%
- team_dynamics: 5%

**Message Topics:**
- Subscribes: `resource_request`, `performance_metric`, `module_completed`
- Publishes: `resource_allocation`, `resource_reallocation`

### TeamDynamicsEngine

**Syfte:** Koordinerar agentteam och samarbete

**Funktionalitet:**
- 4 team patterns (aggressive, conservative, balanced, exploration)
- Team synergy calculation
- Coordination score based på agent interactions
- Team performance evaluation
- Resource boost allocation (1.0x - 1.3x)
- Communication flow logging

**Team Patterns:**

1. **Aggressive Trading**
   - Agents: strategy_agent, execution_agent
   - Resource boost: 1.3x
   - Risk tolerance: high
   - Best för: High-conviction trades, momentum plays

2. **Conservative Trading**
   - Agents: risk_agent, decision_agent
   - Resource boost: 1.0x
   - Risk tolerance: low
   - Best för: Risk-averse trading, capital preservation

3. **Balanced Trading**
   - Agents: strategy_agent, risk_agent, decision_agent
   - Resource boost: 1.1x
   - Risk tolerance: medium
   - Best för: Normal trading conditions

4. **Exploration Phase**
   - Agents: strategy_agent, meta_parameter_agent
   - Resource boost: 0.9x
   - Risk tolerance: medium
   - Best för: Learning and adaptation

**Message Topics:**
- Subscribes: `decision_vote`, `agent_status`, `final_decision`, `form_team`, `dissolve_team`
- Publishes: `team_formed`, `team_dissolved`, `team_metrics`

## Agent Synergies

Enligt evolution_matrix.yaml:

- **strategy_agent** ↔ risk_agent, decision_agent
- **risk_agent** ↔ strategy_agent, portfolio_agent
- **decision_agent** ↔ strategy_agent, execution_agent
- **execution_agent** ↔ decision_agent, portfolio_agent
- **meta_parameter_agent** ↔ all_agents
- **reward_tuner_agent** ↔ rl_controller, meta_parameter_agent

## Integration med Tidigare Sprints

### Sprint 1-2: RL och Belöning
- ResourcePlanner allocates training_budget till rl_controller
- Teams kan använda PPO-agenter med resource-aware training

### Sprint 3: Feedback
- FeedbackAnalyzer använder resource metrics för performance analysis
- Resource efficiency påverkar feedback routing priority

### Sprint 4: Minne och Evolution
- MetaAgentEvolutionEngine considers resource efficiency
- Evolution triggers kan vara resource_constraint
- AgentManager tracks resource usage per agent version

### Sprint 5: Konsensus
- VoteEngine viktar team votes baserat på synergy
- ConsensusEngine considers team_coordination_score

### Sprint 6: Tidsanalys
- TimespanTracker logs resource allocation events
- ActionChainEngine använder resource-aware execution

## Nya Indikatorer

Sprint 7 introducerar 4 nya indikatorer:

1. **VIX (Volatility Index)**
   - Typ: Market sentiment
   - Används av: risk_manager, strategy_engine
   - Syfte: Measure market fear/greed, volatility prediction

2. **Earnings Surprise**
   - Typ: Fundamental
   - Används av: strategy_engine, decision_engine
   - Syfte: React till earnings beats/misses

3. **Short Interest**
   - Typ: Market sentiment
   - Används av: risk_manager, strategy_engine
   - Syfte: Identify heavily shorted stocks, squeeze potential

4. **Put/Call Ratio**
   - Typ: Options data
   - Används av: risk_manager, decision_engine
   - Syfte: Options-based sentiment analysis

## Testning

### Test Coverage

- **ResourcePlanner:** 13 tester
- **TeamDynamicsEngine:** 16 tester
- **Total Sprint 7:** 29 tester
- **System total:** 214 tester (100% pass rate)

### Test Categories

1. **RP (Resource Planner) - RP-001 till RP-013**
   - Initialization, allocation, reallocation
   - Performance tracking, efficiency calculation
   - Dashboard data generation

2. **TD (Team Dynamics) - TD-001 till TD-016**
   - Team formation/dissolution
   - Synergy och coordination calculation
   - Team performance evaluation

3. **IV (Indicator Visualization) - IV-001 till IV-006**
   - Indicator trends visualization
   - Correlation heatmap
   - Real-time updates

4. **SO (System Overview) - SO-001 till SO-008**
   - Module health monitoring
   - System-wide metrics aggregation
   - System health score

5. **INT (Integration) - INT-001 till INT-007**
   - Resource + Team integration
   - System monitor + All modules
   - Full Sprint 1-7 integration

## Resource Allocation Strategies

### 1. Priority-based
```
1. Sortera requests efter priority
2. Allokera critical först
3. Fortsätt med high, medium, low
4. Vid konflikter, använd performance_score
```

### 2. Demand-based
```
1. Beräkna genomsnittlig usage
2. Allokera baserat på predicted demand
3. Reservera buffer (20%) för spikes
4. Dynamisk justering varje 10 beslut
```

### 3. Performance-weighted
```
1. Beräkna performance_score
2. Allokera proportionellt
3. Garantera minimum för alla
4. Bonus till top 20% performers
```

### 4. Team-coordinated
```
1. Identifiera active teams
2. Allokera gruppresurser till teams
3. Team fördelar internt
4. Synergy bonus om team presterar
```

## Metrics och KPIs

### Resource Metrics
- **Utilization Rate:** resources_used / resources_allocated (target: 0.80)
- **Efficiency Score:** performance_gain / resources_consumed (target: 1.20)
- **Waste Detection:** unused resources / allocated (max: 0.20)
- **Bottleneck Score:** pending_requests / available (warning: 2.0)

### Team Metrics
- **Synergy Score:** Natural synergies / total_pairs (0-1 range)
- **Coordination Score:** Interactions / expected_interactions (0-1 range)
- **Team Efficiency:** Team performance / individual sum
- **Resource Boost Effectiveness:** Performance with boost / without boost

### System Metrics
- **System Health Score:** Weighted average of module health
- **Active Modules:** Modules reporting within threshold
- **Stale Modules:** Modules not reporting recently
- **Overall Efficiency:** System performance / total resources

## Visualisering

### Introspection Panel Dashboards

1. **Resource Dashboard**
   - Resource allocation over time
   - Utilization by module
   - Efficiency heatmap
   - Bottleneck alerts

2. **Team Dynamics Dashboard**
   - Active teams overview
   - Synergy scores per team
   - Coordination trends
   - Team performance comparison

3. **Indicator Dashboard**
   - Indicator trends (all 12+ indicators)
   - Correlation heatmap
   - Effectiveness ranking
   - Multi-symbol comparison

4. **System Health Dashboard**
   - Module status overview
   - Health score timeline
   - Performance aggregation
   - Alert history

## CI/CD Pipeline

### Stages

1. **Code Quality** - Linting, formatting, security
2. **Unit Tests** - Per-module tests
3. **Integration Tests** - Module interaction tests
4. **System Validation** - End-to-end tests
5. **Performance Tests** - Benchmarking
6. **Documentation** - YAML validation

### Success Criteria

- 100% test pass rate
- Coverage >= 80%
- No performance regressions
- All modules reporting healthy
- Resource utilization < 90%

## Användningsexempel

### Forma ett Team

```python
# Via message_bus
message_bus.publish('form_team', {
    'team_id': 'aggressive_team_1',
    'pattern': 'aggressive_trading'
})

# Teams får automatiskt resource boost (1.3x för aggressive)
```

### Begär Resurser

```python
# Request compute resources
message_bus.publish('resource_request', {
    'module_id': 'my_module',
    'resource_type': 'compute',
    'amount_requested': 20,
    'priority': 'high'
})

# ResourcePlanner allokerar baserat på priority och performance
```

### Rapportera Performance

```python
# Report performance metrics
message_bus.publish('performance_metric', {
    'module_id': 'my_module',
    'resource_consumed': 15,
    'performance_achieved': 1.8
})

# ResourcePlanner uppdaterar efficiency_score och justerar framtida allocations
```

### Visualisera System Health

```python
# Get system overview
system_view = system_monitor.get_system_view()
print(f"System health: {system_view['health_score']}")
print(f"Active modules: {system_view['active_modules']}")

# Get resource status
resource_status = resource_planner.get_resource_status()
print(f"Compute utilization: {resource_status['compute']['utilization']}")
```

## Nästa Steg

Sprint 7 är den sista sprinten. Systemet är nu komplett med:
- ✅ Kärnsystem och trading (Sprint 1)
- ✅ RL/PPO-agenter och belöning (Sprint 2)
- ✅ Feedbackloopar (Sprint 3)
- ✅ Strategiskt minne och evolution (Sprint 4)
- ✅ Adaptiva parametrar (Sprint 4.2, 4.3)
- ✅ RewardTunerAgent (Sprint 4.4)
- ✅ Konsensus och simulering (Sprint 5)
- ✅ Tidsanalys och action chains (Sprint 6)
- ✅ Resurshantering och teamdynamik (Sprint 7)

**Framtida förbättringar:**
- Live trading integration med broker API
- Advanced visualisering med Dash dashboards
- Multi-market support (stocks, crypto, forex)
- Advanced risk management strategies
- Machine learning för indicator effectiveness
- Real-time collaborative filtering mellan agenter

## Dokumentation

### Sprint 7 YAML Files

- `docs/sprint_7.yaml` - Sprint definition och mål
- `docs/evolution_matrix.yaml` - Agent evolution och team patterns
- `docs/resource_flowchart.yaml` - Resource flow och strategies
- `docs/rl_test_suite_sprint7.yaml` - Test suite (50 test cases)
- `docs/ci_matrix_sprint7.yaml` - CI test matrix
- `docs/ci_pipeline_sprint7.yaml` - CI/CD pipeline
- `docs/flowchart_sprint1_7.yaml` - Complete system flowchart

### Code Files

- `modules/resource_planner.py` - Resource allocation module
- `modules/team_dynamics_engine.py` - Team coordination module
- `tests/test_resource_planner.py` - ResourcePlanner tests
- `tests/test_team_dynamics_engine.py` - TeamDynamics tests

## Kontakt och Support

För frågor om Sprint 7 implementation, se:
- README.md - Huvuddokumentation
- docs/sprint_7.yaml - Sprint definition
- tests/ - Test examples och användning
