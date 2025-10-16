"""
risk_manager.py - Riskhantering

Beskrivning:
    Bedömer risknivå baserat på volatilitet, ESG och fundamentala indikatorer.
    Justerar strategier och beslut baserat på riskprofil.

Roll:
    - Analyserar indikatordata för riskbedömning
    - Beräknar volatilitet och risk metrics
    - Genererar risk_profile för decision_engine
    - Tränas av RL-controller för riskoptimering

Inputs:
    - indicator_data: Dict - Indikatorer från indicator_registry
    - portfolio_status: Dict - Portföljstatus från portfolio_manager

Outputs:
    - risk_profile: Dict - Riskbedömning (risk_level, volatility, recommendations)

Publicerar till message_bus:
    - risk_profile: Riskprofil för decision_engine

Prenumererar på (från functions.yaml):
    - indicator_data (från indicator_registry)
    - portfolio_status (från portfolio_manager)

Använder RL: Ja (från functions.yaml)
Tar emot feedback: Ja (från portfolio_manager, execution_engine)

Anslutningar (från flowchart.yaml - risk_flow):
    Från: indicator_registry (indicator_data)
    Till:
    - decision_engine (risk_profile)
    - strategic_memory_engine (för riskloggning)

RL-anslutning (från feedback_loop.yaml):
    - Tar emot agent_update från rl_controller
    - Belöning baserad på risk-adjusted returns

Indikatorer från indicator_map.yaml:
    Använder (Sprint 1-2):
    - Volume: liquidity assessment, volatility detection
    - ATR: volatility-based risk adjustment (Sprint 2)
    - Analyst Ratings: external confidence (Sprint 2)
    
    Använder (Sprint 4-5):
    - Bollinger Bands: volatility and breakout detection
    - ADX: trend strength confirmation
    
    Använder (Sprint 4):
    - Profit Margin: efficiency and risk
    - ROE, ROA: capital efficiency
    - ESG Score: ethical risk and long-term viability

Används i Sprint: 2, 4
"""

from typing import Dict, Any


class RiskManager:
    """Bedömer risk och genererar riskprofiler för beslut."""
    
    def __init__(self, message_bus):
        """
        Initialiserar riskhanteraren.
        
        Args:
            message_bus: Referens till central message_bus
        """
        self.message_bus = message_bus
        self.current_indicators: Dict[str, Any] = {}
        self.portfolio_status: Dict[str, Any] = {}
        self.rl_agent = None  # Uppdateras från rl_controller
        self.rl_enabled = False
        self.agent_performance = 0.0
        
        # Prenumerera på indicator_data, portfolio_status och agent_update
        self.message_bus.subscribe('indicator_data', self._on_indicator_data)
        self.message_bus.subscribe('portfolio_status', self._on_portfolio_status)
        self.message_bus.subscribe('agent_update', self._on_agent_update)
    
    def _on_indicator_data(self, data: Dict[str, Any]) -> None:
        """
        Callback för nya indikatordata.
        
        Args:
            data: Indikatordata för en symbol
        """
        symbol = data.get('symbol')
        self.current_indicators[symbol] = data
    
    def _on_portfolio_status(self, status: Dict[str, Any]) -> None:
        """
        Callback för portföljstatus.
        
        Args:
            status: Aktuell portföljstatus
        """
        self.portfolio_status = status
    
    def _on_agent_update(self, update: Dict[str, Any]) -> None:
        """
        Callback för agent updates från rl_controller.
        
        Args:
            update: Agent update med nya parametrar
        """
        if update.get('module') == 'risk_manager':
            self.rl_enabled = update.get('policy_updated', False)
            metrics = update.get('metrics', {})
            self.agent_performance = metrics.get('average_reward', 0.0)
    
    def assess_risk(self, symbol: str) -> Dict[str, Any]:
        """
        Bedömer risk för en symbol.
        
        Args:
            symbol: Aktiesymbol att bedöma risk för
            
        Returns:
            Dict med risk_profile (risk_level, volatility, recommendations)
        """
        indicators = self.current_indicators.get(symbol, {})
        
        # Sprint 2: Avancerad riskbedömning med ATR och Analyst Ratings
        risk_profile = {
            'symbol': symbol,
            'risk_level': 'MEDIUM',
            'risk_score': 0.0,  # Läggs till för debug output
            'volatility': 0.5,
            'recommendations': [],
            'confidence': 0.5,
            'rl_enabled': self.rl_enabled
        }
        
        if not indicators:
            return risk_profile
        
        technical = indicators.get('technical', {})
        fundamental = indicators.get('fundamental', {})
        
        # Hämta indikatorer
        volume = technical.get('Volume', 0)
        atr = technical.get('ATR', 2.0)
        analyst_ratings = fundamental.get('AnalystRatings', {})
        
        risk_score = 0  # Negativt = mindre risk, positivt = mer risk
        
        # Volume-baserad riskbedömning
        if volume < 500000:
            risk_score += 2
            risk_profile['recommendations'].append('Låg likviditet, hög risk')
        elif volume > 2000000:
            risk_score -= 1
            risk_profile['recommendations'].append('Hög likviditet, låg risk')
        
        # ATR-baserad volatilitetsrisk (Sprint 2)
        if atr > 5.0:  # Hög volatilitet
            risk_score += 2
            risk_profile['volatility'] = min(1.0, atr / 10.0)
            risk_profile['recommendations'].append(f'Hög volatilitet (ATR: {atr:.1f}), ökad risk')
        elif atr < 2.0:  # Låg volatilitet
            risk_score -= 1
            risk_profile['volatility'] = atr / 5.0
            risk_profile['recommendations'].append(f'Låg volatilitet (ATR: {atr:.1f}), stabil')
        else:
            risk_profile['volatility'] = atr / 5.0
        
        # Analyst Ratings-baserad risk (Sprint 2)
        consensus = analyst_ratings.get('consensus', 'HOLD')
        if consensus in ['STRONG_BUY', 'BUY']:
            risk_score -= 1
            risk_profile['recommendations'].append(f'Positiv analytiker-sentiment ({consensus})')
        elif consensus == 'SELL':
            risk_score += 1
            risk_profile['recommendations'].append(f'Negativ analytiker-sentiment ({consensus})')
        
        # Portföljexponering
        total_value = self.portfolio_status.get('total_value', 1000.0)
        positions = self.portfolio_status.get('positions', {})
        if symbol in positions:
            position_value = positions[symbol]['quantity'] * positions[symbol]['avg_price']
            exposure = position_value / total_value
            if exposure > 0.3:  # Mer än 30% exponering
                risk_score += 1
                risk_profile['recommendations'].append(f'Hög portföljexponering ({exposure*100:.1f}%)')
        
        # Bestäm risknivå baserat på score
        risk_profile['risk_score'] = float(risk_score)  # Spara numerical score
        
        if risk_score >= 3:
            risk_profile['risk_level'] = 'HIGH'
            risk_profile['confidence'] = 0.8
        elif risk_score <= -2:
            risk_profile['risk_level'] = 'LOW'
            risk_profile['confidence'] = 0.8
        else:
            risk_profile['risk_level'] = 'MEDIUM'
            risk_profile['confidence'] = 0.6
        
        # RL-justering (om aktiverad)
        if self.rl_enabled and self.agent_performance > 0:
            # Förbättra confidence om RL-agent presterar bra
            risk_profile['confidence'] = min(1.0, risk_profile['confidence'] * 1.1)
            risk_profile['recommendations'].append(f'RL-förstärkt (perf: {self.agent_performance:.2f})')
        
        return risk_profile
    
    def publish_risk_profile(self, risk_profile: Dict[str, Any]) -> None:
        """
        Publicerar riskprofil till message_bus.
        
        Args:
            risk_profile: Riskprofil att publicera
        """
        self.message_bus.publish('risk_profile', risk_profile)
    
    def receive_feedback(self, feedback: Dict[str, Any]) -> None:
        """
        Tar emot feedback från portfolio och execution.
        
        Args:
            feedback: Feedback om risk assessment accuracy
        """
        # Stub: Implementeras fullt ut i Sprint 3
        pass
    
    def update_from_rl(self, agent_update: Dict[str, Any]) -> None:
        """
        Uppdaterar riskbedömning baserat på RL-träning.
        
        Args:
            agent_update: Uppdatering från rl_controller
        """
        # Stub: Implementeras i Sprint 2
        pass

