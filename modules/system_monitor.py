"""
system_monitor.py - Systemövervakare

Beskrivning:
    Visar systemöversikt, indikatortrender och agentrespons.

Roll:
    - Prenumererar på dashboard_data från olika moduler
    - Aggregerar systemstatus
    - Visar översikt av hela systemet
    - Returnerar system_view via metod, publicerar inte till topic

Inputs:
    - dashboard_data: Dict - Data från alla moduler

Outputs:
    - system_view: Dict - Komplett systemöversikt (returneras från metod)

Publicerar till message_bus:
    - Ingen (konsumerar endast)

Prenumererar på (från functions.yaml):
    - dashboard_data (från olika källor)

Använder RL: Nej
Tar emot feedback: Nej

Används i Sprint: 7
"""

from typing import Dict, Any


class SystemMonitor:
    """Övervakar och visualiserar hela systemet."""
    
    def __init__(self, message_bus):
        self.message_bus = message_bus
        self.system_metrics: Dict[str, Any] = {}

