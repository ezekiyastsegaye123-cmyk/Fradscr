"""
FRADSCR AI Copilot & Hydroclimatic Intelligence Module
======================================================
"""

from treering.copilot.agent import HydroclimaticCopilotAgent, get_copilot_agent
from treering.copilot.analogues import (
    ClimateAnalogueMatcher,
    HistoricalClimateAnalogue,
    get_analogue_matcher,
)
from treering.copilot.guardrails import (
    InputSafetyGuardrail,
    OutputFactualConsistencyGuardrail,
)
from treering.copilot.synthesizer import ClimatologicalAdvisorySynthesizer

__all__ = [
    "HydroclimaticCopilotAgent",
    "get_copilot_agent",
    "ClimateAnalogueMatcher",
    "HistoricalClimateAnalogue",
    "get_analogue_matcher",
    "InputSafetyGuardrail",
    "OutputFactualConsistencyGuardrail",
    "ClimatologicalAdvisorySynthesizer",
]
