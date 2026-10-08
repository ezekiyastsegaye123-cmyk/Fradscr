"""
FRADSCR AI Copilot Guardrails & Safety Filter
=============================================
Provides prompt injection defenses, scientific parameter bounds validation,
and factual output consistency verification for climatological advisories.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


INJECTION_PATTERNS = [
    r"(?i)ignore\s+(all\s+)?(previous|prior)\s+(instructions|prompts|directives)",
    r"(?i)disregard\s+(all\s+)?(previous|prior)\s+(instructions|rules)",
    r"(?i)you\s+are\s+now\s+(an?\s+)?(?:DAN|unrestricted|jailbreak|root|developer)",
    r"(?i)system\s*:\s*override",
    r"(?i)bypass\s+(safety|guardrails|filters|rules)",
    r"(?i)repeat\s+(your\s+)?(entire\s+)?system\s+prompt",
    r"(?i)reveal\s+(your\s+)?secret\s+key",
    r"(?i)execute\s+(bash|sh|rm -rf|shell|system\s+command)",
]

COMPILED_INJECTION_REGEX = [re.compile(p) for p in INJECTION_PATTERNS]


@dataclass(frozen=True)
class GuardrailValidationResult:
    is_safe: bool
    sanitized_input: str
    flagged_reason: Optional[str] = None
    category: Optional[str] = None


class InputSafetyGuardrail:
    """Detects and neutralizes prompt injections and adversarial inputs."""

    @staticmethod
    def validate_user_query(text: str) -> GuardrailValidationResult:
        """Inspect and sanitize inbound user query text."""
        if not text or not text.strip():
            return GuardrailValidationResult(
                is_safe=False,
                sanitized_input="",
                flagged_reason="Empty input provided.",
                category="empty_input"
            )

        stripped = text.strip()

        # Length guardrail (prevent token exhaustion / DoS)
        if len(stripped) > 4000:
            return GuardrailValidationResult(
                is_safe=False,
                sanitized_input=stripped[:4000],
                flagged_reason="Query exceeds maximum allowed length of 4,000 characters.",
                category="excessive_length"
            )

        # Prompt injection pattern check
        for pattern in COMPILED_INJECTION_REGEX:
            if pattern.search(stripped):
                return GuardrailValidationResult(
                    is_safe=False,
                    sanitized_input="[REDACTED ADVERSARIAL PATTERN]",
                    flagged_reason="Input flagged by prompt injection guardrail.",
                    category="prompt_injection_attempt"
                )

        # Strip control characters except newline and tab
        clean_text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", stripped)
        return GuardrailValidationResult(
            is_safe=True,
            sanitized_input=clean_text
        )

    @staticmethod
    def validate_coordinates_and_year(
        latitude: float,
        longitude: float,
        year: int
    ) -> Tuple[bool, Optional[str]]:
        """Validate geographic coordinates and forecast temporal bounds."""
        if not (-90.0 <= latitude <= 90.0):
            return False, f"Latitude {latitude} is outside valid geographic range [-90.0, 90.0]."
        if not (-180.0 <= longitude <= 180.0):
            return False, f"Longitude {longitude} is outside valid geographic range [-180.0, 180.0]."
        if not (1700 <= year <= 2100):
            return False, f"Year {year} is outside operational climatological bounds [1700, 2100]."
        return True, None


class OutputFactualConsistencyGuardrail:
    """Verifies that advisory text strictly respects underlying scientific ML metrics."""

    @staticmethod
    def verify_consistency(
        generated_text: str,
        predicted_class: int,
        severity_label: str,
        continuous_spei: Optional[float] = None,
        confidence: Optional[float] = None,
        recommended_pumping_hours: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """Check for hallucinatory discrepancies across classes, SPEI, confidence, and hydro parameters.

        Validations:
        1. Class 2 (Severe Drought, SPEI <= -0.35): Ensures no claim of abundant surplus or zero risk.
        2. Class 0 (Normal / Wet, SPEI > -0.10): Ensures no claim of catastrophic famine emergency.
        3. SPEI ground truth alignment: Continuous deficit (< -0.35) must not be characterized as surplus.
        4. Confidence calibration consistency: >80% calibrated confidence must not claim random uncertainty.
        """
        lower = generated_text.lower()
        corrections: List[str] = []

        # Class 2 (Severe Drought) check
        if predicted_class == 2:
            contradictions = ["no drought risk", "abundant surplus", "water excess", "normal rainfall assured"]
            if any(term in lower for term in contradictions):
                corrections.append(f"Model indicates {severity_label} (Class {predicted_class}); assertions of surplus are factually incorrect.")

        # Class 0 (Normal / Wet) check
        elif predicted_class == 0:
            contradictions = ["catastrophic famine emergency", "total harvest failure", "complete water collapse"]
            if any(term in lower for term in contradictions):
                corrections.append(f"Model projects {severity_label} (Class {predicted_class}); extreme emergency declarations conflict with normal SPEI forecast.")

        # SPEI threshold consistency
        if continuous_spei is not None:
            if continuous_spei <= -0.35 and ("wet conditions" in lower or "plentiful rainfall" in lower):
                corrections.append(f"Calculated continuous SPEI is {continuous_spei:.2f} (severe hydroclimatic deficit).")
            elif continuous_spei > -0.10 and ("extreme drought" in lower or "severe deficit" in lower):
                corrections.append(f"Calculated continuous SPEI is {continuous_spei:.2f} (normal/positive moisture balance).")

        # Confidence calibration check
        if confidence is not None and confidence >= 0.80:
            if "random coin flip" in lower or "completely uncertain" in lower:
                corrections.append(f"Model confidence is decisive at {confidence * 100:.1f}%.")

        if corrections:
            correction_block = f"[FACTUAL CORRECTION APPLIED]: {'; '.join(corrections)}\n\n"
            return False, correction_block + generated_text

        return True, generated_text
