"""
FRADSCR AI Hydroclimatic Copilot Agent
======================================
The primary intelligence orchestrator: connects scientific machine learning
models, historical climate analogue RAG retrieval, prescriptive RL dispatch,
and multi-provider LLM synthesis with safety guardrails.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import httpx

from treering.copilot.analogues import get_analogue_matcher
from treering.copilot.guardrails import (
    InputSafetyGuardrail,
    OutputFactualConsistencyGuardrail,
)
from treering.copilot.prompts import (
    MULTILINGUAL_GLOSSARY,
    PERSONA_DESCRIPTIONS,
    SYSTEM_PROMPT_COPILOT,
)
from treering.copilot.synthesizer import ClimatologicalAdvisorySynthesizer

logger = logging.getLogger("fradscr.copilot")


class HydroclimaticCopilotAgent:
    """Enterprise AI Agent for Ethiopian hydroclimatic forecasting & advisory."""

    def __init__(
        self,
        prediction_fn: Optional[Any] = None,
        provider: Optional[str] = None,
    ) -> None:
        """Initialize the Copilot Agent.

        Args:
            prediction_fn: Callable wrapping predict_drought(latitude, longitude, year).
                           If None, lazily imported from predict_service.
            provider: Explicit LLM provider ('openai', 'anthropic', 'gemini', 'ollama', 'deterministic').
        """
        self._prediction_fn = prediction_fn
        self.provider = provider or os.getenv("FRADSCR_LLM_PROVIDER", "auto")
        self.matcher = get_analogue_matcher()

    def _get_predict_fn(self) -> Any:
        if self._prediction_fn is not None:
            return self._prediction_fn
        from predict_service import predict_drought
        return predict_drought

    # ── Tool Calling Primitives ──────────────────────────────────────────────

    def tool_predict_drought(
        self, latitude: float, longitude: float, year: int
    ) -> Dict[str, Any]:
        """Tool: Query the high-accuracy ML prediction engine."""
        valid, msg = InputSafetyGuardrail.validate_coordinates_and_year(
            latitude, longitude, year
        )
        if not valid:
            raise ValueError(msg)
        fn = self._get_predict_fn()
        return fn(latitude=latitude, longitude=longitude, year=year)

    def tool_find_climate_analogues(
        self,
        predicted_class: int,
        year: int,
        sunspot_count: Optional[float] = None,
        continuous_spei: Optional[float] = None,
        top_k: int = 2,
    ) -> List[Dict[str, Any]]:
        """Tool: Query the historical climate analogue RAG database."""
        return self.matcher.find_nearest_analogues(
            predicted_class=predicted_class,
            year=year,
            sunspot_count=sunspot_count,
            continuous_spei=continuous_spei,
            top_k=top_k,
        )

    # ── Advisory Generation Workflow ─────────────────────────────────────────

    def generate_advisory_report(
        self,
        latitude: float,
        longitude: float,
        year: int,
        persona: Literal["minister", "pastoralist", "scientist"] = "minister",
        language: Literal["en", "am", "om"] = "en",
    ) -> Dict[str, Any]:
        """Generate an authoritative hydroclimatic early warning advisory report.

        Steps:
        1. Coordinate and temporal boundary validation.
        2. Run physical machine learning forecast (predict_service).
        3. Match nearest historical analogues (RAG).
        4. Synthesize tailored multi-lingual markdown report with guardrails.
        """
        valid_coords, coord_err = InputSafetyGuardrail.validate_coordinates_and_year(
            latitude, longitude, year
        )
        if not valid_coords:
            return {
                "error": coord_err,
                "advisory_markdown": f"**Input Error:** {coord_err}",
            }

        # Step 2: Prediction
        try:
            pred = self.tool_predict_drought(latitude, longitude, year)
        except Exception as e:
            logger.error("Error invoking predict_drought: %s", e)
            return {
                "error": f"ML engine error: {e}",
                "advisory_markdown": f"**Prediction Failure:** Could not execute model inference: {e}",
            }

        # Step 3: Historical analogues
        pred_class = pred.get("predicted_drought_class", 0)
        continuous_spei = pred.get("continuous_spei")
        sunspots = None
        if "solar_cycle" in pred and isinstance(pred["solar_cycle"], dict):
            sunspots = pred["solar_cycle"].get("projected_sunspots")

        analogues = self.tool_find_climate_analogues(
            predicted_class=pred_class,
            year=year,
            sunspot_count=sunspots,
            continuous_spei=continuous_spei,
            top_k=2,
        )

        # Step 4: Synthesize advisory
        advisory = ClimatologicalAdvisorySynthesizer.generate_advisory(
            prediction_result=pred,
            historical_analogues=analogues,
            persona=persona,
            language=language,
        )

        # Step 5: Output consistency verification
        is_consistent, verified_text = (
            OutputFactualConsistencyGuardrail.verify_consistency(
                generated_text=advisory["advisory_markdown"],
                predicted_class=pred_class,
                severity_label=pred.get("severity_label", "Normal"),
            )
        )
        advisory["advisory_markdown"] = verified_text
        advisory["prediction"] = pred
        advisory["analogues"] = analogues
        advisory["is_factually_consistent"] = is_consistent
        return advisory

    # ── Conversational Hydroclimatologist Copilot ─────────────────────────────

    def answer_query(
        self,
        query: str,
        default_lat: float = 9.02,
        default_lon: float = 38.74,
        default_year: int = 2028,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Interactive conversational interface for policymakers and field engineers.

        Interprets natural language queries, automatically extracts temporal or
        geographic intent, invokes necessary ML and RAG tools, and produces a
        grounded conversational response.
        """
        # Step 1: Input safety guardrail
        val_res = InputSafetyGuardrail.validate_user_query(query)
        if not val_res.is_safe:
            return {
                "reply": f"⚠️ **Security Guardrail:** {val_res.flagged_reason}",
                "tools_used": [],
                "flagged": True,
            }

        clean_query = val_res.sanitized_input
        tools_called: List[str] = []

        # Step 2: Extract year from query if mentioned
        year_match = re.search(r"\b(20[2-5][0-9]|19[0-9]{2})\b", clean_query)
        target_year = int(year_match.group(1)) if year_match else default_year

        # Step 3: Check for regional mentions (e.g. Borana, Gondar, GERD, Wollo)
        lat, lon = default_lat, default_lon
        lower_q = clean_query.lower()
        if "borana" in lower_q or "yabelo" in lower_q or "moyale" in lower_q:
            lat, lon = 4.88, 38.08
        elif "gondar" in lower_q or "tana" in lower_q:
            lat, lon = 12.60, 37.45
        elif "gerd" in lower_q or "blue nile" in lower_q or "abbay" in lower_q:
            lat, lon = 11.21, 35.09
        elif "wollo" in lower_q or "dessie" in lower_q:
            lat, lon = 11.13, 39.63
        elif "debrebirkan" in lower_q:
            lat, lon = 9.63, 39.53

        # Step 4: Run prediction if query asks for prediction, risk, future, or specific year
        needs_prediction = any(
            k in lower_q
            for k in [
                "predict",
                "forecast",
                "risk",
                "drought",
                "future",
                "spei",
                "pump",
                "202",
                "203",
                "water",
            ]
        )

        pred_info: Optional[Dict[str, Any]] = None
        if needs_prediction:
            try:
                pred_info = self.tool_predict_drought(lat, lon, target_year)
                tools_called.append(f"predict_drought(lat={lat}, lon={lon}, year={target_year})")
            except Exception as e:
                logger.warning("Query tool prediction failed: %s", e)

        # Step 5: Check if historical analogues are needed
        needs_analogues = any(
            k in lower_q
            for k in ["history", "past", "analogue", "1984", "1973", "1888", "famine", "precedent"]
        )
        analogues_info: List[Dict[str, Any]] = []
        if needs_analogues or (pred_info and pred_info.get("predicted_drought_class", 0) > 0):
            pred_class = pred_info.get("predicted_drought_class", 1) if pred_info else 1
            analogues_info = self.tool_find_climate_analogues(
                predicted_class=pred_class, year=target_year, top_k=2
            )
            tools_called.append("find_climate_analogues")

        # Step 6: Generate response
        # Try external LLM provider if configured
        if self._can_use_external_llm():
            try:
                llm_reply = self._call_external_llm(
                    clean_query, pred_info, analogues_info, tools_called
                )
                return {
                    "reply": llm_reply,
                    "target_year": target_year,
                    "location": {"latitude": lat, "longitude": lon},
                    "prediction": pred_info,
                    "analogues": analogues_info,
                    "tools_used": tools_called,
                    "provider": self.provider,
                }
            except Exception as exc:
                logger.warning("External LLM call failed (%s); falling back to deterministic synthesis.", exc)

        # Deterministic domain synthesis fallback
        reply = self._synthesize_conversational_reply(
            clean_query, target_year, lat, lon, pred_info, analogues_info
        )
        return {
            "reply": reply,
            "target_year": target_year,
            "location": {"latitude": lat, "longitude": lon},
            "prediction": pred_info,
            "analogues": analogues_info,
            "tools_used": tools_called,
            "provider": "deterministic_grounded_engine",
        }

    # ── Internal Helpers ─────────────────────────────────────────────────────

    def _can_use_external_llm(self) -> bool:
        if self.provider == "deterministic":
            return False
        return bool(
            os.getenv("OPENAI_API_KEY")
            or os.getenv("OLLAMA_BASE_URL")
        )

    def _call_external_llm(
        self,
        query: str,
        pred_info: Optional[Dict[str, Any]],
        analogues_info: List[Dict[str, Any]],
        tools_called: List[str],
    ) -> str:
        """Call external LLM API with structured tool outputs injected into context using httpx."""
        context_block = f"""
OBSERVED ML PREDICTION RESULTS:
{json.dumps(pred_info, indent=2) if pred_info else "No model query executed."}

HISTORICAL CLIMATE ANALOGUES:
{json.dumps(analogues_info, indent=2) if analogues_info else "No analogues retrieved."}
"""
        openai_key = os.getenv("OPENAI_API_KEY")
        ollama_url = os.getenv("OLLAMA_BASE_URL")

        if openai_key:
            payload = {
                "model": os.getenv("FRADSCR_OPENAI_MODEL", "gpt-4o-mini"),
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT_COPILOT + "\n\n" + context_block},
                    {"role": "user", "content": query},
                ],
                "temperature": 0.2,
            }
            with httpx.Client(timeout=15.0) as client:
                res = client.post(
                    "https://api.openai.com/v1/chat/completions",
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {openai_key}",
                    },
                )
                res.raise_for_status()
                res_data = res.json()
                return res_data["choices"][0]["message"]["content"]

        elif ollama_url:
            payload = {
                "model": os.getenv("FRADSCR_OLLAMA_MODEL", "llama3"),
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT_COPILOT + "\n\n" + context_block},
                    {"role": "user", "content": query},
                ],
                "stream": False,
            }
            endpoint = ollama_url.rstrip("/") + "/api/chat"
            with httpx.Client(timeout=30.0) as client:
                res = client.post(endpoint, json=payload)
                res.raise_for_status()
                res_data = res.json()
                return res_data["message"]["content"]

        raise RuntimeError("No supported external LLM provider configured (set OPENAI_API_KEY or OLLAMA_BASE_URL).")

    def _synthesize_conversational_reply(
        self,
        query: str,
        year: int,
        lat: float,
        lon: float,
        pred_info: Optional[Dict[str, Any]],
        analogues_info: List[Dict[str, Any]],
    ) -> str:
        """High-fidelity deterministic conversational generator."""
        lower = query.lower()

        # Check for teleconnection / science explanation queries
        if any(w in lower for w in ["how", "explain", "solar", "sunspot", "mechanism", "schwabe"]):
            return (
                f"### Heliophysical & Paleoclimatological Teleconnection\n\n"
                f"The FRADSCR framework couples the **~11-year Schwabe solar magnetic cycle** "
                f"with Ethiopian monsoon dynamics through two documented physical pathways:\n\n"
                f"1. **Top-Down Stratospheric Modulation:** During solar maxima or transitions, "
                f"enhanced solar ultraviolet (UV) radiation increases equatorial stratospheric ozone heating. "
                f"This shifts upper-tropospheric easterly jets and alters the seasonal latitudinal march of the "
                f"**Intertropical Convergence Zone (ITCZ)**, which controls Ethiopia's vital *Kiremt* summer rains.\n\n"
                f"2. **Dendrochronological Memory:** Annual ring-width indices ($RWI$) of high-elevation "
                f"*Juniperus procera* (African Pencilcedar) record not only current-year moisture at lag $\\tau=0$ "
                f"($R = +0.197, p = 0.0215$), but also sustained multi-year carbohydrate reserves and soil moisture "
                f"memory over subsequent years (lags $\\tau=1$ and $2$).\n\n"
                f"This decadal signal provides an observable early warning horizon up to 11 years into the future."
            )

        # If prediction available
        if pred_info:
            sev = pred_info.get("severity_label", "Normal")
            conf = pred_info.get("model_confidence", 0.85) * 100
            cls_idx = pred_info.get("predicted_drought_class", 0)
            spei = pred_info.get("continuous_spei", 0.0)
            prescriptive = pred_info.get("prescriptive_action", "HOLD FUNDS (CONSERVE)")
            hydro = pred_info.get("hydrogeology", {})

            reply_parts = [
                f"### Climatological Assessment for {year} (Lat {lat:.2f}°, Lon {lon:.2f}°)\n",
                f"- **Forecasted Condition:** **{sev}** (Class {cls_idx})",
                f"- **Calibrated Decisive Confidence:** **{conf:.1f}%**",
                f"- **Continuous SPEI Moisture Index:** `{spei:+.2f}`",
                f"- **Aquifer Stress Index:** `{hydro.get('aquifer_stress_index', 0.0):.1f}%`",
                f"- **Recommended Solar Pump Runtime:** `{hydro.get('recommended_solar_pumping_hours', 8.0)} hrs/day`",
                f"- **Prescriptive RL Directive:** `{prescriptive}`\n",
            ]

            if cls_idx >= 1 and analogues_info:
                top = analogues_info[0]
                reply_parts.append(
                    f"**Historical Precedent:** The closest historical hydroclimatic analogue is the **{top.get('name')} ({top.get('year_range')})** "
                    f"[Similarity: {top.get('similarity_percentage')}%]. Key operational lesson: *{top.get('modern_operational_takeaways', [''])[0]}*\n"
                )

            if cls_idx == 2:
                reply_parts.append(
                    "⚠️ **Action Required:** Trigger anticipatory drought financing, review reservoir drawdown curves at GERD/Tekeze, "
                    "and enact livestock destocking protocols across pastoral zones."
                )
            else:
                reply_parts.append(
                    "✅ **Action Required:** Conditions remain favorable. Continue standard seasonal groundwater monitoring and invest in rainwater harvesting recharge."
                )

            return "\n".join(reply_parts)

        # Fallback general answer
        return (
            f"The FRADSCR Intelligence Copilot is active. You can ask me to evaluate drought risks for specific years "
            f"(e.g., *'What is the 2028 drought forecast for Borana?'*), explain solar teleconnections, or generate "
            f"policy briefs for water ministers and borehole operators."
        )


_GLOBAL_COPILOT: Optional[HydroclimaticCopilotAgent] = None


def get_copilot_agent() -> HydroclimaticCopilotAgent:
    """Get or instantiate the global Copilot agent singleton."""
    global _GLOBAL_COPILOT
    if _GLOBAL_COPILOT is None:
        _GLOBAL_COPILOT = HydroclimaticCopilotAgent()
    return _GLOBAL_COPILOT
