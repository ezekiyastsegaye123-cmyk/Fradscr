"""
FRADSCR MLOps Pipeline — Automated Model Validation & Deployment Gating
========================================================================
Enforces quantitative operational benchmarks before any model artifact can be
promoted to active staging or production deployment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

logger = logging.getLogger("fradscr.mlops.gating")


@dataclass
class GateCheck:
    """Individual operational gate result."""
    name: str
    target: float
    observed: float
    comparator: Literal[">=", "<=", "=="]
    passed: bool
    description: str


@dataclass
class GatingScorecard:
    """Consolidated model validation scorecard."""
    model_name: str
    model_version: str
    passed_all_gates: bool
    decision: Literal["DEPLOY_APPROVED", "DEPLOY_REJECTED"]
    checks: List[GateCheck] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class ModelDeploymentGating:
    """Evaluates candidate model performance against strict operational criteria."""

    DEFAULT_GATES = {
        "severe_drought_detection_accuracy": (0.80, ">=", "Severe Drought Holdout Accuracy (Target >= 80%)"),
        "normal_year_accuracy": (0.80, ">=", "Normal Agricultural Year Identification (Target >= 80%)"),
        "famine_recall": (1.00, ">=", "Prescriptive RL Asymmetric Famine Recall (Target 100%)"),
        "brier_score": (0.35, "<=", "Multi-Class Brier Calibration Score (Target <= 0.35)"),
    }

    @classmethod
    def evaluate_model(
        cls,
        metrics: Dict[str, float],
        model_name: str = "CandidateModel",
        model_version: str = "v1.0",
        custom_thresholds: Optional[Dict[str, Tuple[float, str, str]]] = None,
    ) -> GatingScorecard:
        """Run all gates on evaluated model metrics.

        Args:
            metrics: Dictionary of observed metrics (e.g. severe_drought_detection_accuracy).
            model_name: Identifier of the candidate model.
            model_version: Version tag.
            custom_thresholds: Optional override of default gate parameters.

        Returns:
            GatingScorecard with detailed evaluation and decision.
        """
        thresholds = custom_thresholds or cls.DEFAULT_GATES
        checks: List[GateCheck] = []
        all_passed = True

        for metric_key, (threshold, comp, desc) in thresholds.items():
            if metric_key not in metrics:
                logger.warning("Metric %s missing from evaluation dictionary; failing gate.", metric_key)
                checks.append(
                    GateCheck(
                        name=metric_key,
                        target=threshold,
                        observed=0.0,
                        comparator=comp,
                        passed=False,
                        description=f"{desc} [MISSING FROM EVALUATION]",
                    )
                )
                all_passed = False
                continue

            observed = float(metrics[metric_key])
            if comp == ">=":
                passed = observed >= threshold
            elif comp == "<=":
                passed = observed <= threshold
            elif comp == "==":
                passed = abs(observed - threshold) < 1e-6
            else:
                passed = False

            if not passed:
                all_passed = False

            checks.append(
                GateCheck(
                    name=metric_key,
                    target=threshold,
                    observed=round(observed, 4),
                    comparator=comp,
                    passed=passed,
                    description=desc,
                )
            )

        decision = "DEPLOY_APPROVED" if all_passed else "DEPLOY_REJECTED"
        return GatingScorecard(
            model_name=model_name,
            model_version=model_version,
            passed_all_gates=all_passed,
            decision=decision,
            checks=checks,
            diagnostics={
                "total_gates": len(checks),
                "passed_count": sum(1 for c in checks if c.passed),
                "failed_count": sum(1 for c in checks if not c.passed),
            },
        )
