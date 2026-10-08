"""
FRADSCR MLOps Pipeline — Data Quality & Schema Contracts
========================================================
Validates raw paleoclimate, solar, and oceanic observations before ingestion,
and enforces strict schema and data integrity on feature matrices.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("fradscr.mlops.contracts")


@dataclass
class ValidationReport:
    """Standardized report for dataset or feature validation."""
    stage: str
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataDriftReport:
    """Statistical drift evaluation report comparing baseline vs candidate datasets."""
    stage: str
    drift_detected: bool
    drifted_features: List[str] = field(default_factory=list)
    p_values: Dict[str, float] = field(default_factory=dict)
    ks_statistics: Dict[str, float] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)


class DataQualityContracts:
    """Enforces strict pre-training and pre-inference data validation contracts."""

    @staticmethod
    def validate_tucson_rwl(filepath: Path) -> ValidationReport:
        """Validate Tucson .rwl file structure, non-emptiness, and formatting."""
        errors: List[str] = []
        warnings: List[str] = []
        metrics: Dict[str, Any] = {}

        if not filepath.exists():
            errors.append(f"Tucson file does not exist: {filepath}")
            return ValidationReport("tucson_rwl", False, errors, warnings, metrics)

        try:
            from treering.parser import parse_rwl
            df = parse_rwl(filepath)
            n_rows = len(df)
            n_series = df["series_id"].nunique()
            min_yr = int(df["year"].min())
            max_yr = int(df["year"].max())

            metrics = {
                "rows": n_rows,
                "series_count": n_series,
                "year_min": min_yr,
                "year_max": max_yr,
                "time_span": max_yr - min_yr + 1,
            }

            if n_rows == 0 or n_series == 0:
                errors.append(f"Parsed DataFrame is empty for {filepath.name}")

            # Check for negative ring widths
            neg_count = int((df["ring_width"] < 0).sum())
            if neg_count > 0:
                errors.append(f"Found {neg_count} negative ring widths in {filepath.name}")

            # Check time span
            if max_yr - min_yr < 30:
                warnings.append(f"Time span ({max_yr - min_yr + 1} yrs) is short for decadal climate fitting")

        except Exception as exc:
            errors.append(f"Exception parsing Tucson rwl {filepath.name}: {exc}")

        return ValidationReport(
            stage=f"tucson_rwl:{filepath.name}",
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metrics=metrics,
        )

    @staticmethod
    def validate_silso_sunspots(filepath: Path) -> ValidationReport:
        """Validate SILSO solar sunspot dataset for continuity and valid values."""
        errors: List[str] = []
        warnings: List[str] = []
        metrics: Dict[str, Any] = {}

        if not filepath.exists():
            errors.append(f"SILSO sunspot file not found: {filepath}")
            return ValidationReport("silso_sunspots", False, errors, warnings, metrics)

        try:
            from treering.solar_lag import load_sunspot_data
            df = load_sunspot_data(filepath)
            years = df["year"].values
            col = "sunspot" if "sunspot" in df.columns else "sunspot_number"
            sunspots = df[col].values

            metrics = {
                "year_min": int(years.min()),
                "year_max": int(years.max()),
                "total_years": len(years),
                "sunspot_max": float(sunspots.max()),
                "sunspot_mean": float(sunspots.mean()),
            }


            # Check for missing years in sequence
            expected_years = np.arange(years.min(), years.max() + 1)
            diff = set(expected_years) - set(years)
            if diff:
                errors.append(f"SILSO sunspot sequence has {len(diff)} missing years: {sorted(diff)[:5]}...")

            # Check for negative values
            if (sunspots < 0).any():
                errors.append("Negative sunspot counts detected in SILSO table")

            # Check reasonable bounds
            if sunspots.max() > 600:
                warnings.append(f"Unusually high sunspot max: {sunspots.max()}")

        except Exception as exc:
            errors.append(f"Error validating SILSO file: {exc}")

        return ValidationReport(
            stage="silso_sunspots",
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metrics=metrics,
        )

    @staticmethod
    def validate_ocean_indices(filepath: Path) -> ValidationReport:
        """Validate HadISST oceanic ENSO Niño 3.4 and IOD DMI dataset."""
        errors: List[str] = []
        warnings: List[str] = []
        metrics: Dict[str, Any] = {}

        if not filepath.exists():
            errors.append(f"Ocean indices file not found: {filepath}")
            return ValidationReport("ocean_indices", False, errors, warnings, metrics)

        try:
            df = pd.read_csv(filepath)
            required_cols = {"year", "nino34_mean", "dmi_mean"}
            missing_cols = required_cols - set(df.columns)
            if missing_cols:
                errors.append(f"Missing required columns in ocean dataset: {missing_cols}")
                return ValidationReport("ocean_indices", False, errors, warnings, metrics)

            years = df["year"].values
            metrics = {
                "year_min": int(years.min()),
                "year_max": int(years.max()),
                "count": len(df),
                "nino34_range": [float(df["nino34_mean"].min()), float(df["nino34_mean"].max())],
                "dmi_range": [float(df["dmi_mean"].min()), float(df["dmi_mean"].max())],
            }

            # Check nulls
            null_count = int(df[["nino34_mean", "dmi_mean"]].isnull().sum().sum())
            if null_count > 0:
                errors.append(f"Detected {null_count} null values in ocean indices")

            # Validate physical range for anomalies (-5 to +5 deg C)
            if df["nino34_mean"].min() < -6.0 or df["nino34_mean"].max() > 6.0:
                warnings.append("Niño 3.4 index contains values outside typical ±6.0°C anomaly bounds")

        except Exception as exc:
            errors.append(f"Failed to read ocean indices: {exc}")

        return ValidationReport(
            stage="ocean_indices",
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metrics=metrics,
        )

    @staticmethod
    def validate_feature_matrix(
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        required_features: Optional[List[str]] = None,
    ) -> ValidationReport:
        """Validate training feature matrix schema, nulls, and label distributions."""
        errors: List[str] = []
        warnings: List[str] = []
        metrics: Dict[str, Any] = {}

        req = required_features or [
            "sunspot",
            "sunspot_lag1",
            "sunspot_lag2",
            "sunspot_lag3",
            "sunspot_lag4",
            "sunspot_lag5",
            "sunspot_smooth11",
            "sunspot_diff1",
            "sunspot_diff3",
            "solar_phase",
            "solar_phase_sin",
            "solar_phase_cos",
            "rwi",
            "rwi_lag1",
            "rwi_diff1",
            "rwi_smooth5",
            "nino34_mean",
            "dmi_mean",
        ]

        missing_features = [f for f in req if f not in X.columns]
        if missing_features:
            errors.append(f"Missing required feature columns: {missing_features}")

        present_features = [f for f in req if f in X.columns]
        if present_features:
            # Check for nulls or infinite values
            null_features = X[present_features].isnull().sum()
            cols_with_nulls = null_features[null_features > 0].to_dict()
            if cols_with_nulls:
                errors.append(f"Null values detected in feature columns: {cols_with_nulls}")

            inf_mask = np.isinf(X[present_features].values)
            if inf_mask.any():
                errors.append(f"Infinite values detected in feature matrix: {int(inf_mask.sum())} occurrences")

        metrics["feature_count"] = X.shape[1]
        metrics["row_count"] = X.shape[0]


        if y is not None:
            classes = sorted(y.unique())
            metrics["classes"] = [int(c) for c in classes]
            metrics["class_distribution"] = {int(k): int(v) for k, v in y.value_counts().items()}

            # Label validation: must be subset of {0, 1, 2}
            invalid_classes = set(classes) - {0, 1, 2}
            if invalid_classes:
                errors.append(f"Invalid drought classes detected: {invalid_classes}. Must be 0, 1, or 2.")

            # Class imbalance check
            class_counts = y.value_counts()
            if min(class_counts) < 5:
                warnings.append(f"Severe class scarcity: minority class has only {min(class_counts)} samples")

        return ValidationReport(
            stage="feature_matrix",
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metrics=metrics,
        )

    @staticmethod
    def detect_feature_drift(
        baseline_df: pd.DataFrame,
        candidate_df: pd.DataFrame,
        features: List[str],
        significance_level: float = 0.05,
    ) -> DataDriftReport:
        """Two-sample Kolmogorov-Smirnov test to detect significant feature distribution drift."""
        from scipy.stats import ks_2samp

        drifted: List[str] = []
        p_vals: Dict[str, float] = {}
        ks_stats: Dict[str, float] = {}

        for col in features:
            if col in baseline_df.columns and col in candidate_df.columns:
                s1 = baseline_df[col].dropna().values
                s2 = candidate_df[col].dropna().values
                if len(s1) > 5 and len(s2) > 5:
                    res = ks_2samp(s1, s2)
                    p_val = float(res.pvalue)
                    stat = float(res.statistic)
                    p_vals[col] = p_val
                    ks_stats[col] = stat
                    if p_val < significance_level:
                        drifted.append(col)

        return DataDriftReport(
            stage="data_drift_evaluation",
            drift_detected=len(drifted) > 0,
            drifted_features=drifted,
            p_values=p_vals,
            ks_statistics=ks_stats,
            metrics={"tested_features": len(p_vals), "drifted_count": len(drifted)},
        )
