"""
Unit and Integration Tests for FRADSCR MLOps Pipeline & Gating Framework
========================================================================
"""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from treering.mlops.contracts import DataQualityContracts
from treering.mlops.gating import ModelDeploymentGating
from treering.mlops.orchestrator import MLOpsPipelineOrchestrator
from treering.mlops.registry import ModelRegistry


class TestDataQualityContracts:
    """Validate data quality checks for raw datasets and feature matrices."""

    def test_validate_tucson_rwl(self):
        rwl_path = Path("africa/eth007.rwl")
        report = DataQualityContracts.validate_tucson_rwl(rwl_path)
        assert report.is_valid is True
        assert report.metrics["series_count"] > 0
        assert report.metrics["rows"] > 1000

    def test_validate_silso_sunspots(self):
        sun_path = Path("SN_y_tot_V2.0.csv")
        report = DataQualityContracts.validate_silso_sunspots(sun_path)
        assert report.is_valid is True
        assert report.metrics["total_years"] >= 300
        assert report.metrics["year_min"] == 1700

    def test_validate_ocean_indices(self):
        ocean_path = Path("data/ocean_indices_annual.csv")
        report = DataQualityContracts.validate_ocean_indices(ocean_path)
        assert report.is_valid is True
        assert "nino34_range" in report.metrics

    def test_validate_feature_matrix_success(self):
        df = pd.DataFrame({
            "sunspot": [10.0, 20.0, 30.0],
            "rwi": [0.95, 1.05, 0.88],
            "target": [0, 1, 2],
        })
        report = DataQualityContracts.validate_feature_matrix(
            df[["sunspot", "rwi"]],
            y=df["target"],
            required_features=["sunspot", "rwi"],
        )
        assert report.is_valid is True

    def test_validate_feature_matrix_missing_columns(self):
        df = pd.DataFrame({"sunspot": [10.0, 20.0]})
        report = DataQualityContracts.validate_feature_matrix(
            df, required_features=["sunspot", "missing_feat"]
        )
        assert report.is_valid is False
        assert "missing_feat" in report.errors[0]

    def test_detect_feature_drift(self):
        np.random.seed(42)
        df_base = pd.DataFrame({"sunspot": np.random.normal(50, 10, 100)})
        df_cand_same = pd.DataFrame({"sunspot": np.random.normal(50, 10, 100)})
        report_same = DataQualityContracts.detect_feature_drift(
            df_base, df_cand_same, features=["sunspot"]
        )
        assert report_same.drift_detected is False

        df_cand_drifted = pd.DataFrame({"sunspot": np.random.normal(150, 10, 100)})
        report_drifted = DataQualityContracts.detect_feature_drift(
            df_base, df_cand_drifted, features=["sunspot"]
        )
        assert report_drifted.drift_detected is True
        assert "sunspot" in report_drifted.drifted_features


class TestModelDeploymentGating:
    """Validate operational benchmark gating and deployment decision logic."""

    def test_all_gates_pass(self):
        metrics = {
            "severe_drought_detection_accuracy": 0.868,
            "normal_year_accuracy": 0.892,
            "famine_recall": 1.0,
            "brier_score": 0.22,
        }
        scorecard = ModelDeploymentGating.evaluate_model(metrics)
        assert scorecard.passed_all_gates is True
        assert scorecard.decision == "DEPLOY_APPROVED"
        assert len(scorecard.checks) == 4

    def test_gate_failure_on_severe_drought_drop(self):
        metrics = {
            "severe_drought_detection_accuracy": 0.65,  # Below 0.80 target
            "normal_year_accuracy": 0.8923,
            "famine_recall": 1.0,
            "brier_score": 0.22,
        }
        scorecard = ModelDeploymentGating.evaluate_model(metrics)
        assert scorecard.passed_all_gates is False
        assert scorecard.decision == "DEPLOY_REJECTED"
        failed = [c for c in scorecard.checks if not c.passed]
        assert len(failed) == 1
        assert failed[0].name == "severe_drought_detection_accuracy"


class TestModelRegistry:
    """Validate model registration, checksum hashing, and promotion lifecycle."""

    def test_register_and_promote(self, tmp_path):
        dummy_model = tmp_path / "model.joblib"
        dummy_model.write_bytes(b"dummy binary model payload")

        registry_file = tmp_path / "registry.json"
        registry = ModelRegistry(registry_file=registry_file)

        # Register candidate
        e1 = registry.register_model(
            model_id="candidate_1",
            version="1.0.0",
            artifact_path=dummy_model,
            model_type="RandomForest",
            metrics={"severe_drought_detection_accuracy": 0.85},
            gating_decision="DEPLOY_APPROVED",
            training_data_sources=["test_rwl"],
            hyperparameters={"n_estimators": 100},
            feature_names=["sunspot", "rwi"],
            promote_to_champion=False,
        )
        assert e1.lifecycle_status == "candidate"
        assert len(e1.artifact_sha256) == 64

        # Promote to champion
        e2 = registry.register_model(
            model_id="candidate_2",
            version="1.1.0",
            artifact_path=dummy_model,
            model_type="RandomForest",
            metrics={"severe_drought_detection_accuracy": 0.88},
            gating_decision="DEPLOY_APPROVED",
            training_data_sources=["test_rwl"],
            hyperparameters={"n_estimators": 200},
            feature_names=["sunspot", "rwi"],
            promote_to_champion=True,
        )
        assert e2.lifecycle_status == "champion"
        champ = registry.get_champion("RandomForest")
        assert champ is not None
        assert champ.model_id == "candidate_2"


class TestMLOpsPipelineOrchestrator:
    """Validate full end-to-end pipeline execution and individual stages."""

    def test_dry_run_pipeline(self):
        orch = MLOpsPipelineOrchestrator()
        record = orch.run_pipeline(dry_run=True)
        assert record.overall_status == "SUCCESS"
        assert len(record.stages_executed) == 7

    def test_isolated_data_validation_stage(self):
        orch = MLOpsPipelineOrchestrator()
        record = orch.run_pipeline(stages=["data_validation"])
        assert record.overall_status == "SUCCESS"
        assert len(record.stages_executed) == 1
        assert record.stages_executed[0].status == "SUCCESS"

    def test_isolated_data_drift_stage(self):
        orch = MLOpsPipelineOrchestrator()
        record = orch.run_pipeline(stages=["data_drift_monitoring"])
        assert record.overall_status == "SUCCESS"
        assert len(record.stages_executed) == 1
        assert record.stages_executed[0].status == "SUCCESS"

    def test_isolated_gating_stage(self):
        orch = MLOpsPipelineOrchestrator()
        record = orch.run_pipeline(stages=["model_training", "model_gating"])
        assert record.overall_status == "SUCCESS"
        assert record.deployment_decision == "DEPLOY_APPROVED"
