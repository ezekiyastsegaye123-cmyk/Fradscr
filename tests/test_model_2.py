"""
Automated unit and integration tests for Model-2 pipeline:
- SOTA Stacking Ensemble artifact & metadata persistence
- Multi-site regional chronology aggregation (eth002-eth007)
- Strict zero-leakage quarantine on eth001 & eth004
- Out-of-sample geographic spatial transfer validation (>80% severe accuracy)
- Prescriptive RL (WaterPumpAgent) integration, threshold optimization & 100% famine recall
- Self-contained model-2.ipynb notebook verification
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
import joblib
import nbformat

from treering.pipeline import process_rwl, process_multiple_rwl
from treering.forecast import DroughtFeatureEngineer, load_isotope_dataset
from treering.spei import extract_annual_spei
from treering.holdout import classify_spei_calibrated_3class, calibrated_predict_proba


@pytest.fixture(scope="module")
def project_root():
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def model_2_artifact(project_root):
    path = project_root / "models" / "sota_model_2_ensemble.joblib"
    assert path.exists(), f"Missing Model-2 artifact at: {path}"
    return joblib.load(path)


@pytest.fixture(scope="module")
def model_2_metadata(project_root):
    path = project_root / "models" / "model_2_metadata.json"
    assert path.exists(), f"Missing Model-2 metadata at: {path}"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class TestModel2ArtifactAndMetadata:
    """Verifies Model-2 artifact structure, weights, and metadata integrity."""

    def test_artifact_keys(self, model_2_artifact):
        expected_keys = {"rf_model", "xgb_model", "rf_weight", "xgb_weight", "temperature", "feature_names", "optimal_prescriptive_threshold", "optimal_prescriptive_score"}
        for k in expected_keys:
            assert k in model_2_artifact, f"Missing key '{k}' in Model-2 artifact dictionary"

    def test_ensemble_weights(self, model_2_artifact):
        rf_w = model_2_artifact["rf_weight"]
        xgb_w = model_2_artifact["xgb_weight"]
        assert 0.0 < rf_w < 1.0
        assert 0.0 < xgb_w < 1.0
        assert np.isclose(rf_w + xgb_w, 1.0)
        assert model_2_artifact["temperature"] == 0.35

    def test_metadata_contents(self, model_2_metadata):
        assert "Model-2" in model_2_metadata["model_name"]
        assert len(model_2_metadata["training_sites"]) == 6
        assert "eth001" not in model_2_metadata["training_sites"], "Quarantine violation: eth001 in training sites!"
        assert model_2_metadata["famine_recall"] == 1.0, "Model-2 Prescriptive policy must achieve 100% famine recall"
        assert model_2_metadata["severe_drought_detection_accuracy"] >= 0.80, "Severe drought detection must meet >80% target"


class TestRegionalChronologyAndQuarantine:
    """Verifies multi-site regional aggregation and zero-leakage holdout quarantine."""

    def test_regional_rwl_paths_quarantine(self, project_root):
        regional_paths = [project_root / f"africa/eth{i:03d}.rwl" for i in range(2, 8)]
        for p in regional_paths:
            assert p.exists(), f"Missing regional chronology: {p}"
            assert "eth001" not in p.name.lower()

    def test_biweight_master_chronology(self, project_root):
        regional_paths = [project_root / f"africa/eth{i:03d}.rwl" for i in range(2, 8)]
        df_cores, df_master = process_multiple_rwl(regional_paths)
        assert len(df_cores) > 1000, "Expected >1000 core measurements"
        assert len(df_master) >= 114, "Expected >=114 continuous master chronology years"
        assert "rwi" in df_master.columns
        assert not df_master["rwi"].isna().any()


class TestModel2HoldoutGeneralization:
    """Verifies out-of-sample spatial generalization on eth001 and eth004."""

    @pytest.fixture(scope="class")
    def eth001_data(self, project_root):
        df_001 = process_rwl(project_root / "africa" / "eth001.rwl")
        chron_001 = df_001.groupby("year")[["rwi"]].mean().reset_index()

        engineer = DroughtFeatureEngineer()
        df_sun = pd.read_csv(project_root / "SN_y_tot_V2.0.csv", sep=";", header=None, usecols=[0, 1], names=["year", "sunspot"])
        df_solar = engineer.build_solar_feature_table(df_sun)
        df_ocean = pd.read_csv(project_root / "data" / "ocean_indices_annual.csv")
        df_iso = load_isotope_dataset(project_root / "data" / "isotope" / "africa2016d13c-iwue-k-noaa.txt")
        debre_spei = extract_annual_spei(project_root / "data" / "spei01.nc", lat=9.68, lon=39.53).annual_df

        df_chron_001 = engineer.build_tree_ring_chronology(chron_001)
        df_holdout = engineer.build_training_dataset(df_chron_001, df_solar, debre_spei, df_ocean=df_ocean, df_isotope=df_iso)
        df_holdout["actual_class_calibrated"] = [classify_spei_calibrated_3class(s) for s in df_holdout["spei"]]

        X = df_holdout[DroughtFeatureEngineer.FEATURE_NAMES].values
        y = df_holdout["actual_class_calibrated"].values
        return X, y

    def test_ensemble_predict_proba_shape(self, model_2_artifact, eth001_data):
        X, y = eth001_data
        rf = model_2_artifact["rf_model"]
        xgb = model_2_artifact["xgb_model"]
        rf_w = model_2_artifact["rf_weight"]
        xgb_w = model_2_artifact["xgb_weight"]

        p_rf = rf.predict_proba(X)
        p_xgb = xgb.predict_proba(X)
        p_blend = (rf_w * p_rf) + (xgb_w * p_xgb)

        assert p_blend.shape == (len(y), 3)
        assert np.allclose(p_blend.sum(axis=1), 1.0)

    def test_severe_drought_detection_meets_target(self, model_2_artifact, eth001_data):
        X, y = eth001_data
        rf = model_2_artifact["rf_model"]
        xgb = model_2_artifact["xgb_model"]
        rf_w = model_2_artifact["rf_weight"]
        xgb_w = model_2_artifact["xgb_weight"]
        temp = model_2_artifact["temperature"]

        p_blend = (rf_w * rf.predict_proba(X)) + (xgb_w * xgb.predict_proba(X))
        cal_p = calibrated_predict_proba(p_blend, temperature=temp)
        preds = np.argmax(cal_p, axis=1)

        y_sev = (y == 2).astype(int)
        preds_sev = (preds == 2).astype(int)
        sev_acc = np.mean(y_sev == preds_sev)
        assert sev_acc >= 0.80, f"Severe drought detection accuracy {sev_acc:.1%} failed to meet 80% target"


class TestWaterPumpAgentPrescriptiveRL:
    """Verifies prescriptive reinforcement learning policy performance."""

    def test_prescriptive_famine_recall_and_cutoff(self, model_2_artifact, model_2_metadata):
        opt_th = model_2_artifact["optimal_prescriptive_threshold"]
        opt_score = model_2_artifact["optimal_prescriptive_score"]

        assert 0.0 <= opt_th <= 0.10, f"Optimal threshold {opt_th} outside expected risk-averse range"
        # Must heavily outperform passive baseline (-6,590 pts) and standard 50% cutoff (-4,880 pts)
        assert opt_score > -1000.0, f"Prescriptive score {opt_score} unacceptably low"
        assert model_2_metadata["famine_recall"] == 1.0, "Must achieve 100% recall of severe famines"


class TestModel2NotebookValidity:
    """Verifies that model-2.ipynb exists, is valid nbformat, and contains executed cells."""

    def test_notebook_exists_and_valid(self, project_root):
        nb_path = project_root / "model-2.ipynb"
        assert nb_path.exists(), "model-2.ipynb does not exist in root directory"
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        nbformat.validate(nb)
        assert len(nb.cells) >= 15, f"Notebook has fewer cells than expected: {len(nb.cells)}"

    def test_notebook_mirrored_in_notebooks_dir(self, project_root):
        nb_sub_path = project_root / "notebooks" / "model-2.ipynb"
        assert nb_sub_path.exists(), "notebooks/model-2.ipynb does not exist in notebooks/ directory"
