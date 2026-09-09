"""
Automated tests for Model-1 pipeline:
- Tree-ring candidate dataset discovery & validation
- Chronological 80/20 train/test split isolation
- Temporal and target leakage audit
- Random Forest training and prediction validity
- Artifact export and reload reproducibility
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
import joblib
from sklearn.ensemble import RandomForestClassifier

from treering.pipeline import process_rwl
from treering.forecast import DroughtFeatureEngineer
from treering.holdout import classify_spei_calibrated_3class, CLASS_NAMES_3


@pytest.fixture(scope="module")
def project_root():
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def candidate_rwl_files(project_root):
    rwl_dir = project_root / "africa"
    return sorted(list(rwl_dir.glob("eth*.rwl")))


class TestDatasetSelection:
    """Test candidate tree-ring discovery and properties."""

    def test_all_candidates_discovered(self, candidate_rwl_files):
        assert len(candidate_rwl_files) >= 7, f"Expected at least 7 candidates, found {len(candidate_rwl_files)}"
        expected_ids = ["eth001", "eth002", "eth003", "eth004", "eth005", "eth006", "eth007"]
        found_stems = [p.stem for p in candidate_rwl_files]
        for eid in expected_ids:
            assert eid in found_stems, f"Missing candidate dataset: {eid}"

    def test_candidate_data_integrity(self, candidate_rwl_files):
        for rwl_path in candidate_rwl_files:
            df = process_rwl(rwl_path)
            assert "year" in df.columns
            assert "rwi" in df.columns
            assert "series_id" in df.columns
            assert len(df) > 0
            assert not df["year"].isna().any()
            assert not df["rwi"].isna().any()

    def test_selected_dataset_is_eth007(self, candidate_rwl_files):
        # eth007 has the longest modern overlap (1901-2014) and highest recency
        df_eth007 = process_rwl([p for p in candidate_rwl_files if p.stem == "eth007"][0])
        assert df_eth007["year"].max() == 2014, "ETH007 should extend to 2014"
        other_files = [p for p in candidate_rwl_files if p.stem != "eth007"]
        for p in other_files:
            df_other = process_rwl(p)
            assert df_other["year"].max() < 2014, f"{p.stem} unexpectedly extends to 2014 or beyond"


class TestDataSplitAndLeakage:
    """Test chronological 80/20 split and temporal/target leakage."""

    @pytest.fixture(scope="class")
    def dataset(self, project_root):
        df_rwl = process_rwl(project_root / "africa" / "eth007.rwl")
        chron_df = df_rwl.groupby("year")[["rwi"]].mean().reset_index()

        df_sun = pd.read_csv(project_root / "SN_y_tot_V2.0.csv", sep=";", header=None, usecols=[0, 1])
        df_sun.columns = ["year_dec", "sunspot"]
        df_sun["year"] = df_sun["year_dec"].astype(int)
        df_sun = df_sun.dropna(subset=["year", "sunspot"]).drop_duplicates("year").sort_values("year").reset_index(drop=True)

        ocean_path = project_root / "data" / "ocean_indices_annual.csv"
        df_ocean = pd.read_csv(ocean_path) if ocean_path.exists() else None
        df_spei = pd.read_csv(project_root / "results" / "spei_gondar.csv")

        engineer = DroughtFeatureEngineer()
        df_chron = engineer.build_tree_ring_chronology(chron_df)
        df_solar = engineer.build_solar_feature_table(df_sun)
        df_full = engineer.build_training_dataset(df_chron, df_solar, df_spei, df_ocean=df_ocean)
        df_full["target_3class"] = [classify_spei_calibrated_3class(s) for s in df_full["spei"]]
        return df_full

    def test_chronological_80_20_split(self, dataset):
        n_total = len(dataset)
        n_train = int(n_total * 0.80)
        df_train = dataset.iloc[:n_train]
        df_test = dataset.iloc[n_train:]

        assert len(df_train) + len(df_test) == n_total
        assert 0.78 <= len(df_train) / n_total <= 0.82
        assert 0.18 <= len(df_test) / n_total <= 0.22

        # Verify strict chronological isolation
        assert df_train["year"].max() < df_test["year"].min()

    def test_no_target_leakage_in_predictors(self, dataset):
        predictors = DroughtFeatureEngineer.FEATURE_NAMES
        assert "spei" not in predictors
        assert "target_3class" not in predictors
        assert "target" not in predictors
        assert "drought_class" not in predictors

    def test_no_train_test_overlap(self, dataset):
        n_train = int(len(dataset) * 0.80)
        train_years = set(dataset.iloc[:n_train]["year"])
        test_years = set(dataset.iloc[n_train:]["year"])
        overlap = train_years.intersection(test_years)
        assert len(overlap) == 0, f"Detected year overlap between train and test: {overlap}"


class TestModel1TrainingAndArtifact:
    """Test Model-1 training, predictions, feature importances, and serialization."""

    @pytest.fixture(scope="class")
    def trained_artifacts(self, project_root):
        df_rwl = process_rwl(project_root / "africa" / "eth007.rwl")
        chron_df = df_rwl.groupby("year")[["rwi"]].mean().reset_index()

        df_sun = pd.read_csv(project_root / "SN_y_tot_V2.0.csv", sep=";", header=None, usecols=[0, 1])
        df_sun.columns = ["year_dec", "sunspot"]
        df_sun["year"] = df_sun["year_dec"].astype(int)
        df_sun = df_sun.dropna(subset=["year", "sunspot"]).drop_duplicates("year").sort_values("year").reset_index(drop=True)

        ocean_path = project_root / "data" / "ocean_indices_annual.csv"
        df_ocean = pd.read_csv(ocean_path) if ocean_path.exists() else None
        df_spei = pd.read_csv(project_root / "results" / "spei_gondar.csv")

        engineer = DroughtFeatureEngineer()
        df_chron = engineer.build_tree_ring_chronology(chron_df)
        df_solar = engineer.build_solar_feature_table(df_sun)
        df_data = engineer.build_training_dataset(df_chron, df_solar, df_spei, df_ocean=df_ocean)
        df_data["target_3class"] = [classify_spei_calibrated_3class(s) for s in df_data["spei"]]

        n_train = int(len(df_data) * 0.80)
        df_train = df_data.iloc[:n_train]
        df_test = df_data.iloc[n_train:]

        X_train = df_train[DroughtFeatureEngineer.FEATURE_NAMES].values
        y_train = df_train["target_3class"].values
        X_test = df_test[DroughtFeatureEngineer.FEATURE_NAMES].values
        y_test = df_test["target_3class"].values

        clf = RandomForestClassifier(n_estimators=350, max_depth=7, max_features="log2", random_state=42, oob_score=True)
        clf.fit(X_train, y_train)

        return clf, X_test, y_test

    def test_model_training_and_predictions(self, trained_artifacts):
        clf, X_test, y_test = trained_artifacts
        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)

        assert len(preds) == len(X_test)
        assert probs.shape == (len(X_test), 3)
        assert np.allclose(probs.sum(axis=1), 1.0)
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    def test_feature_importance_validity(self, trained_artifacts):
        clf, _, _ = trained_artifacts
        importances = clf.feature_importances_
        assert len(importances) == len(DroughtFeatureEngineer.FEATURE_NAMES)
        assert np.isclose(importances.sum(), 1.0)
        assert np.all(importances >= 0.0)

    def test_model_export_and_reload_reproducibility(self, trained_artifacts, project_root):
        clf, X_test, _ = trained_artifacts
        original_preds = clf.predict(X_test)
        original_probs = clf.predict_proba(X_test)

        artifact_path = project_root / "models" / "random_forest_model_1.joblib"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, artifact_path)

        assert artifact_path.exists()
        assert artifact_path.stat().st_size > 1000

        reloaded_clf = joblib.load(artifact_path)
        reloaded_preds = reloaded_clf.predict(X_test)
        reloaded_probs = reloaded_clf.predict_proba(X_test)

        assert np.array_equal(original_preds, reloaded_preds)
        assert np.allclose(original_probs, reloaded_probs)

    def test_model_metadata_file_exists(self, project_root):
        meta_path = project_root / "models" / "model_1_metadata.json"
        assert meta_path.exists()
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "selected_dataset" in data
        assert "ETH007" in data["selected_dataset"]
        assert data["feature_count"] == 20
        assert data["hyperparameters"]["n_estimators"] == 350


class TestRecommendationsImplementation:
    """Automated tests for all four implemented engineering recommendations."""

    @pytest.fixture(scope="class")
    def dataset_and_split(self, project_root):
        df_rwl = process_rwl(project_root / "africa" / "eth007.rwl")
        chron_df = df_rwl.groupby("year")[["rwi"]].mean().reset_index()

        df_sun = pd.read_csv(project_root / "SN_y_tot_V2.0.csv", sep=";", header=None, usecols=[0, 1])
        df_sun.columns = ["year_dec", "sunspot"]
        df_sun["year"] = df_sun["year_dec"].astype(int)
        df_sun = df_sun.dropna(subset=["year", "sunspot"]).drop_duplicates("year").sort_values("year").reset_index(drop=True)

        ocean_path = project_root / "data" / "ocean_indices_annual.csv"
        df_ocean = pd.read_csv(ocean_path) if ocean_path.exists() else None
        df_spei = pd.read_csv(project_root / "results" / "spei_gondar.csv")

        engineer = DroughtFeatureEngineer()
        df_chron = engineer.build_tree_ring_chronology(chron_df)
        df_solar = engineer.build_solar_feature_table(df_sun)
        df_data = engineer.build_training_dataset(df_chron, df_solar, df_spei, df_ocean=df_ocean)
        df_data["target_3class"] = [classify_spei_calibrated_3class(s) for s in df_data["spei"]]

        n_train = int(len(df_data) * 0.80)
        df_train = df_data.iloc[:n_train]
        df_test = df_data.iloc[n_train:]

        X_train = df_train[DroughtFeatureEngineer.FEATURE_NAMES].values
        y_train = df_train["target_3class"].values
        spei_train = df_train["spei"].values

        X_test = df_test[DroughtFeatureEngineer.FEATURE_NAMES].values
        y_test = df_test["target_3class"].values
        spei_test = df_test["spei"].values

        return {
            "engineer": engineer,
            "df_solar": df_solar,
            "df_ocean": df_ocean,
            "X_train": X_train,
            "y_train": y_train,
            "spei_train": spei_train,
            "X_test": X_test,
            "y_test": y_test,
            "spei_test": spei_test,
        }

    def test_rec1_class_weight_balancing_breaks_majority_collapse(self, dataset_and_split):
        data = dataset_and_split
        clf_bal = RandomForestClassifier(n_estimators=350, max_depth=7, max_features="log2", class_weight="balanced", random_state=42)
        clf_bal.fit(data["X_train"], data["y_train"])
        preds = clf_bal.predict(data["X_test"])

        from sklearn.metrics import recall_score
        c2_recall = recall_score(data["y_test"] == 2, preds == 2, zero_division=0)
        assert c2_recall > 0.0, "Balanced Random Forest should produce > 0% recall on Class 2"

    def test_rec2_temperature_scaling_preserves_ranking_and_sharpens(self, dataset_and_split):
        data = dataset_and_split
        clf_bal = RandomForestClassifier(n_estimators=350, max_depth=7, max_features="log2", class_weight="balanced", random_state=42)
        clf_bal.fit(data["X_train"], data["y_train"])
        raw_probs = clf_bal.predict_proba(data["X_test"])

        def temp_scale(probs, T=0.35):
            eps = 1e-7
            logits = np.log(np.clip(probs, eps, 1.0 - eps))
            scaled = logits / T
            return np.exp(scaled) / np.sum(np.exp(scaled), axis=1, keepdims=True)

        raw_pred = np.argmax(raw_probs, axis=1)
        cal_probs_035 = temp_scale(raw_probs, T=0.35)
        cal_pred_035 = np.argmax(cal_probs_035, axis=1)

        # Monotonic scaling must not permute class decisions
        assert np.array_equal(raw_pred, cal_pred_035)

        # Temperature scaling must strictly sharpen confidence
        assert cal_probs_035.max(axis=1).mean() > raw_probs.max(axis=1).mean()

    def test_rec3_geographic_holdout_sites_evaluable(self, project_root, dataset_and_split):
        data = dataset_and_split
        clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        clf.fit(data["X_train"], data["y_train"])

        # Test ETH001 holdout pipeline
        df_rwl_001 = process_rwl(project_root / "africa" / "eth001.rwl")
        chron_001 = df_rwl_001.groupby("year")[["rwi"]].mean().reset_index()
        df_chron_001 = data["engineer"].build_tree_ring_chronology(chron_001)
        df_spei_deb = pd.read_csv(project_root / "results" / "spei_debrebirkan.csv")
        df_holdout_001 = data["engineer"].build_training_dataset(df_chron_001, data["df_solar"], df_spei_deb, df_ocean=data["df_ocean"])
        assert len(df_holdout_001) >= 100
        preds_001 = clf.predict(df_holdout_001[DroughtFeatureEngineer.FEATURE_NAMES].values)
        assert len(preds_001) == len(df_holdout_001)

    def test_rec4_continuous_spei_regression(self, dataset_and_split):
        data = dataset_and_split
        from sklearn.ensemble import RandomForestRegressor
        reg = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        reg.fit(data["X_train"], data["spei_train"])
        preds = reg.predict(data["X_test"])
        assert len(preds) == len(data["X_test"])
        assert isinstance(preds[0], float) or isinstance(preds[0], np.floating)
