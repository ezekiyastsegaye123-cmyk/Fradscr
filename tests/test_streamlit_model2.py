"""
Unit & Integration Tests for Streamlit Model-2 Acceptance
=========================================================
Tests:
- model-2.ipynb parsing and structure validation in Streamlit
- Model-2 vs Model-1 dual prediction engine switching
- Prescriptive WaterPumpAgent directive validation
- Streamlit entrypoint availability
"""

from pathlib import Path
import pytest

from predict_service import (
    DEFAULT_ETH007_MODEL_PATH,
    DEFAULT_MODEL_2_PATH,
    predict_drought,
)
from streamlit_app import (
    compute_decadal_trajectory,
    get_cached_prediction,
    load_and_parse_notebook,
)


@pytest.fixture(scope="module")
def project_root():
    return Path(__file__).resolve().parents[1]


class TestStreamlitNotebookAcceptance:
    """Validates that model-2.ipynb is parsed and accepted by Streamlit correctly."""

    def test_parse_repository_model_2_notebook(self, project_root):
        nb_path = project_root / "model-2.ipynb"
        assert nb_path.exists(), f"Missing {nb_path}"
        parsed = load_and_parse_notebook(str(nb_path), filename=nb_path.name)

        assert parsed["error"] is None
        assert parsed["filename"] == "model-2.ipynb"
        assert "Model-2" in parsed["title"]
        assert parsed["total_cells"] >= 15
        assert parsed["code_cells"] >= 8
        assert parsed["markdown_cells"] >= 8
        assert parsed["has_water_pump_agent"] is True
        assert parsed["has_ensemble"] is True
        assert parsed["has_monte_carlo"] is True

    def test_parse_notebook_from_bytes(self, project_root):
        nb_path = project_root / "model-2.ipynb"
        with open(nb_path, "rb") as f:
            raw_bytes = f.read()

        parsed = load_and_parse_notebook(raw_bytes, filename="uploaded_test.ipynb")
        assert parsed["error"] is None
        assert parsed["filename"] == "uploaded_test.ipynb"
        assert parsed["has_water_pump_agent"] is True

    def test_streamlit_entrypoint_exists(self, project_root):
        s_path = project_root / "streamlit.py"
        app_path = project_root / "streamlit_app.py"
        assert s_path.exists(), "Missing streamlit.py entrypoint"
        assert app_path.exists(), "Missing streamlit_app.py application"


class TestStreamlitModelPredictionSwitching:
    """Validates prediction execution under Model-2 and Model-1."""

    def test_predict_model_2_prescriptive_action(self):
        res = predict_drought(latitude=4.88, longitude=38.08, year=2026, model_path=DEFAULT_MODEL_2_PATH)
        assert "model_type" in res
        assert "Model-2" in res["model_type"]
        assert "prescriptive_action" in res
        assert res["prescriptive_action"] in {"DEPLOY EMERGENCY PUMPS", "HOLD FUNDS (CONSERVE)"}
        assert res["prescriptive_famine_recall"] == 1.0

    def test_predict_model_1_baseline(self):
        res = predict_drought(latitude=12.60, longitude=37.47, year=2005, model_path=DEFAULT_ETH007_MODEL_PATH)
        assert "model_type" in res
        assert "Model-1" in res["model_type"]
        assert "prescriptive_action" in res

    def test_decadal_trajectory_with_model_2(self):
        df_dec = compute_decadal_trajectory(lat=4.88, lon=38.08, temp=0.35, model_path=str(DEFAULT_MODEL_2_PATH))
        assert len(df_dec) == 11
        assert "Prescriptive Action" in df_dec.columns
        assert "Combined Risk (%)" in df_dec.columns
        assert "Year" in df_dec.columns
