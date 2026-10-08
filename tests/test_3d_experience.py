"""
Unit and Integration Tests for FRADSCR 3D WebGL Digital Twin Experience
=======================================================================
"""

from pathlib import Path
from fastapi.testclient import TestClient
import pytest

from predict_service import app


@pytest.fixture
def client():
    return TestClient(app)


class TestThreeDDigitalTwin:
    """Validate 3D digital twin assets and FastAPI serving endpoints."""

    def test_static_asset_exists_and_contains_shaders(self):
        twin_path = Path("static/3d_twin.html")
        assert twin_path.exists(), "static/3d_twin.html must exist on disk"
        content = twin_path.read_text(encoding="utf-8")
        assert "three.min.js" in content
        assert "UnrealBloomPass" in content
        assert "OrbitControls" in content
        assert "snoise(" in content, "Custom procedural Simplex GLSL shader missing"
        assert "CLIMATE_TIMELINE" in content, "Paleoclimate historical timeline data missing"
        assert "CAMERA_PRESETS" in content, "Cinematic camera fly-to presets missing"

    def test_get_3d_endpoint(self, client):
        response = client.get("/3d")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        assert "FRADSCR 3D Digital Twin" in response.text
        assert len(response.text) > 5000

    def test_get_experience_alias_endpoint(self, client):
        response = client.get("/experience")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        assert "FRADSCR 3D Digital Twin" in response.text

    def test_get_static_mount(self, client):
        response = client.get("/static/3d_twin.html")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    def test_streamlit_app_model_2_has_tab_8(self):
        app_path = Path("app_model_2.py")
        assert app_path.exists()
        content = app_path.read_text(encoding="utf-8")
        assert "tab8" in content
        assert "🌐 3D Digital Twin Experience" in content
        assert "static/3d_twin.html" in content
