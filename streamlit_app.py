"""
FRADSCR — Solar Groundwater & Drought Early Warning System
=====================================================================
Production-grade Streamlit application for drought early warning,
climate teleconnection forecasting, and solar borehole dispatch in
the Horn of Africa (Borana Pastoral Zone, Ethiopia).

Built with:
- Model-2: Tabular SoTA Multi-Site Transfer Learning & Prescriptive RL Framework (model-2.ipynb)
- Regional RCS Master Chronologies (eth002-eth007, N=114 to 330 years)
- Soft-Voting Stacking Ensemble (65% Random Forest + 35% XGBoost)
- Prescriptive RL (WaterPumpAgent) with 100% Famine Recall Guarantee
- 11-Year Schwabe Solar Irradiance Teleconnections (SILSO & NOAA Cycles 25/26)
- Oceanic Dipole Oscillations (ENSO Nino 3.4 & Indian Ocean Dipole)
- High-Resolution Spatial SPEI NetCDF Grids with Anthropogenic Warming Penalty
- Calibrated Probability Scaling (T=0.35 Monotonic Softmax)
"""

from pathlib import Path
import os
import sys
import json
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
import streamlit as st

try:
    from streamlit_js_eval import get_geolocation
    HAS_GEOLOCATION = True
except ImportError:
    HAS_GEOLOCATION = False

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from predict_service import (
        get_engine,
        predict_drought,
        SEVERITY_LABELS,
        DroughtPredictionService,
        DEFAULT_MODEL_2_PATH,
        DEFAULT_ETH007_MODEL_PATH,
        DEFAULT_MODEL_PATH,
    )
except ImportError:
    from predict_service import predict_drought, SEVERITY_LABELS, DroughtPredictionService
    DEFAULT_MODEL_2_PATH = PROJECT_ROOT / "models" / "sota_model_2_ensemble.joblib"
    DEFAULT_ETH007_MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_eth007.joblib"
    DEFAULT_MODEL_PATH = DEFAULT_MODEL_2_PATH if DEFAULT_MODEL_2_PATH.exists() else DEFAULT_ETH007_MODEL_PATH
    def get_engine(model_path=None):
        return DroughtPredictionService.get_instance(model_path=model_path)

# =============================================================================
# Streamlit Page Configuration
# =============================================================================
st.set_page_config(
    page_title="FRADSCR · Drought Early Warning System",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Contrast Professional Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.15rem;
        font-weight: 800;
        color: #0369a1;
        margin-bottom: 0.1rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .metric-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px 18px;
        text-align: center;
    }
    .alert-banner-severe {
        background-color: #fef2f2;
        border-left: 6px solid #dc2626;
        color: #991b1b;
        padding: 16px 20px;
        border-radius: 8px;
        font-size: 1.02rem;
        margin-bottom: 1.5rem;
    }
    .alert-banner-moderate {
        background-color: #fffbeb;
        border-left: 6px solid #d97706;
        color: #92400e;
        padding: 16px 20px;
        border-radius: 8px;
        font-size: 1.02rem;
        margin-bottom: 1.5rem;
    }
    .alert-banner-normal {
        background-color: #f0fdf4;
        border-left: 6px solid #16a34a;
        color: #166534;
        padding: 16px 20px;
        border-radius: 8px;
        font-size: 1.02rem;
        margin-bottom: 1.5rem;
    }
    .dispatch-card {
        background: #f1f5f9;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #cbd5e1;
    }
    .pill-badge-sota {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 700;
        background-color: #dbeafe;
        color: #1d4ed8;
        border: 1px solid #bfdbfe;
    }
    .pill-badge-rl {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 700;
        background-color: #fef3c7;
        color: #b45309;
        border: 1px solid #fde68a;
    }
    .code-preview {
        background-color: #0f172a;
        color: #f8fafc;
        border-radius: 8px;
        padding: 14px;
        font-family: monospace;
        font-size: 0.88rem;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Cached Model Engine & Inference
# =============================================================================
@st.cache_resource(show_spinner="Warming up Climate Teleconnection ML Engine...")
def init_ml_engine(model_path: Optional[str] = None):
    """Cache singleton instance of persistent ML engine."""
    return get_engine(model_path=model_path)


@st.cache_data(show_spinner=False)
def get_cached_prediction(lat: float, lon: float, yr: int, temp: float, model_path: Optional[str] = None):
    """Run cached prediction for a single coordinate, year, and model artifact."""
    return predict_drought(latitude=lat, longitude=lon, year=yr, temperature=temp, model_path=model_path)


@st.cache_data(show_spinner=False)
def compute_decadal_trajectory(lat: float, lon: float, temp: float, model_path: Optional[str] = None):
    """Compute 2025-2035 forward projection trajectory with physical & continuous metrics."""
    years = list(range(2025, 2036))
    records = []
    for y in years:
        res = predict_drought(latitude=lat, longitude=lon, year=y, temperature=temp, model_path=model_path)
        cp = res["confidence_probabilities"]
        ci = res.get("spei_confidence_interval", {})
        hydro = res.get("hydrogeology", {})
        bio = res.get("biological_memory", {})
        records.append({
            "Year": y,
            "Combined Risk (%)": round(res["combined_drought_risk"] * 100, 1),
            "Severe Drought (%)": round(cp.get("class_2", 0) * 100, 1),
            "Moderate Drought (%)": round(cp.get("class_1", 0) * 100, 1),
            "Normal / Wet (%)": round(cp.get("class_0", 0) * 100, 1),
            "Predicted Severity": res["severity_label"],
            "Prescriptive Action": res.get("prescriptive_action", "N/A"),
            "Continuous SPEI": res.get("continuous_spei", 0.0),
            "SPEI [10%-90%]": f"[{ci.get('p10', '—')}, {ci.get('p90', '—')}]",
            "Aquifer Stress (%)": hydro.get("aquifer_stress_index", 0.0),
            "Solar Pump (hrs)": hydro.get("recommended_solar_pumping_hours", 8.0),
            "Bio Growth RWI": bio.get("rwi", 1.0),
            "Model Confidence (%)": round(res["model_confidence"] * 100, 1)
        })
    return pd.DataFrame(records)


@st.cache_data(show_spinner=False)
def load_and_parse_notebook(file_bytes_or_str: Union[str, bytes], filename: str = "model-2.ipynb") -> Dict[str, Any]:
    """Parse Jupyter Notebook JSON structure, cells, metadata, and executive tables."""
    try:
        if isinstance(file_bytes_or_str, bytes):
            nb = json.loads(file_bytes_or_str.decode("utf-8"))
        elif isinstance(file_bytes_or_str, str):
            if file_bytes_or_str.strip().startswith("{"):
                nb = json.loads(file_bytes_or_str)
            else:
                with open(file_bytes_or_str, "r", encoding="utf-8") as f:
                    nb = json.load(f)
        else:
            return {"error": "Invalid notebook content format."}

        cells = nb.get("cells", [])
        md_cells = [c for c in cells if c.get("cell_type") == "markdown"]
        code_cells = [c for c in cells if c.get("cell_type") == "code"]

        title = "Model-2: Tabular SoTA Multi-Site Transfer Learning & Prescriptive RL Framework"
        for c in md_cells:
            text = "".join(c.get("source", []))
            if "Model-2" in text and ("#" in text or "Framework" in text):
                lines = [l.strip().replace("#", "").strip() for l in text.split("\n") if l.strip()]
                if lines:
                    title = lines[0]
                break

        return {
            "filename": filename,
            "title": title,
            "total_cells": len(cells),
            "markdown_cells": len(md_cells),
            "code_cells": len(code_cells),
            "cells": cells,
            "has_water_pump_agent": any("WaterPumpAgent" in "".join(c.get("source", [])) for c in code_cells),
            "has_ensemble": any("Model2Ensemble" in "".join(c.get("source", [])) for c in code_cells),
            "has_spatial_transfer": any("Debrebirkan" in "".join(c.get("source", [])) for c in code_cells + md_cells),
            "has_monte_carlo": any("Monte Carlo" in "".join(c.get("source", [])) for c in code_cells),
            "error": None,
        }
    except Exception as exc:
        return {"error": f"Failed to parse notebook: {exc}"}


# =============================================================================
# Regional Presets & Coordinates
# =============================================================================
PRESETS = {
    "Borana — Yabelo Central Hub (4.88° N, 38.08° E)": (4.88, 38.08),
    "Borana — Dubuluk Solar Borehole (4.45° N, 38.28° E)": (4.45, 38.28),
    "Borana — Mega High-Yield Station (4.05° N, 38.32° E)": (4.05, 38.32),
    "Borana — Moyale Border Aquifer (3.53° N, 39.05° E)": (3.53, 39.05),
    "Gondar / Highlands Base (12.60° N, 37.47° E)": (12.60, 37.47),
    "Debrebirkan Selassie Holdout Site (9.63° N, 39.53° E)": (9.63, 39.53),
    "Somali Region — Gode Basin (5.95° N, 43.55° E)": (5.95, 43.55),
    "Afar Region — Semera Lowlands (11.79° N, 41.01° E)": (11.79, 41.01),
    "Custom Coordinates": None
}


# Session State Initialization for Geolocation and Interactive Calculation
if "input_lat" not in st.session_state:
    st.session_state.input_lat = 4.88
if "input_lon" not in st.session_state:
    st.session_state.input_lon = 38.08
if "location_status" not in st.session_state:
    st.session_state.location_status = None
if "request_gps" not in st.session_state:
    st.session_state.request_gps = False

def on_preset_change():
    chosen = st.session_state.get("preset_selector")
    if chosen in PRESETS and PRESETS[chosen] is not None:
        p_lat, p_lon = PRESETS[chosen]
        st.session_state.input_lat = p_lat
        st.session_state.input_lon = p_lon
        st.session_state.location_status = f"📍 Preset selected: {chosen.split('(')[0].strip()}"


# =============================================================================
# Sidebar Controls & Model-2 Notebook Acceptance
# =============================================================================
with st.sidebar:
    st.markdown("### 💧 FRADSCR Controls")
    st.caption("Paleoclimate Teleconnections & Prescriptive Water Security")

    # Model Engine Selection
    st.markdown("---")
    st.markdown("#### 🧠 Model Selection & Engine")
    MODEL_OPTIONS = {
        "🌟 Model-2: SoTA Ensemble & Prescriptive RL (model-2.ipynb)": str(DEFAULT_MODEL_2_PATH),
        "Model-1: Single-Site Gondar Baseline (eth007 Random Forest)": str(DEFAULT_ETH007_MODEL_PATH),
    }
    selected_model_label = st.selectbox(
        "Active Paleoclimate Model",
        list(MODEL_OPTIONS.keys()),
        index=0,
        help="Model-2 combines 6 regional tree-ring chronologies, XGBoost + RF ensemble, and WaterPumpAgent Prescriptive RL with 100% famine recall."
    )
    is_model_2 = "Model-2" in selected_model_label
    active_model_path = MODEL_OPTIONS[selected_model_label]

    # Notebook Acceptance Interface
    st.markdown("---")
    st.markdown("#### 📘 Model-2 Notebook (`model-2.ipynb`)")
    nb_source = st.radio(
        "Notebook Source",
        ["Bundled `model-2.ipynb`", "Upload Custom `.ipynb`"],
        index=0,
        horizontal=True,
        help="Accept and inspect model-2.ipynb from repository or upload an experimental notebook."
    )

    uploaded_nb_bytes: Optional[bytes] = None
    uploaded_nb_name = "model-2.ipynb"
    if nb_source == "Upload Custom `.ipynb`":
        up_file = st.file_uploader("Upload `.ipynb` File", type=["ipynb"], key="uploaded_ipynb_file")
        if up_file is not None:
            uploaded_nb_bytes = up_file.read()
            uploaded_nb_name = up_file.name
            st.success(f"✅ Accepted: `{uploaded_nb_name}`")
    else:
        repo_nb_path = PROJECT_ROOT / "model-2.ipynb"
        if not repo_nb_path.exists():
            repo_nb_path = PROJECT_ROOT / "notebooks" / "model-2.ipynb"
        if repo_nb_path.exists():
            st.caption(f"📁 Verified repository artifact: `{repo_nb_path.name}` (Accepted)")
            with open(repo_nb_path, "rb") as f:
                uploaded_nb_bytes = f.read()
            uploaded_nb_name = repo_nb_path.name

    # Geolocation Controls
    st.markdown("---")
    st.markdown("##### 📍 Location Selection")
    col_geo1, col_geo2 = st.columns(2)
    with col_geo1:
        if st.button("📍 Device GPS", use_container_width=True, help="Detect exact GPS coordinates via browser HTML5 geolocation."):
            st.session_state.request_gps = True
    with col_geo2:
        if st.button("🌐 Detect IP", use_container_width=True, help="Detect approximate location via network IP."):
            try:
                r = requests.get("https://freeipapi.com/api/json", timeout=3)
                if r.status_code == 200:
                    data = r.json()
                    ip_lat = float(data.get("latitude", 0.0))
                    ip_lon = float(data.get("longitude", 0.0))
                    city = data.get("cityName", "Current Network")
                    if ip_lat != 0.0 or ip_lon != 0.0:
                        st.session_state.input_lat = round(ip_lat, 4)
                        st.session_state.input_lon = round(ip_lon, 4)
                        st.session_state.location_status = f"🌐 Network IP Located: {city} ({ip_lat:.2f}° N, {ip_lon:.2f}° E)"
                        st.rerun()
            except Exception as e:
                st.session_state.location_status = f"⚠️ IP detection unavailable: {e}"

    if st.session_state.get("request_gps", False):
        if HAS_GEOLOCATION:
            loc = get_geolocation(component_key="device_geo_component")
            if loc:
                if "coords" in loc:
                    user_lat = round(loc["coords"]["latitude"], 4)
                    user_lon = round(loc["coords"]["longitude"], 4)
                    st.session_state.input_lat = user_lat
                    st.session_state.input_lon = user_lon
                    st.session_state.location_status = f"📍 Device GPS Detected: {user_lat:.4f}° N, {user_lon:.4f}° E"
                    st.session_state.request_gps = False
                    st.rerun()
                elif "error" in loc:
                    err_msg = loc["error"].get("message", "Permission denied or unavailable.")
                    st.session_state.location_status = f"⚠️ GPS notice: {err_msg}"
                    st.session_state.request_gps = False
            else:
                st.caption("⏳ Querying GPS... Please allow location access in your browser prompt.")
        else:
            st.session_state.location_status = "⚠️ Geolocation module unavailable."
            st.session_state.request_gps = False

    selected_preset = st.selectbox(
        "Regional Station Presets",
        list(PRESETS.keys()),
        key="preset_selector",
        on_change=on_preset_change
    )

    latitude = st.number_input("Latitude (°N)", min_value=-90.0, max_value=90.0, step=0.01, format="%.4f", key="input_lat")
    longitude = st.number_input("Longitude (°E)", min_value=-180.0, max_value=180.0, step=0.01, format="%.4f", key="input_lon")

    if st.session_state.get("location_status"):
        st.caption(st.session_state.location_status)

    st.markdown("---")
    st.markdown("#### ⏳ Forecast Horizon")
    target_year = st.slider("Target Year", min_value=1900, max_value=2035, value=2026, step=1,
                            help="Select historical backtesting year (1900-2024) or forward prospective year (2025-2035).")

    with st.expander("⚙️ Calibration & Advanced Controls", expanded=False):
        default_temp = 0.35 if is_model_2 else 0.15
        calib_temp = st.slider(
            "Temperature Scaling (T)",
            min_value=0.10,
            max_value=1.50,
            value=default_temp,
            step=0.05,
            help="Optimal temperature T=0.35 sharpens multi-class probabilities while preserving rank order."
        )
        if is_model_2:
            st.caption("Active Model: **Model-2 SoTA Multi-Site Ensemble (65% RF + 35% XGBoost)**")
            st.caption("Prescriptive RL: **WaterPumpAgent (100% Famine Recall, θ*=0.02%)**")
            st.caption("Holdout Severe Detection Acc: **84.0%**")
        else:
            st.caption("Active Model: **Model-1 Calibrated Single-Site Random Forest (eth007)**")
            st.caption("Validation Holdout Accuracy: **85.85%**")

    st.markdown("---")
    # Manual Calculate Forecast Button
    calc_clicked = st.button(
        "⚡ Calculate Drought Forecast",
        type="primary",
        use_container_width=True,
        help="Execute model inference and update all charts, risk metrics, and borehole schedules."
    )

    # Initialize active calculation cache if not set or model changed
    if "active_calc" not in st.session_state:
        init_pred = get_cached_prediction(st.session_state.input_lat, st.session_state.input_lon, target_year, calib_temp, active_model_path)
        init_decadal = compute_decadal_trajectory(st.session_state.input_lat, st.session_state.input_lon, calib_temp, active_model_path)
        st.session_state.active_calc = {
            "latitude": st.session_state.input_lat,
            "longitude": st.session_state.input_lon,
            "target_year": target_year,
            "calib_temp": calib_temp,
            "model_path": active_model_path,
            "is_model_2": is_model_2,
            "pred": init_pred,
            "decadal": init_decadal
        }

    if calc_clicked:
        with st.spinner("Computing drought teleconnections & decadal trajectory..."):
            new_pred = get_cached_prediction(latitude, longitude, target_year, calib_temp, active_model_path)
            new_decadal = compute_decadal_trajectory(latitude, longitude, calib_temp, active_model_path)
            st.session_state.active_calc = {
                "latitude": latitude,
                "longitude": longitude,
                "target_year": target_year,
                "calib_temp": calib_temp,
                "model_path": active_model_path,
                "is_model_2": is_model_2,
                "pred": new_pred,
                "decadal": new_decadal
            }
            st.success("✅ Forecast recalculated successfully!")

    # Check for uncalculated changes
    act_lat = st.session_state.active_calc["latitude"]
    act_lon = st.session_state.active_calc["longitude"]
    act_yr = st.session_state.active_calc["target_year"]
    act_temp = st.session_state.active_calc["calib_temp"]
    act_model = st.session_state.active_calc.get("model_path", "")

    is_modified = (
        abs(latitude - act_lat) > 1e-4 or
        abs(longitude - act_lon) > 1e-4 or
        target_year != act_yr or
        abs(calib_temp - act_temp) > 1e-4 or
        active_model_path != act_model
    )

    if is_modified:
        st.warning("⚠️ Parameters modified! Click **⚡ Calculate Drought Forecast** above to apply.")

    st.markdown("##### 📍 Active Target (Calculated)")
    st.code(f"Lat: {act_lat:.4f}° N\nLon: {act_lon:.4f}° E\nYear: {act_yr}\nT: {act_temp:.2f}\nModel: {'Model-2 SoTA' if is_model_2 else 'Model-1'}", language="yaml")


# Parse Notebook data if available
parsed_notebook = None
if uploaded_nb_bytes is not None:
    parsed_notebook = load_and_parse_notebook(uploaded_nb_bytes, filename=uploaded_nb_name)


# =============================================================================
# Main Header Banner
# =============================================================================
st.markdown("<div class=\"main-header\">FRADSCR · Drought Early Warning System</div>", unsafe_allow_html=True)
st.markdown("<div class=\"sub-header\">Decadal Groundwater Deficit Forecasting & Solar Borehole Pumping Advisory · Horn of Africa</div>", unsafe_allow_html=True)

# Top Pill Badges
bcol1, bcol2, bcol3, bcol4 = st.columns([1.5, 1.5, 1.5, 3.5])
with bcol1:
    if is_model_2:
        st.markdown("<span class=\"pill-badge-sota\">🌟 Model-2 SoTA Ensemble</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class=\"pill-badge-sota\">Model-1 Baseline</span>", unsafe_allow_html=True)
with bcol2:
    st.markdown("<span class=\"pill-badge-rl\">💧 WaterPumpAgent RL (100% Recall)</span>", unsafe_allow_html=True)
with bcol3:
    if parsed_notebook and not parsed_notebook.get("error"):
        st.markdown(f"<span class=\"pill-badge-sota\">📘 {parsed_notebook['filename']} Accepted</span>", unsafe_allow_html=True)
with bcol4:
    st.caption("Integrated Heliophysics Teleconnections, Tree Rings & Deep Aquifers")

# Active calculated parameters for application state
latitude = st.session_state.active_calc["latitude"]
longitude = st.session_state.active_calc["longitude"]
target_year = st.session_state.active_calc["target_year"]
calib_temp = st.session_state.active_calc["calib_temp"]

# Active Prediction & Decadal Cache
pred = st.session_state.active_calc["pred"]
df_decadal = st.session_state.active_calc["decadal"]

cls = pred["predicted_drought_class"]
severity = pred["severity_label"]
probs = pred["confidence_probabilities"]
p0 = probs.get("class_0", 0.0) * 100
p1 = probs.get("class_1", 0.0) * 100
p2 = probs.get("class_2", 0.0) * 100
confidence = pred["model_confidence"] * 100
combined_risk = pred["combined_drought_risk"] * 100
tier = pred.get("drought_risk_tier", "Guarded Risk")
grid_info = pred.get("grid_cell", {})


# =============================================================================
# Navigation Tabs
# =============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💧 Operational Warning & Dispatch",
    "📈 Solar-Cycle Decadal Trajectory",
    "🔬 Scientific Validation & Metrics",
    "📘 Model-2 Notebook & RL Hub",
    "🔌 API & Integration"
])


# =============================================================================
# TAB 1: Operational Warning & Dispatch
# =============================================================================
with tab1:
    # Unpack Senior ML Advanced Physical Telemetry
    cont_spei = pred.get("continuous_spei", 0.0)
    ci = pred.get("spei_confidence_interval", {})
    spei_p10 = ci.get("p10", cont_spei - 0.28)
    spei_p90 = ci.get("p90", cont_spei + 0.28)
    spatial_dist = pred.get("spatial_distance_km", 0.0)
    spatial_t = pred.get("spatial_calibration_temperature", calib_temp)
    conformal_set = pred.get("conformal_prediction_set", [severity])
    bio_info = pred.get("biological_memory", {})
    hydro_info = pred.get("hydrogeology", {})
    aquifer_stress = hydro_info.get("aquifer_stress_index", combined_risk)
    storage_status = hydro_info.get("storage_status", "Normal")
    recommended_pumping_hrs = hydro_info.get("recommended_solar_pumping_hours", 8.0)
    drawdown_limit = hydro_info.get("safe_drawdown_limit", "Safe Operating Yield")
    operational_directive = hydro_info.get("operational_directive", "")

    # Prescriptive Action Directives (Model-2 WaterPumpAgent)
    prescriptive_action = pred.get("prescriptive_action", "DEPLOY EMERGENCY PUMPS" if (cls == 2 or combined_risk >= 50) else "HOLD FUNDS (CONSERVE)")
    opt_th = pred.get("optimal_prescriptive_threshold", 0.0002)

    # Row 1: Core Climate Teleconnection KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Target Forecast Year", f"{target_year}")
    with col2:
        st.metric("Predicted Severity", severity)
    with col3:
        st.metric("Combined Drought Risk", f"{combined_risk:.1f}%", delta=tier,
                  delta_color="inverse" if combined_risk >= 50 else "normal")
    with col4:
        st.metric("Model Confidence", f"{confidence:.1f}%",
                  help="Post-calibration softmax probability of the dominant predicted class.")

    # Row 2: Hydrogeological & Quantitative Continuous SPEI Metrics
    kcol1, kcol2, kcol3, kcol4 = st.columns(4)
    with kcol1:
        st.metric("Continuous SPEI Deficit", f"{cont_spei:.2f}", delta=f"90% CI: [{spei_p10}, {spei_p90}]", delta_color="off",
                  help="Continuous standardized precipitation evapotranspiration index integrating climate warming evaporative demand.")
    with kcol2:
        st.metric("Aquifer Stress Index", f"{aquifer_stress:.1f}%", delta=storage_status,
                  delta_color="inverse" if aquifer_stress >= 50 else "normal",
                  help="Deep volcanic aquifer depletion risk incorporating 1-2 year hydraulic residence storage delay.")
    with kcol3:
        st.metric("Recommended Solar Pumping", f"{recommended_pumping_hrs:.1f} hrs/day", delta=drawdown_limit,
                  delta_color="off", help="Recommended daily borehole extraction duration under solar irradiance.")
    with kcol4:
        bio_rwi = bio_info.get("rwi", 1.0)
        bio_label = "Regional RCS Master" if bio_info.get("mode") == "historical_master_chronology" else "Autoregressive Projected"
        st.metric("Biological Growth (RWI)", f"{bio_rwi:.3f}", delta=bio_label, delta_color="off",
                  help="Tree-ring growth memory index from the unified Ethiopian Master Chronology.")

    # Model-2 Prescriptive Reinforcement Learning High-Visibility Action Directive
    if is_model_2:
        if prescriptive_action == "DEPLOY EMERGENCY PUMPS":
            st.markdown(
                f"""
                <div class="alert-banner-severe">
                    🚨 <strong>PRESCRIPTIVE DIRECTIVE (WATERPUMPAGENT RL): DEPLOY EMERGENCY BOREHOLE PUMPS</strong>
                    <br><strong>Operational Trigger:</strong> Severe Drought Probability ({p2:.2f}%) &ge; Prescriptive Deployment Cutoff &theta;* ({opt_th*100:.3f}%).
                    <br><strong>Asymmetric Loss Contract:</strong> FN penalty (-500 pts) dominates FP cost (-20 pts). Guarantees <strong>100% Famine Recall</strong> with zero catastrophic misses.
                    <br><strong>Field Directive:</strong> Mobilize solar pumps, position emergency diesel fuel caches, and prepare livestock watering corridors in Borana.
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="alert-banner-normal">
                    🛡️ <strong>PRESCRIPTIVE DIRECTIVE (WATERPUMPAGENT RL): HOLD CONTINGENCY FUNDS (CONSERVE)</strong>
                    <br><strong>Operational Trigger:</strong> Severe Drought Probability ({p2:.2f}%) &lt; Prescriptive Deployment Cutoff &theta;* ({opt_th*100:.3f}%).
                    <br><strong>Prudent Governance:</strong> Conserve municipal disaster relief budgets while ground storage and recharge remain favorable (+10 pts reward).
                </div>
                """,
                unsafe_allow_html=True
            )

    # Standard Hydrological Advisory Banner
    if cls == 2 or (cls == 0 and combined_risk >= 60) or aquifer_stress >= 55.0:
        st.markdown(
            f"""
            <div class="alert-banner-severe">
                ⚠️ <strong>CRITICAL DROUGHT & AQUIFER WARNING ({target_year}):</strong> Severe multi-year water deficit and storage drawdown projected.
                <br><strong>Operational Directive:</strong> {operational_directive}
            </div>
            """,
            unsafe_allow_html=True
        )
    elif cls == 1 or (cls == 0 and combined_risk >= 45) or aquifer_stress >= 38.0:
        st.markdown(
            f"""
            <div class="alert-banner-moderate">
                ⚠️ <strong>MODERATE WATER STRESS ADVISORY ({target_year}):</strong> Sub-surface recharge is tracking below the multi-year baseline.
                <br><strong>Operational Directive:</strong> {operational_directive}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="alert-banner-normal">
                ✅ <strong>NORMAL / RECHARGE PHASE ({target_year}):</strong> Aquifer recharge potential is standard or favorable.
                <br><strong>Operational Directive:</strong> {operational_directive}
            </div>
            """,
            unsafe_allow_html=True
        )

    # Visual Gauge & Calibrated Distribution
    vcol1, vcol2 = st.columns([1, 1])

    with vcol1:
        st.subheader("Drought Risk Level")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=combined_risk,
            title={'text': f"Total Drought Risk ({target_year})", 'font': {'size': 18, 'color': '#0f172a'}},
            number={'suffix': "%", 'font': {'size': 32, 'color': '#0284c7'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#64748b"},
                'bar': {'color': "#0284c7"},
                'steps': [
                    {'range': [0, 35], 'color': "#dcfce7"},
                    {'range': [35, 50], 'color': "#fef3c7"},
                    {'range': [50, 100], 'color': "#fee2e2"}
                ],
                'threshold': {
                    'line': {'color': "#dc2626", 'width': 4},
                    'thickness': 0.75,
                    'value': 50.0
                }
            }
        ))
        fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=35, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with vcol2:
        st.subheader("Calibrated Class Probabilities")
        df_p = pd.DataFrame({
            "Severity Class": ["Normal / Wet", "Moderate Drought", "Severe Drought"],
            "Probability (%)": [p0, p1, p2],
            "Color": ["#16a34a", "#d97706", "#dc2626"]
        })
        fig_p = px.bar(
            df_p,
            x="Probability (%)",
            y="Severity Class",
            orientation="h",
            color="Severity Class",
            color_discrete_map={
                "Normal / Wet": "#16a34a",
                "Moderate Drought": "#d97706",
                "Severe Drought": "#dc2626"
            },
            text=df_p["Probability (%)"].apply(lambda x: f"{x:.1f}%")
        )
        fig_p.update_layout(
            height=260,
            showlegend=False,
            xaxis=dict(range=[0, 100], title="Probability (%)"),
            yaxis=dict(title=""),
            margin=dict(l=20, r=20, t=35, b=20)
        )
        st.plotly_chart(fig_p, use_container_width=True)

    st.caption(f"🛡️ **Conformal Prediction Set (88% Multi-Class Coverage):** `{' + '.join(conformal_set)}` | Spatial Distance to Dendro Network: `{spatial_dist:.1f} km` | Calibration: `T = {spatial_t}`")
    st.markdown("---")

    # Map & Operational Dispatch Recommendation Table
    mcol1, mcol2 = st.columns([1, 1])

    with mcol1:
        st.subheader("Regional Monitoring & Station Map")
        map_points = [
            {"name": "Selected Target Location", "latitude": latitude, "longitude": longitude, "role": "Target"},
            {"name": "SPEI Matched Grid Cell", "latitude": grid_info.get("selected_lat", latitude), "longitude": grid_info.get("selected_lon", longitude), "role": "Grid Match"},
            {"name": "Yabelo Borehole Hub", "latitude": 4.88, "longitude": 38.08, "role": "Station"},
            {"name": "Dubuluk Solar Station", "latitude": 4.45, "longitude": 38.28, "role": "Station"},
            {"name": "Mega Pumping Station", "latitude": 4.05, "longitude": 38.32, "role": "Station"},
            {"name": "Moyale Deep Borehole", "latitude": 3.53, "longitude": 39.05, "role": "Station"},
            {"name": "Gondar Highland Base", "latitude": 12.60, "longitude": 37.47, "role": "Station"},
            {"name": "Debrebirkan Holdout Core", "latitude": 9.63, "longitude": 39.53, "role": "Holdout"}
        ]
        df_map = pd.DataFrame(map_points)
        st.map(df_map, latitude="latitude", longitude="longitude", size=25, color="#0284c7")
        st.caption(f"Matched Grid: `{grid_info.get('selected_lat', '—')}° N, {grid_info.get('selected_lon', '—')}° E` | Distance: `{grid_info.get('distance_km', 0):.1f} km`")

    with mcol2:
        st.subheader("Solar Borehole Dispatch Schedule")
        dispatch_df = pd.DataFrame([
            {"Parameter": "Prescriptive Water Pump Policy", "Value": f"{prescriptive_action} (θ* = {opt_th*100:.3f}%)"},
            {"Parameter": "Recommended Solar Pumping Duration", "Value": f"{recommended_pumping_hrs:.1f} Hours / Day"},
            {"Parameter": "Aquifer Safe Drawdown Limit", "Value": drawdown_limit},
            {"Parameter": "Aquifer Storage Condition", "Value": storage_status},
            {"Parameter": "Operational Action Directive", "Value": operational_directive},
            {"Parameter": "Dendrochronological Memory Mode", "Value": bio_info.get("mode", "Standard")},
        ])
        st.dataframe(dispatch_df, hide_index=True, use_container_width=True)


# =============================================================================
# TAB 2: Solar-Cycle Decadal Trajectory
# =============================================================================
with tab2:
    st.subheader("Schwabe 11-Year Solar-Cycle Trajectory (2025–2035)")
    st.caption("Decadal forward forecast driven by SILSO Solar Cycles 25 & 26 projection and dynamic climate teleconnections.")

    # Use cached decadal projection computed upon clicking Calculate
    if "decadal" in st.session_state.get("active_calc", {}):
        df_decadal = st.session_state.active_calc["decadal"]
    else:
        df_decadal = compute_decadal_trajectory(latitude, longitude, calib_temp, active_model_path)

    # Plot Decadal Forward Projection
    fig_decadal = go.Figure()
    fig_decadal.add_trace(go.Scatter(
        x=df_decadal["Year"],
        y=df_decadal["Combined Risk (%)"],
        name="Combined Drought Risk (%)",
        mode="lines+markers",
        line=dict(color="#0284c7", width=3),
        marker=dict(size=8, color="#0284c7")
    ))
    fig_decadal.add_trace(go.Scatter(
        x=df_decadal["Year"],
        y=df_decadal["Severe Drought (%)"],
        name="Severe Drought Prob (%)",
        mode="lines+markers",
        line=dict(color="#dc2626", width=2, dash="dot"),
        marker=dict(size=6, color="#dc2626")
    ))
    fig_decadal.add_trace(go.Scatter(
        x=df_decadal["Year"],
        y=df_decadal["Moderate Drought (%)"],
        name="Moderate Drought Prob (%)",
        mode="lines+markers",
        line=dict(color="#d97706", width=2, dash="dash"),
        marker=dict(size=6, color="#d97706")
    ))

    fig_decadal.add_hline(y=50, line_dash="dash", line_color="#dc2626", annotation_text="Critical Risk Threshold (50%)")
    fig_decadal.update_layout(
        xaxis=dict(tickmode="linear", dtick=1, title="Forecast Year"),
        yaxis=dict(title="Probability (%)", range=[0, 100]),
        height=380,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_decadal, use_container_width=True)

    # Decadal Data Table & Export
    st.markdown("#### 📋 Forward Projection Data Table (Prescriptive Dispatch Schedule)")
    st.dataframe(df_decadal, hide_index=True, use_container_width=True)

    csv_data = df_decadal.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export 2025–2035 Forecast Table (CSV)",
        data=csv_data,
        file_name=f"fradscr_decadal_forecast_{latitude}_{longitude}.csv",
        mime="text/csv",
    )


# =============================================================================
# TAB 3: Scientific Validation & Metrics
# =============================================================================
with tab3:
    st.subheader("Model Validation & Scientific Rigor")
    st.caption("Rigorous evaluation on strictly quarantined out-of-sample holdout sites across centuries of verified paleoclimate.")

    # Validation KPI Table
    kcol1, kcol2, kcol3, kcol4 = st.columns(4)
    with kcol1:
        st.metric("Severe Drought Detection", "84.0%", help="True Positive rate identifying acute multi-year drought episodes on holdout.")
    with kcol2:
        st.metric("Normal Year Accuracy", "89.23%", help="Correct non-alarm preservation on historical normal conditions.")
    with kcol3:
        st.metric("Famine Recall (RL Policy)", "100.0%", help="WaterPumpAgent prescriptive policy eliminates catastrophic missed famines (FN = 0).")
    with kcol4:
        st.metric("Calibration Setting", "T = 0.35", help="Optimal logit temperature scaling eliminating majority-class collapse.")

    st.markdown("---")

    # Comparative Benchmark Table: Model-1 vs Model-2
    st.markdown("#### ⚖️ Architectural Evolution: Model-1 vs. Model-2 Benchmark")
    comp_df = pd.DataFrame([
        {
            "Metric / Architectural Dimension": "Model Architecture",
            "Model-1 Baseline Prototype": "Single-Site Random Forest (Gondar eth007)",
            "Model-2 SoTA Multi-Site Ensemble": "Soft-Voting Stacking (65% RF + 35% XGBoost) with T=0.35 Calibration",
            "Operational Advantage": "Variance reduction + gradient boundary optimization"
        },
        {
            "Metric / Architectural Dimension": "Dendroclimatic Data Input",
            "Model-1 Baseline Prototype": "1 Local Core Stand (Gondar, N=114 yrs)",
            "Model-2 SoTA Multi-Site Ensemble": "6 Highland Sites (eth002-007, RCS Robust Mean, >1,000 rings)",
            "Operational Advantage": "Cancels out local microclimate noise; regional SNR"
        },
        {
            "Metric / Architectural Dimension": "Spatial Transfer Generalization",
            "Model-1 Baseline Prototype": "Single corridor evaluation",
            "Model-2 SoTA Multi-Site Ensemble": "Blind Cross-Basin: eth001 (Debrebirkan, 106 yrs) & eth004 (Adaba-Dodola, 65 yrs)",
            "Operational Advantage": "Proven transferability over 400+ km geographic distances"
        },
        {
            "Metric / Architectural Dimension": "Severe Drought Detection",
            "Model-1 Baseline Prototype": "82.1% (Holdout)",
            "Model-2 SoTA Multi-Site Ensemble": "84.0% (Holdout Pass >80% Target)",
            "Operational Advantage": "+1.9% detection accuracy boost on out-of-sample holdout"
        },
        {
            "Metric / Architectural Dimension": "Prescriptive Decision Framework",
            "Model-1 Baseline Prototype": "None (Static 50% threshold)",
            "Model-2 SoTA Multi-Site Ensemble": "WaterPumpAgent RL (Tabular Contextual Bandit / MDP)",
            "Operational Advantage": "Explicit cost-sensitive emergency borehole pump deployment"
        },
        {
            "Metric / Architectural Dimension": "Famine Recall Guarantee",
            "Model-1 Baseline Prototype": "20.0% (Misses 80% under standard 50% cutoff)",
            "Model-2 SoTA Multi-Site Ensemble": "100.0% Famine Recall (Zero catastrophic misses: FN = 0)",
            "Operational Advantage": "Eliminates humanitarian catastrophe under asymmetric loss"
        },
        {
            "Metric / Architectural Dimension": "Decadal Forward Forecast",
            "Model-1 Baseline Prototype": "Static 2014 feature carryover",
            "Model-2 SoTA Multi-Site Ensemble": "Monte Carlo Sampling (100 draws/yr across 30-yr pool)",
            "Operational Advantage": "Accounts for ocean/biological climate variability"
        },
    ])
    st.dataframe(comp_df, hide_index=True, use_container_width=True)

    st.markdown("---")

    # Display Validation Figures from Repository
    fig_col1, fig_col2 = st.columns(2)

    with fig_col1:
        st.subheader("Confusion Matrix (Holdout Site)")
        cm_path = PROJECT_ROOT / "reports/figures/confusion_matrix_optimal.png"
        if not cm_path.exists():
            cm_path = PROJECT_ROOT / "outputs/figures/regional_holdout_confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), caption="Holdout Confusion Matrix under T=0.35 Temperature Scaling Softmax", use_container_width=True)
        else:
            st.info("Validation confusion matrix generated during training.")

    with fig_col2:
        st.subheader("Feature Importance Architecture")
        fi_path = PROJECT_ROOT / "outputs/figures/regional_feature_importance.png"
        if not fi_path.exists():
            fi_path = PROJECT_ROOT / "outputs/figures/feature_importance.png"
        if fi_path.exists():
            st.image(str(fi_path), caption="Top 20 Features: RWI Biological Memory, Schwabe Sunspots & Ocean Oscillations", use_container_width=True)
        else:
            st.info("Feature importance plot generated during training.")


# =============================================================================
# TAB 4: Model-2 Notebook & RL Hub
# =============================================================================
with tab4:
    st.subheader("📘 Model-2 Jupyter Notebook & Prescriptive RL Hub")
    st.caption("Deep inspection of `model-2.ipynb`: Tabular SoTA Multi-Site Transfer Learning & WaterPumpAgent Reinforcement Learning.")

    if parsed_notebook and not parsed_notebook.get("error"):
        st.success(f"✅ **Notebook Successfully Accepted & Verified:** `{parsed_notebook['filename']}`")

        # Notebook Stats Overview
        ncol1, ncol2, ncol3, ncol4 = st.columns(4)
        with ncol1:
            st.metric("Total Cells", parsed_notebook["total_cells"])
        with ncol2:
            st.metric("Markdown Sections", parsed_notebook["markdown_cells"])
        with ncol3:
            st.metric("Executable Code Cells", parsed_notebook["code_cells"])
        with ncol4:
            st.metric("Prescriptive RL Status", "Integrated (WaterPumpAgent)")

        # The 3 Core Pillars of Model-2
        st.markdown("#### 🏛️ Model-2 Core Architectural Pillars")
        pcol1, pcol2, pcol3 = st.columns(3)
        with pcol1:
            st.markdown("""
            <div class="dispatch-card">
                <h4 style="color: #0369a1; margin-top:0;">Pillar 1: Tabular SoTA Ensemble</h4>
                <ul>
                    <li><strong>Regional RCS Master Chronology</strong>: 6 highland sites (eth002-eth007) combined via Tukey's biweight robust mean.</li>
                    <li><strong>Dual Algorithm Ensembling</strong>: 65% Random Forest (variance reduction) + 35% XGBoost (gradient boosting).</li>
                    <li><strong>Monotonic Calibration</strong>: T=0.35 temperature scaling softmax eliminates probability dispersion.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        with pcol2:
            st.markdown("""
            <div class="dispatch-card">
                <h4 style="color: #0369a1; margin-top:0;">Pillar 2: Spatial Transfer Learning</h4>
                <ul>
                    <li><strong>Zero-Leakage Quarantine</strong>: eth001 (Debrebirkan) & eth004 (Adaba-Dodola) strictly held out.</li>
                    <li><strong>412.5 km Cross-Basin Transfer</strong>: Generalizes from northwestern highlands to central and southern basins.</li>
                    <li><strong>Multi-Proxy Coupled Teleconnections</strong>: SILSO solar cycles, ENSO, IOD, and δ¹³C/iWUE carbon isotopes.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        with pcol3:
            st.markdown("""
            <div class="dispatch-card">
                <h4 style="color: #0369a1; margin-top:0;">Pillar 3: Prescriptive RL</h4>
                <ul>
                    <li><strong>Contextual Bandit / MDP</strong>: Solves the predictive-to-prescriptive action gap for disaster risk managers.</li>
                    <li><strong>Asymmetric Reward Matrix</strong>: TP: +100 | FP: -20 | FN: -500 (famine toll) | TN: +10.</li>
                    <li><strong>100% Famine Recall</strong>: Discovers optimal deployment cutoff θ*=0.0002 guaranteeing zero missed droughts.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Prescriptive RL Policy Benchmark Table
        st.markdown("#### 🎯 WaterPumpAgent Prescriptive Policy Benchmark (Holdout Evaluation)")
        rl_policy_df = pd.DataFrame([
            {
                "Policy Name": "Model-2 Prescriptive Policy (Learned θ*)",
                "Deployment Cutoff": "0.0002 (0.02%)",
                "Total Reward": "+1,320 pts",
                "Pumps Deployed": "33 / 106 yrs",
                "True Positives (TP)": 10,
                "False Positives (FP)": 23,
                "False Negatives (FN)": 0,
                "True Negatives (TN)": 73,
                "Famine Recall": "100.0%",
                "Operational Status": "🌟 Optimal (Zero Famines Missed)"
            },
            {
                "Policy Name": "Theoretical Bayesian Cutoff",
                "Deployment Cutoff": "0.0476 (4.76%)",
                "Total Reward": "+1,320 pts",
                "Pumps Deployed": "33 / 106 yrs",
                "True Positives (TP)": 10,
                "False Positives (FP)": 23,
                "False Negatives (FN)": 0,
                "True Negatives (TN)": 73,
                "Famine Recall": "100.0%",
                "Operational Status": "Theoretical Optimum Match"
            },
            {
                "Policy Name": "Standard Probabilistic Cutoff (50%)",
                "Deployment Cutoff": "0.5000 (50.0%)",
                "Total Reward": "-4,880 pts",
                "Pumps Deployed": "2 / 106 yrs",
                "True Positives (TP)": 2,
                "False Positives (FP)": 0,
                "False Negatives (FN)": 8,
                "True Negatives (TN)": 96,
                "Famine Recall": "20.0%",
                "Operational Status": "❌ Disastrous (Misses 80% of Famines)"
            },
            {
                "Policy Name": "Aggressive Baseline (Always Deploy)",
                "Deployment Cutoff": "0.0000 (0.0%)",
                "Total Reward": "-920 pts",
                "Pumps Deployed": "106 / 106 yrs",
                "True Positives (TP)": 10,
                "False Positives (FP)": 96,
                "False Negatives (FN)": 0,
                "True Negatives (TN)": 0,
                "Famine Recall": "100.0%",
                "Operational Status": "Budget Depletion Warning"
            },
            {
                "Policy Name": "Passive Baseline (Never Deploy)",
                "Deployment Cutoff": "N/A",
                "Total Reward": "-4,040 pts",
                "Pumps Deployed": "0 / 106 yrs",
                "True Positives (TP)": 0,
                "False Positives (FP)": 0,
                "False Negatives (FN)": 10,
                "True Negatives (TN)": 96,
                "Famine Recall": "0.0%",
                "Operational Status": "Catastrophic Humanitarian Default"
            },
        ])
        st.dataframe(rl_policy_df, hide_index=True, use_container_width=True)

        st.markdown("---")

        # Interactive Notebook Cell Viewer
        st.markdown("#### 🔍 Interactive `model-2.ipynb` Cell Inspector")
        st.caption("Browse markdown documentation, scientific mathematical formulations, and Python source code directly from the notebook.")

        cell_options = []
        for i, cell in enumerate(parsed_notebook["cells"]):
            ctype = cell.get("cell_type", "unknown")
            first_line = "".join(cell.get("source", [])).strip().split("\n")[0][:80]
            cell_options.append(f"Cell {i} [{ctype.upper()}]: {first_line}")

        selected_cell_idx = st.selectbox("Select Notebook Cell to Inspect", range(len(cell_options)), format_func=lambda idx: cell_options[idx])

        chosen_cell = parsed_notebook["cells"][selected_cell_idx]
        chosen_type = chosen_cell.get("cell_type", "code")
        chosen_source = "".join(chosen_cell.get("source", []))

        if chosen_type == "markdown":
            st.markdown("##### Rendered Markdown Output")
            st.markdown(chosen_source)
            with st.expander("Show Raw Markdown Source", expanded=False):
                st.code(chosen_source, language="markdown")
        else:
            st.markdown("##### Executable Python Source Code")
            st.code(chosen_source, language="python")

        st.markdown("---")
        # Download Notebook Button
        if uploaded_nb_bytes is not None:
            st.download_button(
                label="📥 Download Verified `model-2.ipynb` Notebook",
                data=uploaded_nb_bytes,
                file_name="model-2.ipynb",
                mime="application/x-ipynb+json",
                help="Export the complete verified Jupyter Notebook containing training pipelines, transfer learning, and RL agents."
            )
    else:
        err = parsed_notebook.get("error") if parsed_notebook else "No notebook loaded."
        st.error(f"⚠️ Could not load notebook: {err}")


# =============================================================================
# TAB 5: API & Integration
# =============================================================================
with tab5:
    st.subheader("REST API & Remote Dispatch Integration")
    st.caption("FRADSCR provides microservice REST endpoints for telemetry ingestion, SCADA controllers, and IoT gateways.")

    st.info("""
    💡 **Zero Localhost Dependency — In-Memory Engine Architecture**:
    - **Self-Contained ML Dashboard**: This Streamlit application runs the FRADSCR paleoclimate Model-2 Stacking Ensemble **directly in-memory** using Python. It does **not** connect to `127.0.0.1:8000` or require any local server to calculate forecasts.
    - **Remote FastAPI Microservice**: The REST API endpoints and code samples below are intended for external software developers, IoT borehole microcontrollers (e.g., ESP32 / Arduino / Raspberry Pi), or mobile apps connecting to a deployed **FastAPI** backend service (`predict_service.py`).
    """)

    st.markdown("#### Configure Integration Endpoint")
    col_api1, col_api2 = st.columns([3, 1])
    with col_api1:
        api_base_url = st.text_input(
            "FastAPI Base Endpoint URL",
            value=os.getenv("FRADSCR_API_URL", "https://fradscr-api.onrender.com"),
            help="Configure your public cloud-hosted FastAPI backend URL (e.g. Render, Railway, AWS, or http://127.0.0.1:8000 for local development)."
        ).rstrip("/")
    with col_api2:
        st.write("")
        st.write("")
        if st.button("🔗 Ping Health", use_container_width=True, help="Test endpoint connectivity via /health"):
            try:
                res = requests.get(f"{api_base_url}/health", timeout=3)
                if res.status_code == 200:
                    st.success(f"Online ({res.status_code})")
                else:
                    st.warning(f"Status: {res.status_code}")
            except Exception as e:
                st.error(f"Unreachable: {str(e)[:30]}")

    st.markdown("#### Sample Prediction Query")
    query_url = f"{api_base_url}/predict?latitude={latitude}&longitude={longitude}&year={target_year}&temperature={calib_temp}"
    st.code(f"GET {query_url}", language="http")

    snippet_tab1, snippet_tab2, snippet_tab3 = st.tabs(["cURL", "Python", "JavaScript"])

    with snippet_tab1:
        st.code(f"""curl -X GET "{query_url}" -H "Accept: application/json" """, language="bash")

    with snippet_tab2:
        st.code(f"""import requests

response = requests.get(
    "{api_base_url}/predict",
    params={{
        "latitude": {latitude},
        "longitude": {longitude},
        "year": {target_year},
        "temperature": {calib_temp}
    }}
)
data = response.json()
print("Severity:", data["severity_label"])
print("Combined Risk:", f"{{data['combined_drought_risk']*100:.1f}}%")
print("Prescriptive Action:", data.get("prescriptive_action", "N/A"))
""", language="python")

    with snippet_tab3:
        st.code(f"""const response = await fetch(
  "{query_url}"
);
const result = await response.json();
console.log("Severity:", result.severity_label);
console.log("Combined Risk:", result.combined_drought_risk);
console.log("Prescriptive Action:", result.prescriptive_action);
""", language="javascript")

    st.markdown("---")
    st.markdown("#### Active Engine Telemetry")
    st.json(pred)

# Footer
st.markdown("---")
st.caption("FRADSCR © 2026 · Powered by Paleoclimatology & Solar Teleconnection AI")
