"""
FRADSCR — Drought Early Warning & Solar Borehole Advisory System
=================================================================
Master Application Entrypoint & Dual-Model Hub.

This launcher hosts two complete, standalone scientific applications:
1. Model-2 SoTA Ensemble & Prescriptive RL (app_model_2.py / model-2.ipynb)
2. Model-1 Baseline Prototype Stand (app_model_1.py / model-1.ipynb)

Both applications can also be launched directly and independently:
    streamlit run app_model_2.py
    streamlit run app_model_1.py
"""

from pathlib import Path
import sys
import runpy
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Backward-compatible exports for tests and external scripts
from app_model_2 import (
    compute_decadal_trajectory,
    get_cached_prediction,
    load_and_parse_notebook,
)

# Master Page Config (must be called first)
st.set_page_config(
    page_title="FRADSCR · Drought Early Warning System",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar Top-Level App Switcher
with st.sidebar:
    st.markdown("## 🧭 Application Hub")
    app_choice = st.radio(
        "Active Application Architecture",
        [
            "🌟 Model-2: SoTA Regional Ensemble & Prescriptive RL (model-2.ipynb)",
            "🌲 Model-1: Single-Site Gondar Baseline Prototype (model-1.ipynb)",
        ],
        index=0,
        help="Select which scientific model application to inspect and run."
    )
    st.markdown("---")

# Route to the selected standalone application
if "Model-2" in app_choice:
    app_target = PROJECT_ROOT / "app_model_2.py"
else:
    app_target = PROJECT_ROOT / "app_model_1.py"

# Run the selected application module
runpy.run_path(str(app_target), run_name="__main__")
