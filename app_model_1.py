"""
FRADSCR — Model-1 Baseline Prototype Application (model-1.ipynb)
================================================================
Dedicated standalone Streamlit dashboard for the Model-1 paleoclimate
baseline prototype.

Covers:
- Systematic candidate tree-ring selection (eth001 to eth007) & quality audit
- Selected Stand: ETH007 (Gondar Highland Stand, Juniperus procera, N=114 yrs)
- Transparent 80/20 chronological train-test split (1901-1991 vs 1992-2014)
- Single-Stand Calibrated Random Forest inference (random_forest_eth007.joblib)
- Baseline limitations analysis that motivated the Model-2 SoTA ensemble
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

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from predict_service import (
    get_engine,
    predict_drought,
    SEVERITY_LABELS,
    DEFAULT_ETH007_MODEL_PATH,
)

# Page configuration
try:
    st.set_page_config(
        page_title="FRADSCR · Model-1 Baseline Prototype",
        page_icon="🌲",
        layout="wide",
        initial_sidebar_state="expanded"
    )
except Exception:
    pass

# Custom Styling
st.markdown("""
<style>
    .m1-header {
        font-size: 2.15rem;
        font-weight: 800;
        color: #1e3a5f;
        margin-bottom: 0.1rem;
    }
    .m1-sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .pill-badge-m1 {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 700;
        background-color: #f1f5f9;
        color: #334155;
        border: 1px solid #cbd5e1;
    }
    .pill-badge-selected {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 700;
        background-color: #dcfce7;
        color: #166534;
        border: 1px solid #bbf7d0;
    }
    .dispatch-card-m1 {
        background: #f8fafc;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)


# Load Model-1 Metadata
def _load_model1_metadata() -> Dict[str, Any]:
    meta_path = PROJECT_ROOT / "models" / "model_1_metadata.json"
    if not meta_path.exists():
        meta_path = PROJECT_ROOT / "models" / "eth007_model_metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "model_name": "Random Forest Model-1",
        "selected_dataset": "ETH007 (Gondar)",
        "training_period": "1901-1991",
        "testing_period": "1992-2014",
        "n_total_observations": 114,
        "training_observations": 91,
        "testing_observations": 23,
    }

MODEL1_META = _load_model1_metadata()


# Candidate tree-ring datasets catalog (from model-1.ipynb)
CANDIDATES = [
    {"id": "ETH007", "site": "Gondar Highland Stand", "lat": 13.01, "lon": 37.80, "elev": 2471, "species": "Juniperus procera", "span": "1901–2014 (114 yrs)", "overlap": "114 yrs", "cores": 24, "missing": 0, "quality": 0.963, "status": "✅ SELECTED"},
    {"id": "ETH002", "site": "Simien Mountains", "lat": 13.25, "lon": 38.00, "elev": 3100, "species": "Erica arborea", "span": "1849–2014 (166 yrs)", "overlap": "114 yrs", "cores": 20, "missing": 0, "quality": 0.910, "status": "Alternative Stand"},
    {"id": "ETH003", "site": "Guna Mountain Stand", "lat": 11.71, "lon": 38.24, "elev": 3400, "species": "Juniperus procera", "span": "1780–2014 (235 yrs)", "overlap": "114 yrs", "cores": 18, "missing": 1, "quality": 0.885, "status": "Alternative Stand"},
    {"id": "ETH005", "site": "Choke Mountains", "lat": 10.70, "lon": 37.85, "elev": 3300, "species": "Hagenia abyssinica", "span": "1880–2014 (135 yrs)", "overlap": "114 yrs", "cores": 16, "missing": 0, "quality": 0.870, "status": "Alternative Stand"},
    {"id": "ETH006", "site": "Wof-Washa Forest", "lat": 9.78, "lon": 39.75, "elev": 2800, "species": "Juniperus procera", "span": "1750–2014 (265 yrs)", "overlap": "114 yrs", "cores": 22, "missing": 2, "quality": 0.860, "status": "Alternative Stand"},
    {"id": "ETH001", "site": "Debrebirkan Selassie", "lat": 9.68, "lon": 39.53, "elev": 2750, "species": "Juniperus procera", "span": "1901–2006 (106 yrs)", "overlap": "106 yrs", "cores": 19, "missing": 0, "quality": 0.845, "status": "Quarantined Holdout"},
    {"id": "ETH004", "site": "Adaba-Dodola (Bale)", "lat": 6.85, "lon": 39.20, "elev": 2600, "species": "Podocarpus falcatus", "span": "1901–2003 (103 yrs)", "overlap": "103 yrs", "cores": 15, "missing": 0, "quality": 0.810, "status": "Quarantined Holdout"},
]

PRESETS = {
    "Gondar Highland Stand Base (13.01° N, 37.80° E)": (13.01, 37.80),
    "Borana — Yabelo Station (4.88° N, 38.08° E)": (4.88, 38.08),
    "Debrebirkan Selassie (9.68° N, 39.53° E)": (9.68, 39.53),
    "Custom Coordinates": None,
}

# Sidebar
with st.sidebar:
    st.markdown("### 🌲 Model-1 Controls")
    st.caption("Baseline Prototype · Single-Site Gondar Stand (`eth007`)")

    st.markdown("---")
    st.markdown("#### 📍 Location")
    chosen_preset = st.selectbox("Preset Location", list(PRESETS.keys()), index=0)
    if PRESETS[chosen_preset] is not None:
        init_lat, init_lon = PRESETS[chosen_preset]
    else:
        init_lat, init_lon = 13.01, 37.80

    latitude = st.number_input("Latitude (°N)", value=init_lat, min_value=-90.0, max_value=90.0, step=0.01)
    longitude = st.number_input("Longitude (°E)", value=init_lon, min_value=-180.0, max_value=180.0, step=0.01)

    target_year = st.slider("Forecast Year", min_value=1901, max_value=2035, value=2024, step=1,
                            help="1901-1991: Training set | 1992-2014: Holdout test set | 2015+: Forward prospective")

    calib_temp = st.slider("Softmax Temperature (T)", min_value=0.05, max_value=1.5, value=0.15, step=0.05,
                           help="Model-1 uses T=0.15 for single-site probability sharpening.")

    st.markdown("---")
    st.markdown("#### ⚖️ Architectural Details")
    st.caption("• **Model Type**: Random Forest Classifier (350 trees, max_depth=7)")
    st.caption("• **Chronology**: Single-stand Gondar `eth007.rwl` (N=114 yrs)")
    st.caption("• **Split**: 80% Train (1901-1991) / 20% Test (1992-2014)")
    st.caption("• **Threshold**: Static 50% cutoff (No Reinforcement Learning)")

    st.markdown("---")
    st.info("💡 **Looking for the SoTA Model?**\nModel-2 with 6-site RCS master chronology and Prescriptive RL is in `app_model_2.py`.")


# Compute prediction
@st.cache_data(show_spinner=False)
def get_m1_pred(lat: float, lon: float, yr: int, temp: float):
    return predict_drought(
        latitude=lat, longitude=lon, year=yr, temperature=temp,
        model_path=DEFAULT_ETH007_MODEL_PATH
    )

pred = get_m1_pred(latitude, longitude, target_year, calib_temp)
cls = pred["predicted_drought_class"]
severity = pred["severity_label"]
probs = pred["confidence_probabilities"]
p0, p1, p2 = probs.get("class_0", 0)*100, probs.get("class_1", 0)*100, probs.get("class_2", 0)*100
conf = pred["model_confidence"] * 100
combined_risk = pred["combined_drought_risk"] * 100

# Main Header
st.markdown("<div class=\"m1-header\">FRADSCR · Model-1 Baseline Prototype</div>", unsafe_allow_html=True)
st.markdown("<div class=\"m1-sub-header\">Single-Stand Gondar Paleoclimate Reconstruction (<code>model-1.ipynb</code>) · Random Forest Baseline</div>", unsafe_allow_html=True)

# Pill badges
b1, b2, b3, b4 = st.columns([1.8, 1.8, 1.8, 3.0])
with b1:
    st.markdown("<span class=\"pill-badge-selected\">🌲 ETH007 Gondar Stand</span>", unsafe_allow_html=True)
with b2:
    st.markdown("<span class=\"pill-badge-m1\">📊 80/20 Chronological Split</span>", unsafe_allow_html=True)
with b3:
    st.markdown("<span class=\"pill-badge-m1\">⚡ 350-Tree Bagging</span>", unsafe_allow_html=True)
with b4:
    st.caption("Baseline Prototype: Single-Site Dendroclimatic Reconstruction")

st.markdown("---")

# Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌲 Candidate Selection & Quality Audit",
    "🧪 Data Processing & 80/20 Split",
    "💧 Operational Drought Warning",
    "🔬 Scientific Evaluation & Limitations",
    "📘 Model-1 Notebook Inspector",
])

# ── TAB 1: Candidate Selection ───────────────────────────────────────────────
with tab1:
    st.subheader("Systematic Candidate Stand Selection & Quality Audit")
    st.caption("Audit of all 7 African ITRDB tree-ring chronologies across Ethiopia to determine the optimal stand for Model-1.")

    df_cand = pd.DataFrame(CANDIDATES)
    st.dataframe(df_cand, hide_index=True, use_container_width=True)

    st.markdown("#### 🏆 Why Was ETH007 Selected as the Primary Stand?")
    qcol1, qcol2, qcol3 = st.columns(3)
    with qcol1:
        st.markdown("""
        <div class="dispatch-card-m1">
            <h4 style="color:#1e3a5f; margin-top:0;">1. Maximum Modern Overlap</h4>
            <p><strong>114 continuous years (1901–2014)</strong> of complete temporal overlap with SILSO solar cycles and instrumental SPEI ground truth. Zero missing rings.</p>
        </div>
        """, unsafe_allow_html=True)
    with qcol2:
        st.markdown("""
        <div class="dispatch-card-m1">
            <h4 style="color:#1e3a5f; margin-top:0;">2. High Signal-to-Noise Ratio</h4>
            <p><strong>24 individual tree cores</strong> from <em>Juniperus procera</em> at 2,471m elevation. High inter-series correlation (r &gt; 0.65) and strong common moisture sensitivity.</p>
        </div>
        """, unsafe_allow_html=True)
    with qcol3:
        st.markdown("""
        <div class="dispatch-card-m1">
            <h4 style="color:#1e3a5f; margin-top:0;">3. High Quality Score (0.963)</h4>
            <p>Weighted composite quality score: 40% modern overlap, 30% total span, 20% replication count, and 10% record continuity.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Geographic Distribution of Candidate Tree-Ring Stands")
    map_cand = pd.DataFrame([
        {"name": f"{c['id']} — {c['site']} ({c['status']})", "latitude": c["lat"], "longitude": c["lon"]}
        for c in CANDIDATES
    ])
    st.map(map_cand, latitude="latitude", longitude="longitude", size=25, color="#1e3a5f")


# ── TAB 2: Data Processing & 80/20 Split ─────────────────────────────────────
with tab2:
    st.subheader("Data Processing Pipeline & 80/20 Chronological Split")
    st.caption("Transparent review of the preprocessing, feature engineering, and train/test quarantine.")

    st.info("""
    **Model-1 Data Pipeline Flow:**
    1. `eth007.rwl` raw core measurements → cubic smoothing spline detrending → standardized RWI
    2. SILSO international sunspot records (SN_y_tot_V2.0.csv) → 11-yr Schwabe moving averages & lags
    3. Regional SPEI ground truth from CRU/SPEIbase NetCDF at Gondar (13.01°N, 37.80°E)
    4. 20 multi-proxy indicators constructed across 114 continuous annual records (1901–2014)
    5. **Strict Chronological Quarantine**: 1901–1991 (Train, N=91) | 1992–2014 (Test, N=23)
    """)

    # Train-test split visualization
    st.markdown("#### Chronological 80/20 Train vs. Test Split Timeline")
    years_all = np.arange(1901, 2015)
    train_mask = years_all <= 1991
    test_mask = years_all > 1991

    fig_split = go.Figure()
    fig_split.add_trace(go.Bar(
        x=years_all[train_mask], y=[1]*np.sum(train_mask),
        name="Training Set: 1901–1991 (N=91 yrs, 79.8%)",
        marker_color="#1e3a5f", opacity=0.85
    ))
    fig_split.add_trace(go.Bar(
        x=years_all[test_mask], y=[1]*np.sum(test_mask),
        name="Holdout Test Set: 1992–2014 (N=23 yrs, 20.2%)",
        marker_color="#d97706", opacity=0.85
    ))
    fig_split.add_vline(x=1991.5, line_dash="dash", line_color="#dc2626", annotation_text="Chronological Quarantine Cutoff (1991)")
    fig_split.update_layout(
        barmode="stack", height=240,
        xaxis=dict(title="Calendar Year (CE)", tickmode="linear", dtick=10),
        yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_split, use_container_width=True)

    scol1, scol2 = st.columns(2)
    with scol1:
        st.markdown("##### Training Set Class Balance (1901–1991, N=91)")
        train_cls_df = pd.DataFrame({
            "Class": ["Normal (0)", "Moderate (1)", "Severe (2)"],
            "Years": [56, 23, 12],
            "Pct": ["61.5%", "25.3%", "13.2%"]
        })
        st.dataframe(train_cls_df, hide_index=True, use_container_width=True)
    with scol2:
        st.markdown("##### Holdout Test Set Class Balance (1992–2014, N=23)")
        test_cls_df = pd.DataFrame({
            "Class": ["Normal (0)", "Moderate (1)", "Severe (2)"],
            "Years": [14, 5, 4],
            "Pct": ["60.9%", "21.7%", "17.4%"]
        })
        st.dataframe(test_cls_df, hide_index=True, use_container_width=True)


# ── TAB 3: Operational Drought Warning ───────────────────────────────────────
with tab3:
    st.subheader(f"Model-1 Baseline Drought Inference ({target_year})")
    st.caption("Live prediction generated by the Single-Stand Gondar Random Forest baseline.")

    # Status period indicator
    if target_year <= 1991:
        period_badge = "📚 Training Period (1901–1991)"
    elif target_year <= 2014:
        period_badge = "🧪 Holdout Test Period (1992–2014)"
    else:
        period_badge = "🔭 Forward Prospective Projection (2015–2035)"
    st.caption(f"Status: **{period_badge}**")

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Forecast Year", f"{target_year}")
    with k2:
        st.metric("Predicted Severity", severity)
    with k3:
        st.metric("Combined Drought Risk", f"{combined_risk:.1f}%")
    with k4:
        st.metric("Model Confidence", f"{conf:.1f}%", help="Softmax probability under T=0.15")

    # Gauges
    gcol1, gcol2 = st.columns(2)
    with gcol1:
        st.subheader("Drought Risk Level")
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=combined_risk,
            number={'suffix': "%", 'font': {'size': 30, 'color': '#1e3a5f'}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': '#1e3a5f'},
                'steps': [
                    {'range': [0, 35], 'color': '#dcfce7'},
                    {'range': [35, 50], 'color': '#fef3c7'},
                    {'range': [50, 100], 'color': '#fee2e2'},
                ]
            }
        ))
        fig_g.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_g, use_container_width=True)

    with gcol2:
        st.subheader("Calibrated Class Probabilities")
        fig_p = px.bar(
            pd.DataFrame({
                "Class": ["Normal / Wet", "Moderate Drought", "Severe Drought"],
                "Probability (%)": [p0, p1, p2],
            }),
            x="Probability (%)", y="Class", orientation="h",
            color="Class",
            color_discrete_map={"Normal / Wet": "#16a34a", "Moderate Drought": "#d97706", "Severe Drought": "#dc2626"},
            text=[f"{p0:.1f}%", f"{p1:.1f}%", f"{p2:.1f}%"]
        )
        fig_p.update_layout(height=260, showlegend=False, margin=dict(l=20, r=20, t=30, b=20), xaxis=dict(range=[0, 100]))
        st.plotly_chart(fig_p, use_container_width=True)

    # Dispatch Banner (Standard 50% Threshold)
    if p2 >= 50.0 or combined_risk >= 50.0:
        st.error(f"🚨 **MODEL-1 ALERT ({target_year}):** Severe Drought Detected (Prob: {p2:.1f}% ≥ 50% cutoff). Deploy emergency relief.")
    elif p1 >= 50.0 or combined_risk >= 35.0:
        st.warning(f"⚠️ **MODEL-1 ADVISORY ({target_year}):** Moderate Moisture Deficit. Monitor ground storage.")
    else:
        st.success(f"✅ **MODEL-1 NORMAL ({target_year}):** Standard moisture conditions projected.")


# ── TAB 4: Scientific Evaluation & Limitations ───────────────────────────────
with tab4:
    st.subheader("Scientific Evaluation & Baseline Limitations")
    st.caption("How Model-1 performed on the 1992–2014 holdout test set, and what weaknesses led to Model-2.")

    # Validation KPIs
    vk1, vk2, vk3, vk4 = st.columns(4)
    with vk1:
        st.metric("OOB Score (Train)", "62.6%", help="Out-Of-Bag accuracy on 1901–1991 training set")
    with vk2:
        st.metric("Holdout Detection Acc", "85.85%", help="Severe drought detection under balanced recalibration")
    with vk3:
        st.metric("Test Macro F1", "0.489", help="Macro-averaged F1 score on test set")
    with vk4:
        st.metric("Tree Count", "350 Trees", help="Random forest ensemble estimators")

    st.markdown("---")

    # The 3 Fundamental Limitations of Model-1
    st.markdown("#### ⚠️ Why Model-1 Was Not Sufficient for Production (The Need for Model-2)")
    lcol1, lcol2, lcol3 = st.columns(3)
    with lcol1:
        st.markdown("""
        <div class="dispatch-card-m1" style="border-left: 5px solid #dc2626;">
            <h4 style="color:#dc2626; margin-top:0;">Limitation 1: Stand Microclimate Bias</h4>
            <p>Model-1 relies on a <strong>single tree stand (Gondar)</strong>. Local hill-slope exposure, localized rainfall anomalies, and tree-level competition introduce non-climatic noise that does not generalize across Ethiopia.</p>
            <p><em>👉 Solved in Model-2 by aggregating 6 regional stands into the RCS Master Chronology.</em></p>
        </div>
        """, unsafe_allow_html=True)
    with lcol2:
        st.markdown("""
        <div class="dispatch-card-m1" style="border-left: 5px solid #d97706;">
            <h4 style="color:#d97706; margin-top:0;">Limitation 2: Majority-Class Collapse</h4>
            <p>Because Normal years represent &gt;60% of history, standard training creates an algorithmic bias toward predicting "Normal", causing the model to miss acute droughts during validation.</p>
            <p><em>👉 Solved in Model-2 with dual RF+XGBoost ensembling and T=0.35 temperature calibration.</em></p>
        </div>
        """, unsafe_allow_html=True)
    with lcol3:
        st.markdown("""
        <div class="dispatch-card-m1" style="border-left: 5px solid #0284c7;">
            <h4 style="color:#0284c7; margin-top:0;">Limitation 3: The Prescriptive Action Gap</h4>
            <p>Model-1 operates on a static 50% probability threshold. Under asymmetric disaster loss (where a missed famine is 25× worse than a false alarm), it <strong>misses 80% of famines</strong> (Famine Recall = 20%).</p>
            <p><em>👉 Solved in Model-2 with WaterPumpAgent RL (100% Famine Recall Guarantee).</em></p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Images if present
    f1, f2 = st.columns(2)
    with f1:
        st.subheader("Model-1 Confusion Matrix")
        cm_p = PROJECT_ROOT / "reports/figures/confusion_matrix_optimal.png"
        if cm_p.exists():
            st.image(str(cm_p), caption="Model-1 Holdout Confusion Matrix (Optimal Balanced Thresholds)", use_container_width=True)
        else:
            st.info("Confusion matrix available in reports/figures/")
    with f2:
        st.subheader("Model-1 Feature Importance")
        fi_p = PROJECT_ROOT / "outputs/figures/feature_importance.png"
        if fi_p.exists():
            st.image(str(fi_p), caption="Model-1 Gini Importance (Top features: sunspot_smooth11, rwi_smooth5)", use_container_width=True)
        else:
            st.info("Feature importance plot available in outputs/figures/")


# ── TAB 5: Notebook Inspector ────────────────────────────────────────────────
with tab5:
    st.subheader("📘 Model-1 Jupyter Notebook Inspector (`model-1.ipynb`)")
    st.caption("Inspect the complete scientific training code and historical cells for Model-1.")

    nb1_path = PROJECT_ROOT / "model-1.ipynb"
    if nb1_path.exists():
        with open(nb1_path, "r", encoding="utf-8") as f:
            nb1_data = json.load(f)

        cells = nb1_data.get("cells", [])
        st.write(f"Total Notebook Cells: **{len(cells)}**")

        cell_opts = [f"Cell {i} [{c.get('cell_type','').upper()}]: {''.join(c.get('source',[]))[:70].strip()}" for i, c in enumerate(cells)]
        sel_idx = st.selectbox("Select Notebook Cell", range(len(cell_opts)), format_func=lambda i: cell_opts[i])

        chosen = cells[sel_idx]
        ctype = chosen.get("cell_type", "code")
        csrc = "".join(chosen.get("source", []))

        if ctype == "markdown":
            st.markdown("##### Rendered Markdown")
            st.markdown(csrc)
        else:
            st.markdown("##### Executable Python Code")
            st.code(csrc, language="python")

        with open(nb1_path, "rb") as f:
            st.download_button("📥 Download `model-1.ipynb`", data=f.read(), file_name="model-1.ipynb", mime="application/x-ipynb+json")
    else:
        st.warning("model-1.ipynb not found.")

# Footer
st.markdown("---")
st.caption("FRADSCR Model-1 Baseline Prototype · Single-Site Gondar Reconstruction")
