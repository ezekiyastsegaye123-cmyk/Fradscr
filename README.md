# FRADSCR — Tree-Ring Paleoclimatology & Solar Teleconnection AI

[![Open Model-2 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ezekiyastsegaye123-cmyk/Fradscr/blob/main/model-2.ipynb)
[![Open Model-1 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ezekiyastsegaye123-cmyk/Fradscr/blob/main/model-1.ipynb)

A production-ready machine learning framework and Streamlit advisory system for predicting decadal groundwater deficits and optimizing solar water pump dispatch across the Horn of Africa using tree-ring chronologies, SILSO solar teleconnections, and reinforcement learning.

## Installation

Requires Python ≥ 3.10.

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install numpy pandas scipy

# For development/testing
pip install pytest
```

## Usage

### Command Line

```bash
python -m treering <input.rwl> <output.csv> [options]
```

**Arguments:**

| Argument     | Description                          |
| ------------ | ------------------------------------ |
| `input_rwl`  | Path to input `.rwl` file (Tucson format) |
| `output_csv` | Path for output CSV file             |

**Options:**

| Option          | Description                                          |
| --------------- | ---------------------------------------------------- |
| `--overwrite`   | Overwrite existing output CSV file                   |
| `--skip-failed` | Skip series that fail curve fitting (log a warning)  |
| `-v, --verbose` | Enable debug-level logging                           |
| `-h, --help`    | Show help message                                    |

**Examples:**

```bash
# Basic usage
python -m treering africa/eth007.rwl results.csv

# Skip series with too few observations for fitting
python -m treering africa/eth001.rwl results.csv --skip-failed

# Overwrite existing output, verbose logging
python -m treering africa/eth007.rwl results.csv --overwrite --verbose
```

### Python API

```python
from treering import process_rwl, export_csv

# Run the full pipeline
df = process_rwl("africa/eth007.rwl", skip_failed_series=True)
print(df.head())

# Export to CSV
export_csv(df, "output.csv", overwrite=True)
```

## Input Format — Tucson `.rwl`

The pipeline reads standard **Tucson Decadal Format** `.rwl` files as used by
the International Tree-Ring Data Bank (ITRDB).

**Structure:**

- Optional 3-line header (site name, location/species, investigators).
- Data lines: `<series_id>  <decade_start_year>  <value_1> ... <value_N>`
- Series ID: up to 8 characters.
- Values: integer ring widths (units depend on dataset).
- `999` = end-of-series stop marker.
- Multiple series stored sequentially.

**Example:**

```
TST01   1900   500   480   460   440   420   400   380   360   340   320
TST01   1910   300   280   260   250   240   230   220   215   210   205
TST01   1920   200   198   196   999
TST02   1920   400   380   360   340   320   300   285   270   260   250
TST02   1930   240   235   230   999
```

## Processing Pipeline

```
.rwl file
    │
    ▼
┌──────────────┐
│  parse_rwl() │  → Parse Tucson format, extract (series_id, year, ring_width)
└──────┬───────┘
       │
       ▼
┌────────────────────┐
│  For each series:  │
│                    │
│  fit_growth_curve()│  → Fit G(t) = a·exp(-b·t) + c via scipy.optimize.curve_fit
│                    │
│  calculate_rwi()   │  → RWI_t = raw_t / G(t)
└────────┬───────────┘
         │
         ▼
┌──────────────┐
│ export_csv() │  → Write series_id, year, raw_ring_width, fitted_growth, rwi
└──────────────┘
```

### Negative Exponential Growth Model

```
G(t) = a · exp(-b · t) + c
```

| Parameter | Meaning                        | Bounds      |
| --------- | ------------------------------ | ----------- |
| `a`       | Amplitude of decaying component | ≥ 0         |
| `b`       | Decay rate                     | ≥ 0         |
| `c`       | Asymptotic minimum growth      | ≥ 0         |
| `t`       | Age index (year − min(year))   | ≥ 0         |

Fitting uses `scipy.optimize.curve_fit` with bounded parameters.

### Ring Width Index (RWI)

```
RWI_t = RawRingWidth_t / G(t)
```

A well-detrended series has a mean RWI near 1.0. Values > 1.0 indicate
above-average growth; values < 1.0 indicate below-average growth.

## Output Schema

The CSV output contains these columns in order:

| Column           | Type    | Description                           |
| ---------------- | ------- | ------------------------------------- |
| `series_id`      | string  | Tree/core series identifier           |
| `year`           | integer | Calendar year                         |
| `raw_ring_width` | integer | Original measurement from `.rwl` file |
| `fitted_growth`  | float   | Fitted G(t) value                     |
| `rwi`            | float   | Ring Width Index (raw / fitted)       |

- UTF-8 encoded.
- No pandas index column.
- Sorted by `(series_id, year)`.

## Error Handling

The pipeline validates at every stage and provides specific error messages:

| Error                       | Behavior                                            |
| --------------------------- | --------------------------------------------------- |
| Missing input file          | `FileNotFoundError` with path                       |
| Empty / no-data file        | `RWLParseError` with context                        |
| Non-numeric measurements    | `RWLParseError` identifying line and position       |
| Too few observations (<10)  | `FittingError` identifying the series               |
| Non-finite input values     | `FittingError` identifying the series               |
| Curve fit failure           | `FittingError` with scipy error details             |
| Near-zero fitted growth     | `FittingError` or `RWIError` with tolerance info    |
| Non-finite RWI              | `RWIError` identifying the series                   |
| Output file exists          | `ExportError` (use `--overwrite` to replace)        |

With `--skip-failed`, series that fail fitting are logged and omitted rather
than aborting the entire pipeline.

## Testing

```bash
python -m pytest tests/ -v
```

Tests cover:
- Parser: valid files, multi-series, headers, stop markers, edge cases, errors.
- Model: mathematical correctness, parameter recovery from synthetic data.
- RWI: arithmetic correctness, division safety.
- Pipeline: schema validation, independent fitting, real data, determinism.
- Export: column order, index exclusion, overwrite protection.
- End-to-end: `.rwl` → CSV round-trip on both fixtures and real data.

## Assumptions and Limitations

1. **Tucson format variant**: Supports the standard decadal format with
   optional 3-line headers.  Non-standard header formats or non-Tucson
   `.rwl` variants may not parse correctly.

2. **Stop marker**: Only `999` is recognized as an end-of-series marker.
   The missing-value marker `-9999` is not specially handled (treated as a
   regular measurement).

3. **Minimum observations**: At least 10 measurements per series are required
   for curve fitting (3 parameters + margin).

4. **Growth model**: Only the negative exponential model
   `G(t) = a·exp(-b·t) + c` is supported.  Alternative detrending methods
   (spline, linear regression) are not implemented.

5. **Measurement units**: The pipeline does not convert or validate units.
   Ring widths are processed as-is from the `.rwl` file.

## Solar-Cycle Lag Analysis (RWI vs. Sunspot Number)

The package includes a solar-cycle lag analysis module (`treering.solar_lag`) to investigate decadal statistical associations between solar activity (historical Sunspot Number, $SN$) and regional tree growth ($RWI$).

### Analysis Workflow
1. **Merge on Calendar Year**: Exact inner join on integer calendar years $[1700, \dots]$.
2. **11-Year Centered Moving Average**: Isolates the ~11-year Schwabe solar cycle:
   $$x_{\text{smoothed}}(t) = \frac{1}{11} \sum_{k=-5}^{5} x(t+k)$$
   *First 5 and last 5 observations strictly receive `NaN`.*
3. **Standardization**: z-score anomalies with sample standard deviation ($\text{ddof}=1$):
   $$z_t = \frac{x_t - \mu}{\sigma}$$
4. **Lag Correlation**: Evaluates Pearson correlation $R(\tau) = \text{corr}(RWI(t), SN(t-\tau))$ for $\tau \in \{0, 1, 2, 3, 4, 5\}$ years.
5. **Optimal Lag Selection**: $\tau^* = \arg\max_{\tau} |R(\tau)|$ (evaluating strongest positive or negative correlation).

### Python API Example

```python
from treering import run_solar_lag_analysis

# Run end-to-end solar lag analysis
result = run_solar_lag_analysis(
    rwi_input="results/rwi_eth007.csv",
    sunspot_input="SN_y_tot_V2.0.csv",
    max_lag=5,
    output_dir="results",
    overwrite=True,
)

print(f"Optimal lag: tau = {result.optimal_lag.optimal_lag} years")
print(f"Pearson R: {result.optimal_lag.optimal_correlation:.4f}")
print(result.lag_correlations)
```

### Interactive Jupyter Notebook

An interactive notebook with visualizations and step-by-step auditability is available at:
- `notebooks/rwi_sunspot_lag_analysis.ipynb`
- `Model.ipynb`

## Project Structure

```
treering/
├── __init__.py      # Public API exports
├── __main__.py      # python -m treering entry point
├── cli.py           # Argument parsing and CLI logic
├── export.py        # CSV export with validation
├── model.py         # Negative exponential model and curve fitting
├── parser.py        # Tucson .rwl file parser
├── pipeline.py      # End-to-end detrending orchestration
├── rwi.py           # Ring Width Index calculation
└── solar_lag.py     # RWI / Sunspot 11-yr Schwabe lag analysis

notebooks/
└── rwi_sunspot_lag_analysis.ipynb # Interactive analysis & visualization

tests/
├── fixtures/
├── test_model.py
├── test_parser.py
├── test_pipeline.py
├── test_rwi.py
└── test_solar_lag.py
```

### FRADSCR — Streamlit Early Warning & Solar Borehole Advisory System

**FRADSCR** is an interactive scientific application built with **Streamlit** for drought forecasting and solar groundwater advisory across the **Borana Zone, Oromia, Ethiopia**. It links regional tree-ring climate memory (*Juniperus procera*) and solar cycle teleconnections directly to community water point operators and solar-powered borehole pumps.

### Architecture

```text
User / Field Operator Browser
             │
             │ HTTPS / WebSocket
             ▼
Streamlit Application Hub (streamlit_app.py)
   │
   ├── Model-2: SoTA Regional Ensemble & Prescriptive RL (app_model_2.py)
   │     ├── Pan-Ethiopian RCS Master Chronology (eth002 to eth007)
   │     ├── Stacking Ensemble (Random Forest + XGBoost)
   │     ├── Softmax Calibration (T = 0.35)
   │     └── WaterPumpAgent (Prescriptive RL Dispatch)
   │
   ├── Model-1: Single-Site Gondar Baseline (app_model_1.py)
   │
   └── In-Memory Scientific ML Engine (predict_service.py)
```

### Running the Streamlit Application

```bash
# Launch master dual-model hub
streamlit run streamlit_app.py

# Or run Model-2 directly
streamlit run app_model_2.py

# Or run Model-1 directly
streamlit run app_model_1.py
```

### Production Docker Deployment

```bash
# Build and run the Streamlit container
docker compose up -d --build
# Or run directly via Docker
docker build -t fradscr-streamlit .
docker run -p 8501:8501 fradscr-streamlit
```

### Automated Testing

```bash
# Run complete Python test suite
pytest -v
```

---


## Satellite & Climatological Auxiliary Observation Ingestion

In addition to physical tree-ring dendrochronology and Schwabe solar cycles, FRADSCR accommodates continuous cross-validation against high-resolution Earth Observation (EO) feeds:

### 1. Satellite Observation Ingestion Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│              Multi-Source Validation Pipeline                    │
└─────────────────────────────────────────────────────────────────┘
         │                                       │
         ▼                                       ▼
┌───────────────────────────────┐     ┌───────────────────────────┐
│   Satellite Earth Observation │     │  Operator Ground-Truth    │
│   - CHIRPS Precipitation (0.05°)│     │  - Static Water Levels    │
│   - MODIS/Sentinel NDVI/EVI   │     │  - Pump Operational Quota │
│   - SMAP L4 Soil Moisture      │     │  - Livestock Stress Notes │
└───────────────────────────────┘     └───────────────────────────┘
         │                                       │
         └───────────────────┬───────────────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │ Ground-Truth Reconciliation │
              │ - Confusion Matrix Tracking │
              │ - Brier Reliability Score   │
              │ - False Alarm Mitigation    │
              └─────────────────────────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │ Retraining & Model Updating │
              │ (Zero-Leakage Holdout Rule) │
              └─────────────────────────────┘
```

### 2. Supported Auxiliary Datasets
1. **CHIRPS v2.0 (Rainfall Anomalies)**: High-resolution ($0.05^\circ$) quasi-global precipitation estimates used to independently confirm Kiremt / Belg monsoon deficit timing.
2. **MODIS Terra/Aqua & Sentinel-2 (NDVI/NDWI)**: Normalized Difference Vegetation and Water Indices providing real-time pasture greenness and water canopy metrics.
3. **NASA SMAP L4 (Root-Zone Soil Moisture)**: $9\,\text{km}$ global soil moisture anomalies measuring deep aquifer replenishment potential.
4. **NOAA PSL Oceanic Teleconnections**: Operational updates for El Niño–Southern Oscillation (Niño 3.4 index) and Indian Ocean Dipole (Dipole Mode Index - DMI).



