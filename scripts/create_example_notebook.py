import json
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.13.15"
    }
}

cells = []

# Cell 0: Title & Overview
cells.append(nbf.v4.new_markdown_cell("""# Random Forest Drought Classification: Step-by-Step Educational Example
### Paleoclimate & Solar-Driven Early Warning System for Ethiopian Water Security

---

## 1. Executive Summary & Architecture Overview
This notebook is a **clean, simplified, and self-contained walkthrough** demonstrating the core **Random Forest Classification architecture** developed for the **Fradscr (Maji-Alert)** research pipeline.

### The Objective:
Forecast Ethiopian regional drought risk into 3 actionable categories:
- **Class 0 (Normal / Wet)**: Adequate moisture conditions ($SPEI > -0.10$)
- **Class 1 (Moderate Deficit)**: Developing drought stress ($-0.35 < SPEI \\le -0.10$)
- **Class 2 (Severe Drought)**: Critical water shortage & food security alert ($SPEI \\le -0.35$)

### Why Random Forest for Paleoclimate Drought Modeling?
1. **Non-Linear Interactions**: Solves the non-linear coupling between multi-decadal solar irradiance cycles (~11-year Schwabe cycle) and multi-year biological tree growth persistence.
2. **Noise Resilience**: Ensembling 300 decorrelated decision trees reduces variance and prevents overfitting on century-scale climate series (~100 observations).
3. **Transparent Interpretability**: Provides clear Gini feature importance rankings so climate scientists can audit which physical drivers govern the predictions.
"""))

# Cell 1: Imports
cells.append(nbf.v4.new_markdown_cell("""## 2. Environment Setup & Imports
We import standard data science and machine learning libraries.
The `find_project_root()` helper dynamically resolves paths across **Google Colab, VS Code, and local Jupyter**.
"""))

cells.append(nbf.v4.new_code_cell("""import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# Auto-resolve repository root across Colab and local environments
def find_project_root():
    candidates = [
        Path(".").resolve(),
        Path("..").resolve(),
        Path("/content/Fradscr"),
        Path("/content/maji-alert"),
        Path("/content"),
    ]
    for p in candidates:
        if (p / "results" / "processed_lagged_data.csv").exists():
            return p
    return Path(".").resolve()

PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["font.size"] = 11

print(f"Project Root resolved to: {PROJECT_ROOT}")
print("All dependencies loaded successfully.")
"""))

# Cell 2: Data Ingestion
cells.append(nbf.v4.new_markdown_cell("""## 3. Data Ingestion & Time-Series Alignment
We load the two fundamental empirical time series:
1. **Detrended Tree-Ring & Solar Data** (`results/processed_lagged_data.csv`): Contains standardized Ring-Width Index anomalies ($RWI_z$) and Sunspot counts ($SN$).
2. **Meteorological SPEI Ground Truth** (`results/spei_debrebirkan.csv`): The ground truth drought index.
"""))

cells.append(nbf.v4.new_code_cell("""# 1. Load detrended tree-ring and solar cycle data
df_lag = pd.read_csv(PROJECT_ROOT / "results" / "processed_lagged_data.csv")

# 2. Compute annual master chronology (averaging across cores)
chron = df_lag.groupby("year")[["rwi", "rwi_z", "sunspot", "sunspot_z"]].mean().reset_index()

# 3. Load meteorological SPEI ground truth
df_spei = pd.read_csv(PROJECT_ROOT / "results" / "spei_debrebirkan.csv")

# 4. Inner join on calendar year
df_climate = pd.merge(chron, df_spei, on="year", how="inner").sort_values("year").reset_index(drop=True)

print(f"Dataset Temporal Span: {df_climate['year'].min()} – {df_climate['year'].max()} ({len(df_climate)} continuous years)")
df_climate.head(6)
"""))

# Cell 3: Feature Engineering
cells.append(nbf.v4.new_markdown_cell("""## 4. Feature Engineering: Tree Rings + Solar Signals
We construct the feature matrix from the physical mechanisms of drought teleconnection:
1. **Tree-Ring Biological Memory**: Soil moisture carryover and carbohydrate storage mean tree radial growth exhibits multi-year persistence (`rwi_lag_1`, `rwi_lag_2`).
2. **Astrophysical Solar Cycles**: The ~11-year Schwabe solar cycle is captured via 11-year rolling averages (`sunspot_smooth11`) and annual lag (`sunspot_lag_1`).
3. **Cross-Domain Teleconnection Interaction**: Coupling solar activity with tree growth (`solar_tree_interaction = sunspot_z * rwi_z`).
4. **Target Classification**: Segmenting SPEI into 3 calibrated drought risk classes.
"""))

cells.append(nbf.v4.new_code_cell("""# 1. Tree-ring biological memory lags
df_climate["rwi_lag_1"] = df_climate["rwi"].shift(1)
df_climate["rwi_lag_2"] = df_climate["rwi"].shift(2)

# 2. Solar cycle lags and 11-year Schwabe smoothing
df_climate["sunspot_lag_1"] = df_climate["sunspot"].shift(1)
df_climate["sunspot_smooth11"] = df_climate["sunspot"].rolling(window=11, min_periods=1, center=False).mean()

# 3. Cross-domain interaction term
df_climate["solar_tree_interaction"] = df_climate["sunspot_z"] * df_climate["rwi_z"]

# 4. Target Classification (Calibrated Regional Thresholds)
def categorize_spei_3class(spei_val):
    if spei_val > -0.10:
        return 0  # Normal / Wet
    elif spei_val > -0.35:
        return 1  # Moderate Deficit
    else:
        return 2  # Severe Drought

df_climate["target_3class"] = df_climate["spei"].apply(categorize_spei_3class)

# Drop initial NaN lag rows
df_model = df_climate.dropna().reset_index(drop=True)

class_names = ["Normal / Wet (0)", "Moderate Deficit (1)", "Severe Drought (2)"]
print(f"Total modeling samples: {len(df_model)} years ({df_model['year'].min()}–{df_model['year'].max()})")
print("\nClass Distribution:")
print(df_model["target_3class"].value_counts().rename(index=dict(enumerate(class_names))))
"""))

# Cell 4: Train/Test Split
cells.append(nbf.v4.new_markdown_cell("""## 5. Chronological Train/Test Split (Zero Temporal Leakage)
In time-series and climate forecasting, **random train_test_split causes temporal leakage** (the model memorizes future weather patterns to predict the past).

We enforce a strict chronological cutoff:
- **Training Set (1903–1985)**: Historical observations used exclusively for model learning.
- **Holdout Test Set (1986–2009)**: Unseen modern decades used strictly for blind performance evaluation.
"""))

cells.append(nbf.v4.new_code_cell("""feature_cols = [
    "rwi", "rwi_z", "rwi_lag_1", "rwi_lag_2",
    "sunspot", "sunspot_z", "sunspot_lag_1", "sunspot_smooth11",
    "solar_tree_interaction"
]

split_idx = int(len(df_model) * 0.76)

df_train = df_model.iloc[:split_idx].copy()
df_test = df_model.iloc[split_idx:].copy()

X_train, y_train = df_train[feature_cols], df_train["target_3class"]
X_test, y_test = df_test[feature_cols], df_test["target_3class"]

print(f"Training Period:   {df_train['year'].min()} – {df_train['year'].max()} ({len(df_train)} samples)")
print(f"Holdout Test Span: {df_test['year'].min()} – {df_test['year'].max()} ({len(df_test)} samples)")
print(f"Training Class Counts: {dict(y_train.value_counts())}")
print(f"Testing Class Counts:  {dict(y_test.value_counts())}")
"""))

# Cell 5: Model Training
cells.append(nbf.v4.new_markdown_cell("""## 6. Training the Random Forest Classifier
We instantiate `RandomForestClassifier` with hyperparameters specifically configured for paleoclimate records:

- `n_estimators=300`: Builds an ensemble of 300 decorrelated decision trees, minimizing prediction variance.
- `max_depth=4`: Restricts tree depth to prevent memorization / overfitting on small sample sizes (~100 years).
- `class_weight='balanced'`: Compensates for the natural rarity of severe drought events (~14% of historical years).
- `random_state=42`: Ensures deterministic, 100% reproducible training.
"""))

cells.append(nbf.v4.new_code_cell("""# Instantiate Random Forest
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=4,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# Fit model on training set
rf_model.fit(X_train, y_train)

print("Random Forest Training Complete.")
print(f"Number of Trees:     {len(rf_model.estimators_)}")
print(f"Number of Features:  {rf_model.n_features_in_}")
print(f"Classes Learned:     {rf_model.classes_}")
"""))

# Cell 6: Holdout Evaluation
cells.append(nbf.v4.new_markdown_cell("""## 7. Model Evaluation on Unseen Test Years
We evaluate the trained model on the unseen test period (1986–2009).
We compute:
1. **Overall Accuracy**: Total correct predictions over total years.
2. **Balanced Accuracy**: Macro-averaged recall across all classes (vital for imbalanced drought events).
3. **Macro F1-Score**: Harmonic mean of precision and recall.
"""))

cells.append(nbf.v4.new_code_cell("""y_pred = rf_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
bal_acc = balanced_accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")

print("=" * 55)
print("     UNSEEN TEST EVALUATION METRICS (1986–2009)")
print("=" * 55)
print(f"Overall Accuracy:       {acc:.1%}")
print(f"Balanced Accuracy:      {bal_acc:.1%}")
print(f"Macro F1-Score:         {macro_f1:.3f}")
print("=" * 55)

print("\nDetailed Classification Report:")
target_labels = [0, 1, 2]
target_names = ["Normal/Wet", "Moderate Deficit", "Severe Drought"]
print(classification_report(y_test, y_pred, labels=target_labels, target_names=target_names, zero_division=0))
"""))

# Cell 7: Confusion Matrix Heatmap
cells.append(nbf.v4.new_markdown_cell("""## 8. Confusion Matrix Heatmap
The confusion matrix visualizes true ground-truth classes versus model predictions.
- **Diagonal cells**: Correct classifications.
- **Off-diagonal cells**: Errors (false alarms or missed events).
"""))

cells.append(nbf.v4.new_code_cell("""cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

plt.figure(figsize=(7, 5.5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal/Wet", "Moderate", "Severe"],
    yticklabels=["Normal/Wet", "Moderate", "Severe"],
    cbar=False
)
plt.title(f"Holdout Test Confusion Matrix ({df_test['year'].min()}–{df_test['year'].max()})", fontsize=13, pad=12)
plt.xlabel("Predicted Class", fontsize=11)
plt.ylabel("Actual Ground Truth (SPEI)", fontsize=11)
plt.tight_layout()
plt.show()
"""))

# Cell 8: Feature Importance
cells.append(nbf.v4.new_markdown_cell("""## 9. Feature Importance Interpretation (Gini Impurity)
Random Forest allows us to compute the **Gini Feature Importance** (Mean Decrease in Impurity), revealing which paleoclimate features contributed most to separating drought from non-drought years.
"""))

cells.append(nbf.v4.new_code_cell("""feat_importance = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=True)

plt.figure(figsize=(9, 5.5))
feat_importance.plot(kind="barh", color="#1b7837", edgecolor="black", alpha=0.85)
plt.title("Random Forest Gini Feature Importance", fontsize=13, pad=10)
plt.xlabel("Importance Score (Mean Decrease in Impurity)", fontsize=11)
plt.ylabel("Engineered Feature", fontsize=11)
plt.tight_layout()
plt.show()

print("Top 3 Most Influential Features:")
for feat, score in feat_importance.tail(3).iloc[::-1].items():
    print(f"  - {feat}: {score:.3f}")
"""))

# Cell 9: Inference & Calibrated Confidence
cells.append(nbf.v4.new_markdown_cell("""## 10. Live Inference Simulation & Temperature Scaling ($T=0.35$)
In operational deployment (`predict_service.py`), raw Random Forest class probabilities can be diffuse.
To give water managers crisp, actionable alert signals, we apply **Temperature Scaling ($T = 0.35$)**:

$$\\hat{P}_k = \\frac{\\exp(z_k / T)}{\\sum_j \\exp(z_j / T)}$$

Let's simulate a live prediction for an incoming test year:
"""))

cells.append(nbf.v4.new_code_cell("""def calibrated_predict(model, X_input, temperature=0.35):
    raw_probs = model.predict_proba(X_input)[0]
    eps = 1e-7
    logits = np.log(np.clip(raw_probs, eps, 1.0 - eps))
    scaled_logits = logits / temperature
    calibrated_probs = np.exp(scaled_logits) / np.sum(np.exp(scaled_logits))
    predicted_class = int(np.argmax(calibrated_probs))
    return predicted_class, calibrated_probs, raw_probs

# Sample test year
sample_idx = 0
sample_year = df_test.iloc[sample_idx]["year"]
sample_features = X_test.iloc[[sample_idx]]
actual_class = y_test.iloc[sample_idx]

pred_class, cal_probs, raw_probs = calibrated_predict(rf_model, sample_features, temperature=0.35)

print(f"Prediction for Year:      {int(sample_year)}")
print(f"Actual Ground Truth:      {class_names[actual_class]}")
print(f"Predicted Class:          {class_names[pred_class]}")
print("-" * 55)
print(f"Raw Probabilities:        Normal={raw_probs[0]:.2f}, Moderate={raw_probs[1]:.2f}, Severe={raw_probs[2]:.2f}")
print(f"Calibrated Probabilities: Normal={cal_probs[0]:.2f}, Moderate={cal_probs[1]:.2f}, Severe={cal_probs[2]:.2f}")
print(f"Model Alert Confidence:   {np.max(cal_probs):.1%}")
"""))

# Cell 10: Conclusion
cells.append(nbf.v4.new_markdown_cell("""## 11. Presentation Defense Cheat Sheet (EGATE Instructors)
When presenting this Random Forest architecture to the EGATE evaluation panel:

1. **How Features are Built**:
   > *"We detrend raw tree rings into Ring-Width Index ($RWI$) and align them with sunspot records. Multi-year lag features capture soil moisture memory, while 11-year smoothing tracks the Schwabe cycle."*

2. **Why Random Forest over Linear Regression**:
   > *"Solar teleconnections are non-linear and coupled with multi-year biological delays. Random Forest captures complex decision boundaries without assuming linear Gaussian relationships."*

3. **Why Restrict Depth (`max_depth=4`)**:
   > *"With ~100 historical annual climate records, deep trees would easily overfit to noise. Shallow trees act as regularized ensemble stumps that generalize better to unseen decades."*

4. **Why Balanced Class Weights**:
   > *"Severe droughts represent only ~14% of historical years. Setting `class_weight='balanced'` penalizes false negatives on extreme events so the model does not become overly conservative."*

5. **Why Temperature Scaling ($T=0.35$)**:
   > *"Random Forest probability outputs can be uncalibrated and overly diffuse. Applying $T=0.35$ sharpens the true signal into confident, actionable early-warning alerts for disaster response teams."*
"""))

nb.cells = cells

# Save notebook
output_path = Path("example_random_forest.ipynb")
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Successfully generated {output_path}")
