# AGY — Production Tree-Ring Dataset Selection + Model-1 Development

## ROLE

Act as a **senior machine-learning engineer, dendrochronology data scientist, scientific Python engineer, and ML validation engineer**.

We are developing the next machine-learning model for the EGATE Heliophysics / Maji Alert project:

> **Using Tree Rings to Study Solar-Driven Climate Cycles: A Scientific Framework for Ethiopian Water Security**

My teacher has instructed me to determine **which available tree-ring dataset is the most suitable for the machine-learning model** by comparing model performance, prediction confidence, and data quality.

After identifying the most appropriate tree-ring dataset, build a new supervised machine-learning model using that dataset.

The complete training workflow must be implemented and documented in:

```text
model-1.ipynb
```

The notebook must contain both:

1. **Tree-ring dataset comparison/selection**
2. **Final Model-1 training and evaluation**

Do not create a superficial comparison based only on accuracy.

---

# 1. FIRST — INSPECT THE PROJECT

Before writing or modifying anything, inspect the repository.

Identify all available tree-ring datasets/chronologies/series that could reasonably be candidates for the model.

Inspect:

```text id="sl9k2q"
tree-ring data
.rwl files
RWI CSV files
processed RWI files
existing drought datasets
solar datasets
SPEI datasets
existing notebooks
existing ML scripts
existing model artifacts
tests
```

Determine:

- tree-ring dataset names;
- site names;
- chronology/series identifiers;
- time periods;
- number of observations;
- available RWI variables;
- whether data are raw ring widths or already detrended;
- whether multiple tree/core series exist;
- existing preprocessing methodology.

Do not assume a particular dataset is the best one.

Do not assume the first dataset found should be used.

---

# 2. DEFINE THE DECISION PROBLEM

The objective is to answer:

> **Which tree-ring dataset provides the most reliable predictor information for the drought-classification model?**

The selection must consider at least four dimensions:

### A. Data Quality

Evaluate:

- temporal coverage;
- number of valid observations;
- missing values;
- duplicate observations;
- year continuity;
- consistency of tree-ring measurements;
- availability over the common analysis period.

### B. Predictive Performance

Evaluate:

- accuracy;
- balanced accuracy;
- macro F1;
- weighted F1;
- precision;
- recall;
- Class 2 F1;
- Class 2 precision;
- Class 2 recall.

### C. Prediction Probability / Confidence

Evaluate the model's probability outputs.

Do not automatically call raw Random Forest probabilities "calibrated confidence."

Use precise terminology such as:

```text
predicted probability
```

unless calibration has been explicitly performed.

Evaluate where appropriate:

- mean maximum predicted probability;
- probability distribution;
- probability for the predicted class;
- uncertainty/ambiguity;
- calibration metrics if enough data exist.

If the project requires a user-facing "confidence" number, clearly distinguish it from statistically calibrated confidence.

### D. Scientific Relevance

Consider:

- temporal coverage;
- geographical relevance;
- relationship to the target SPEI region;
- consistency with the research question;
- availability of valid observations during the drought-analysis period.

Do not select a dataset solely because it produces the highest numerical score if its scientific/data quality is inadequate.

---

# 3. DO NOT USE THE TEST SET TO CHOOSE THE WINNER

This is a critical ML requirement.

Do not select the best tree-ring dataset by repeatedly evaluating candidates on the final test/holdout set and choosing the one that performs best.

Use a validation strategy appropriate to the available data.

For historical time-series data, prefer:

```text id="r4pjgx"
training data
    ↓
validation strategy
    ↓
candidate comparison
    ↓
select best tree-ring dataset
    ↓
final untouched test/holdout evaluation
```

The final test set must remain untouched until the selected dataset/model has been determined.

If the existing project has `TimeSeriesSplit`, use it for candidate comparison where appropriate.

---

# 4. IDENTIFY CANDIDATE TREE-RING DATASETS

Automatically identify all candidate tree-ring datasets supported by the existing repository.

For each candidate, record:

```text id="6aw4lt"
dataset_id
site
series_id if applicable
start_year
end_year
number_of_observations
missing_value_count
year_gap_count
RWI availability
```

If a dataset contains multiple individual tree/core series, determine whether the existing project methodology treats:

- each series independently;
- an established chronology;
- an already aggregated RWI series.

Do not invent an aggregation method.

---

# 5. COMMON ANALYSIS PERIOD

Candidate tree-ring datasets may cover different periods.

Determine the common valid period between:

```text id="5zpwlf"
tree-ring data
solar data
SPEI data
```

Do not compare candidates on completely different target periods without documenting the difference.

Prefer a consistent evaluation window.

Calculate and report:

```text id="4afj2p"
common start year
common end year
number of valid years
```

for each candidate.

---

# 6. PREPROCESSING CONSISTENCY

Every candidate tree-ring dataset must pass through the same scientifically defined preprocessing pipeline.

Where established by the project, this includes:

```text id="xplk1y"
RWL parsing
↓
biological detrending
↓
negative exponential fitting
↓
RWI calculation
↓
annual alignment
↓
11-year smoothing where required
↓
solar feature alignment
↓
SPEI target alignment
```

Do not give one candidate special preprocessing treatment unless there is a scientifically documented reason.

The comparison must be fair.

---

# 7. TARGET DEFINITION

Use the established drought classification:

```text id="j7q6om"
Class 0 — Normal / Wet:
SPEI > -1.0

Class 1 — Moderate Drought:
-1.5 < SPEI <= -1.0

Class 2 — Severe Drought:
SPEI <= -1.5
```

Verify the implementation carefully at:

```text id="gwm6q3"
SPEI = -1.0
SPEI = -1.5
```

Do not change the target boundaries for individual candidate tree-ring datasets.

The target definition must remain identical across comparisons.

---

# 8. FEATURE ENGINEERING

For each candidate tree-ring dataset, construct the same model feature set.

Use the actual existing feature names discovered in the project.

Potential features include:

```text id="njtz2o"
RWI standardized
RWI raw
smoothed solar activity
solar lag 0
solar lag 1
solar lag 2
solar lag 3
solar lag 4
solar lag 5
```

Do not automatically include every available column.

Do not include:

```text id="s7f6jv"
SPEI
drought_class
future target information
```

as predictors.

---

# 9. TEMPORAL LEAKAGE AUDIT

Inspect all existing transformations for future information.

Pay particular attention to:

```text id="f7g6d1"
11-year centered rolling averages
lagged variables
standardization
imputation
feature selection
```

If centered smoothing uses:

```text id="xv8y39"
t-5 ... t ... t+5
```

document that it incorporates future observations relative to year `t`.

Determine whether the model is intended for:

```text id="8ob7q3"
retrospective scientific reconstruction
```

or:

```text id="k5ac9p"
prospective forecasting
```

Do not silently present future-dependent features as real-time forecasting features.

---

# 10. CANDIDATE MODEL

For dataset comparison, use the same model configuration for every candidate.

Unless the existing project specifies otherwise, use:

```python id="w6l8b0"
RandomForestClassifier
```

with a fixed:

```text id="4r3u9v"
random_state
```

and consistent hyperparameters.

Do not tune each candidate differently.

The goal is to compare the information content of the tree-ring datasets, not to compare unequal amounts of hyperparameter optimization.

---

# 11. VALIDATION STRATEGY

Use an appropriate chronological validation strategy.

Preferred:

```python id="1fkk6t"
TimeSeriesSplit(n_splits=5)
```

if the candidate datasets have enough observations.

For every candidate:

1. Maintain chronological ordering.
2. Train only on earlier observations.
3. Validate on later observations.
4. Record fold metrics.
5. Aggregate out-of-fold predictions.
6. Calculate overall metrics.

Do not shuffle historical observations.

If `TimeSeriesSplit` cannot be used because the dataset is too small, document the reason and use the most appropriate alternative supported by the project.

---

# 12. CANDIDATE EVALUATION METRICS

For every tree-ring candidate, calculate:

```text id="1k0h0k"
Accuracy
Balanced Accuracy
Macro F1
Weighted F1
```

and:

```text id="cyja2a"
Class 0 Precision
Class 0 Recall
Class 0 F1

Class 1 Precision
Class 1 Recall
Class 1 F1

Class 2 Precision
Class 2 Recall
Class 2 F1
```

Also record:

```text id="7zsvo0"
Class 2 support
Class 2 false positives
Class 2 false negatives
```

where meaningful.

Do not use accuracy alone.

---

# 13. PROBABILITY / CONFIDENCE ANALYSIS

For each candidate, obtain:

```python id="rkgu94"
model.predict_proba(...)
```

where supported.

Analyze:

```text id="9oef8i"
mean maximum predicted probability
median maximum predicted probability
probability distribution
probability assigned to actual class
```

If feasible, calculate a proper probabilistic metric such as:

```text id="5h5j2g"
log loss
Brier score
```

and calibration diagnostics.

Do not interpret high predicted probability as proof of correctness.

A model can be highly confident and still be wrong.

---

# 14. DATA QUALITY SCORE

Create a transparent data-quality summary.

Consider:

```text id="1pqn66"
temporal coverage
valid observation count
missing data
calendar-year continuity
target overlap
feature completeness
```

Do not invent arbitrary weights without justification.

If you use a composite score, document the exact formula.

Prefer showing the individual metrics alongside any composite ranking.

---

# 15. TREE-RING DATASET SELECTION

Create a candidate comparison table similar to:

| Dataset | Years | Observations | Accuracy | Balanced Accuracy | Macro F1 | Class 2 Precision | Class 2 Recall | Mean Predicted Probability | Data Quality |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

Use **actual calculated values**.

Do not choose the winning dataset until all candidates have been evaluated under the same methodology.

The winning candidate should be selected using a documented decision rule that balances:

```text id="n5x76p"
predictive performance
probability quality
data quality
scientific relevance
```

Do not select a candidate solely because of accuracy.

---

# 16. SELECTION DECISION

Write an explicit selection statement in `model-1.ipynb`.

Example structure:

> "Dataset X was selected because it provided the strongest combination of balanced predictive performance, Class 2 performance, prediction probability behavior, temporal coverage, and scientific suitability."

Do not use this example text without replacing it with the actual evidence.

The notebook must explain:

```text id="7q2nki"
why the selected dataset won
why other datasets were not selected
what trade-offs existed
```

---

# 17. MODEL-1

After selecting the best tree-ring dataset, create:

```text id="fwm3e8"
model-1.ipynb
```

This notebook must contain the **complete training pipeline**.

Do not create a notebook that only loads an already-trained model.

The notebook must actually contain:

```text id="ln0qna"
data loading
data validation
feature preparation
train/test or temporal validation setup
model initialization
model.fit()
prediction
evaluation
model export
```

---

# 18. MODEL-1 TRAINING DATA

Use the selected tree-ring dataset identified in the candidate comparison.

Document:

```text id="9cfh1h"
selected dataset
site
series/chronology
time period
number of observations
target definition
feature list
```

Do not silently substitute another tree-ring dataset later in the notebook.

---

# 19. FINAL MODEL TRAIN/TEST SPLIT

For the final Model-1 experiment, use the teacher's requested approximately:

```text id="f9h9e4"
80% training
20% testing
```

If the data are historical time-series data:

```text id="43wx2t"
earliest 80% → training
latest 20% → testing
```

Do not shuffle the observations.

Verify:

```python id="gb2yq9"
max(train_year) < min(test_year)
```

when a year column exists.

Print:

```text id="7v8d8r"
Training observations
Testing observations
Training percentage
Testing percentage
Training period
Testing period
```

---

# 20. DO NOT CONFUSE VALIDATION WITH FINAL TESTING

The workflow must distinguish:

```text id="5g8zpo"
Candidate selection / validation
```

from:

```text id="xj5j6t"
Final 20% test evaluation
```

The candidate-selection process must not repeatedly use the final test set.

After selecting the tree-ring dataset and final model configuration, evaluate the final Model-1 once on the untouched 20% test period.

---

# 21. MODEL-1 TRAINING

Train the final Random Forest using the selected dataset.

Use the verified project model configuration.

Explicitly execute:

```python id="wpz5gq"
model.fit(X_train, y_train)
```

inside `model-1.ipynb`.

The notebook must show the actual training code.

Do not only call a pre-trained artifact.

---

# 22. MODEL-1 PREDICTIONS

Generate:

```python id="00jx7n"
y_pred = model.predict(X_test)
```

and, where appropriate:

```python id="fgl56o"
y_prob = model.predict_proba(X_test)
```

Do not modify probabilities after prediction unless an explicitly documented calibration/decision rule is part of the model.

---

# 23. MODEL-1 EVALUATION

Generate:

```python id="bwp4y9"
classification_report
confusion_matrix
```

and calculate:

```text id="s5m05f"
accuracy
balanced accuracy
macro F1
weighted F1
```

Explicitly report:

```text id="tuw0gs"
Class 2 precision
Class 2 recall
Class 2 F1
```

Do not hide poor minority-class performance behind overall accuracy.

---

# 24. MODEL CONFIDENCE REPORTING

For the final test set, calculate probability-based diagnostics.

Report where appropriate:

```text id="81bd5v"
mean predicted probability
median predicted probability
prediction probability by class
```

If displaying a user-facing confidence value, clearly state:

> The Random Forest probability is a model-estimated probability and is not automatically a calibrated confidence level.

If calibration has not been performed, do not call it "95% certain."

---

# 25. FEATURE IMPORTANCE

Calculate Random Forest feature importance.

Create:

```text id="mbqf60"
feature
importance
```

Sort descending.

Verify:

```text id="59qtgn"
number of importance values == number of features
```

and:

```text id="8sjrju"
sum(importances) ≈ 1
```

Create a publication-quality feature importance figure.

Do not interpret feature importance as causation.

---

# 26. CONFUSION MATRIX VISUALIZATION

Create a readable `matplotlib` confusion matrix.

Use the actual class labels:

```text id="6uk55u"
Class 0 = Normal / Wet
Class 1 = Moderate Drought
Class 2 = Severe Drought
```

only after verifying this mapping from the project.

---

# 27. MODEL ARTIFACT

Save Model-1 to an explicit path such as:

```text id="v70fuj"
models/random_forest_model_1.joblib
```

Use the project's existing artifact conventions when available.

After saving:

1. Reload the model.
2. Run predictions again.
3. Verify predictions match the original model.

The model artifact must not be considered valid until reload testing passes.

---

# 28. MODEL-1 METADATA

Save metadata documenting:

```text id="dhv4q0"
selected tree-ring dataset
feature names
feature count
target definition
Random Forest parameters
random_state
training period
test period
model path
software versions
```

Use actual values.

---

# 29. REQUIRED NOTEBOOK STRUCTURE

`model-1.ipynb` should contain:

```text id="blz9tb"
1. Project Objective
2. Environment
3. Inspect Available Tree-Ring Datasets
4. Candidate Dataset Quality Analysis
5. Candidate Feature Construction
6. Validation Strategy
7. Candidate Model Comparison
8. Confidence / Probability Analysis
9. Tree-Ring Dataset Selection
10. Selected Dataset Description
11. Final 80/20 Split
12. Feature Preparation
13. Random Forest Configuration
14. Model-1 Training
15. Model-1 Predictions
16. Classification Report
17. Confusion Matrix
18. Probability / Confidence Analysis
19. Feature Importance
20. Model Export
21. Model Reload Verification
22. Scientific Interpretation
23. Limitations
24. Recommendations
25. Final Conclusion
```

The notebook must execute from top to bottom from a clean kernel.

---

# 30. AUTOMATED TESTING

Create or update automated tests.

Test:

### Dataset selection

- candidate datasets discovered;
- required fields available;
- metrics calculated;
- selection result reproducible.

### Split

Verify approximately:

```text id="5f4hlh"
80% train
20% test
```

and chronological ordering.

### Leakage

Verify:

```text id="prc7li"
train/test overlap = 0
future information leakage = not detected
target leakage = not detected
```

### Model

Verify:

- training succeeds;
- predictions succeed;
- probability output is valid;
- feature importance is valid.

### Artifact

Verify:

- model saves;
- model reloads;
- predictions are reproducible.

---

# 31. RED-TEAM THE DATASET SELECTION

Actively try to make the wrong tree-ring dataset appear better.

Check whether one candidate wins because of:

- shorter but easier time period;
- class imbalance;
- missing drought years;
- fewer difficult observations;
- leakage;
- preprocessing differences;
- accidental test-set selection;
- different feature counts;
- artificially favorable missing-value handling.

If a candidate's performance appears suspiciously strong, investigate before selecting it.

---

# 32. SCIENTIFIC QA

For the selected dataset verify:

- temporal coverage is appropriate;
- tree-ring data quality is acceptable;
- RWI methodology is consistent;
- the selected site is scientifically relevant;
- features are aligned correctly with solar and SPEI data;
- target labels are correctly defined.

Do not select a dataset solely because the Random Forest score is higher.

---

# 33. MODEL QUALITY CHECK

At the end, answer explicitly:

### Which tree-ring dataset was selected?

### Why was it selected?

### What was its accuracy?

### What was its balanced accuracy?

### What was its macro F1?

### What was its Class 2 precision?

### What was its Class 2 recall?

### What was its prediction-probability behavior?

### How did it compare with the other candidates?

### What is the final Model-1 test performance?

### Is the result strong enough for the intended research purpose?

Do not fabricate any answers.

---

# 34. RECOMMENDATIONS

Based on the actual results, provide recommendations covering:

### Dataset

Whether additional tree-ring datasets should be investigated.

### Model

Whether Random Forest remains appropriate.

### Validation

Whether 80/20 evaluation is sufficient or should be supplemented with:

```text id="4h3n7s"
TimeSeriesSplit
walk-forward validation
geographic holdout
```

### Probability quality

Whether probability calibration should be considered.

### Scientific analysis

Whether more chronology/site information is needed.

### Production use

Whether the model is appropriate for the Maji Alert application or should remain a research prototype.

Do not recommend more complexity unless the observed results justify it.

---

# 35. FINAL QA — ACT LIKE A REAL ML ENGINEER

Before declaring completion, perform all of the following.

## Code QA

Run:

- formatter;
- linter;
- type checks where available;
- unit tests;
- integration tests.

## Data QA

Check:

- missing values;
- duplicates;
- year gaps;
- temporal coverage;
- class distribution;
- candidate comparability.

## ML QA

Check:

- training/test isolation;
- no leakage;
- reproducibility;
- prediction validity;
- probability validity;
- feature importance.

## Scientific QA

Check:

- tree-ring preprocessing consistency;
- target correctness;
- temporal alignment;
- scientific interpretation.

## Notebook QA

Run `model-1.ipynb` from a clean kernel.

Verify:

- no hidden state;
- no undefined variables;
- every cell runs;
- charts render;
- model trains;
- artifact is created.

---

# 36. PRODUCTION READINESS GATE

Do not call Model-1 production-ready unless:

```text id="ah4pse"
[ ] All candidate tree-ring datasets inspected
[ ] Candidate comparison completed
[ ] Selection criteria documented
[ ] Final test set protected from selection
[ ] No target leakage
[ ] No temporal leakage
[ ] Dataset quality checked
[ ] 80/20 split verified
[ ] Model actually trained inside model-1.ipynb
[ ] Metrics calculated
[ ] Confusion matrix generated
[ ] Probability analysis completed
[ ] Feature importance generated
[ ] Model saved
[ ] Model reload verified
[ ] Predictions reproducible
[ ] Automated tests pass
[ ] Notebook executes cleanly
[ ] Limitations documented
[ ] Recommendations documented
```

If any critical item fails:

```text id="ca3x5s"
NOT PRODUCTION-READY
```

Do not hide the failure.

---

# 37. REQUIRED FINAL REPORT

Return the following.

## 1. Candidate Dataset Comparison

Provide an actual table:

| Tree-Ring Dataset | Years | Observations | Accuracy | Balanced Accuracy | Macro F1 | Class 2 Precision | Class 2 Recall | Mean Predicted Probability |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

## 2. Selected Dataset

State exactly which tree-ring dataset was selected and why.

## 3. Model-1 Configuration

Report the actual Random Forest configuration.

## 4. Train/Test Split

Report:

```text
Training observations:
Testing observations:
Training percentage:
Testing percentage:
Training period:
Testing period:
```

## 5. Model-1 Results

Report:

```text
Accuracy:
Balanced Accuracy:
Macro F1:
Weighted F1:
Class 0 Precision/Recall/F1:
Class 1 Precision/Recall/F1:
Class 2 Precision/Recall/F1:
```

## 6. Confidence / Probability Analysis

Report the actual model probability diagnostics.

## 7. Feature Importance

Report the most important features.

## 8. Model Artifact

Report the exact artifact path.

## 9. Testing

Report actual:

```text
Unit tests:
Integration tests:
Red-team tests:
Notebook execution:
Lint:
Type checking:
```

## 10. Scientific Interpretation

Explain what the model results mean without claiming causation.

## 11. Limitations

Clearly state remaining limitations.

## 12. Recommendations

Provide evidence-based next steps.

## 13. Final Status

Return exactly one:

```text
MODEL-1 READY
```

or:

```text
MODEL-1 NOT READY
```

---

# NON-NEGOTIABLE RULES

1. **Inspect all available tree-ring datasets before choosing one.**
2. **Do not choose a dataset based only on accuracy.**
3. **Evaluate data quality and scientific relevance.**
4. **Do not use the final test/holdout set to select the winning dataset.**
5. **Keep candidate evaluation methodology consistent.**
6. **Do not randomly shuffle historical time-series data for the primary evaluation.**
7. **Use chronological validation where appropriate.**
8. **Do not allow target leakage.**
9. **Do not allow future-information leakage.**
10. **Do not silently change preprocessing between candidate datasets.**
11. **Do not fabricate probability/confidence results.**
12. **Do not describe Random Forest probabilities as calibrated confidence unless calibration has been performed.**
13. **Train the final Model-1 inside `model-1.ipynb`.**
14. **Do not merely load a previously trained Model-1.**
15. **Use the selected tree-ring dataset consistently throughout final training.**
16. **Actually save and reload the final model.**
17. **Verify that reloaded predictions match original predictions.**
18. **Run automated tests.**
19. **Execute the notebook from a clean kernel.**
20. **Attempt to break the dataset-selection process with red-team tests.**
21. **Report poor performance honestly.**
22. **Do not force a tree-ring dataset to win simply because the project needs a winner.**
23. **Do not claim scientific causation from predictive performance.**
24. **Do not claim nationwide generalization from limited datasets or sites.**
25. **Make recommendations based on actual evidence.**