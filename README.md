# Neurological & Cognitive Health Classification
**COMP 3608 - B-Rank Mission**

*Team 21 Members:*
- Aaron Maharaj (Project Lead / Model Architecture)
- Jonathan Dass (Data Engineering)
- Levi Ali (Baseline Models & ML Engineering)

## Problem Definition
Neurological and cognitive disorders such as Alzheimer's, Parkinson's, and Autism Spectrum Disorder (ASD) affect millions worldwide. The early detection of these conditions is paramount to improving patient outcomes and planning clinical interventions. However, diagnostic procedures (such as MRI scans) are incredibly expensive, time-consuming, and require specialized medical personnel.

The key **stakeholders** in this scenario are understaffed medical clinics and hospital triage centers. Their fundamental need is a fast, cheap, and reliable method to screen patients *before* escalating them to expensive, specialized diagnostics.

By predicting disease likelihood via cheap tabular data (voice recordings, mobile surveys, simple cognitive tests), our models provide immense **benefits**: drastically cutting hospital triage costs and accelerating the timeline for critical patient intervention.

## Datasets
| Disease | Dataset | Source |
|---|---|---|
| Alzheimer's | OASIS-2 longitudinal (MRI + non-invasive variants) | [Kaggle: MRI and Alzheimers](https://www.kaggle.com/datasets/jboysen/mri-and-alzheimers?select=oasis_longitudinal.csv) |
| Parkinson's | Sakar PD speech features (753 acoustic features) | [Kaggle: Parkinson's Disease (PD) classification](https://www.kaggle.com/datasets/dipayanbiswas/parkinsons-disease-speech-signal-features) |
| Autism (ASD) | Autism Screening on Adults (AQ-10) | [Kaggle: Autism Screening on Adults](https://www.kaggle.com/datasets/andrewmvd/autism-screening-on-adults) |

Raw CSVs live in [data/raw/](data/raw/).

## Repository Layout
```
.
├── main.py                 # Entry: CV evaluation across all datasets/models
├── train_production.py     # Trains final models on 100% data + calibrates
├── make_visuals.py         # Renders all figures from CV artifacts
├── src/
│   ├── data_loader.py      # Per-dataset loaders (returns X, y[, groups])
│   ├── preprocessing.py    # ColumnTransformer (KNN impute + scale + OHE)
│   ├── pipeline_factory.py # RandomizedSearchCV builders for LR & RF
│   ├── models_sklearn.py   # sklearn fit/score helper
│   ├── model_pytorch.py    # TabularFNN + train loop with early stopping
│   ├── evaluation.py       # Outer CV, threshold sweep, Platt calibration
│   ├── persistence.py      # Versioned joblib artifacts (model + threshold)
│   └── visualizations.py   # Pure plotting layer (no I/O)
├── data/raw/               # Raw input CSVs
├── artifacts/cv_arrays/    # Per-fold (y_true, y_score, threshold) .npz
├── models/production/      # Deployable artifacts (calibrated, with thresholds)
├── models/thresholds.json  # Per-(dataset, model) operating thresholds from CV
├── figures/                # Output PNGs (ROC, PR, calibration, DCA, SHAP, …)
└── project_results_summary.csv   # Aggregated CV metrics table
```

## Pipeline Overview

```
data/raw/*.csv
      │
      ▼
[data_loader] ──► (X, y[, groups])
      │
      ▼
[evaluation.evaluate_pipeline]   ◄── 5-fold (Stratified or GroupKFold)
      │  for each fold:
      │     1. Carve val slice (group-disjoint, class-balanced)
      │     2. Fit LR / RF (RandomizedSearchCV, recall_macro) and FNN
      │     3. Platt-calibrate val + test scores
      │     4. Sweep threshold on val (maximise balanced recall)
      │     5. Apply threshold to outer test fold
      ▼
artifacts/cv_arrays/*.npz   +   models/thresholds.json   +   project_results_summary.csv
      │
      ▼
[train_production] ──► retrain on 100% data, wrap in CalibratedClassifierCV
      │
      ▼
models/production/*.joblib  (model + threshold + cv_metrics bundle)
      │
      ▼
[make_visuals] ──► figures/*.png  (ROC, PR, calibration, DCA, threshold-cost,
                                   confusion @ swept t, RF importance, FNN SHAP)
```

## Core Functionality

### 1. Data loading ([src/data_loader.py](src/data_loader.py))
- **`load_alzheimers`** — OASIS-2 longitudinal. Each visit is one row, labelled by its concurrent CDR (`CDR > 0 → Demented`). Drops identifiers and the `Group/Visit/MR Delay` leakage columns. Returns `Subject ID` as a **group vector** so downstream splits never put visits from the same patient on both sides of the train/test boundary.
- **`load_alzheimers_noninvasive`** — same as above but drops `eTIV / nWBV / ASF` (all MRI-derived), simulating the realistic triage scenario where only demographics + the MMSE pen-and-paper test are available.
- **`load_parkinsons_v3`** — full Sakar dataset, all 3 recordings per patient retained to preserve biological variance; `id` returned as the group vector.
- **`load_autism`** — AQ-10 screening data. Drops the `result` column (literal sum of A1–A10 → direct target leakage) and the constant `age_desc` column. NaNs deferred to the preprocessing pipeline.

### 2. Preprocessing ([src/preprocessing.py](src/preprocessing.py))
A single `ColumnTransformer` shared by every model:
- **Numeric** → `KNNImputer(n_neighbors=5)` + `StandardScaler` (KNN preserves the non-linear SES/Education/Age relationships better than median imputation on small clinical datasets).
- **Low-cardinality categorical** (≤10 levels) → `"Unknown"` constant fill + `OneHotEncoder` (constant fill avoids demographic bias from mode-imputation).
- **High-cardinality categorical** → `"Unknown"` fill + `OneHotEncoder(min_frequency=0.05, max_categories=10, handle_unknown="infrequent_if_exist")`.

### 3. Models
| Model | Builder | Key choices |
|---|---|---|
| Logistic Regression | [pipeline_factory.build_lr_search](src/pipeline_factory.py) | `saga` solver, `class_weight="balanced"`, RandomizedSearchCV over L1 / L2 / ElasticNet (`l1_ratio ∈ {0, 0.25, 0.5, 0.75, 1}`), `C ∈ loguniform(1e-3, 1e1)`, scoring=`recall_macro` |
| Random Forest | [pipeline_factory.build_rf_search](src/pipeline_factory.py) | `class_weight="balanced"`, RandomizedSearchCV over `n_estimators`, `max_depth`, splits/leaves, `max_features`, `criterion`, scoring=`recall_macro` |
| Feed-Forward NN | [model_pytorch.TabularFNN](src/model_pytorch.py) | Architecture auto-sized from input dim (≤20 → 64-32, ≤100 → 128-64-32, else 256-128-64). BatchNorm + ReLU + Dropout (0.3 / 0.4 for wide inputs). `BCEWithLogitsLoss` with `pos_weight = neg/pos` to handle imbalance. AdamW + ReduceLROnPlateau + early stopping (patience 20). SMOTE applied to training data only. |

All sklearn pipelines also wrap **SMOTE** via `imblearn.pipeline.Pipeline` so oversampling happens *inside* each CV fold (no leakage).

### 4. Evaluation ([src/evaluation.py](src/evaluation.py))
The heart of the project. `evaluate_pipeline(X, y, groups, run_threshold_sweep)` performs:

1. **Outer split** — `GroupKFold(5)` when `groups` is provided, else `StratifiedKFold(5, shuffle=True, seed=67)`. Group overlap between train and test is asserted at runtime.
2. **Per-fold validation carve** ([_carve_validation_slice](src/evaluation.py)) — 10% slice that is group-disjoint, class-balanced in val, and class-balanced in sub-train. Up to 15 retries with different seeds; falls back to fixed thresholds for that fold if all attempts fail.
3. **Fit all three models** on the same sub-train so the threshold sweep is comparable across LR / RF / FNN.
4. **Platt (sigmoid) calibration** ([_calibrate_scores](src/evaluation.py)) — fitted on raw val scores, applied to both val and test scores. Necessary because SMOTE + class-weighted losses systematically miscalibrate `predict_proba`. Sigmoid (vs isotonic) is chosen because the ~30–75-sample val slice would collapse under isotonic's step-function plateaus.
5. **Threshold sweep** ([_sweep_threshold](src/evaluation.py)) — picks the threshold maximising **balanced accuracy** (= mean of sensitivity & specificity = `recall_macro` for binary, = Youden's J + 0.5). Sweep is exact over unique score boundaries when ≤500, else a 99-point linspace. Tie-break: pick the candidate closest to 0.5.
6. **Apply threshold to the outer test fold** and record accuracy, F1, precision, recall, AUC-ROC, sensitivity, specificity, balanced recall, Youden J, F2.
7. **Aggregate** mean ± std across folds; emit per-fold thresholds (median across folds becomes the deployment threshold).

`run_threshold_sweep=False` is used for **Autism Screening** because `Class/ASD` is a deterministic function of A1–A10 (sum ≥ 6) — sweeping is uninformative.

### 5. Orchestration ([main.py](main.py))
- Splits datasets into "standard" (Autism) and "grouped" (OASIS variants + Parkinson's).
- For each: load → evaluate → print per-model summary table → collect rows for CSV → save per-fold CV arrays to `artifacts/cv_arrays/<dataset>__<model>.npz` → build `thresholds.json` record.
- Final outputs: `models/thresholds.json` (consumed by `train_production.py`) and `project_results_summary.csv`.

### 6. Production training ([train_production.py](train_production.py))
- Re-trains LR / RF / FNN on **100%** of each dataset (no held-out test).
- Wraps the fitted sklearn pipelines in `CalibratedClassifierCV(method="sigmoid", cv=GroupKFold)` so deployment probabilities stay clinically interpretable.
- Loads thresholds from `models/thresholds.json` (falls back to `{LR: 0.5, RF: 0.5, FNN: 0.35}`) and **bundles** the operating threshold + CV metrics into the saved artifact via [src/persistence.py](src/persistence.py) — downstream consumers don't need to re-derive anything.

### 7. Visualizations ([make_visuals.py](make_visuals.py) + [src/visualizations.py](src/visualizations.py))
For each dataset, renders:
- **ROC** & **Precision-Recall** curves (LR / RF / FNN overlaid)
- **Calibration** curves (predicted probability vs observed frequency)
- **Decision Curve Analysis (DCA)** — net benefit vs `treat-all` / `treat-none` over the clinically relevant probability band
- **Threshold-cost curve** — `Z(t) = 5·FN + 1·FP` with both the sweep-chosen `t*` and the cost-argmin marked, exposing any gap between "what the sweep picked" and "what the cost function would prefer"
- **Confusion matrix** at the swept threshold
- **RF feature importance** (top 10, averaged across the calibrator's component pipelines)
- **FNN SHAP** (mean |SHAP| via `GradientExplainer`, top 20 features)

The plotting layer is pure (`Axes` + arrays in, no I/O), so it's testable without matplotlib backend setup.

## Reproducibility
- Global seeds: `torch.manual_seed(67)`, `np.random.seed(67)`, plus per-fold reseeding (`67 + fold`) inside `evaluate_pipeline`.
- All randomised search and SMOTE calls receive `random_state=67`.
- Group leakage is asserted at runtime; the carve will retry rather than emit a bad split.

## Running the Pipeline
Requires Python 3.12+. Dependencies declared in [pyproject.toml](pyproject.toml) (`uv sync`) or [requirements.txt](requirements.txt) (`pip install -r requirements.txt`).

```bash
# 1) Cross-validated evaluation across all datasets/models.
#    Writes: artifacts/cv_arrays/, models/thresholds.json, project_results_summary.csv
python main.py

# 2) Train final deployable artifacts on 100% of the data.
#    Writes: models/production/<dataset>_{lr,rf,fnn}.joblib
python train_production.py

# 3) Render report figures.
#    Writes: figures/<dataset>__*.png
python make_visuals.py
```

## Outputs
- `project_results_summary.csv` — flat table of mean ± std metrics per (dataset, model)
- `models/thresholds.json` — operating thresholds + per-fold values + CV metrics, keyed by `(dataset, model)`
- `models/production/*.joblib` — versioned artifacts bundling the calibrated model, decision threshold, and CV-derived metrics
- `figures/*.png` — every plot used in the report
