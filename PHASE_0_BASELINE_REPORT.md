# PHASE 0 BASELINE AUDIT REPORT
## SIH Hybrid Quantum–Classical Disease Detection Platform
**Audit Execution Date:** 2026-09-14  
**Audit Status:** COMPLETE — FOUNDATION / AUDIT ONLY  
**Branch:** `sync-quantum-model`  
**Host Environment:** Windows 11 / Python 3.12 / `.venv`

---

## 1. EXECUTIVE SUMMARY

This forensic audit establishes the definitive baseline of the **Hybrid Quantum–Classical Disease Detection Platform** repository prior to any Phase 1 modifications. The audit was conducted under strict rules of evidence: no artifact was assumed to exist based on UI claims, legacy comments, or variable names; every dataset, model, weight file, schema, and metric was verified through direct file system inspection and execution in the active project environment (`.venv`).

### Key Audit Findings:
1. **Three Genuine Biomedical Datasets Exist:**
   - **WDBC (Breast Cancer Cytology):** 569 samples, 30 numerical features, binary classification (357 Benign, 212 Malignant). Raw data file `data/raw/wdbc/wdbc.data` is present.
   - **UCI Heart Disease:** 297 complete patient records (303 raw with 6 missing values dropped), 13 clinical features, binary classification (160 Absence, 137 Presence). Raw file `data/raw/UCI/processed.cleveland.data` is present.
   - **Golub Leukemia:** 72 microarray expression samples (38 train, 34 independent test), 7,129 genes, binary lineage classification (47 ALL, 25 AML). Raw files `data_set_ALL_AML_train.csv`, `data_set_ALL_AML_independent.csv`, and `actual.csv` are present in `data/raw/golub/`.

2. **Model Inventory & Readiness Status:**
   - Across the 3 datasets, exactly **9 dataset–model combinations** exist in the project evaluation records.
   - **4 models are INFERENCE_READY:** WDBC Logistic Regression, WDBC SVM, WDBC Random Forest, and WDBC 8-Qubit VQC.
   - **1 model is LOAD_ERROR (XGBoost on WDBC):** Serialized artifact `models/xgboost.joblib` exists (272 KB), but deserialization fails with `ModuleNotFoundError: No module named 'xgboost'` because `xgboost` is absent from `.venv` and `requirements.txt`.
   - **4 models are EVALUATION_ONLY:** UCI Logistic Regression, UCI QSVC, Golub Logistic Regression, and Golub QSVC. Their performance metrics are recorded in authoritative evaluation JSONs, and preprocessing artifacts exist, but trained model weights were never serialized to disk.

3. **Quantum Stack & Circuit Reconstructibility:**
   - WDBC VQC uses an 8-qubit parameter-shift circuit: `ZZFeatureMap(8, reps=1, linear)` composed with `RealAmplitudes(8, reps=2, linear)` containing exactly 24 trainable parameters. Trained weights `results/best_model/best_vqc_weights_8q.npy` match the 24 parameters exactly and execute inference on the Qiskit Aer statevector simulator.
   - UCI and Golub QSVC use an 8-qubit `ZZFeatureMap(8, reps=1, full)` quantum kernel. While interactive inference cannot be run (due to unexported dual-alpha vectors), their quantum circuit representations are dynamically reconstructible from verified backend configurations.

4. **Critical Backend & Architectural Issues (Blockers for Phase 1):**
   - **Missing Environment Dependency:** `xgboost` is missing, preventing full classical model baseline execution.
   - **Evaluation Split Discrepancy on WDBC:** `results/classical_test_results.csv` was generated using a 60/20/20 train/val/test split (114 test samples), while `results/wdbc_metrics.json` classical baseline was generated using an 80/20 split (114 test samples), yielding slight metric divergences (97.37% vs 96.49% accuracy).
   - **Synthetic Fallback Code in Ingestion:** `src/data_loader.py` contains fallback generators (`_generate_synthetic_heart`, `_generate_synthetic_leukemia`) that produce random numbers via `np.random` if disk paths or network URLs fail.
   - **Hardcoded Cohort Statistics in Service Layer:** `src/data_service.py` hardcodes sample counts (569, 297, 72) and class counts in dictionary returns rather than dynamically querying verified datasets.

---

## 2. REPOSITORY STRUCTURE

```text
Hybrid_Quantum_Machine_Learning_Platform_for_Early_Disease_Detection/
├── .streamlit/                                  # Streamlit configuration
├── .venv/                                       # Active Python 3.12 virtual environment
├── README.md                                    # Project documentation
├── requirements.txt                             # Pinned Python package dependencies
├── audit.py                                     # Presentation-adapter audit verification script
├── audit_backend.py                             # Deep backend artifact & inference verification script
├── test_backend_full.py                         # End-to-end backend registry & inference test suite
├── test_registry.py                             # Registry discovery test
├── fresh_wdbc_eval.py                           # WDBC evaluation execution script
├── fresh_uci_eval.py                            # UCI Heart evaluation execution script
├── fresh_golub_eval.py                          # Golub Leukemia evaluation execution script
├── data/
│   ├── raw/
│   │   ├── wdbc/                                # WDBC raw data (wdbc.data, wdbc.names, WDBC_raw.csv)
│   │   ├── UCI/                                 # UCI Heart raw data (processed.cleveland.data, etc.)
│   │   └── golub/                               # Golub Leukemia raw data (train, independent, actual)
│   └── processed/                               # Pre-split CSVs (WDBC_train_8_features.csv, etc.)
├── models/                                      # Serialized classical model weights (.joblib)
│   ├── classical_scaler.joblib                  # WDBC StandardScaler (30 features)
│   ├── logistic_regression.joblib               # WDBC Logistic Regression model
│   ├── svm.joblib                               # WDBC Support Vector Classifier (RBF)
│   ├── random_forest.joblib                     # WDBC Random Forest Classifier
│   └── xgboost.joblib                           # WDBC XGBoost Classifier (unpickleable without xgboost)
├── results/
│   ├── classical_test_results.csv               # Classical models test metrics (WDBC)
│   ├── classical_validation_results.csv         # Classical models validation metrics (WDBC)
│   ├── random_forest_feature_importance.csv     # WDBC RF feature importance rankings
│   ├── wdbc_metrics.json                        # Authoritative WDBC evaluation metrics
│   ├── uci_metrics.json                         # Authoritative UCI Heart evaluation metrics
│   ├── golub_metrics.json                       # Authoritative Golub Leukemia evaluation metrics
│   ├── artifacts/                               # Preprocessing pipelines & Autoencoders
│   │   ├── wdbc_logistic_regression_preprocessing.joblib
│   │   ├── wdbc_vqc_preprocessing.joblib        # StandardScaler + PCA(8)
│   │   ├── uci_logistic_regression_preprocessing.joblib
│   │   ├── uci_qsvc_preprocessing.pt            # StandardScaler + PyTorch Autoencoder (13->8)
│   │   ├── golub_logistic_regression_preprocessing.joblib
│   │   └── golub_qsvc_preprocessing.pt          # SelectKBest(64) + StandardScaler + Autoencoder (64->8)
│   ├── best_model/
│   │   ├── best_model_metrics.json              # Historical VQC hyperparameter search record
│   │   ├── best_vqc_weights_8q.npy              # 24 trained VQC parameter weights
│   │   └── model_card.txt                       # VQC model specification
│   └── figures/                                 # Authoritative evaluation visualizations
│       ├── wdbc_clinical_radar.png              # WDBC clinical performance radar
│       ├── wdbc_classical_model_comparison.png  # WDBC 4-model classical bar chart
│       ├── wdbc_vqc_vs_classical.png            # WDBC VQC vs classical delta
│       ├── uci_clinical_radar.png               # UCI Heart clinical performance radar
│       ├── golub_clinical_radar.png             # Golub clinical performance radar
│       ├── golub_qsvc_vs_classical.png          # Golub QSVC vs classical delta
│       ├── quantum_vs_classical_delta.png       # Master cross-cohort delta plot
│       └── qxai_shap_summary.png                # Golub QSVC 8-latent QXAI SHAP summary
├── src/                                         # Core backend scientific modules
│   ├── __init__.py
│   ├── autoencoder.py                           # PyTorch dimensionality reduction autoencoder
│   ├── classical_models.py                      # Classical model wrapper class
│   ├── classical_models/
│   │   └── train_classical.py                   # Script that trained models/ joblib files
│   ├── data_loader.py                           # Data ingestion engine (contains fallbacks)
│   ├── data_service.py                          # Authoritative backend data service layer
│   ├── disagreement_protocol.py                 # Discordance & clinical triage protocol
│   ├── evaluate.py                              # Classical & quantum evaluation routines
│   ├── explainability.py                        # Model-agnostic SHAP explainability utilities
│   ├── generate_figures.py                      # Matplotlib visualization generation script
│   ├── inference.py                             # Live inference engine
│   ├── kernel_alignment.py                      # Quantum kernel target alignment
│   ├── preprocessing.py                         # Classical preprocessing pipelines
│   ├── q_xai.py                                 # Quantum Explainable AI (SHAP on Q-kernel)
│   ├── quantum_circuit.py                       # Qiskit circuit construction routines
│   ├── quantum_kernel.py                        # Qiskit QSVC kernel matrix & classifier
│   ├── quantum_model.py                         # Qiskit VQC wrapper class
│   ├── result_registry.py                       # Authoritative backend catalog & registry
│   ├── train_quantum.py                         # VQC training optimization routines
│   └── zne_mitigation.py                        # Zero-Noise Extrapolation error mitigation
└── Streamlit_dashboard/                         # User Interface layer
    ├── app.py                                   # Streamlit application entry point
    └── dashboard_core/                          # Dashboard core modules & workspaces
        ├── backend_adapter.py                   # Presentation adapter (delegates to src)
        ├── ui_components.py                     # Presentation badges & UI widgets
        └── pages/                               # 8 Research Workspaces
            ├── overview.py                      # Overview workspace
            ├── data_preview.py                  # Data exploration workspace
            ├── models.py                        # Models landscape workspace
            ├── quantum_model.py                 # Architecture workspace
            ├── prediction.py                    # Interactive prediction workspace
            ├── evaluation.py                    # Analytical evaluation workspace
            ├── benchmark.py                     # Comparative benchmarks workspace
            └── explainability.py                # Explainability workspace
```

---

## 3. DATASET INVENTORY

| Dataset Key | Display Name | Source / Origin | Data File on Disk | Sample Count | Feature Count | Target Classes | Train / Test Split | Verification Status |
|---|---|---|---|---|---|---|---|---|
| `wdbc` | Solid Oncology — WDBC Breast Cancer Cytology | University of Wisconsin Hospitals (FNA biopsy) | `data/raw/wdbc/wdbc.data` (124,672 B) | 569 | 30 continuous | Benign: 357 (62.7%)<br>Malignant: 212 (37.3%) | 60% Train (341)<br>20% Val (114)<br>20% Test (114) | **VERIFIED** |
| `uci` | Cardiology — UCI Ischemic Heart EHR Telemetry | Cleveland Clinic Foundation EHR | `data/raw/UCI/processed.cleveland.data` (18,764 B) | 297 complete<br>(303 raw, 6 missing dropped) | 13 clinical | Absence: 160 (53.9%)<br>Presence: 137 (46.1%) | 75% Train (222)<br>25% Test (75) Stratified | **VERIFIED** |
| `golub` | Hematologic Oncology — Golub Leukemia Genomics | Whitehead Institute / MIT Center for Genome Research | `data/raw/golub/data_set_ALL_AML_train.csv`<br>`data_set_ALL_AML_independent.csv`<br>`data/raw/golub/actual.csv` | 72 total<br>(38 train, 34 test) | 7,129 gene expression probes | ALL: 47 (65.3%)<br>AML: 25 (34.7%) | 38 Train (27 ALL, 11 AML)<br>34 Independent Test (20 ALL, 14 AML) | **VERIFIED** |

*Note on counts:* While raw files on disk independently confirm sample counts of 569, 297, and 72, `src/data_service.py` currently returns these as hardcoded static integer literals rather than computing `len(df)` dynamically.

---

## 4. MODEL INVENTORY

| Dataset Key | Model Key | Display Name | Model Type | Implementation Location | Weight Artifact Location | Artifact Exists | Artifact Loadable | Evaluated | Inference Status | Explainability | Quantum Config |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `wdbc` | `logistic_regression` | Logistic Regression | Classical | `src/classical_models/train_classical.py` | `models/logistic_regression.joblib` | YES | YES | YES | **INFERENCE_READY** | UNAVAILABLE | UNAVAILABLE |
| `wdbc` | `svm` | Support Vector Machine (SVM) | Classical | `src/classical_models/train_classical.py` | `models/svm.joblib` | YES | YES | YES | **INFERENCE_READY** | UNAVAILABLE | UNAVAILABLE |
| `wdbc` | `random_forest` | Random Forest | Classical | `src/classical_models/train_classical.py` | `models/random_forest.joblib` | YES | YES | YES | **INFERENCE_READY** | **VERIFIED** (Gini CSV) | UNAVAILABLE |
| `wdbc` | `xgboost` | XGBoost | Classical | `src/classical_models/train_classical.py` | `models/xgboost.joblib` | YES | **NO** (ModuleNotFoundError) | YES | **LOAD_ERROR** | UNAVAILABLE | UNAVAILABLE |
| `wdbc` | `vqc` | 8-Qubit VQC | Quantum | `src/quantum_model.py`<br>`src/quantum_circuit.py` | `results/best_model/best_vqc_weights_8q.npy` | YES | YES | YES | **INFERENCE_READY** | UNAVAILABLE | **VERIFIED** |
| `uci` | `logistic_regression` | Logistic Regression | Classical | `fresh_uci_eval.py` | None (weights not exported) | NO | N/A | YES | **EVALUATION_ONLY** | UNAVAILABLE | UNAVAILABLE |
| `uci` | `qsvc` | Quantum QSVC (8-qubit) | Quantum | `src/quantum_kernel.py` | None (dual vectors not exported) | NO | N/A | YES | **EVALUATION_ONLY** | UNAVAILABLE | **VERIFIED** |
| `golub` | `logistic_regression` | Logistic Regression | Classical | `fresh_golub_eval.py` | None (weights not exported) | NO | N/A | YES | **EVALUATION_ONLY** | UNAVAILABLE | UNAVAILABLE |
| `golub` | `qsvc` | Quantum QSVC (8-qubit) | Quantum | `src/quantum_kernel.py` | None (dual vectors not exported) | NO | N/A | YES | **EVALUATION_ONLY** | **VERIFIED** (QXAI SHAP) | **VERIFIED** |

---

## 5. ARTIFACT INVENTORY

### Serialized Model Weights & Pipelines
- `models/classical_scaler.joblib`: 2,071 bytes. `sklearn.preprocessing.StandardScaler` fitted on 30 WDBC features. **VERIFIED LOADABLE**.
- `models/logistic_regression.joblib`: 1,103 bytes. `sklearn.linear_model.LogisticRegression` (C=1.0, lbfgs). **VERIFIED LOADABLE**.
- `models/svm.joblib`: 24,603 bytes. `sklearn.svm.SVC` (kernel='rbf', probability=True, C=1.0). **VERIFIED LOADABLE**.
- `models/random_forest.joblib`: 819,849 bytes. `sklearn.ensemble.RandomForestClassifier` (n_estimators=100). **VERIFIED LOADABLE**.
- `models/xgboost.joblib`: 272,277 bytes. Serialized XGBoost model. **LOAD FAILED**: `ModuleNotFoundError: No module named 'xgboost'`.
- `results/best_model/best_vqc_weights_8q.npy`: 320 bytes. `numpy.ndarray` with shape `(24,)`, dtype `float64`. **VERIFIED LOADABLE**.
- `results/artifacts/wdbc_vqc_preprocessing.joblib`: 4,353 bytes. Dictionary with keys `['scaler', 'pca']`. **VERIFIED LOADABLE**.
- `results/artifacts/wdbc_logistic_regression_preprocessing.joblib`: 1,320 bytes. Dictionary with key `['scaler']`. **VERIFIED LOADABLE**.
- `results/artifacts/uci_logistic_regression_preprocessing.joblib`: 896 bytes. Dictionary with key `['scaler']`. **VERIFIED LOADABLE**.
- `results/artifacts/uci_qsvc_preprocessing.pt`: 8,217 bytes. PyTorch checkpoint with keys `['scaler', 'autoencoder_state']`. **VERIFIED LOADABLE**.
- `results/artifacts/golub_logistic_regression_preprocessing.joblib`: 116,465 bytes. Dictionary with keys `['selector', 'scaler']`. **VERIFIED LOADABLE**.
- `results/artifacts/golub_qsvc_preprocessing.pt`: 196,805 bytes. PyTorch checkpoint with keys `['selector', 'scaler', 'autoencoder_state']`. **VERIFIED LOADABLE**.

### Visual Benchmark Figures
- `results/figures/wdbc_clinical_radar.png`: 135,138 bytes. **VERIFIED EXISTS**.
- `results/figures/wdbc_classical_model_comparison.png`: 446,825 bytes. **VERIFIED EXISTS**.
- `results/figures/wdbc_vqc_vs_classical.png`: 105,600 bytes. **VERIFIED EXISTS**.
- `results/figures/uci_clinical_radar.png`: 120,433 bytes. **VERIFIED EXISTS**.
- `results/figures/golub_clinical_radar.png`: 120,938 bytes. **VERIFIED EXISTS**.
- `results/figures/golub_qsvc_vs_classical.png`: 118,032 bytes. **VERIFIED EXISTS**.
- `results/figures/quantum_vs_classical_delta.png`: 95,537 bytes. **VERIFIED EXISTS**.
- `results/figures/qxai_shap_summary.png`: 38,461 bytes. **VERIFIED EXISTS**.

---

## 6. PREPROCESSING INVENTORY

| Dataset / Model Pipeline | Preprocessing Sequence | Input Dimension | Output Dimension | Fit on Train Only? | Required for Inference? | Verification Status |
|---|---|---|---|---|---|---|
| **WDBC Classical**<br>(LR, SVM, RF, XGB) | `StandardScaler` | 30 | 30 | YES (fit on 341 train samples) | YES | **VERIFIED** |
| **WDBC VQC** | `StandardScaler` → `PCA(8, random_state=42)` → `np.tanh(X) * np.pi` | 30 | 8 | YES (fit on train set) | YES | **VERIFIED** |
| **UCI Classical** | `StandardScaler` | 13 | 13 | YES (fit on 222 train samples) | YES (if deployed) | **VERIFIED** |
| **UCI QSVC** | `StandardScaler` → `Autoencoder(13 → 8, PyTorch, 100 epochs)` | 13 | 8 | YES (fit on train set) | YES (if deployed) | **VERIFIED** |
| **Golub Classical** | `SelectKBest(f_classif, k=64)` → `StandardScaler` | 7,129 | 64 | YES (fit on 38 train samples) | YES (if deployed) | **VERIFIED** |
| **Golub QSVC** | `SelectKBest(64)` → `StandardScaler` → `Autoencoder(64 → 8, PyTorch, 100 epochs)` | 7,129 | 8 | YES (fit on 38 train samples) | YES (if deployed) | **VERIFIED** |

*Note on WDBC VQC Tanh Scaling:* In `fresh_wdbc_eval.py:61-62` and `src/inference.py:153`, the PCA components are scaled as `np.tanh(X_pca) * np.pi`. This bounds the quantum rotation angles to $(-\pi, \pi)$ for the Qiskit `ZZFeatureMap`.

---

## 7. FEATURE SCHEMA INVENTORY

| Dataset Key | Feature Names Available | Feature Count | Expected Input Type | Source of Schema | Verification Status |
|---|---|---|---|---|---|
| `wdbc` | YES (30 exact clinical names) | 30 | `float64` (continuous real) | `data/raw/wdbc/wdbc.names`<br>`src/classical_models/train_classical.py:93-132` | **VERIFIED AUTHORITATIVE** |
| `uci` | YES (13 clinical names) | 13 | `float64` / integer categorical | `data/raw/UCI/heart-disease.names` | **VERIFIED AUTHORITATIVE** |
| `golub` | PARTIAL (7,129 probe IDs in raw CSV; fallback generates `Gene_1`..`Gene_7129`) | 7,129 raw<br>64 selected | `float64` (continuous expression) | Raw CSV header in `data/raw/golub/data_set_ALL_AML_train.csv` | **SCHEMA UNSTABLE / FALLBACK RELIANT** |

### Schema Vulnerability Finding:
In `src/data_service.py:22-29`, the WDBC schema is retrieved by dynamically importing `from sklearn.datasets import load_breast_cancer` rather than parsing `data/raw/wdbc/wdbc.names` or `data/raw/wdbc/wdbc.data`. For Golub, `src/data_loader.py:130,135,139` synthesizes artificial names `[f"Gene_{i+1}" for i in range(7129)]` instead of reading the gene accessions from `data/raw/golub/data_set_ALL_AML_train.csv`.

---

## 8. EVALUATION ARTIFACT INVENTORY

### Metric Completeness by Model
| Dataset | Model | Accuracy | Precision | Sensitivity | Specificity | F1 Score | ROC-AUC | MCC | Confusion Matrix |
|---|---|---|---|---|---|---|---|---|---|
| `wdbc` | Logistic Regression (CSV) | 0.9737 | 0.9756 | 0.9524 | 0.9861 | 0.9639 | 0.9954 | 0.9419 | `[[71, 1], [2, 40]]` |
| `wdbc` | Logistic Regression (JSON) | 0.9649 | 0.9750 | 0.9286 | 0.9861 | 0.9512 | 0.9960 | 0.9245 | `[[71, 1], [3, 39]]` |
| `wdbc` | SVM | 0.9737 | 0.9756 | 0.9524 | 0.9861 | 0.9639 | 0.9940 | 0.9419 | `[[71, 1], [2, 40]]` |
| `wdbc` | Random Forest | 0.9561 | 0.9744 | 0.9048 | 0.9861 | 0.9383 | 0.9967 | 0.9048 | `[[71, 1], [4, 38]]` |
| `wdbc` | XGBoost | 0.9737 | 1.0000 | 0.9286 | 1.0000 | 0.9630 | 0.9950 | 0.9442 | `[[72, 0], [3, 39]]` |
| `wdbc` | 8-Qubit VQC | 0.7368 | 0.6154 | 0.7619 | 0.7222 | 0.6809 | 0.7801 | 0.4689 | `[[52, 20], [10, 32]]` |
| `uci` | Logistic Regression | 0.8533 | 0.8529 | 0.8286 | 0.8750 | 0.8406 | 0.9168 | 0.7047 | `[[35, 5], [6, 29]]` |
| `uci` | Quantum QSVC (8Q) | 0.8133 | 0.8000 | 0.8000 | 0.8250 | 0.8000 | null | 0.6250 | `[[33, 7], [7, 28]]` |
| `golub` | Logistic Regression | 0.8529 | 1.0000 | 0.6429 | 1.0000 | 0.7826 | 0.8929 | null | `[[20, 0], [5, 9]]` |
| `golub` | Quantum QSVC (8Q) | 0.8824 | 1.0000 | 0.7143 | 1.0000 | 0.8333 | null | null | `[[20, 0], [4, 10]]` |

*Preservation of Unavailable Values:* As verified above, missing values (such as ROC-AUC on QSVC models, which do not output continuous posterior probabilities without calibration, or MCC on Golub) remain `null` rather than being converted to zero.

---

## 9. EXPLAINABILITY INVENTORY

| Dataset / Model | Artifact File | Method | Feature Space Explained | Status |
|---|---|---|---|---|
| `golub` / `qsvc` | `results/figures/qxai_shap_summary.png` | `shap.KernelExplainer` using `shap.kmeans(k=15)` | 8 Autoencoder latent components (`Latent_AE_Q0`–`Latent_AE_Q7`) | **VERIFIED (Quantum Post-Hoc Attribution)** |
| `wdbc` / `random_forest` | `results/random_forest_feature_importance.csv` | Gini impurity feature importance | 30 raw cytological descriptors | **VERIFIED (Tree Feature Attribution)** |
| `wdbc` / `vqc` | None | None | None | **UNAVAILABLE** |
| `uci` / `qsvc` | None | None | None | **UNAVAILABLE** |
| All other pairs | None | None | None | **UNAVAILABLE** |

### Explainability Methodology Classification:
- `results/figures/qxai_shap_summary.png` is **QUANTUM POST-HOC ATTRIBUTION** applied to the 8 compressed latent autoencoder features feeding the quantum kernel. It does **not** attribute importance back to individual Affymetrix genes.
- `results/random_forest_feature_importance.csv` is **TREE-BASED GINI ATTRIBUTION** across the original 30 cytology features.

---

## 10. QUANTUM MODEL INVENTORY

| Model | Dataset | Qubits | Feature Map | Ansatz / Kernel | Trainable Params | Simulator / Backend | Optimized Threshold ($\tau$) | Circuit Reconstruction |
|---|---|---|---|---|---|---|---|---|
| **VQC** | `wdbc` | 8 | `ZZFeatureMap`<br>(reps=1, linear) | `RealAmplitudes`<br>(reps=2, linear) | 24 | `QMLSampler` / Aer Statevector | 0.65 | **VERIFIED** (Dynamic reconstruction via Qiskit) |
| **QSVC** | `uci` | 8 | `ZZFeatureMap`<br>(reps=1, full) | `QuantumKernelClassifier`<br>(C=10.0, scale=0.05) | 0 (Dual $\alpha$ vectors) | Qiskit Aer Kernel | 0.50 | **VERIFIED** (Feature map reconstructible) |
| **QSVC** | `golub` | 8 | `ZZFeatureMap`<br>(reps=1, full) | `QuantumKernelClassifier`<br>(C=10.0, scale=0.05) | 0 (Dual $\alpha$ vectors) | Qiskit Aer Kernel | 0.50 | **VERIFIED** (Feature map reconstructible) |

---

## 11. INFERENCE READINESS MATRIX

| Classification | Count | Models Included | Criteria Justification |
|---|---|---|---|
| **INFERENCE_READY** | 4 | `wdbc:logistic_regression`<br>`wdbc:svm`<br>`wdbc:random_forest`<br>`wdbc:vqc` | Validated model weights exist, scaler/preprocessing artifacts exist, runtime dependencies are installed, and test inference executes successfully. |
| **LOAD_ERROR** | 1 | `wdbc:xgboost` | Artifact `models/xgboost.joblib` exists, but deserialization fails due to uninstalled dependency `xgboost`. |
| **EVALUATION_ONLY** | 4 | `uci:logistic_regression`<br>`uci:qsvc`<br>`golub:logistic_regression`<br>`golub:qsvc` | Rigorous evaluation metrics and preprocessing artifacts exist, but trained model weights were not serialized to disk. |
| **EXPERIMENTAL** | 0 | None | No unverified experimental model stubs are active in the registry. |
| **UNAVAILABLE** | 0 | None | Missing models are not falsely advertised in the catalog. |

---

## 12. BACKEND REGISTRY AUDIT

1. **Authoritative Registry:** `src/result_registry.py` is the single authoritative source for the presentation layer. It discovers and formats model metadata, capabilities, protocols, and artifacts.
2. **Hardcoded Mappings:** `DATASET_MODELS` (lines 28–44 in `src/result_registry.py`) statically enumerates the 9 models rather than scanning the directory dynamically. However, it accurately reflects the actual evaluation outputs present on disk.
3. **Competing Registries / Metadata:**
   - `results/best_model/best_model_metrics.json` records an older optimization run of VQC.
   - `src/classical_models.py` defines a legacy `ClassicalBaselines` wrapper not used by the current registry.
4. **Evaluation Discrepancy on WDBC:** `src/result_registry.py:128-175` reads WDBC classical model metrics from `results/classical_test_results.csv`, while `wdbc:vqc` metrics are read from `results/wdbc_metrics.json`.
5. **No Synthetic Fallbacks in Registry:** `result_registry.py` sets `"inference_ready": False` and provides an explicit `"offline_reason"` for models lacking weights, preventing silent mock predictions.

---

## 13. DASHBOARD INTEGRATION AUDIT

| Page Module | File Path | Integration Status | Data Origin |
|---|---|---|---|
| **Overview** | `dashboard_core/pages/overview.py` | **BACKEND-DERIVED** | `backend.get_dataset_metadata()`, `backend.get_available_models()`, `backend.get_model_metadata()` |
| **Data Explorer** | `dashboard_core/pages/data_preview.py` | **MIXED** | Schema and sample records from `data_service`; class counts hardcoded in `data_service` |
| **Models Landscape** | `dashboard_core/pages/models.py` | **BACKEND-DERIVED** | All cards, badges, and capability checklists derived from `backend_adapter` |
| **Architecture** | `dashboard_core/pages/quantum_model.py` | **BACKEND-DERIVED** | Pipeline flow, hyperparameter table, and Qiskit circuit text drawn from `backend.get_quantum_circuit()` |
| **Prediction** | `dashboard_core/pages/prediction.py` | **BACKEND-DERIVED** | Inputs validated against schema; executes `backend.predict_instance()`; renders offline cards for non-ready models |
| **Evaluation** | `dashboard_core/pages/evaluation.py` | **BACKEND-DERIVED** | KPI metrics, confusion matrices, and ROC plots loaded from backend results |
| **Benchmarks** | `dashboard_core/pages/benchmark.py` | **BACKEND-DERIVED** | Model comparison table and visual radar figures loaded from backend artifacts |
| **Explainability** | `dashboard_core/pages/explainability.py` | **BACKEND-DERIVED** | Displays verified QXAI SHAP summary for Golub QSVC, Gini importance for WDBC RF; displays explicit unavailable card for all others |

---

## 14. HARDCODING AUDIT

| Location | Category | Code Snippet / Value | Finding Description |
|---|---|---|---|
| `src/data_service.py:70-76` | C (Static scientific value) | `"total_samples": 569, "classes": {"Benign": 357, "Malignant": 212}` | Hardcoded WDBC class counts in backend data service |
| `src/data_service.py:85-91` | C (Static scientific value) | `"total_samples": 297, "classes": {"Absence": 160, "Presence": 137}` | Hardcoded UCI class counts in backend data service |
| `src/data_service.py:100-107` | C (Static scientific value) | `"total_samples": 72, "classes": {"ALL": 47, "AML": 25}` | Hardcoded Golub class counts in backend data service |
| `src/data_loader.py:31-51` | E (Suspicious / synthetic fallback) | `np.random.randint(...)`, `risk_score = 0.03*age + ...` | Synthetic heart disease EHR generator triggered on network failure |
| `src/data_loader.py:56-66` | E (Suspicious / synthetic fallback) | `np.random.randn(72, 7129) * 500.0 + 1000.0` | Synthetic leukemia gene matrix generator triggered on missing CSVs |
| `src/data_loader.py:135,139` | E (Mock feature names) | `feature_names = [f"Gene_{i+1}" for i in range(7129)]` | Synthesizes artificial gene names rather than reading CSV headers |
| `src/quantum_model.py:187-192` | E (Hard approximation) | `proba = np.column_stack([1 - preds, preds]).astype(float)` | `predict_proba` returns hard step values instead of continuous probabilities |
| `Streamlit_dashboard/dashboard_core/pages/prediction.py:126` | D (Legitimate UI default) | `default_val = 0.0` | Fallback form field initial value |

---

## 15. TERMINOLOGY AUDIT

Occurrences of clinical and medical terms across python source files:
- **`patient`**:
  - `Streamlit_dashboard/dashboard_core/pages/prediction.py:93`: `"Enter patient feature telemetry or load a representative cohort sample."`
  - `Streamlit_dashboard/dashboard_core/pages/evaluation.py:103`: `ax.set_ylabel('True Patient Label', ...)`
  - `src/disagreement_protocol.py:6,20,22,73`: Repeated references to `"patient cases"`, `"triage_patient_case"`, `"Total Evaluated Patient Cohort: {n_samples} patients"`.
- **`biopsy`**:
  - `src/disagreement_protocol.py:10,45`: `"🟡 Targeted Biopsy / Multi-Disciplinary Review"`.
- **`quantum superiority` / `advantage`**:
  - `src/generate_figures.py:106`: `ax2.set_title(r'Quantum Performance Advantage $\Delta$ ...')`.
  - `Streamlit_dashboard/dashboard_core/pages/benchmark.py:135`: `"No claims of absolute quantum superiority are implied."` (Explicit disclaimer).

---

## 16. DEPENDENCY AUDIT

| Package Name | Required For | Specified in `requirements.txt`? | Installed in `.venv`? | Installed Version | Status |
|---|---|---|---|---|---|
| `streamlit` | Dashboard UI | `==1.40.1` | YES | 1.40.1 | **OK** |
| `qiskit` | Quantum Circuit & Stack | `==2.5.2` | YES | 2.5.2 | **OK** |
| `qiskit-aer` | Quantum Simulator | `==0.17.2` | YES | 0.17.2 | **OK** |
| `qiskit-machine-learning`| VQC & Quantum Algorithms | `==0.9.1` | YES | 0.9.1 | **OK** |
| `qiskit-algorithms` | Quantum Optimizers | `==0.4.0` | YES | 0.4.0 | **OK** |
| `scikit-learn` | Classical ML & Preprocessing | `==1.9.1` | YES | 1.9.1 | **OK** (Version warning on 1.9.0 unpickle) |
| `numpy` | Numerical arrays | `==2.1.2` | YES | 2.1.2 | **OK** |
| `pandas` | Tabular data processing | `==2.2.2` | YES | 2.2.2 | **OK** |
| `torch` | Autoencoder dimensionality reduction | `>=2.2.0` | YES | 2.14.0 | **OK** |
| `shap` | QXAI Explainability | `==0.52.0` | YES | 0.52.0 | **OK** |
| `matplotlib` | Figure generation & display | `==3.11.2` | YES | 3.11.2 | **OK** |
| `seaborn` | Heatmap plotting | `==0.13.2` | YES | 0.13.2 | **OK** |
| `plotly` | Interactive chart widgets | `==5.24.1` | YES | 5.24.1 | **OK** |
| `pylatexenc` | Qiskit circuit drawing | `==2.11` | YES | 2.11 | **OK** |
| `joblib` | Model & pipeline deserialization | Implicit | YES | 1.6.0 | **OK** |
| `xgboost` | WDBC XGBoost inference | **NO** | **NO** | None | **MISSING (LOAD_ERROR)** |

---

## 17. EXISTING TEST RESULTS

Three existing test and audit scripts were executed in `.venv`:
1. **`audit.py`**:
   - Command: `.venv\Scripts\python.exe audit.py`
   - Exit Code: `0`
   - Verified that `get_dataset_catalog()` returns all 3 datasets, model capabilities match backend state, and preprocessing/protocol metadata serialize properly.
2. **`audit_backend.py`**:
   - Command: `.venv\Scripts\python.exe audit_backend.py`
   - Exit Code: `0`
   - Verified all 9 models. Emitted `ModuleNotFoundError` for `xgboost.joblib` and unpickling version warnings (`StandardScaler` fitted with 1.9.0, running 1.9.1).
3. **`test_backend_full.py`**:
   - Command: `.venv\Scripts\python.exe test_backend_full.py`
   - Exit Code: `0`
   - Tested registry discovery and executed live inference on all 4 inference-ready models (`logistic_regression`, `svm`, `random_forest`, `vqc`). All 4 returned correct predicted classes and confidence scores.

---

## 18. RUNTIME ISSUES DISCOVERED

1. **XGBoost Unpickling Failure:**
   - Attempting to load `models/xgboost.joblib` raises `ModuleNotFoundError: No module named 'xgboost'`.
2. **Scikit-Learn Version Warnings:**
   - Joblib files saved with scikit-learn 1.9.0 trigger `InconsistentVersionWarning` when loaded under scikit-learn 1.9.1.
3. **Streamlit Port Binding Warning:**
   - Streamlit background process was previously launched on port 8501. Headless service operates stably.

---

## 19. STALE & DUPLICATE IMPLEMENTATION FINDINGS

1. **Deleted Directory in Working Tree:**
   - `Streamlit_dashboard/src/` has 10 deleted files staged/unstaged in git status. The new architecture is located in `Streamlit_dashboard/dashboard_core/`.
2. **Duplicate Classical Baseline Wrappers:**
   - `src/classical_models.py` duplicates functionality now residing in `src/classical_models/train_classical.py` and `src/result_registry.py`.
3. **Unused Root Scripts:**
   - `quick_quantum_test.py`, `run_fast_qsvc.py`, `tune_qsvc_c.py`, `save_best.py` in the root directory represent ad-hoc development experiments that should be archived.

---

## 20. EXACT BLOCKERS FOR PHASE 1

To proceed from Phase 0 to Phase 1, the following items are established as the mandatory scope:
1. **Install `xgboost` into `.venv` and update `requirements.txt`** so that `models/xgboost.joblib` can be deserialized and verified for inference.
2. **Align WDBC Classical Split Documentation:** Harmonize `results/classical_test_results.csv` and `results/wdbc_metrics.json` protocol definitions.
3. **Eliminate Synthetic Fallbacks:** Remove `_generate_synthetic_heart` and `_generate_synthetic_leukemia` in `src/data_loader.py`, replacing them with strict assertions requiring the verified local raw files.
4. **Decouple Data Service Hardcoding:** Update `src/data_service.py` to derive sample and class counts dynamically from raw data files.
5. **Clean Repository Working Tree:** Safely remove obsolete legacy files from `Streamlit_dashboard/src/` and ad-hoc root scripts without affecting Git commit history.

---
