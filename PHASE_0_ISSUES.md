# PHASE 0 — ISSUE REGISTER
## SIH Hybrid Quantum–Classical Disease Detection Platform

**Audit Status:** Active Baseline Findings  
**Action Rule:** DO NOT FIX IN PHASE 0 — RECORD AND AUDIT ONLY

---

### Priority Classification Legend:
- **P0:** Prevents reliable execution or causes runtime crash / import error.
- **P1:** Important scientific or pipeline defect (data leakage, split divergence, synthetic fallbacks, approximation errors).
- **P2:** Important integration or architectural defect (hardcoded registries, static service data, decoupled adapters).
- **P3:** Codebase cleanup, dead script pruning, and clinical terminology adjustments.

---

## 1. P0 ISSUES (Operational Blockers)

### [ISSUE P0-01] Missing `xgboost` Dependency in Python Environment
- **File:** `requirements.txt` / `.venv` environment
- **Location:** Global environment / `models/xgboost.joblib`
- **Description:** The `xgboost` package is neither pinned in `requirements.txt` nor installed in `.venv`, preventing deserialization of `models/xgboost.joblib`.
- **Evidence:** Running `joblib.load('models/xgboost.joblib')` raises:
  ```text
  ModuleNotFoundError: No module named 'xgboost'
  ```
- **Impact:** WDBC XGBoost cannot be loaded or run for live inference, forcing it into `LOAD_ERROR` status despite existing model weights and CSV test metrics.
- **Recommended Phase:** Phase 1 (Dependency & Environment Standardization).

---

### [ISSUE P0-02] Unserialized Support Vector Weights for UCI QSVC & Golub QSVC
- **File:** `fresh_uci_eval.py:93` and `fresh_golub_eval.py:122`
- **Location:** Pipeline persistence stage
- **Description:** Evaluation scripts for UCI and Golub train `QuantumKernelClassifier` and serialize the preprocessing scalers and PyTorch autoencoder weights (`.pt`), but do NOT serialize the trained `QuantumKernelClassifier` instance, support vectors, or dual-alpha coefficients.
- **Evidence:** `results/artifacts/uci_qsvc_preprocessing.pt` contains only `scaler` and `autoencoder_state`. The metadata file `results/uci_metrics.json` records `"weights": null`.
- **Impact:** UCI and Golub QSVC models cannot be executed for interactive prediction in the dashboard and remain permanently constrained to `EVALUATION_ONLY`.
- **Recommended Phase:** Phase 1 / Phase 2 (Backend Artifact Export Enhancement).

---

## 2. P1 ISSUES (Scientific & Pipeline Defects)

### [ISSUE P1-01] Synthetic / Random Number Fallbacks in Clinical Ingestion Engine
- **File:** `src/data_loader.py`
- **Location:** Lines 29–67 (`_generate_synthetic_heart`, `_generate_synthetic_leukemia`) and Lines 105, 134, 138
- **Description:** When remote URL download fails for UCI Heart, or when local files are not found at `data/data_set_ALL_AML_train.csv`, `load_clinical_data()` silently falls back to generating random synthetic integers and Gaussian distributions via `np.random`.
- **Evidence:**
  ```python
  # src/data_loader.py:104-105
  except Exception as e:
      print(f"[DataLoader] Network/URL unavailable ({e}). Using verified fallback Heart dataset generator.")
      X, y = _generate_synthetic_heart(seed=seed)
  ```
- **Impact:** Risk of silent scientific invalidation if data directory paths are misconfigured; models could train on synthetic random data rather than genuine patient cohorts.
- **Recommended Phase:** Phase 1 (Strict Deterministic Ingestion Enforcement).

---

### [ISSUE P1-02] Split Protocol Discrepancy on WDBC Classical Baseline
- **File:** `src/classical_models/train_classical.py:208-220` vs `fresh_wdbc_eval.py:41`
- **Location:** Dataset split definitions
- **Description:** WDBC classical models in `models/*.joblib` and `results/classical_test_results.csv` were trained with a 60% Train / 20% Validation / 20% Test split (114 test samples, LR Accuracy: 97.37%). Conversely, `fresh_wdbc_eval.py` evaluated an 80% Train / 20% Test split (114 test samples, LR Accuracy: 96.49%), and wrote this divergent metric to `results/wdbc_metrics.json`.
- **Evidence:**
  - `results/classical_test_results.csv`: Logistic Regression Accuracy = 0.97368, CM = `[[71, 1], [2, 40]]`.
  - `results/wdbc_metrics.json`: Logistic Regression Accuracy = 0.96491, CM = `[[71, 1], [3, 39]]`.
- **Impact:** Presentation layer displays divergent accuracy and confusion matrix values depending on whether it queries `classical_test_results.csv` or `wdbc_metrics.json`.
- **Recommended Phase:** Phase 1 (Scientific Registry Consolidation).

---

### [ISSUE P1-03] Hard Step Probability Approximation in Quantum Model Wrapper
- **File:** `src/quantum_model.py`
- **Location:** Lines 180–193 (`QuantumClassifier.predict_proba`)
- **Description:** `predict_proba()` in `src/quantum_model.py` constructs pseudo-probabilities by thresholding `preds = self.vqc_.predict(X)` and stacking `[1 - preds, preds]`.
- **Evidence:**
  ```python
  # src/quantum_model.py:190-192
  preds = self.vqc_.predict(X)
  proba = np.column_stack([1 - preds, preds]).astype(float)
  return proba
  ```
- **Impact:** Downstream explainability routines (e.g. `shap.KernelExplainer`) receive step functions rather than smooth posterior expectation values, distorting gradient/marginal perturbation calculations.
- **Recommended Phase:** Phase 1 (Quantum Probability Calibration).

---

### [ISSUE P1-04] Synthetic Feature Names Synthesized for Golub Genomics
- **File:** `src/data_loader.py`
- **Location:** Lines 130, 135, 139
- **Description:** When loading the Golub microarray dataset, `load_clinical_data()` assigns feature names as `[f"Gene_{i+1}" for i in range(7129)]` instead of parsing gene accession names from the raw CSV header in `data/raw/golub/data_set_ALL_AML_train.csv`.
- **Evidence:**
  ```python
  feature_names = [f"Gene_{i+1}" for i in range(X.shape[1])]
  ```
- **Impact:** Biological interpretability is lost; features appear as anonymous numbers (`Gene_1` to `Gene_7129`) rather than recognized oncological markers.
- **Recommended Phase:** Phase 1 (Schema & Header Alignment).

---

## 3. P2 ISSUES (Integration & Architectural Defects)

### [ISSUE P2-01] Hardcoded Dataset Characteristics in Data Service Layer
- **File:** `src/data_service.py`
- **Location:** Lines 60–111 (`get_dataset_characteristics`)
- **Description:** Sample counts (569, 297, 72) and class counts (357/212, 160/137, 47/25) are hardcoded as static integer literals in Python dictionaries rather than queried from verified datasets.
- **Evidence:**
  ```python
  # src/data_service.py:70-75
  "total_samples": 569,
  "feature_count": 30,
  "classes": {"Benign": 357, "Malignant": 212},
  ```
- **Impact:** Violates the single-source-of-truth rule; if raw dataset files change, the service layer displays stale or incorrect statistics.
- **Recommended Phase:** Phase 1 (Dynamic Dataset Introspection).

---

### [ISSUE P2-02] Sklearn Ingestion Coupling for WDBC Feature Schema
- **File:** `src/data_service.py`
- **Location:** Lines 20–31 (`get_authoritative_schema`)
- **Description:** To fetch the WDBC feature names, `data_service.py` imports `sklearn.datasets.load_breast_cancer()` rather than parsing the local authoritative repository schema file `data/raw/wdbc/wdbc.names`.
- **Evidence:**
  ```python
  # src/data_service.py:22-26
  from sklearn.datasets import load_breast_cancer
  raw = load_breast_cancer()
  return {"feature_names": list(raw.feature_names), ...}
  ```
- **Impact:** Unnecessary external dependency coupling; fails to verify that the local file `data/raw/wdbc/wdbc.data` matches the Scikit-Learn built-in version.
- **Recommended Phase:** Phase 1 (Local File Schema Parser).

---

### [ISSUE P2-03] Static Model Catalog in Result Registry
- **File:** `src/result_registry.py`
- **Location:** Lines 28–44 (`DATASET_MODELS`)
- **Description:** Available models are enumerated in a hardcoded dictionary mapping rather than discovered dynamically through artifact introspection.
- **Evidence:**
  ```python
  DATASET_MODELS = {
      "wdbc": {"logistic_regression": "...", "svm": "...", "random_forest": "...", "xgboost": "...", "vqc": "..."},
      "uci": {"logistic_regression": "...", "qsvc": "..."},
      "golub": {"logistic_regression": "...", "qsvc": "..."}
  }
  ```
- **Impact:** Requires manual code edits whenever new models are trained or evaluated.
- **Recommended Phase:** Phase 1 (Dynamic Catalog Discovery).

---

## 4. P3 ISSUES (Cleanup, Dead Code & Terminology)

### [ISSUE P3-01] Deleted Legacy Directory Staged/Unstaged in Git Status
- **File:** `Streamlit_dashboard/src/`
- **Location:** Entire directory
- **Description:** 10 files in `Streamlit_dashboard/src/` were replaced by `Streamlit_dashboard/dashboard_core/` but remain listed as unstaged deletions in `git status`.
- **Evidence:** `git status` shows:
  ```text
  deleted: Streamlit_dashboard/src/backend_adapter.py
  deleted: Streamlit_dashboard/src/pages/overview.py
  ...
  ```
- **Impact:** Clutters git status and risks confusion regarding which adapter is authoritative.
- **Recommended Phase:** Phase 1 (Working Tree Hygiene).

---

### [ISSUE P3-02] Ad-Hoc Experimental Scripts in Repository Root
- **File:** Root directory
- **Location:** `quick_quantum_test.py`, `run_fast_qsvc.py`, `tune_qsvc_c.py`, `save_best.py`
- **Description:** Ad-hoc prototyping scripts left in the root directory from earlier development phases.
- **Evidence:** Files exist at root with hardcoded paths and manual print loops.
- **Impact:** Code bloat and ambiguity for external contributors.
- **Recommended Phase:** Phase 1 (Archive to `scratch/` or `experiments/`).

---

### [ISSUE P3-03] Clinical Triage & Medical Diagnostic Terminology
- **File:** `src/disagreement_protocol.py` and `Streamlit_dashboard/dashboard_core/pages/prediction.py`
- **Location:** Comments, docstrings, and caption text
- **Description:** Uses terms such as `"Targeted Biopsy"`, `"Clinical Triage Bands"`, `"Patient Case"`, and `"Telepathology"` for machine learning prototype outputs.
- **Evidence:**
  ```python
  # src/disagreement_protocol.py:10
  "🟡 Targeted Biopsy / Multi-Disciplinary Review"
  ```
- **Impact:** May suggest clinical validation or diagnostic readiness that exceeds the research scope of the platform.
- **Recommended Phase:** Phase 1 / Phase 2 (Scientific Neutrality & Disclaimer Alignment).

---
