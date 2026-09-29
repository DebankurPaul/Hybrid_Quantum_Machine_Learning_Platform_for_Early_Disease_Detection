# PHASE 0 — MODEL CAPABILITY & READINESS MATRIX
## SIH Hybrid Quantum–Classical Disease Detection Platform

This matrix provides the forensic, evidence-based status of every dataset–model pair in the repository.
All statuses are derived from direct file inspection, unpickling verification, and execution tests in `.venv`.

| Dataset | Model | Model Type | Artifact | Loadable | Evaluated | Inference Status | Explainability | Quantum Config | Circuit | Notes / Evidence |
|---|---|---|---|---|---|---|---|---|---|---|
| **WDBC** | Logistic Regression | Classical | `models/logistic_regression.joblib` | **YES** | **YES** | **INFERENCE_READY** | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | Verified live inference; test accuracy 97.37% (CSV) / 96.49% (JSON) |
| **WDBC** | SVM | Classical | `models/svm.joblib` | **YES** | **YES** | **INFERENCE_READY** | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | Verified live inference; test accuracy 97.37% (CSV) |
| **WDBC** | Random Forest | Classical | `models/random_forest.joblib` | **YES** | **YES** | **INFERENCE_READY** | **VERIFIED** | UNAVAILABLE | UNAVAILABLE | Verified live inference; Gini feature importance in `results/random_forest_feature_importance.csv` |
| **WDBC** | XGBoost | Classical | `models/xgboost.joblib` | **NO** | **YES** | **LOAD_ERROR** | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | Artifact exists (272 KB), but unpickling fails with `ModuleNotFoundError: No module named 'xgboost'` |
| **WDBC** | 8-Qubit VQC | Quantum | `results/best_model/best_vqc_weights_8q.npy` | **YES** | **YES** | **INFERENCE_READY** | UNAVAILABLE | **VERIFIED** | **VERIFIED** | Verified live inference on Aer Statevector simulator; 24 parameters; ZZFeatureMap + RealAmplitudes |
| **UCI Heart** | Logistic Regression | Classical | None (No weights exported) | N/A | **YES** | **EVALUATION_ONLY** | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | Scaler saved in `results/artifacts/`, weights not saved; test accuracy 85.33% in `results/uci_metrics.json` |
| **UCI Heart** | Quantum QSVC (8Q) | Quantum | None (Dual vectors not exported) | N/A | **YES** | **EVALUATION_ONLY** | UNAVAILABLE | **VERIFIED** | **VERIFIED** | Autoencoder + Scaler in `results/artifacts/`, kernel weights not saved; circuit dynamically reconstructible |
| **Golub** | Logistic Regression | Classical | None (No weights exported) | N/A | **YES** | **EVALUATION_ONLY** | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | SelectKBest(64) + Scaler saved, weights not saved; test accuracy 85.29% in `results/golub_metrics.json` |
| **Golub** | Quantum QSVC (8Q) | Quantum | None (Dual vectors not exported) | N/A | **YES** | **EVALUATION_ONLY** | **VERIFIED** | **VERIFIED** | **VERIFIED** | Autoencoder + Scaler saved, weights not saved; QXAI SHAP artifact `results/figures/qxai_shap_summary.png` present |

---

### Column Legend & Definitions:
- **Dataset:** Target clinical disease vertical (`wdbc`, `uci`, `golub`).
- **Model:** Specific algorithm variant.
- **Model Type:** `Classical` or `Quantum`.
- **Artifact:** Path to trained model weights / serialized estimator file.
- **Loadable:** Whether the model artifact can be successfully loaded and deserialized into memory in `.venv`.
- **Evaluated:** Whether authoritative performance metrics exist on disk in `results/*.json` or `results/*.csv`.
- **Inference Status:**
  - `INFERENCE_READY`: Required model + preprocessing + schema + runtime dependencies are verified, and live inference executes without errors.
  - `EVALUATION_ONLY`: Evaluation metrics exist, but model weights were not serialized to disk.
  - `LOAD_ERROR`: Artifact exists on disk, but fails to load due to missing environment dependencies.
  - `UNAVAILABLE`: Neither weights nor implementation exist.
- **Explainability:**
  - `VERIFIED`: Authoritative feature importance / SHAP summary artifact exists on disk for this specific model.
  - `UNAVAILABLE`: No explainability artifact exists for this model.
- **Quantum Config:** Whether verified quantum hyperparameters (qubits, feature map, reps, entanglement, ansatz/kernel) are recorded in backend metadata.
- **Circuit:** Whether the quantum circuit diagram can be dynamically reconstructed from backend parameters.
