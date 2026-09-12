# Repository Audit Report

## A. Repository structure
- `README.md` (Empty)
- `requirements.txt` (Empty)
- `Hybrid_QML_Streamlit_Dashboard_Implementation_Plan.md` (Implementation specification)
- `data/raw/` containing `wdbc.data`, `wdbc.names`, `WDBC_raw.csv`
- `data/processed/` containing 8-feature subset CSVs (`WDBC_train_8_features.csv`, etc.) and a feature ranking CSV.
- `notebooks/01_wdbc_preprocessing.ipynb` (A Jupyter notebook performing basic EDA and feature selection).
- **Missing:** There is no `src/`, `backend/`, or `models/` directory. No Python modules exist.

## B. Existing backend components
- **None.** There are no Python scripts or modules for data preprocessing, model inference, or evaluation in the repository.

## C. Preprocessing pipeline
- The only implemented preprocessing is in the notebook.
- It uses `StandardScaler` followed by `SelectKBest` (Mutual Information) to select 8 features.
- **Discrepancy:** This contradicts the spec, which requires PCA for dimensionality reduction, followed by `MinMaxScaler` to `[0, π]`.
- No preprocessing artifacts (e.g., `.pkl` or `.joblib` files) have been saved.

## D. Exact inference pipeline
- **None.** No inference pipeline is implemented in the repository.

## E. Existing model artifacts
- **None.** No `.npy`, `.pt`, `.h5`, or `.pkl` weight files exist.

## F. Existing evaluation artifacts
- **None.** No JSON or CSV files containing metrics, ROC data, or confusion matrices exist.

## G. Existing classical models
- **None.**

## H. Existing quantum models
- **None.** No Qiskit VQC or QSVC code is present.

## I. Exact model input/output contracts
- **Missing.** Because no model code exists, the exact programmatic contract is unknown. We must assume the contract described in the spec: 30 features in -> probability out.

## J. Exact probability/class-label conventions
- The notebook defines: `0 = Benign`, `1 = Malignant`.
- The string labels in the dataset are `B` and `M`.

## K. Required artifacts for raw 30-feature patient inference
Based on the spec, the following artifacts *should* exist but are currently **missing**:
- `scaler.pkl` (StandardScaler)
- `pca.pkl` (PCA)
- `minmax_scaler.pkl` (MinMaxScaler)
- `vqc_4q_weights.npy`
- `vqc_8q_weights.npy`

## L. Current dependencies and versions
- `requirements.txt` is empty. The notebook uses `numpy`, `pandas`, and `scikit-learn`, but versions are unspecified.

## M. Existing code that can be directly reused
- The feature name list from the notebook can be reused to structure the 30-feature UI input.

## N. Existing code that should NOT be duplicated
- We should not duplicate the notebook's `SelectKBest` logic, as the spec strictly calls for PCA. We must wait for the backend teammate to provide the PCA artifacts.

## O. Missing components
- Qiskit quantum circuits (ZZFeatureMap, RealAmplitudes).
- Preprocessing artifacts.
- Model weights.
- Evaluation metrics.
- Backend Python API / service layer.

## P. Integration risks
- **High Risk:** The backend teammate's work is completely missing, and the one existing notebook directly contradicts the specified PCA preprocessing step. The Streamlit dashboard will have to rely entirely on a mock service layer until the real backend is committed.

## Q. Recommended Streamlit integration architecture
- **Service Layer Pattern:** Create a `backend_adapter.py` that defines all necessary functions (e.g., `load_metadata()`, `predict()`).
- This adapter will initially return `None` or structured empty states, triggering the UI's empty/loading states as required by the "no-hardcoding" policy.
- Once the backend is available, only the adapter needs to be updated.

## R. Recommended implementation order
- **Phase 1:** Backend Integration Contract (Create the mock `backend_adapter.py`).
- **Phase 2:** Streamlit Application Shell (Theme, layout, navigation).
- **Phase 3:** Prediction Experience (30-feature input form, CSV upload, mock inference).
- **Phase 4:** Quantum Model Experience (Circuit visualization container, configuration table).
- **Phase 5:** Evaluation (Metrics and charts empty states).
- **Phase 6:** Classical-vs-Quantum Benchmark (Table and charts empty states).
- **Phase 7:** Explainability (Empty states).
- **Phase 8:** Visual refinement.

## S. Any discrepancy between the project Markdown specification and the actual repository
1. **Preprocessing:** The spec mandates PCA and MinMaxScaler. The notebook uses SelectKBest and does not use MinMaxScaler.
2. **Models:** The spec implies 8Q and 4Q VQC models and evaluation artifacts are available (e.g., "The current handoff reports..."). In reality, the repository is empty of all model and evaluation artifacts.
3. **Dependencies:** `requirements.txt` is completely empty.
