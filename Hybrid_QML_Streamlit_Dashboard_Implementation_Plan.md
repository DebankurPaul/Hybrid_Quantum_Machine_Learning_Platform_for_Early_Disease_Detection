# Hybrid-QML Streamlit Dashboard

## Detailed Implementation & UI/UX Design Specification

**Project:** Hybrid Quantum--Classical Disease Detection\
**Hackathon:** Smart India Hackathon 2026\
**Dataset:** WDBC / Breast Cancer Wisconsin (Diagnostic)\
**Primary Quantum Model:** 8-Qubit VQC\
**Secondary Quantum Model:** 4-Qubit VQC\
**Quantum Stack:** Qiskit + Qiskit Aer\
**Frontend:** Streamlit

------------------------------------------------------------------------

## 1. Purpose

This document is the implementation blueprint for the Streamlit
dashboard. It combines the SIH project brief, the later dashboard
functional specification, and the latest implementation handoff. The
latest implementation handoff is treated as the highest-authority source
for the current implemented state. fileciteturn5file0

The dashboard is the user-facing layer of the hybrid platform. It must
demonstrate the complete workflow without duplicating model training or
inventing backend results.

### Non-negotiable rule

> **Design everything now, connect nothing fictitious.**

Every backend-dependent component must be fully designed, but its values
must remain dynamic placeholders until the real artifacts and APIs are
available.

------------------------------------------------------------------------

# 2. Product Vision

The dashboard must feel like a deliberately designed
scientific/engineering product, not a generic AI-generated dashboard.

It should communicate:

-   precision,
-   trust,
-   technical competence,
-   scientific transparency,
-   restraint,
-   and a clear classical-to-quantum progression.

The central story is:

**Biomedical data → classical preprocessing → PCA reduction → quantum
encoding → VQC/QSVC → prediction → evaluation → explanation.**

The UI should make this understandable within a short SIH demonstration.

------------------------------------------------------------------------

# 3. UI/UX Design Principles

## 3.1 Avoid the generic AI aesthetic

Do **not** use:

-   excessive gradients,
-   neon purple/blue styling,
-   glowing cards,
-   glassmorphism everywhere,
-   random futuristic illustrations,
-   decorative quantum particles,
-   stock AI-brain imagery,
-   excessive emojis,
-   meaningless animations,
-   oversized cards for every metric,
-   fake circuit-board backgrounds.

The real Qiskit circuit, real WDBC statistics, real PCA analysis, real
ROC curves, and real model results are enough to establish the product
identity.

## 3.2 Visual language

The visual direction should resemble:

-   scientific research software,
-   biomedical analytics,
-   engineering instrumentation,
-   a professional analysis workstation.

Use hierarchy through typography, spacing, borders, alignment,
restrained color, and meaningful charts.

## 3.3 Color system

Use a restrained palette:

-   warm/off-white or very light neutral background,
-   white/slightly tinted surfaces,
-   deep charcoal/navy text,
-   restrained indigo/navy or blue as the primary accent,
-   muted teal for data/preprocessing,
-   green only for successful system status,
-   amber for warnings,
-   red for errors or carefully scoped result emphasis.

Centralize colors as theme tokens rather than scattering hex values
throughout the application.

Suggested tokens:

``` text
COLOR_BG
COLOR_SURFACE
COLOR_SURFACE_ALT
COLOR_TEXT
COLOR_TEXT_MUTED
COLOR_PRIMARY
COLOR_SECONDARY
COLOR_BORDER
COLOR_SUCCESS
COLOR_WARNING
COLOR_ERROR
```

------------------------------------------------------------------------

# 4. Application Structure

Use four main functional areas:

1.  **Prediction**
2.  **Quantum Model**
3.  **Evaluation**
4.  **Benchmark**

A compact overview should appear at the top/landing state.

This is intentionally more compact than six separate pages because it is
better suited to a short judge demonstration.

------------------------------------------------------------------------

# 5. Global Application Shell

## Header

Example:

``` text
HYBRID-QML
Quantum-Classical Intelligence for Early Disease Detection

WDBC · Breast Cancer Wisconsin (Diagnostic)
```

The title and descriptive text are static UI content. Dataset status and
model status must be dynamic.

## Sidebar

Suggested structure:

``` text
HYBRID-QML

DATASET
WDBC

TASK
Disease Detection

QUANTUM BACKEND
Qiskit Aer

MODEL
[ 8-Qubit VQC ▼ ]

DECISION THRESHOLD
τ = [ 0.650 ]

SYSTEM STATUS
✓ Dataset
✓ Preprocessor
✓ Quantum Backend
✓ Model Artifacts
```

Do not expose unnecessary training hyperparameters.

Model selection must correspond to real available artifacts.

------------------------------------------------------------------------

# 6. Overview / Landing Experience

The landing state should answer:

1.  What is this?
2.  What data does it use?
3.  What makes it hybrid?

Suggested visual structure:

``` text
HYBRID-QML
Quantum-Classical Intelligence for Early Disease Detection

[ Samples ] [ Original Features ] [ PCA Components ] [ Qubits ]

30 Biomedical Features
        ↓
Classical Preprocessing
        ↓
PCA Reduction
        ↓
Quantum-Compatible Representation
        ↓
VQC / QSVC
        ↓
Model Prediction
```

The metric values must come from backend metadata.

Before backend integration, use an intentional placeholder:

``` text
—
Waiting for dataset metadata
```

Never display invented numbers.

------------------------------------------------------------------------

# 7. Core Classical-to-Quantum Visual Story

The implemented quantum pipeline is:

``` text
30 Original WDBC Measurements
            ↓
      StandardScaler
            ↓
      PCA: 30 → 4/8
            ↓
      MinMaxScaler [0, π]
            ↓
       ZZFeatureMap
            ↓
      RealAmplitudes
            ↓
        Qiskit Aer
            ↓
     P(malignant)
            ↓
       Threshold τ
            ↓
     Model Prediction
```

For the recommended 8Q configuration:

-   8 PCA components
-   8 qubits
-   ZZFeatureMap, reps=1
-   RealAmplitudes, reps=2
-   24 trainable parameters
-   Aer simulator
-   current recommended threshold τ=0.65

The 4Q configuration uses 4 components, 4 qubits, 12 parameters, and the
current recommended threshold τ=0.60. These values come from the latest
implementation handoff. fileciteturn5file0

The UI should load configuration from actual model metadata wherever
possible.

------------------------------------------------------------------------

# 8. Prediction Tab

This is the primary live demonstration.

## Patient input

The actual inference flow begins with **30 original WDBC measurements**:

``` text
30 raw measurements
→ saved StandardScaler
→ saved PCA
→ saved MinMaxScaler
→ 8/4 reduced components
→ trained VQC
→ P(malignant)
→ threshold
→ predicted class
```

Do not replace the 30 required raw measurements with eight hand-selected
measurements.

## Input organization

Avoid a flat wall of 30 controls.

Group the WDBC fields according to the actual backend schema, for
example:

-   Mean measurements
-   Standard error measurements
-   Worst measurements

Exact field names must be obtained from the repository.

## CSV input

Support patient CSV upload if the backend supports it.

Validate:

-   required columns,
-   numeric types,
-   missing values,
-   unexpected columns,
-   row count.

Use clear user-facing validation messages.

------------------------------------------------------------------------

# 9. Prediction Interaction

Primary action:

``` text
[ Run Model Prediction ]
```

During inference, show meaningful progress:

``` text
Preparing patient record
Applying saved preprocessing
Running quantum inference
Calculating model output
```

Only display stages that actually occur.

Do not retrain models during dashboard interaction.

------------------------------------------------------------------------

# 10. Prediction Result

Use a research-oriented result panel:

``` text
MODEL PREDICTION

Predicted Class
MALIGNANT

Estimated Model Probability
—

Decision Threshold
τ = 0.650
```

The probability and class must be backend-generated.

Use:

-   Model Prediction
-   Predicted Class
-   Estimated Model Probability

Avoid:

-   Clinical Diagnosis
-   Diagnostic Decision
-   Patient is diagnosed with...

Persistent notice:

> **Research prototype:** This result is a machine-learning prediction
> for demonstration/research purposes and is not a medical diagnosis or
> medical advice.

------------------------------------------------------------------------

# 11. Model Switching

Support:

### 8-Qubit VQC

Default recommended model.

### 4-Qubit VQC

Lightweight comparison/demo model.

Switching must load the corresponding existing model artifacts. It must
not trigger model retraining.

------------------------------------------------------------------------

# 12. Threshold Interaction

The threshold control may range from 0.10 to 0.90.

Current recommended defaults:

-   8Q: τ=0.65
-   4Q: τ=0.60

Changing τ changes the classification rule:

``` text
P(malignant) >= τ
```

It must not retrain the model.

For evaluation data, the UI should dynamically update:

-   sensitivity,
-   specificity,
-   accuracy,
-   F1,
-   confusion matrix,

when the required stored probabilities/labels exist.

Do not hard-code claims such as "τ above 0.70 causes missed tumors."
Show measured trade-offs instead.

------------------------------------------------------------------------

# 13. Quantum Model Tab

Purpose:

> Show judges exactly what happens in the quantum stage.

## Architecture

Display the actual:

``` text
30 Features
→ StandardScaler
→ PCA
→ 8 Components
→ MinMaxScaler [0,π]
→ ZZFeatureMap
→ RealAmplitudes
→ Aer
→ Probability
```

## Configuration cards

Show dynamically:

  Property                  Value
  ------------------------- -------
  PCA components            ---
  Qubits                    ---
  Feature map               ---
  Feature map repetitions   ---
  Ansatz                    ---
  Ansatz repetitions        ---
  Trainable parameters      ---
  Backend                   ---
  Threshold                 ---

These fields remain blank/dynamic until real model metadata is
connected.

------------------------------------------------------------------------

# 14. Quantum Circuit

Render the **actual Qiskit circuit**.

Do not draw a decorative fake circuit.

The circuit must change when the selected 4Q/8Q model changes.

This is one of the most important technical visuals in the application.

------------------------------------------------------------------------

# 15. Trained Parameters

Show the actual learned variational parameters.

Use the label:

**Trained Variational Parameters**

not:

**Quantum Explainability**

Parameter visualization describes the model, not the reason for an
individual patient prediction.

------------------------------------------------------------------------

# 16. PCA Visualization

Make the dimensionality reduction visually obvious.

For 8Q:

``` text
30 original features
        ↓
      PCA
        ↓
8 components
        ↓
92.4% variance retained
```

For 4Q:

``` text
30 original features
        ↓
      PCA
        ↓
4 components
        ↓
81.2% variance retained
```

The latest handoff reports these variance figures. They should
ultimately be loaded from the fitted PCA/model metadata rather than
duplicated in UI code. fileciteturn5file0

------------------------------------------------------------------------

# 17. PCA Component Loadings

Show how original WDBC measurements contribute to the reduced
components.

Correct title:

**PCA Component Loadings**

Do not call this:

**Quantum Feature Importance**

PCA loadings describe the dimensionality-reduction representation and
are not automatically patient-level prediction explanations.

------------------------------------------------------------------------

# 18. Evaluation Tab

Purpose:

> Show the measured performance of the models.

Display available:

-   Accuracy
-   Sensitivity / Recall
-   Specificity
-   F1-score
-   ROC-AUC
-   Training time
-   Inference time

Missing values remain:

``` text
—
```

Never fabricate metrics.

The current handoff reports the 8Q model at approximately:

-   Accuracy: 75.4%
-   Sensitivity: 83.3%
-   Specificity: 70.8%
-   ROC-AUC: 0.831

These are reference results, not values to hard-code into the UI.
fileciteturn5file0

------------------------------------------------------------------------

# 19. ROC Curve

Show:

-   ROC curve,
-   AUC,
-   selected threshold operating point where appropriate.

The operating point must be calculated from actual evaluation
probabilities and labels.

------------------------------------------------------------------------

# 20. Confusion Matrix

Generate from actual evaluation artifacts.

The current handoff reports for the 8Q result:

-   TP=35
-   TN=51
-   FP=21
-   FN=7

These sum to 114, matching the stated test-set size.

Again, the dashboard must load/generate these values dynamically rather
than hard-code them. fileciteturn5file0

------------------------------------------------------------------------

# 21. Threshold Trade-off Chart

Show threshold against:

-   Sensitivity
-   Specificity

Optionally:

-   Accuracy
-   F1

Highlight the selected threshold.

This gives judges evidence for why a threshold was chosen.

------------------------------------------------------------------------

# 22. Benchmark Tab

Compare actual available models.

### Classical

-   Logistic Regression
-   SVM
-   Random Forest
-   XGBoost only if actually implemented

### Quantum

-   VQC 4Q
-   VQC 8Q
-   QSVC if actually implemented
-   QNN only if actually implemented

Do not create rows for models that do not exist merely because they
appear in a specification.

------------------------------------------------------------------------

# 23. Benchmark Table

Suggested structure:

  Model                   Accuracy   Sensitivity   Specificity    F1   ROC-AUC   Time
  --------------------- ---------- ------------- ------------- ----- --------- ------
  Logistic Regression          ---           ---           ---   ---       ---    ---
  SVM                          ---           ---           ---   ---       ---    ---
  Random Forest                ---           ---           ---   ---       ---    ---
  VQC 4Q                       ---           ---           ---   ---       ---    ---
  VQC 8Q                       ---           ---           ---   ---       ---    ---

Populate only from actual evaluation artifacts.

------------------------------------------------------------------------

# 24. Honest Benchmarking

The UI must never imply that quantum models are automatically better.

If Random Forest wins, say:

> **Best measured model: Random Forest**

If VQC wins, say:

> **Best measured model: 8-Qubit VQC**

The dashboard reports the experimental result rather than deciding the
conclusion beforehand.

------------------------------------------------------------------------

# 25. Explainability

Keep these concepts separate:

### PCA analysis

How original features contribute to principal components.

### Classical explainability

SHAP or another appropriate attribution method.

### Quantum attribution

Only use a real post-hoc attribution method if the backend actually
supports it.

Never invent quantum feature importance.

------------------------------------------------------------------------

# 26. Natural-Language Explanation

If explanation data exists, provide a concise evidence-based summary:

``` text
Model Explanation

The prediction was generated from the reduced representation of the
30 WDBC measurements.

The strongest available contributing features were:
1. —
2. —
3. —

This describes model behavior and is not a clinical assessment.
```

Every statement about a specific feature must be generated from actual
attribution data.

------------------------------------------------------------------------

# 27. Backend Integration Boundary

Preferred architecture:

``` text
Streamlit UI
      ↓
Application / Service Layer
      ↓
Preprocessing + Model Inference
      ↓
Saved Model Artifacts
      ↓
Evaluation Results
```

Streamlit must not own model training.

------------------------------------------------------------------------

# 28. Artifact Management

The latest handoff identifies artifacts including:

``` text
results/best_model/best_vqc_weights_8q.npy
results/best_model/best_model_metrics.json
results/vqc_weights_4q.npy
```

The actual repository must be inspected before implementation.

The dashboard also needs the fitted preprocessing artifacts required
for:

``` text
30 raw values
→ StandardScaler
→ PCA
→ MinMaxScaler
```

If those artifacts do not exist, Streamlit must not silently refit
preprocessing during inference.

------------------------------------------------------------------------

# 29. Centralized Artifact Loader

Use one integration layer responsible for loading:

-   dataset metadata,
-   preprocessing artifacts,
-   4Q VQC,
-   8Q VQC,
-   classical models,
-   evaluation results,
-   ROC data,
-   confusion matrices,
-   attribution data.

Pages should consume this layer instead of loading files independently.

------------------------------------------------------------------------

# 30. Caching

Use Streamlit caching for expensive reusable objects such as:

-   preprocessing artifacts,
-   VQC model/circuit,
-   model weights,
-   evaluation data.

Threshold changes should not recreate or retrain the model.

------------------------------------------------------------------------

# 31. Session State

Potential session state:

``` text
selected_model
threshold
patient_input
prediction_result
```

Do not place unnecessary large model objects into session state if
cached resources can be used.

------------------------------------------------------------------------

# 32. Empty States

Every backend-dependent component must have a polished empty state.

Examples:

### Metrics

``` text
Accuracy
—
Awaiting evaluation results
```

### Circuit

``` text
Quantum Circuit
Awaiting model artifact
```

### Prediction

``` text
No prediction yet

Enter a patient record and run the model.
```

### Benchmark

``` text
Evaluation results are not connected yet.
The comparison will populate automatically when
the current model results are available.
```

This allows the UI to be completed before the backend teammate pushes
the implementation.

------------------------------------------------------------------------

# 33. Loading States

Use meaningful loading indicators for expensive operations:

``` text
Preparing model...
Applying saved preprocessing...
Running quantum simulation...
Calculating prediction...
```

Only show steps that actually execute.

------------------------------------------------------------------------

# 34. Error States

Never expose raw Python tracebacks to judges by default.

Instead:

``` text
The evaluation artifact could not be loaded.

Please verify that the current model results have been generated.
```

Developer details may be placed behind an optional debug expander.

------------------------------------------------------------------------

# 35. No-Hard-Coding Policy

## Must come from backend/data

Never hard-code:

-   sample count,
-   feature count,
-   class count,
-   missing-value count,
-   PCA/component count,
-   variance retained,
-   model metrics,
-   confusion matrix values,
-   ROC data,
-   probabilities,
-   predictions,
-   feature attributions,
-   trained parameters,
-   circuit configuration,
-   timing measurements,
-   model availability,
-   backend-generated feature names,
-   artifact-dependent paths when centrally configurable.

## May be static UI configuration

-   application title,
-   labels,
-   explanatory copy,
-   disclaimer,
-   theme tokens,
-   layout constants,
-   navigation structure,
-   control ranges where appropriate.

Even model defaults such as thresholds should preferably come from model
metadata once available.

------------------------------------------------------------------------

# 36. Design Before Backend Connection

The dashboard must be fully designed before the backend is available.

This means implementing:

-   layout,
-   cards,
-   tables,
-   charts,
-   navigation,
-   controls,
-   circuit container,
-   result container,
-   loading states,
-   empty states,
-   error states,
-   responsive layout.

Backend-dependent content should show `—`, an empty state, or a loading
state.

When the teammate pushes the backend, we connect the real data through
the service layer.

**The UI should not need to be redesigned simply because real data
becomes available.**

------------------------------------------------------------------------

# 37. Judge Demonstration Flow

## 0--20 seconds: Overview

Show:

-   project identity,
-   WDBC,
-   hybrid architecture,
-   system status.

## 20--40 seconds: Reduction

Show:

**30 original features → PCA → 8 components**

## 40--70 seconds: Quantum Model

Show:

-   ZZFeatureMap,
-   RealAmplitudes,
-   8 qubits,
-   24 parameters,
-   Qiskit Aer,
-   actual circuit.

## 70--100 seconds: Prediction

Enter a patient record and show:

-   probability,
-   threshold,
-   model prediction.

## 100--120 seconds: Evaluation

Show:

-   benchmark,
-   ROC-AUC,
-   sensitivity/specificity,
-   confusion matrix,
-   explanation.

This matches the current implementation handoff. fileciteturn5file0

------------------------------------------------------------------------

# 38. Development Plan With Antigravity

## Stage 0 --- Read-only repository audit

Do not modify files.

Prompt:

``` text
Audit the current repository specifically for Streamlit integration.

Identify:
1. Repository structure.
2. Exact WDBC feature schema.
3. Complete preprocessing pipeline.
4. StandardScaler/PCA/MinMaxScaler artifacts.
5. VQC implementation.
6. 4-qubit and 8-qubit configurations.
7. QSVC implementation.
8. Classical models.
9. Saved model artifacts.
10. Evaluation artifacts.
11. Prediction/inference functions.
12. Qiskit/Aer versions.
13. Dependencies.
14. Missing artifacts required for raw-patient inference.

Return a structured integration report.

Do not modify any files.
Do not recreate existing ML/QML logic.
```

Review the report before code changes.

------------------------------------------------------------------------

## Stage 1 --- Integration Contract

Define the smallest interface Streamlit needs.

Conceptually:

``` python
load_dataset_metadata()
load_preprocessor()
load_quantum_model(model_name)
load_classical_models()
load_evaluation_results()
predict_patient(data, model_name, threshold)
get_quantum_circuit(model_name)
get_model_configuration(model_name)
get_explainability(...)
```

These are conceptual examples only. Actual interfaces must match the
repository.

------------------------------------------------------------------------

## Stage 2 --- Streamlit Shell

Implement:

-   page configuration,
-   theme,
-   header,
-   sidebar,
-   navigation,
-   system status,
-   empty states.

Do not implement duplicated ML logic.

------------------------------------------------------------------------

## Stage 3 --- Prediction

Implement:

-   30-feature input,
-   CSV upload,
-   validation,
-   model selection,
-   threshold,
-   inference,
-   result panel,
-   disclaimer.

------------------------------------------------------------------------

## Stage 4 --- Quantum Model

Implement:

-   architecture flow,
-   configuration,
-   actual circuit,
-   trained parameter visualization,
-   PCA reduction visualization.

------------------------------------------------------------------------

## Stage 5 --- Evaluation

Implement:

-   metrics,
-   ROC,
-   confusion matrix,
-   threshold analysis,
-   timing.

------------------------------------------------------------------------

## Stage 6 --- Benchmark

Implement:

-   classical models,
-   quantum models,
-   comparison table,
-   comparison charts,
-   best-model identification.

------------------------------------------------------------------------

## Stage 7 --- Explainability

Implement:

-   PCA loadings,
-   classical SHAP,
-   supported quantum attribution.

------------------------------------------------------------------------

## Stage 8 --- Visual Polish

Refine:

-   spacing,
-   typography,
-   chart hierarchy,
-   empty/loading/error states,
-   responsive layout,
-   consistency.

Do not add decorative AI effects simply to make the dashboard look
"futuristic."

------------------------------------------------------------------------

# 39. Testing

## Functional

-   App launches.
-   Navigation works.
-   Model switching works.
-   30-feature input validates.
-   CSV validation works.
-   Prediction works.
-   Threshold changes work without retraining.
-   Circuit renders.
-   Metrics load.
-   Benchmark loads.
-   Explainability loads where supported.

## Data integrity

-   Same preprocessing at training and inference.
-   No inference-time fitting.
-   No hard-coded evaluation results.
-   No fabricated predictions.
-   No fabricated classical results.

## Visual

-   Consistent typography.
-   Consistent spacing.
-   No clutter.
-   Charts readable on a projector.
-   Tables readable at a distance.
-   Error states are understandable.
-   No generic AI visual tropes.

------------------------------------------------------------------------

# 40. Definition of Done

The dashboard is complete when:

-   [ ] The hybrid architecture is understandable within seconds.
-   [ ] Classical-to-quantum transition is visually obvious.
-   [ ] 30 → PCA → 8 is represented correctly.
-   [ ] 8Q VQC is the default recommended model.
-   [ ] 4Q VQC works when its artifact exists.
-   [ ] Actual Qiskit circuit is rendered.
-   [ ] Patient inference uses saved preprocessing.
-   [ ] Models are not retrained on page load.
-   [ ] Threshold changes do not retrain the model.
-   [ ] Evaluation values come from actual artifacts.
-   [ ] Classical and quantum models are compared honestly.
-   [ ] PCA analysis is distinguished from explainability.
-   [ ] Medical disclaimer is visible.
-   [ ] No clinical-diagnosis claims are made.
-   [ ] Backend-dependent components have designed empty/loading states.
-   [ ] No backend-dependent values are fabricated.
-   [ ] The interface feels like a professional scientific software
    product rather than a generic AI-generated dashboard.
-   [ ] The judge demonstration can be completed comfortably within
    approximately two minutes.

------------------------------------------------------------------------

# 41. Final Design Philosophy

The dashboard should not try to impress judges by looking futuristic.

It should impress them by looking **intentional**.

The strongest visual identity should come from the actual project:

**real WDBC data → real preprocessing → real PCA → real quantum circuit
→ real model output → real evaluation.**

The UI is the carefully designed layer that makes those real components
understandable.

> **Design everything now. Connect only what actually exists. Never
> fabricate what the backend has not produced.**
