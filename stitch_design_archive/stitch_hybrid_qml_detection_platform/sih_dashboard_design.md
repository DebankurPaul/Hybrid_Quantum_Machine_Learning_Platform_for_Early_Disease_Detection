# SIH Hybrid Quantum–Classical Disease Detection
# Dashboard Design System & Product UI Specification

**File:** `SIH_Dashboard_DESIGN.md`  
**Role:** Authoritative visual/UX/design specification for Google Stitch and the Streamlit implementation.  
**Project:** Smart India Hackathon 2026 — Hybrid Quantum Machine Learning Platform for Early Disease Detection.

---

## 0. Authority

This document is the **design source of truth**. It controls visual language, information architecture, interaction patterns, page composition, terminology, and presentation.

It does **not** control scientific facts. The backend remains authoritative for datasets, feature schemas, preprocessing, model inventory, model configuration, artifacts, predictions, metrics, confusion matrices, ROC data, explainability, and availability.

**Rule: Backend truth controls content. Design controls presentation.**

Never fabricate a metric, probability, model, feature, chart, explanation, or capability to fill a visual component.

The dashboard is a **research platform/prototype**, not a validated clinical diagnostic system.

---

# 1. Research Basis

This specification was developed from:

- the SIH project brief;
- the existing functional dashboard specification;
- the current backend/result-registry architecture and artifact metadata;
- the supplied Figma dashboard reference;
- Figma's current dashboard template ecosystem;
- Google Stitch's current design-system and design-to-code workflow;
- Streamlit's current layout/container capabilities;
- Carbon Design System dashboard/data-visualization guidance;
- current healthcare analytics dashboard examples from 2025–2026.

The SIH brief requires an end-to-end interface covering preprocessing, classical and quantum models, prediction/risk-style outputs, evaluation, explainability, and honest classical-vs-quantum benchmarking. It explicitly says not to assume quantum models outperform classical baselines and requires outputs to be presented as model predictions rather than clinical diagnoses.

The existing functional specification calls for Overview, Data & Preprocessing, Quantum Models, Disease Prediction, Model Comparison/Evaluation, and Explainability. Those capabilities remain, but this redesign changes the experience from a documentation/report layout into an interactive research workspace.

Google's current Stitch workflow supports high-fidelity UI generation, reusable `DESIGN.md` design rules, iteration, and Antigravity-oriented design-to-code workflows. Streamlit provides persistent sidebars, containers, columns, tabs, expanders, popovers, and responsive layout primitives. Carbon's dashboard guidance emphasizes prioritization, limited non-essential metrics, consistent visual encoding, and whitespace.

---

# 2. Product Vision

The product should feel like a **real scientific research application**:

- precise;
- calm;
- modern;
- intelligent;
- trustworthy;
- data-rich;
- technically sophisticated without looking like developer tooling.

It should **not** look like:

- a static report;
- an academic paper;
- a generic SaaS admin panel;
- a hospital administration system;
- neon/cyberpunk quantum software;
- glassmorphism;
- crypto/fintech;
- a “futuristic AI” concept;
- a page made entirely from text and metric cards.

The central visual story is:

**Data → Preparation → Representation → Classical + Quantum Models → Prediction → Evaluation → Explanation**

The user should understand this story visually.

---

# 3. Global Application Shell

## 3.1 Persistent sidebar

The left sidebar is a compact product navigation area.

### Brand block

**Hybrid-QML**

Small descriptor:

> Research platform for evaluating classical and quantum machine learning approaches.

Use a restrained scientific mark/icon if available. Do not use an oversized emoji as the brand.

### Active context

Show:

**DATASET**  
`WDBC Breast Cancer`

**MODEL**  
`Quantum VQC · 8 qubits`

These values are dynamic.

Only show a model selector when models actually exist for the active dataset.

### Navigation

- Overview
- Data
- Models
- Prediction
- Evaluation
- Explainability

Use a clear active state.

### System status

At the bottom:

- Data
- Preprocessing
- Model artifacts
- Quantum backend

Each status must be independent. Do not collapse them into one generic “Ready”.

## 3.2 Main workspace header

Every page starts with:

**Page title**

One-line description.

Right side, when relevant:

- Dataset selector
- Model selector
- primary action

The header should be compact rather than a giant hero.

---

# 4. Visual Language

## 4.1 Direction

Use a **light, neutral, scientific interface**.

- off-white application canvas;
- white surfaces;
- charcoal typography;
- cool gray secondary text;
- restrained teal/blue product accent;
- subtle indigo/purple reserved for quantum content;
- muted semantic status colors.

Absolutely avoid:

- neon;
- gradients;
- glowing borders;
- glassmorphism;
- heavy shadows;
- dark sci-fi HUD styling.

## 4.2 Design tokens

| Token | Suggested value | Purpose |
|---|---|---|
| Canvas | `#F6F8FA` | App background |
| Surface | `#FFFFFF` | Main panels |
| Surface subtle | `#F1F4F6` | Secondary panels |
| Border | `#DCE2E7` | Borders/dividers |
| Text primary | `#17212B` | Headings |
| Text secondary | `#5F6B76` | Supporting text |
| Text muted | `#87929C` | Metadata |
| Accent | `#176B87` | Main interaction/scientific accent |
| Accent soft | `#E7F3F6` | Accent backgrounds |
| Quantum | `#6558C8` | Quantum-specific emphasis |
| Quantum soft | `#F0EEFC` | Quantum panels |
| Success | `#237A57` | Available/success state |
| Warning | `#9A6A16` | Partial/attention |
| Error | `#B74343` | Actual error |

Semantic colors must never imply clinical safety.

## 4.3 Typography

Use one modern, highly readable sans-serif.

Suggested hierarchy:

- display: 30–36 px;
- page title: 26–30 px;
- section title: 18–20 px;
- card title: 14–16 px;
- body: 14–15 px;
- metadata: 12–13 px;
- metric value: 24–32 px.

Avoid decorative futuristic fonts and excessive uppercase.

## 4.4 Spacing

Use an 8-point rhythm:

`4 / 8 / 12 / 16 / 24 / 32 / 40 / 48`

Typical:

- page padding: 32–40 px;
- section gap: 28–36 px;
- card padding: 20–24 px;
- card gap: 16–20 px.

Whitespace is intentional.

---

# 5. Cards and Panels

Cards are structural units, not decoration.

A standard analytical card contains:

1. small category/eyebrow;
2. title;
3. short description if needed;
4. main value/visual;
5. optional metadata footer.

Prefer subtle borders over strong shadows.

Do not turn every paragraph into a card.

Do not put large blocks of explanatory prose inside metric cards.

---

# 6. Overview Dashboard

The Overview page must immediately look like a dashboard.

## 6.1 Hero/context area

Left:

**Hybrid Quantum–Classical Disease Detection**

> Evaluate classical and quantum machine learning approaches across biomedical datasets through one transparent research workflow.

Below:

`Dataset: dynamic`  
`Models: dynamic`  
`Backend: dynamic`

Right:

A compact **Active Experiment / System State** module.

No marketing-style oversized hero.

## 6.2 KPI strip

Use 4–5 meaningful dynamic KPIs, for example:

- Samples
- Input Features
- Reduced Features
- Available Models
- Evaluation Status

Do not show global Accuracy unless there is a single authoritative context.

## 6.3 Main composition

### Row 1
**Hybrid Pipeline** — wide  
**Active Model** — narrow

### Row 2
**Performance Snapshot** — wide  
**Dataset Composition** — narrow

### Row 3
**Model Landscape** — full width

This is the major change from the previous report-like design: charts, comparisons, and active state are first-class content.

## 6.4 Pipeline visualization

Visual horizontal chain:

`Dataset → Preprocessing → Feature Reduction → Classical + Quantum → Evaluation → Prediction/Explanation`

The classical/quantum branch should be visually obvious.

Dimensions such as “30 → 8” must be read from backend metadata, never hardcoded globally.

---

# 7. Data Page

The Data page is a data-exploration workspace.

## 7.1 Summary strip

Dynamic:

- Samples
- Features
- Classes
- Missing values
- Evaluation cohort/status

## 7.2 Main composition

Wide panel:

**Class Distribution**

Narrow panel:

**Dataset Profile**

Include source/name, feature type, class labels, evaluation protocol, and availability.

## 7.3 Feature explorer

Searchable/sortable table:

- Feature
- Type
- Range
- Missing
- Selected/Used
- Stage

Only show fields actually exposed by the backend.

## 7.4 Preprocessing flow

Visual:

`Raw Data → Split → Transform → Selection/Reduction → Model Input`

Each stage may expose method, parameters, artifact status, and train-only fitting status where available.

This is important because preprocessing is a central part of the hybrid architecture.

## 7.5 Dataset states

Distinguish:

**Available** — dataset can be inspected.

**Evaluated offline** — valid evaluation exists.

**Inference-ready** — valid preprocessing and model artifacts exist for live inference.

**Experimental** — present for research/evaluation but not deployable.

Never label all of these simply “Ready”.

---

# 8. Models Page

This is a **model exploration workspace**, not a technical report.

## 8.1 Dynamic model discovery

Models must come from backend metadata.

Never hardcode a permanent sidebar list such as Logistic Regression, SVM, Random Forest, VQC, QSVC.

The UI should support any number of backend models.

## 8.2 Model cards

Each card:

- name;
- family: Classical/Quantum;
- evaluation availability;
- inference availability;
- explainability availability;
- compact configuration;
- performance summary where available.

## 8.3 Model detail

Header:

`Quantum VQC`

Badges:

`Quantum` `8 qubits` `Inference available`

Main layout:

### Architecture panel

`Input → Feature Map → Ansatz → Measurement`

### Configuration panel

- qubits;
- feature map;
- repetitions;
- ansatz;
- depth;
- backend;
- shots when applicable;
- trainable parameters when available.

All dynamic.

## 8.4 Circuit panel

The quantum circuit is a major visual element.

Use a large panel with:

- rendered circuit;
- horizontal scrolling if required;
- model metadata;
- short explanation.

If the circuit cannot be generated, show a designed unavailable state. Never allow a Python traceback.

---

# 9. Prediction Workspace

Prediction should feel like a focused research tool.

## 9.1 Header

**Model Prediction**

> Run the selected trained model against a valid input record using the stored preprocessing and model artifacts.

Show Dataset / Model / Inference Status.

## 9.2 Input modes

- Manual Input
- Upload Record
- Batch, only if actually implemented

## 9.3 Manual input

Feature fields must come from backend feature schema.

Group fields according to backend metadata. For WDBC, Mean/SE/Worst grouping may be used if actually exposed.

Never hardcode ranges or silently insert zeroes.

## 9.4 Result panel

### Primary

**Model Prediction**

`Predicted class: <backend class label>`

### Probability

If supported:

**Estimated Model Probability**

Display the model-produced probability.

Use language such as:

> Probability produced by the selected model under the configured inference pipeline.

Do not imply clinical risk validation.

### Details

- model;
- threshold;
- preprocessing;
- inference status;
- timestamp if available.

## 9.5 Multi-model comparison

If multiple models support inference:

| Model | Prediction | Probability | Agreement |
|---|---|---:|---|

Do not show unsupported probabilities.

## 9.6 Persistent disclaimer

> Research prototype. Model outputs are computational predictions and are not clinical diagnoses or medical advice.

---

# 10. Evaluation Page

This should be one of the strongest pages.

## 10.1 Header

**Evaluation**

> Compare measured model performance under the selected evaluation protocol.

Protocol strip:

- Dataset
- Test protocol
- Test size
- Split strategy
- Seed, if exposed
- Independent cohort status, if exposed

## 10.2 KPI row

Show only metrics actually available:

- Accuracy
- Sensitivity
- Specificity
- F1
- ROC-AUC
- MCC

Missing metrics must say `Not available`, never `0`.

## 10.3 Performance chart

Use a horizontal model comparison.

Allow a compact metric selector:

`Accuracy | Sensitivity | Specificity | F1 | ROC-AUC`

Models are dynamically populated.

## 10.4 Confusion matrix

Large visual.

Adjacent metadata:

- model;
- test size;
- class labels;
- protocol.

Render the matrix directly from backend data and labels. Never reconstruct a matrix from assumed TN/FP/FN/TP fields.

## 10.5 ROC

When ROC data exists:

- ROC curve;
- AUC;
- operating threshold if provided.

When absent:

> ROC curve data was not provided for this evaluation artifact.

Never fabricate a curve.

## 10.6 Threshold analysis

Only show if backend provides threshold sweep data.

Plot:

`Threshold ↔ Sensitivity ↔ Specificity`

The control is an analysis tool, not decoration.

## 10.7 Interpretation

Use factual language:

> “QSVC achieved higher held-out accuracy than Logistic Regression under this evaluation protocol.”

Avoid automatically calling this “quantum advantage” or “superiority”.

---

# 11. Benchmark Page

The benchmark should make the classical-vs-quantum story visually compelling.

## 11.1 Main view

Use:

1. visual comparison;
2. compact table;
3. methodology context.

Do not make the table the whole page.

## 11.2 Comparison matrix

| Model | Family | Accuracy | F1 | Sensitivity | Specificity | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| Dynamic | Classical/Quantum | … | … | … | … | … |

Highlight the strongest measured value **per metric**, not one universal winner.

## 11.3 Delta view

If valid paired comparisons exist:

- Δ Accuracy
- Δ F1
- Δ Sensitivity
- Δ Specificity

Do not label positive deltas “quantum advantage” automatically.

## 11.4 Methodology panel

Expandable:

**Evaluation protocol**

Display exact backend-provided context.

---

# 12. Explainability

Explainability must be strictly scoped to Dataset + Model.

## 12.1 Header

**Explainability**

> Inspect feature-level attribution for the selected dataset and model where a validated explanation artifact is available.

Show:

`Dataset: dynamic`  
`Model: dynamic`

## 12.2 Classical

Where supported:

- global feature attribution;
- local prediction contribution;
- top contributing features.

Use SHAP or the actual backend-provided method.

## 12.3 Quantum

Never label ordinary feature importance “quantum explainability”.

If a validated post-hoc quantum attribution exists, show:

- method;
- model;
- dataset;
- limitations.

Otherwise:

**Explainability unavailable**

> The backend does not expose a validated explanation artifact for this dataset/model combination.

## 12.4 PCA

If applicable, use the exact label:

**PCA Component Loadings**

Never call PCA loadings “feature importance”.

---

# 13. Empty, Partial, Loading, Error States

These are designed states.

### Unavailable

**Not available**

> This dataset/model does not currently expose the required artifact.

### Evaluated but not deployable

**Evaluated offline**

> Valid evaluation results are available, but interactive inference artifacts are not currently available.

### Experimental

**Experimental dataset**

> Included for research evaluation. Interactive deployment is not currently supported.

### Loading

Use skeletons matching final content geometry.

### Error

**Something could not be loaded**

Show what failed and what remains available. Put technical detail in an expander/log, not a traceback.

---

# 14. Interaction Rules

## Dataset switch

Changing dataset must:

1. update metadata;
2. invalidate previous model;
3. load models for the new dataset;
4. update capabilities;
5. clear incompatible prediction/explainability state.

No stale content.

## Model switch

Changing model updates:

- metadata;
- capabilities;
- threshold if model-specific;
- prediction;
- circuit;
- evaluation;
- explainability.

## Loading

Use meaningful messages such as:

- Loading evaluation artifact…
- Loading circuit…
- Preparing model metadata…

Do not create artificial delays.

---

# 15. Charts & Visualization

Charts must answer questions.

Recommended:

### Data
- class distribution;
- feature distributions;
- compact feature-selection/reduction visual.

### Evaluation
- horizontal metric comparison;
- ROC;
- confusion matrix;
- threshold curves;
- timing comparison if available.

### Benchmark
- grouped bars;
- delta bars;
- comparison matrix.

### Explainability
- SHAP summary;
- ranked attribution;
- local contribution.

Avoid:

- 3D charts;
- decorative gauges;
- radar charts as the primary comparison;
- excessive pie charts;
- decorative donuts;
- gratuitous animation.

Do not rely on color alone. Use labels, position, markers, or patterns.

---

# 16. Streamlit Implementation Mapping

Use Streamlit deliberately:

- `st.sidebar` → persistent navigation/context;
- `st.container` → analytical panels;
- `st.columns` → controlled grids;
- horizontal containers → flexible rows;
- `st.tabs` → secondary views;
- `st.expander` → methodology/advanced detail;
- `st.popover` → optional filters;
- placeholders → dynamic result areas.

Avoid deeply nested columns.

Do not implement the design as a page full of `st.metric()` widgets.

Do not build a second frontend framework inside Streamlit.

---

# 17. Backend Contract

Conceptual architecture:

```text
Streamlit UI
     ↓
Application/UI adapter
     ↓
Backend registry + metadata
     ↓
Preprocessing / models / evaluation artifacts
```

UI must not:

- scan arbitrary folders for scientific meaning;
- infer model availability from filename guesses;
- load arbitrary model files;
- reconstruct missing metrics;
- fabricate probabilities;
- import a dataset loader just to obtain display metadata;
- silently fall back to synthetic data;
- train models during normal page rendering.

Backend metadata should expose concepts equivalent to:

```text
dataset
model
model_type
family
display_name
feature_schema
preprocessing
quantum_config
evaluation_protocol
artifacts
capabilities
metrics
availability
```

## Capability model

Use independent flags/states for:

- data_view;
- evaluation;
- interactive_inference;
- probability;
- circuit;
- explainability;
- benchmark.

This prevents an evaluated model from being incorrectly treated as inference-ready.

---

# 18. Model Discovery

Required flow:

```text
backend registry
      ↓
selected dataset
      ↓
models belonging to dataset
      ↓
capabilities
      ↓
UI
```

This is mandatory.

The design must accommodate:

- one model;
- two models;
- ten models;
- different classical/quantum combinations;

without changing the source code for each model.

---

# 19. Terminology

Use:

- Model Prediction
- Estimated Model Probability
- Predicted Class
- Research Prototype
- Evaluation
- Held-out test set
- Independent test cohort
- Classical baseline
- Quantum model
- Benchmark
- Model attribution
- PCA Component Loadings

Avoid:

- clinical diagnosis;
- diagnostic alert;
- clinical standard;
- production clinical model;
- medically safe;
- quantum superiority;
- quantum advantage;
- patient diagnosis.

---

# 20. Judge Demonstration Flow

The UI should support a smooth 2-minute walkthrough.

### 0–15 sec — Overview
Project, active dataset, model inventory, hybrid pipeline.

### 15–35 sec — Data
Dataset size, class distribution, preprocessing, reduction.

### 35–55 sec — Models
Classical branch, quantum branch, circuit/configuration.

### 55–85 sec — Prediction
Input record, inference, model prediction, probability where supported.

### 85–110 sec — Evaluation
Accuracy, sensitivity, specificity, F1, ROC-AUC where available, confusion matrix.

### 110–120 sec — Benchmark / Explainability
Classical-vs-quantum comparison and explanation artifact where available.

The interface must support this flow without navigating through report-like documentation.

---

# 21. Google Stitch Strategy

Create one Stitch project representing the whole product.

Required screens:

1. Overview
2. Data
3. Models
4. Prediction
5. Evaluation
6. Benchmark
7. Explainability
8. Unavailable state
9. Evaluated-but-not-deployable state
10. Loading state
11. Error state

The Stitch project should use one coherent design system.

The resulting `DESIGN.md` should define:

- colors;
- typography;
- spacing;
- radii;
- borders;
- navigation;
- buttons;
- inputs;
- cards;
- tables;
- chart styling;
- status states;
- responsive rules.

The approved Stitch screens are the **visual source of truth** for the Streamlit implementation.

However, placeholder data from Stitch must always be replaced by authoritative backend data.

---

# 22. Stitch Generation Direction

Use the following design intent for the first high-fidelity generation:

> Create a high-fidelity desktop scientific research dashboard for a hybrid quantum–classical machine learning platform for biomedical disease detection. The interface should feel like a premium scientific analytics product: calm light neutral surfaces, charcoal typography, restrained teal/blue accents, subtle indigo/purple reserved for quantum concepts, strong information hierarchy, modular analytical panels, precise charts, structured tables, and an elegant persistent navigation rail. Avoid dark mode, neon, gradients, glassmorphism, cyberpunk, futuristic HUDs, crypto aesthetics, excessive rounded cards, and generic AI visual clichés. The product must look like a real research application rather than a report. Design an Overview dashboard, Data exploration workspace, Model architecture workspace, Prediction workspace, Evaluation analytics page, Classical-vs-Quantum Benchmark page, and Explainability page. Make the classical-to-quantum pipeline visually obvious. Do not imply clinical diagnosis. Prioritize readability, evidence, interaction, and scientific trust.

Refine the prompt after reviewing the first Stitch result rather than accepting the first generation blindly.

---

# 23. Design QA

## Product

- [ ] Immediately recognizable as a dashboard
- [ ] Meaningful interaction
- [ ] Clear hierarchy
- [ ] Distinct page purposes
- [ ] No report-like wall of text

## Visual

- [ ] Light and restrained
- [ ] No neon
- [ ] No gradients
- [ ] No glassmorphism
- [ ] No excessive rounded cards
- [ ] No giant marketing hero
- [ ] Strong typography
- [ ] Consistent spacing
- [ ] Projector readable

## Scientific

- [ ] No fabricated metrics
- [ ] No fabricated probabilities
- [ ] No simulated results presented as measured
- [ ] No unsupported quantum advantage
- [ ] No clinical diagnosis claims
- [ ] No invented feature importance
- [ ] Dataset/model state is visible
- [ ] Evaluation protocol is visible

## Backend

- [ ] Dynamic model discovery
- [ ] Dynamic dataset metadata
- [ ] Dynamic feature schema
- [ ] Dynamic model configuration
- [ ] Artifact-bound prediction
- [ ] Artifact-bound explainability
- [ ] Backend confusion matrices
- [ ] Backend ROC data
- [ ] No page-load retraining

---

# 24. Streamlit QA

Test:

1. Launch.
2. Every dataset.
3. Every dynamically discovered model.
4. Dataset switching.
5. Model switching.
6. Data page.
7. Models page.
8. Prediction.
9. Supported inference.
10. Evaluation.
11. Benchmark.
12. Explainability.
13. Every unavailable state.
14. Invalid input.
15. Missing artifact.
16. Missing metric.
17. Missing ROC data.
18. Missing explainability.
19. Narrow browser.
20. 1280/1440/1920 px desktop.
21. Console/application logs.
22. Confirm no traceback is visible.
23. Confirm no stale dataset/model data remains after context changes.

---

# 25. Acceptance Criteria

The redesign is complete only when:

### Product
- It looks and behaves like a dashboard.
- Overview is visually engaging without gimmicks.
- Every page has a distinct purpose.
- The 2-minute demonstration flow is smooth.

### Backend
- All actual backend models can be discovered.
- No manual model inventory in UI.
- Dataset/model context is authoritative.
- Evaluation is artifact-driven.
- Prediction is artifact-driven.
- Explainability is dataset/model scoped.
- Offline evaluation and inference availability are distinct.

### Reliability
- Data page cannot crash due to missing helpers.
- Circuit page cannot crash from mismatched adapter signatures.
- Prediction cannot silently fall back to fake results.
- Explainability cannot show another model's artifact.
- Evaluation cannot reconstruct missing scientific data.
- Missing capabilities render polished states.

### Scientific integrity
- No mock metrics.
- No mock probabilities.
- No simulated results presented as measured.
- No unsupported quantum advantage.
- No clinical diagnosis claims.
- No invented feature importance.

### Design
- Approved Stitch visual language is followed.
- Streamlit components are consistent.
- Whitespace is intentional.
- Charts are readable.
- Tables are secondary to visual analysis.
- Sidebar is compact and useful.
- The product feels like a real scientific research platform.

---

# 26. Final Principle

The application is **not a report about the project**.

It is the **working interface through which the project can be explored**.

Every page should answer:

1. Where am I?
2. What dataset/model is active?
3. What can I do here?
4. What is the important result?
5. What evidence supports it?
6. What can I inspect next?

The final visual story is:

**Biomedical Data → Classical Preparation → Compact Representation → Classical + Quantum Models → Prediction → Evaluation → Explanation**

---

# 27. Authority Order

If future implementation decisions conflict:

1. Scientific/backend artifacts
2. Project scientific requirements
3. This design document
4. Approved Stitch design
5. Streamlit implementation constraints
6. Developer convenience

Never reverse this order.
