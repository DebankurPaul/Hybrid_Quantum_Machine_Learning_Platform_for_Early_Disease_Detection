# 🧬 Hybrid Quantum Machine Learning Platform for Early Disease Detection

> **A complete beginner-to-implementation guide.**
> *This README assumes you know Python and basic ML — but zero quantum computing.*

---

## Table of Contents

1. [What is Quantum Computing? (Plain English)](#1-what-is-quantum-computing-plain-english)
2. [What is Quantum Machine Learning?](#2-what-is-quantum-machine-learning)
3. [Why Hybrid? Why Not Pure Quantum?](#3-why-hybrid-why-not-pure-quantum)
4. [The WDBC Dataset](#4-the-wdbc-dataset)
5. [Our Architecture — The Big Picture](#5-our-architecture--the-big-picture)
6. [Project Structure](#6-project-structure)
7. [Environment Setup](#7-environment-setup)
8. [Module 1 — Data Loading & EDA](#8-module-1--data-loading--eda)
9. [Module 2 — Pre-processing Pipeline](#9-module-2--pre-processing-pipeline)
10. [Module 3 — The Quantum Circuit (Core)](#10-module-3--the-quantum-circuit-core)
11. [Module 4 — The Quantum Model Class](#11-module-4--the-quantum-model-class)
12. [Module 5 — Training Loop](#12-module-5--training-loop)
13. [Module 6 — Classical Baselines](#13-module-6--classical-baselines)
14. [Module 7 — Evaluation & Benchmarking](#14-module-7--evaluation--benchmarking)
15. [Module 8 — Explainability](#15-module-8--explainability)
16. [Notebook Walkthrough](#16-notebook-walkthrough)
17. [Expected Results](#17-expected-results)
18. [Glossary](#18-glossary)
19. [References](#19-references)

---

## 1. What is Quantum Computing? (Plain English)

### Classical Bit vs Qubit

In your laptop, everything is **bits** — 0 or 1. A classical bit is like a light switch: either OFF or ON.

A **qubit** (quantum bit) is like a *spinning coin*. While it is in the air, it is **both heads AND tails at the same time**. This is called **superposition**. The moment you catch it (measure it), it collapses to either heads or tails.

```
Classical bit:   0 or 1                (definite)
Qubit:           a|0> + b|1>           (both, until measured)
                 a,b are probability amplitudes
                 |a|^2 + |b|^2 = 1
```

---

### Key Quantum Phenomena

#### Superposition
A qubit exists in a combination of 0 and 1 simultaneously. Mathematically:

```
|psi> = a|0> + b|1>

|a|^2 = probability of measuring 0
|b|^2 = probability of measuring 1
```

#### Entanglement
Two qubits can be **entangled** — the state of one instantly determines the state of the other.

```
Bell state: (1/sqrt(2))(|00> + |11>)
If you measure qubit 1 = 0, qubit 2 is GUARANTEED = 0.
```

Why it matters for ML: Entanglement allows a quantum model to capture feature correlations natively.

#### Measurement
When you measure a qubit, superposition collapses to 0 or 1. We use **expectation value** <Z> — the average result — which gives a continuous value in [-1, 1]:

```
<Z> = +1  means qubit in |0>  = class 0 (Benign)
<Z> = -1  means qubit in |1>  = class 1 (Malignant)
<Z> =  0  means 50/50         = uncertain
```

---

### Quantum Gates

| Gate | What it does |
|------|-------------|
| Hadamard (H) | Puts qubit in 50/50 superposition |
| RY(theta) | Rotates qubit around Y-axis by angle theta |
| RZ(phi) | Rotates qubit around Z-axis by angle phi |
| CNOT | Entangles 2 qubits: flips target if control=|1> |

Think of the qubit as a point on a **Bloch sphere** (a 3D ball). Gates rotate this point.

---

## 2. What is Quantum Machine Learning?

### Classical ML Recap

```
Classical Neural Net:
Input x -> Linear(Wx + b) -> Activation -> Output y_hat
Loss = BCE(y_hat, y)
Gradient -> update W  (backpropagation)
```

### Quantum ML (QML)

In QML, the model is a **Parameterized Quantum Circuit (PQC)** — also called a **Variational Quantum Classifier (VQC)**. Parameters are rotation angles of quantum gates.

```
Quantum ML Pipeline:
Classical Data -> [Encoding] -> Quantum Circuit(theta) -> Measurement -> Loss
                                       ^
                               Trainable angles theta
                               optimized by gradient descent
```

The gradient is computed using the **Parameter-Shift Rule** — the quantum analog of backpropagation.

---

## 3. Why Hybrid? Why Not Pure Quantum?

We are in the **NISQ era** (Noisy Intermediate-Scale Quantum). Current quantum computers have:
- 50-1000 qubits (but very noisy)
- Short coherence times
- Gate errors (~0.1-1% per gate)

**The Hybrid Solution:**

```
Classical Computer:              Quantum Processor:
- Data loading                   - Feature encoding
- Pre-processing                 - VQC forward pass
- Loss computation               - Expectation values
- Gradient + weight updates      - Entanglement + Superposition
```

---

## 4. The WDBC Dataset

**Wisconsin Diagnostic Breast Cancer (WDBC)** — UCI Machine Learning Repository

- **569 samples** (212 Malignant, 357 Benign)
- **30 features** — computed from digitized images of fine needle aspirate (FNA)
- **Binary target**: M (Malignant = 1), B (Benign = 0)
- **Load**: `sklearn.datasets.load_breast_cancer()`

### The 30 Features (10 measurements x 3 statistics each)

| Nucleus Characteristic | mean | SE | worst |
|---|---|---|---|
| Radius | radius_mean | radius_se | radius_worst |
| Texture | texture_mean | texture_se | texture_worst |
| Perimeter | perimeter_mean | perimeter_se | perimeter_worst |
| Area | area_mean | area_se | area_worst |
| Smoothness | smoothness_mean | smoothness_se | smoothness_worst |
| Compactness | compactness_mean | compactness_se | compactness_worst |
| Concavity | concavity_mean | concavity_se | concavity_worst |
| Concave points | concave_mean | concave_se | concave_worst |
| Symmetry | symmetry_mean | symmetry_se | symmetry_worst |
| Fractal dimension | fractal_mean | fractal_se | fractal_worst |

---

## 5. Our Architecture — The Big Picture

```
WDBC Dataset (569 x 30)
        |
        v
DATA PIPELINE
  1. Load + split (80/10/10)
  2. StandardScaler
  3. PCA -> 4 components (87% variance)
  4. tanh normalization -> [-pi, pi]
        |
        +----------------+----------------+
        |                |                |
        v                v                v
CLASSICAL BRANCH    QUANTUM BRANCH     EVALUATION
LR / SVM / RF       VQC (4 qubits)     Accuracy, AUC-ROC
All 30 features     Angle Encoding     Sensitivity
~96-97% acc         RY+RZ+CNOT layers  Specificity, F1
                    <Z> -> sigmoid
                    ~92-95% acc
        |                |
        +----------------+
                |
                v
        EXPLAINABILITY
        SHAP + Circuit diagrams
```

---

## 6. Project Structure

```
Hybrid_Quantum_Machine_Learning_Platform_for_Early_Disease_Detection/
|
+-- data/
|   +-- raw/                         # Store original WDBC CSV here
|   +-- processed/                   # Numpy arrays after preprocessing
|
+-- notebooks/
|   +-- 01_wdbc_preprocessing.ipynb  # EDA + data prep
|   +-- 02_classical_baselines.ipynb # LR, SVM, RF training
|   +-- 03_quantum_vqc_model.ipynb   # Build + train quantum model
|   +-- 04_benchmarking.ipynb        # Compare all models
|   +-- 05_explainability.ipynb      # SHAP + circuit viz
|
+-- src/
|   +-- __init__.py
|   +-- data_loader.py               # load_wdbc()
|   +-- preprocessing.py             # scale(), pca_reduce()
|   +-- classical_models.py          # train_lr(), train_svm(), train_rf()
|   +-- quantum_circuit.py           # THE CORE QUANTUM CIRCUIT
|   +-- quantum_model.py             # QuantumClassifier class
|   +-- train_quantum.py             # Training loop
|   +-- evaluate.py                  # All metrics + plots
|   +-- explainability.py            # SHAP analysis
|
+-- results/
|   +-- figures/
|   +-- metrics.json
|
+-- requirements.txt
+-- README.md
```

---

## 7. Environment Setup

```bash
pip install pennylane pennylane-lightning scikit-learn torch shap pandas numpy matplotlib seaborn jupyterlab plotly ucimlrepo
```

### requirements.txt

```
pennylane>=0.38.0
pennylane-lightning>=0.38.0
scikit-learn>=1.4.0
torch>=2.2.0
shap>=0.45.0
pandas>=2.1.0
numpy>=1.26.0
matplotlib>=3.8.0
seaborn>=0.13.0
jupyterlab>=4.0.0
plotly>=5.18.0
ucimlrepo>=0.0.3
```

### Verify Installation

```python
import pennylane as qml
print(qml.version())   # 0.38.x or higher
print(qml.about())     # Lists all backends
```

---

## 8. Module 1 — Data Loading & EDA

### src/data_loader.py

```python
"""
data_loader.py
Loads WDBC dataset and performs train/val/test split.
"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def load_wdbc():
    """
    Returns WDBC as a pandas DataFrame.
    We encode: 1 = Malignant, 0 = Benign (clinical convention)
    sklearn reverses these, so we flip.
    """
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = 1 - data.target   # flip: 1=Malignant, 0=Benign
    df['diagnosis'] = df['target'].map({1: 'Malignant', 0: 'Benign'})
    return df

def get_splits(df, test_size=0.2, val_size=0.1, random_state=42):
    """
    Splits DataFrame into train, validation, and test sets.
    Returns: X_train, X_val, X_test, y_train, y_val, y_test
    """
    X = df.drop(columns=['target', 'diagnosis']).values
    y = df['target'].values

    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_ratio, random_state=random_state, stratify=y_tv
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
```

### EDA Checklist (Notebook 01)

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = load_wdbc()

# 1. Class distribution
print(df['diagnosis'].value_counts())
# Benign: 357  |  Malignant: 212

# 2. Correlation heatmap
top_features = df.drop(columns=['target','diagnosis']).corrwith(df['target'])
top_features = top_features.abs().sort_values(ascending=False)[:10]
sns.heatmap(df[top_features.index].corr(), annot=True, fmt='.2f', cmap='coolwarm')

# 3. PCA variance explained
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop(columns=['target','diagnosis']))
pca = PCA()
pca.fit(X_scaled)
cumvar = pca.explained_variance_ratio_.cumsum()
print(f"Variance in 4 PCs:  {cumvar[3]:.1%}")   # ~87%
print(f"Variance in 6 PCs:  {cumvar[5]:.1%}")   # ~93%
```

---

## 9. Module 2 — Pre-processing Pipeline

### src/preprocessing.py

```python
"""
preprocessing.py
Feature scaling, PCA reduction, and angle normalization.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def preprocess(X_train, X_val, X_test, n_qubits=4):
    """
    Full pre-processing pipeline:
    1. StandardScaler: zero mean, unit variance
    2. PCA: reduce to n_qubits dimensions
    3. tanh normalization: map to [-pi, pi] for angle encoding

    Args:
        X_train, X_val, X_test: raw feature arrays (n_samples, 30)
        n_qubits: number of qubits = number of PCA components

    Returns:
        Processed arrays + fitted scaler + fitted PCA
    """
    # Step 1: Standardize
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)

    # Step 2: PCA reduction
    pca = PCA(n_components=n_qubits, random_state=42)
    X_train_pca = pca.fit_transform(X_train_sc)
    X_val_pca   = pca.transform(X_val_sc)
    X_test_pca  = pca.transform(X_test_sc)

    print(f"PCA: {n_qubits} components explain "
          f"{pca.explained_variance_ratio_.sum():.1%} of variance")

    # Step 3: Map to [-pi, pi] using tanh
    # tanh squashes to (-1,1), multiply by pi -> (-pi, pi)
    def angle_normalize(X):
        return np.tanh(X) * np.pi

    X_train_q = angle_normalize(X_train_pca)
    X_val_q   = angle_normalize(X_val_pca)
    X_test_q  = angle_normalize(X_test_pca)

    return X_train_q, X_val_q, X_test_q, scaler, pca


def save_processed(arrays, path_prefix='../data/processed/'):
    """Save processed numpy arrays to disk."""
    names = ['X_train_q', 'X_val_q', 'X_test_q', 'y_train', 'y_val', 'y_test']
    for name, arr in zip(names, arrays):
        np.save(f"{path_prefix}{name}.npy", arr)
    print(f"Saved {len(arrays)} arrays to {path_prefix}")
```

### Why Each Step Matters

| Step | Why? |
|------|------|
| StandardScaler | Zero-mean, unit-variance data works best for both classical and quantum models |
| PCA | Quantum simulators limited to ~4-20 qubits; PCA gives most informative features |
| tanh normalization | Quantum RY(theta) gates need angles in [-pi, pi]; tanh smoothly maps any value |

---

## 10. Module 3 — The Quantum Circuit (Core)

### src/quantum_circuit.py

```python
"""
quantum_circuit.py
Defines the Variational Quantum Classifier (VQC) circuit.

ARCHITECTURE:
  1. ENCODING LAYER   -- maps classical features to qubit rotations
  2. VARIATIONAL LAYERS -- parameterized gates (what gets trained)
  3. MEASUREMENT        -- extracts classical output from quantum state
"""
import pennylane as qml
import numpy as np

N_QUBITS = 4      # Must equal number of PCA components
N_LAYERS = 2      # Number of variational blocks
# Total trainable parameters: N_LAYERS x N_QUBITS x 2 = 16

# Device selection
try:
    dev = qml.device("lightning.qubit", wires=N_QUBITS)
    print("Using lightning.qubit (fast C++ simulator)")
except Exception:
    dev = qml.device("default.qubit", wires=N_QUBITS)
    print("Falling back to default.qubit")


@qml.qnode(dev, interface="torch", diff_method="parameter-shift")
def vqc_circuit(inputs, weights):
    """
    VQC circuit with angle encoding + variational layers.

    Args:
        inputs  : shape (N_QUBITS,) -- one feature per qubit, in (-pi, pi)
        weights : shape (N_LAYERS, N_QUBITS, 2) -- trainable rotation angles

    Returns:
        Expectation value <Z_0> in [-1, 1]

    Circuit layout (4 qubits, 2 layers):
      q0: RY(x0) - RY(w00) - RZ(w01) - C - - - - - X - <Z>
      q1: RY(x1) - RY(w10) - RZ(w11) - X - C - - - |
      q2: RY(x2) - RY(w20) - RZ(w21) - - - X - C - |
      q3: RY(x3) - RY(w30) - RZ(w31) - - - - - X - C (ring)
           Encoding     Layer 1         CNOT entanglement ring
    """

    # BLOCK 1: ENCODING LAYER
    # Each classical feature xi becomes a rotation angle for qubit i.
    # RY(xi)|0> = cos(xi/2)|0> + sin(xi/2)|1>
    # The qubit is in superposition weighted by the feature value.
    for i in range(N_QUBITS):
        qml.RY(inputs[i], wires=i)

    # BLOCK 2: VARIATIONAL LAYERS (trainable)
    for layer in range(N_LAYERS):

        # Single-qubit rotations (trainable)
        # RY rotates on Y-axis, RZ on Z-axis
        # Together they can reach any point on the Bloch sphere
        for i in range(N_QUBITS):
            qml.RY(weights[layer, i, 0], wires=i)
            qml.RZ(weights[layer, i, 1], wires=i)

        # Entangling gates (CNOT ring topology)
        # Creates quantum correlations between adjacent qubits
        # This is where quantum advantage comes from!
        for i in range(N_QUBITS):
            qml.CNOT(wires=[i, (i + 1) % N_QUBITS])

    # BLOCK 3: MEASUREMENT
    # Measure expectation value of Pauli-Z on qubit 0.
    # <Z> = P(measure 0) - P(measure 1) in [-1, +1]
    return qml.expval(qml.PauliZ(0))


def draw_circuit(sample_input=None):
    """Print a text diagram of the circuit."""
    if sample_input is None:
        sample_input = np.zeros(N_QUBITS)
    sample_weights = np.zeros((N_LAYERS, N_QUBITS, 2))
    print(qml.draw(vqc_circuit)(sample_input, sample_weights))


def count_parameters():
    """Return total number of trainable parameters."""
    total = N_LAYERS * N_QUBITS * 2
    print(f"Trainable parameters: {N_LAYERS} x {N_QUBITS} x 2 = {total}")
    return total
```

### Understanding the Parameter-Shift Rule

This is the quantum equivalent of backpropagation. You cannot directly differentiate a quantum measurement. Instead:

```
dL/dtheta = [L(theta + pi/2) - L(theta - pi/2)] / 2

Run the circuit twice per parameter, shifted by +/-pi/2.
PennyLane handles this automatically with diff_method="parameter-shift".
```

---

## 11. Module 4 — The Quantum Model Class

### src/quantum_model.py

```python
"""
quantum_model.py
QuantumClassifier wraps the VQC circuit in a PyTorch nn.Module,
compatible with standard training loops and sklearn-style APIs.
"""
import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
from src.quantum_circuit import N_QUBITS, N_LAYERS

class QuantumClassifier(nn.Module):
    """
    Hybrid Quantum-Classical Classifier using VQC.
    Parameters are torch.nn.Parameter tensors.
    """
    def __init__(self, n_qubits=N_QUBITS, n_layers=N_LAYERS):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        try:
            self.dev = qml.device("lightning.qubit", wires=n_qubits)
        except Exception:
            self.dev = qml.device("default.qubit", wires=n_qubits)

        self.qnode = qml.QNode(
            self._circuit, self.dev,
            interface="torch", diff_method="parameter-shift"
        )

        # Initialize small random angles to avoid barren plateaus
        self.weights = nn.Parameter(
            torch.randn(n_layers, n_qubits, 2) * 0.1
        )

    def _circuit(self, inputs, weights):
        for i in range(self.n_qubits):
            qml.RY(inputs[i], wires=i)
        for layer in range(self.n_layers):
            for i in range(self.n_qubits):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)
            for i in range(self.n_qubits):
                qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
        return qml.expval(qml.PauliZ(0))

    def forward(self, X):
        """
        Forward pass: process a batch.
        X shape: (batch_size, n_qubits)
        Returns: probabilities in [0, 1]
        """
        raw_outputs = torch.stack([
            self.qnode(x, self.weights) for x in X
        ])
        return torch.sigmoid(raw_outputs)   # <Z> in [-1,1] -> prob in [0,1]

    def predict(self, X_numpy):
        """sklearn-compatible predict. Returns class labels 0/1."""
        X_tensor = torch.tensor(X_numpy, dtype=torch.float32)
        with torch.no_grad():
            probs = self.forward(X_tensor).numpy()
        return (probs > 0.5).astype(int)

    def predict_proba(self, X_numpy):
        """sklearn-compatible predict_proba. Returns [P(0), P(1)]."""
        X_tensor = torch.tensor(X_numpy, dtype=torch.float32)
        with torch.no_grad():
            probs = self.forward(X_tensor).numpy()
        return np.column_stack([1 - probs, probs])

    def summary(self):
        total = self.n_layers * self.n_qubits * 2
        print(f"QuantumClassifier: {self.n_qubits} qubits, "
              f"{self.n_layers} layers, {total} parameters")
```

### The Barren Plateau Problem

A critical challenge in training VQCs:

```
Problem: Deep random circuits -> gradients become exponentially small
Effect:  dL/dtheta ~= 0 everywhere -> optimizer is blind

Our fix:
- Initialize weights * 0.1 (small angles)
- Keep layers shallow (N_LAYERS = 2)
- Monitor gradient norms during training
```

---

## 12. Module 5 — Training Loop

### src/train_quantum.py

```python
"""
train_quantum.py
Training loop for QuantumClassifier with Adam optimizer and BCE loss.
"""
import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
import time

def train_quantum_model(
    model, X_train, y_train,
    X_val=None, y_val=None,
    n_epochs=50, lr=0.01, batch_size=16,
    patience=10, save_path=None
):
    """
    Train QuantumClassifier.

    Args:
        model     : QuantumClassifier instance
        X_train   : numpy array (n_train, n_qubits)
        y_train   : numpy array (n_train,) with 0/1 labels
        n_epochs  : maximum training epochs
        lr        : learning rate (0.01 works well for VQCs)
        batch_size: samples per gradient update
        patience  : early stopping patience
        save_path : save best model weights here (optional)

    Returns:
        history dict with loss, train_acc, val_acc per epoch
    """
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn   = nn.BCELoss()

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32)

    if X_val is not None:
        X_v = torch.tensor(X_val, dtype=torch.float32)
        y_v = torch.tensor(y_val, dtype=torch.float32)

    history = {"epoch": [], "loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    no_improve   = 0

    print(f"Training QuantumClassifier | Epochs={n_epochs} | LR={lr} | Batch={batch_size}")

    t_start = time.time()

    for epoch in range(1, n_epochs + 1):
        model.train()
        perm = torch.randperm(len(X_tr))
        epoch_loss, n_batches = 0.0, 0

        for i in range(0, len(X_tr), batch_size):
            idx  = perm[i : i + batch_size]
            xb, yb = X_tr[idx], y_tr[idx]

            optimizer.zero_grad()
            pred = model(xb).squeeze()
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / n_batches

        model.eval()
        with torch.no_grad():
            tr_acc = ((model(X_tr).squeeze() > 0.5).float() == y_tr).float().mean().item()
            val_acc = None
            if X_val is not None:
                val_acc = ((model(X_v).squeeze() > 0.5).float() == y_v).float().mean().item()

        history["epoch"].append(epoch)
        history["loss"].append(avg_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t_start
        val_str = f"Val: {val_acc:.4f}" if val_acc else ""
        print(f"Ep {epoch:3d}/{n_epochs} | Loss: {avg_loss:.4f} | "
              f"Train: {tr_acc:.4f} | {val_str} | {elapsed:.0f}s")

        # Early stopping
        if val_acc and val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            if save_path:
                torch.save(model.state_dict(), save_path)
        elif val_acc:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}. Best val: {best_val_acc:.4f}")
                break

    return history


def plot_training_history(history, save_path=None):
    """Plot loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history["epoch"], history["loss"], 'b-o', markersize=4)
    ax1.set(xlabel='Epoch', ylabel='BCE Loss', title='Training Loss')
    ax1.grid(alpha=0.3)

    ax2.plot(history["epoch"], history["train_acc"], 'g-o', markersize=4, label='Train')
    if history["val_acc"][0]:
        ax2.plot(history["epoch"], history["val_acc"], 'r-s', markersize=4, label='Val')
    ax2.set(xlabel='Epoch', ylabel='Accuracy', title='Accuracy', ylim=[0.5, 1.05])
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
```

> **Expected training time**: 3-10 minutes for 50 epochs on CPU with lightning.qubit backend.

---

## 13. Module 6 — Classical Baselines

### src/classical_models.py

```python
"""
classical_models.py
Trains three classical ML baselines on ALL 30 WDBC features.
"""
from sklearn.linear_model  import LogisticRegression
from sklearn.svm            import SVC
from sklearn.ensemble       import RandomForestClassifier
from sklearn.preprocessing  import StandardScaler
from sklearn.metrics        import (accuracy_score, roc_auc_score, f1_score)
import time

def train_classical_baselines(X_train, X_test, y_train, y_test):
    """
    Train all classical baselines on ALL 30 features (no PCA).
    This gives them full advantage for a fair comparison.
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM (RBF Kernel)"   : SVC(probability=True, random_state=42),
        "Random Forest"      : RandomForestClassifier(n_estimators=100, random_state=42),
    }

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    results = {}
    for name, clf in models.items():
        t0 = time.time()
        clf.fit(X_tr, y_train)
        train_time = time.time() - t0

        y_pred  = clf.predict(X_te)
        y_proba = clf.predict_proba(X_te)[:, 1]

        results[name] = {
            "accuracy"   : accuracy_score(y_test, y_pred),
            "roc_auc"    : roc_auc_score(y_test, y_proba),
            "f1_score"   : f1_score(y_test, y_pred),
            "train_time" : train_time,
            "model"      : clf,
            "scaler"     : scaler
        }
        print(f"{name}: Acc={results[name]['accuracy']:.4f} | "
              f"AUC={results[name]['roc_auc']:.4f} | Time={train_time:.3f}s")

    return results
```

---

## 14. Module 7 — Evaluation & Benchmarking

### src/evaluate.py

```python
"""
evaluate.py
Comprehensive evaluation metrics and visualization.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve,
    classification_report, accuracy_score, f1_score,
    precision_score, recall_score
)
import json

def compute_metrics(y_true, y_pred, y_proba, model_name="Model"):
    """Compute all relevant clinical metrics."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "model"       : model_name,
        "accuracy"    : accuracy_score(y_true, y_pred),
        "precision"   : precision_score(y_true, y_pred, zero_division=0),
        "sensitivity" : recall_score(y_true, y_pred),          # TP/(TP+FN)
        "specificity" : tn / (tn + fp) if (tn + fp) > 0 else 0, # TN/(TN+FP)
        "f1_score"    : f1_score(y_true, y_pred),
        "roc_auc"     : roc_auc_score(y_true, y_proba),
    }

    print(f"\n{'='*50}\n  {model_name}\n{'='*50}")
    print(classification_report(y_true, y_pred,
                                 target_names=["Benign","Malignant"]))
    print(f"  Specificity : {metrics['specificity']:.4f}")
    print(f"  AUC-ROC     : {metrics['roc_auc']:.4f}")
    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred Benign','Pred Malignant'],
                yticklabels=['True Benign','True Malignant'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150)
    plt.show()


def plot_roc_curves(model_names, y_true_list, y_proba_list, save_path=None):
    """Overlay ROC curves for all models."""
    plt.figure(figsize=(8, 6))
    colors = ['#2ECC71','#3498DB','#9B59B6','#E74C3C','#F39C12']
    for name, y_true, y_proba, color in zip(model_names, y_true_list, y_proba_list, colors):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        plt.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC={auc:.3f})')
    plt.plot([0,1],[0,1],'k--', lw=1)
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('ROC Curves - All Models')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150)
    plt.show()


def benchmark_table(all_metrics):
    """Print comparison table."""
    import pandas as pd
    cols = ['model','accuracy','sensitivity','specificity','f1_score','roc_auc']
    df = pd.DataFrame(all_metrics)[cols].sort_values('roc_auc', ascending=False)
    print("\n" + "="*80 + "\nBENCHMARKING RESULTS\n" + "="*80)
    print(df.to_string(index=False))
    return df
```

---

## 15. Module 8 — Explainability

### src/explainability.py

```python
"""
explainability.py
Model-agnostic explainability using SHAP KernelExplainer.
"""
import numpy as np
import shap
import matplotlib.pyplot as plt
import pennylane as qml

def shap_analysis(predict_fn, X_train_sample, X_test_sample,
                  feature_names, model_name="Model", save_path=None):
    """
    Run SHAP KernelExplainer on any model.

    KernelExplainer is model-agnostic -- treats model as a black box.
    Works with quantum model's predict_proba wrapper.

    Args:
        predict_fn    : function(X_numpy) -> P(class=1), shape (n,)
        X_train_sample: background dataset (first 50 rows)
        X_test_sample : samples to explain (first 20 rows)
        feature_names : list of names, e.g. ['PC1','PC2','PC3','PC4']
    """
    print(f"SHAP analysis for {model_name} (takes 2-5 min for quantum)...")

    explainer  = shap.KernelExplainer(predict_fn, X_train_sample[:50])
    shap_vals  = explainer.shap_values(X_test_sample[:20], nsamples=100)

    shap.summary_plot(shap_vals, X_test_sample[:20],
                      feature_names=feature_names, show=False)
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return shap_vals


def plot_circuit_diagram(circuit_fn, n_qubits=4, n_layers=2):
    """Draw the VQC circuit using PennyLane's built-in drawer."""
    import pennylane.numpy as pnp
    sample_inputs  = pnp.zeros(n_qubits)
    sample_weights = pnp.zeros((n_layers, n_qubits, 2))

    print("VQC Circuit Diagram:")
    print(qml.draw(circuit_fn)(sample_inputs, sample_weights))

    fig, ax = qml.draw_mpl(circuit_fn)(sample_inputs, sample_weights)
    fig.suptitle("Variational Quantum Classifier Circuit", fontsize=14)
    plt.savefig('../results/figures/vqc_circuit_diagram.png', dpi=150)
    plt.show()
```

---

## 16. Notebook Walkthrough

### Notebook 01: Pre-processing (01_wdbc_preprocessing.ipynb)

```python
# Cell 1
import sys; sys.path.append('..')
from src.data_loader import load_wdbc, get_splits
from src.preprocessing import preprocess, save_processed

# Cell 2
df = load_wdbc()
print(df.shape)
print(df['diagnosis'].value_counts())

# Cell 3
X_train, X_val, X_test, y_train, y_val, y_test = get_splits(df)
X_train_q, X_val_q, X_test_q, scaler, pca = preprocess(
    X_train, X_val, X_test, n_qubits=4
)
print("Quantum-ready shapes:", X_train_q.shape, X_test_q.shape)

# Cell 4
save_processed([X_train_q, X_val_q, X_test_q, y_train, y_val, y_test])
```

### Notebook 03: Quantum VQC (03_quantum_vqc_model.ipynb)

```python
# Cell 1: Imports and load data
import sys; sys.path.append('..')
from src.quantum_model import QuantumClassifier
from src.train_quantum import train_quantum_model, plot_training_history
from src.quantum_circuit import draw_circuit, count_parameters
import numpy as np

X_train_q = np.load('../data/processed/X_train_q.npy')
X_val_q   = np.load('../data/processed/X_val_q.npy')
y_train   = np.load('../data/processed/y_train.npy')
y_val     = np.load('../data/processed/y_val.npy')

# Cell 2: Inspect circuit
draw_circuit()     # prints ASCII diagram
count_parameters() # 16 trainable params

# Cell 3: Build model
model = QuantumClassifier(n_qubits=4, n_layers=2)
model.summary()

# Cell 4: Train (takes 3-10 minutes on CPU)
history = train_quantum_model(
    model, X_train_q, y_train,
    X_val=X_val_q, y_val=y_val,
    n_epochs=50, lr=0.01, batch_size=16,
    patience=10,
    save_path='../results/best_quantum_model.pt'
)

# Cell 5: Plot
plot_training_history(history, save_path='../results/figures/training_curves.png')
```

---

## 17. Expected Results

| Model | Features | Accuracy | Sensitivity | Specificity | AUC-ROC | Train Time | Params |
|-------|----------|----------|-------------|-------------|---------|------------|--------|
| Logistic Regression | All 30 | ~96.5% | ~95.1% | ~97.4% | ~0.990 | <1s | 30 |
| SVM (RBF) | All 30 | ~97.4% | ~96.2% | ~98.2% | ~0.994 | <1s | -- |
| Random Forest | All 30 | ~96.5% | ~95.1% | ~97.4% | ~0.993 | <2s | -- |
| **VQC (4q, 2L)** | **PCA-4** | **~92-95%** | **~91-94%** | **~93-96%** | **~0.970** | **3-10 min** | **16** |

### Key Takeaway

The VQC achieves competitive cancer detection accuracy with **only 16 trainable parameters** vs complex classical models using all 30 features. This demonstrates quantum-enhanced learning viability for biomedical classification.

### How to Improve Performance

| Approach | Expected improvement |
|---|---|
| Increase N_QUBITS to 6 | +1-2% accuracy |
| Increase N_LAYERS to 3 | +1-2% expressibility |
| Add hybrid classical linear layer after circuit | +2-4% |
| Use AmplitudeEmbedding (encode all 30 features in 5 qubits) | More information preserved |

---

## 18. Glossary

| Term | Plain English |
|------|--------------|
| **Qubit** | Quantum bit. Can be 0, 1, or both at once (superposition) |
| **Superposition** | A qubit being in multiple states simultaneously |
| **Entanglement** | Two qubits whose states are correlated; measuring one instantly affects the other |
| **Measurement** | Forcing a qubit to collapse to 0 or 1 |
| **Expectation value <Z>** | Average measurement outcome, continuous in [-1,1] |
| **Bloch sphere** | 3D sphere representing all possible states of a single qubit |
| **Quantum gate** | A reversible rotation operation on a qubit |
| **RY(theta)** | Rotates qubit by angle theta around the Y-axis |
| **RZ(phi)** | Rotates qubit by angle phi around the Z-axis |
| **CNOT** | Entangles 2 qubits: flips target if control is |1> |
| **PQC / VQC** | Parameterized/Variational Quantum Circuit — the quantum neural network |
| **Angle encoding** | Mapping a classical feature to a qubit rotation angle |
| **Parameter-shift rule** | Quantum analog of backpropagation |
| **Barren plateau** | Training failure where gradients vanish exponentially for deep circuits |
| **NISQ** | Noisy Intermediate-Scale Quantum — era of current quantum computers |
| **PennyLane** | Python library for quantum ML |
| **QNode** | A PennyLane quantum function that can be differentiated |
| **SHAP** | SHapley Additive exPlanations — explains any model's predictions |
| **Sensitivity** | TP/(TP+FN) — how often malignant cases are correctly detected |
| **Specificity** | TN/(TN+FP) — how often benign cases are correctly identified |

---

## 19. References

### Papers
1. **Biamonte et al. (2017)** — "Quantum machine learning" — *Nature* 549:195-202 (start here)
2. **Cerezo et al. (2021)** — "Variational quantum algorithms" — *Nature Reviews Physics*
3. **Havlicek et al. (2019)** — "Supervised learning with quantum-enhanced feature spaces" — *Nature* 567
4. **McClean et al. (2018)** — "Barren plateaus in quantum neural network training" — *Nature Communications*

### Datasets
5. **Street et al. (1993)** — WDBC original paper
6. UCI ML Repository: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

### Libraries
7. PennyLane: https://pennylane.ai/
8. PennyLane QML Demos: https://pennylane.ai/qml/demos/
9. SHAP: https://shap.readthedocs.io/

### Free Learning Resources
10. **PennyLane Codebook** (interactive, free): https://pennylane.ai/codebook/
11. **Qiskit Textbook** (free): https://qiskit.org/learn/
12. **Nielsen & Chuang** — "Quantum Computation and Quantum Information" (the bible)

---

*Built for Smart India Hackathon — Problem Statement 26139*
*Organization: Egreen Quanta | Theme: MedTech / BioTech / HealthTech*
*Platform: Hybrid Quantum Machine Learning for Early Disease Detection (WDBC)*
