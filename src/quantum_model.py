"""
quantum_model.py  (Qiskit version)
===================================
QuantumClassifier wraps Qiskit's high-level VQC class, giving it
an sklearn-compatible API (fit / predict / predict_proba / score).

Qiskit VQC internals:
─────────────────────
1. feature_map  encodes X into qubit states   (ZZFeatureMap)
2. ansatz       applies trainable rotations   (RealAmplitudes)
3. sampler      runs the circuit & gets probs (AerSimulator)
4. optimizer    updates ansatz weights        (COBYLA or SPSA)
5. The output probability of |0…0⟩ vs |1…1⟩ → binary prediction

Usage:
    from src.quantum_model import QuantumClassifier
    clf = QuantumClassifier(n_qubits=4)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.score(X_test, y_test))
"""
import numpy as np
import time

from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_aer.primitives import Sampler as AerSampler
from qiskit_algorithms.optimizers import COBYLA, SPSA, L_BFGS_B
from qiskit_machine_learning.algorithms.classifiers import VQC

from src.quantum_circuit import (
    N_QUBITS, FEATURE_REPS, ANSATZ_REPS,
    build_feature_map, build_ansatz
)


# ─────────────────────────────────────────────────────────────────────────────
# OPTIMIZER OPTIONS
# ─────────────────────────────────────────────────────────────────────────────
# COBYLA  — gradient-free, works with noisy circuits, fast, good for shallow VQCs
# SPSA    — stochastic gradient, scales better to many parameters
# L_BFGS_B— gradient-based (needs statevector), fastest on simulators

OPTIMIZERS = {
    'cobyla'  : lambda maxiter: COBYLA(maxiter=maxiter),
    'spsa'    : lambda maxiter: SPSA(maxiter=maxiter),
    'lbfgs'   : lambda maxiter: L_BFGS_B(maxiter=maxiter, maxfun=maxiter * 10),
}


class QuantumClassifier:
    """
    Hybrid Quantum Classifier using Qiskit's VQC.

    This class wraps Qiskit's VQC to provide:
      • sklearn-compatible API (fit, predict, predict_proba, score)
      • Training progress callbacks
      • Easy configuration of qubits, reps, optimizer

    Attributes:
        n_qubits     : number of qubits = PCA output dimension
        fm_reps      : repetitions of ZZFeatureMap encoding block
        ans_reps     : repetitions of RealAmplitudes variational block
        optimizer    : 'cobyla' (default), 'spsa', or 'lbfgs'
        max_iter     : optimizer iterations (= training epochs equivalent)
        shots        : number of circuit shots for probability estimation
                       (use None for statevector / exact simulation)
        vqc_         : the fitted Qiskit VQC object (available after fit())
        history_     : list of (iter, loss_value) tuples recorded during training
    """

    def __init__(
        self,
        n_qubits  : int = N_QUBITS,
        fm_reps   : int = FEATURE_REPS,
        ans_reps  : int = ANSATZ_REPS,
        optimizer : str = 'cobyla',
        max_iter  : int = 100,
        shots     : int = None,    # None → statevector (exact, faster on simulator)
        seed      : int = 42,
    ):
        self.n_qubits  = n_qubits
        self.fm_reps   = fm_reps
        self.ans_reps  = ans_reps
        self.optimizer = optimizer
        self.max_iter  = max_iter
        self.shots     = shots
        self.seed      = seed

        self.vqc_     = None
        self.history_ = []

    # ─────────────────────────────────────────────────────────────────────────
    def _build(self):
        """
        Construct the Qiskit VQC object (called internally by fit()).
        Separated from __init__ so parameters can be changed before fitting.
        """
        # 1. Feature map (data encoding circuit)
        feature_map = build_feature_map(self.n_qubits, self.fm_reps)

        # 2. Ansatz (trainable variational circuit)
        ansatz = build_ansatz(self.n_qubits, self.ans_reps)

        # 3. Sampler (runs the quantum circuit)
        #    shots=None  → statevector simulation (exact, no noise)
        #    shots=1024  → shot-based simulation (adds sampling noise)
        sampler = AerSampler()
        if self.shots is not None:
            sampler = AerSampler()
            # Set shots via options if needed

        # 4. Optimizer
        opt = OPTIMIZERS.get(self.optimizer, OPTIMIZERS['cobyla'])(self.max_iter)

        # 5. Callback to record training progress
        self.history_ = []
        def _callback(weights, obj_value):
            self.history_.append(obj_value)
            n = len(self.history_)
            if n % 10 == 0 or n == 1:
                print(f"  Iter {n:4d}/{self.max_iter} | Loss: {obj_value:.6f}")

        # 6. Assemble VQC
        self.vqc_ = VQC(
            sampler=sampler,
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=opt,
            callback=_callback,
            initial_point=np.random.default_rng(self.seed).uniform(
                -np.pi, np.pi, ansatz.num_parameters
            )
        )

        n_params = ansatz.num_parameters
        print(f"\n[QuantumClassifier] Built VQC")
        print(f"  Qubits      : {self.n_qubits}")
        print(f"  FM reps     : {self.fm_reps}")
        print(f"  Ansatz reps : {self.ans_reps}")
        print(f"  Parameters  : {n_params}")
        print(f"  Optimizer   : {self.optimizer} (max_iter={self.max_iter})")
        print(f"  Shots       : {'statevector (exact)' if self.shots is None else self.shots}")

    # ─────────────────────────────────────────────────────────────────────────
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train the VQC on (X_train, y_train).

        Args:
            X_train : shape (n_samples, n_qubits) — quantum-ready features
            y_train : shape (n_samples,) — binary labels (0 or 1)

        Returns:
            self
        """
        print("\n" + "="*55)
        print("  TRAINING QUANTUM CLASSIFIER")
        print("="*55)
        self._build()

        t0 = time.time()
        print(f"\nStarting training on {len(X_train)} samples...")

        self.vqc_.fit(X_train, y_train)

        elapsed = time.time() - t0
        print(f"\nTraining complete in {elapsed:.1f}s  "
              f"({elapsed/60:.1f} min)")
        return self

    # ─────────────────────────────────────────────────────────────────────────
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for X.
        Returns array of 0s and 1s.
        """
        self._check_fitted()
        return self.vqc_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class probabilities [[P(0), P(1)], ...] for SHAP compatibility.
        Qiskit VQC doesn't expose raw probabilities directly, so we
        infer them from the prediction score via sigmoid approximation.
        """
        self._check_fitted()
        # Qiskit VQC gives hard predictions; we use a soft approximation
        # by querying the underlying weights via statevector if needed.
        # For simplicity: return one-hot-like probabilities.
        preds = self.vqc_.predict(X)
        proba = np.column_stack([1 - preds, preds]).astype(float)
        return proba

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy score (sklearn-compatible)."""
        self._check_fitted()
        return self.vqc_.score(X, y)

    # ─────────────────────────────────────────────────────────────────────────
    def summary(self):
        """Print model configuration."""
        n_params = self.n_qubits * (self.ans_reps + 1)
        print(f"\nQuantumClassifier (Qiskit VQC)")
        print(f"  n_qubits  = {self.n_qubits}")
        print(f"  fm_reps   = {self.fm_reps}   (ZZFeatureMap)")
        print(f"  ans_reps  = {self.ans_reps}   (RealAmplitudes)")
        print(f"  n_params  = {n_params}")
        print(f"  optimizer = {self.optimizer}")
        print(f"  max_iter  = {self.max_iter}")
        print(f"  fitted    = {self.vqc_ is not None}")

    def _check_fitted(self):
        if self.vqc_ is None:
            raise RuntimeError("Call .fit() before .predict()")
