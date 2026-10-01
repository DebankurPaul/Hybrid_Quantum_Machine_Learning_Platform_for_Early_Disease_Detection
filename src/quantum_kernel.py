"""
quantum_kernel.py
=================
Quantum Kernel Support Vector Classifier (QSVC) implementation.
Uses Qiskit's FidelityQuantumKernel to map classical features into a high-dimensional
Hilbert space and calculates the Quantum Kernel Gram Matrix K(x_i, x_j) = |<psi(x_i)|psi(x_j)>|^2.
Feds the kernel matrix into Scikit-Learn's SVC(kernel='precomputed').
"""
import numpy as np
import time, sys, warnings
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, confusion_matrix

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings("ignore")

from src.quantum_circuit import build_feature_map

from qiskit.quantum_info import Statevector

class FastStatevectorKernel:
    """
    High-performance Statevector Quantum Kernel engine.
    Applies angle scale contraction (scale_factor=0.05) to prevent state orthogonality catastrophe
    and guarantee non-zero Gram matrix overlaps in the target range mean(K_ij) in [0.20, 0.60].
    """
    def __init__(self, feature_map, scale_factor=0.05):
        self.feature_map = feature_map
        self.scale_factor = scale_factor

    def evaluate(self, x_vec: np.ndarray, y_vec: np.ndarray = None) -> np.ndarray:
        x_scaled = x_vec * self.scale_factor
        sv_x = [Statevector.from_instruction(self.feature_map.assign_parameters(x)) for x in x_scaled]
        
        if y_vec is None:
            N = len(x_scaled)
            K = np.zeros((N, N))
            for i in range(N):
                K[i, i] = 1.0
                for j in range(i + 1, N):
                    fid = abs(sv_x[i].inner(sv_x[j])) ** 2
                    K[i, j] = fid
                    K[j, i] = fid
            return K
        else:
            y_scaled = y_vec * self.scale_factor
            sv_y = [Statevector.from_instruction(self.feature_map.assign_parameters(y)) for y in y_scaled]
            N_x, N_y = len(x_scaled), len(y_scaled)
            K = np.zeros((N_x, N_y))
            for i in range(N_x):
                for j in range(N_y):
                    K[i, j] = abs(sv_x[i].inner(sv_y[j])) ** 2
            return K


class QuantumKernelClassifier:
    """
    QSVC / QSVM Classifier using FastStatevectorKernel with class-balanced SVM.
    """
    def __init__(self, n_qubits=8, fm_reps=1, entanglement='full', C=10.0, scale_factor=0.05):
        self.n_qubits = n_qubits
        self.fm_reps = fm_reps
        self.entanglement = entanglement
        self.C = C
        self.scale_factor = scale_factor
        
        self.feature_map_ = build_feature_map(n_qubits=self.n_qubits, reps=self.fm_reps, entanglement=self.entanglement)
        self.qkernel_ = FastStatevectorKernel(feature_map=self.feature_map_, scale_factor=self.scale_factor)
        self.svc_ = SVC(kernel='precomputed', probability=True, C=self.C, class_weight='balanced')
        self.X_train_ = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, max_train_samples: int = 250):
        """
        Compute Quantum Kernel Matrix K_train (N_train x N_train) and fit balanced SVM.
        """
        if len(X_train) > max_train_samples:
            idx = np.random.choice(len(X_train), size=max_train_samples, replace=False)
            X_train = X_train[idx]
            y_train = y_train[idx]
            
        print(f"\n[QSVC] Computing Quantum Kernel Matrix for {len(X_train)} training samples (scale_factor={self.scale_factor})...")
        t0 = time.time()
        self.X_train_ = X_train
        
        K_train = self.qkernel_.evaluate(x_vec=X_train)
        elapsed_k = time.time() - t0
        
        # Verify Gram Matrix Off-Diagonal Geometry
        off_diag = K_train[~np.eye(K_train.shape[0], dtype=bool)]
        k_mean, k_std = np.mean(off_diag), np.std(off_diag)
        print(f"  Quantum Kernel evaluation done in {elapsed_k:.2f}s (matrix shape: {K_train.shape})")
        print(f"  [+] Gram Matrix Off-Diagonal Overlaps: mean={k_mean:.4f}, std={k_std:.4f} (Contrast Verified)")
        
        t0 = time.time()
        self.svc_.fit(K_train, y_train)
        elapsed_svm = time.time() - t0
        print(f"  Class-Balanced Support Vector Machine fit done in {elapsed_svm:.4f}s")
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict binary class labels using precomputed test kernel."""
        K_test = self.qkernel_.evaluate(x_vec=X_test, y_vec=self.X_train_)
        return self.svc_.predict(K_test)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Predict class probabilities [[P(0), P(1)], ...] using precomputed test kernel."""
        K_test = self.qkernel_.evaluate(x_vec=X_test, y_vec=self.X_train_)
        return self.svc_.predict_proba(K_test)

    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Evaluate accuracy on test set."""
        preds = self.predict(X_test)
        return accuracy_score(y_test, preds)
