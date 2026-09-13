"""
quantum_circuit.py  (Qiskit version)
=====================================
Builds the two quantum circuit halves of the VQC:
  1. Feature Map  — encodes classical data into qubit states
  2. Ansatz       — parameterized circuit that gets trained

Qiskit terminology vs PennyLane terminology
────────────────────────────────────────────
PennyLane term      │ Qiskit term
────────────────────┼───────────────────────────────
Encoding layer      │ Feature Map  (ZZFeatureMap)
Variational layer   │ Ansatz       (RealAmplitudes)
QNode               │ QuantumCircuit assembled as one
Trainable weights   │ ParameterVector (theta_*)
diff_method=...     │ optimizer handles gradients

Usage:
    from src.quantum_circuit import build_feature_map, build_ansatz, build_vqc_circuit
    feature_map = build_feature_map(n_qubits=4, reps=1)
    ansatz      = build_ansatz(n_qubits=4, reps=2)
    qc          = build_vqc_circuit(feature_map, ansatz)
    qc.draw('text')
"""
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes, TwoLocal
from qiskit import QuantumCircuit
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULTS  (change here to experiment)
# ─────────────────────────────────────────────────────────────────────────────
N_QUBITS       = 4   # must equal n_qubits used in preprocessing PCA
FEATURE_REPS   = 1   # repetitions of the feature map block
ANSATZ_REPS    = 2   # repetitions of the variational block


# ─────────────────────────────────────────────────────────────────────────────
# 1. FEATURE MAP — ZZFeatureMap
# ─────────────────────────────────────────────────────────────────────────────
def build_feature_map(n_qubits: int = N_QUBITS, reps: int = FEATURE_REPS, entanglement: str = 'linear'):
    """
    Build a ZZFeatureMap for data encoding.
    """
    try:
        from qiskit.circuit.library import zz_feature_map
        feature_map = zz_feature_map(feature_dimension=n_qubits, reps=reps, entanglement=entanglement)
    except ImportError:
        feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=reps, entanglement=entanglement).decompose()
    
    print(f"[Feature Map] ZZFeatureMap | qubits={n_qubits} | reps={reps} | entanglement={entanglement}")
    print(f"  Parameters: {feature_map.num_parameters} input features")
    return feature_map


def build_ansatz(n_qubits: int = N_QUBITS, reps: int = ANSATZ_REPS, entanglement: str = 'linear'):
    """
    Build a RealAmplitudes ansatz as the variational (trainable) part.
    """
    try:
        from qiskit.circuit.library import real_amplitudes
        ansatz = real_amplitudes(num_qubits=n_qubits, reps=reps, entanglement=entanglement)
    except ImportError:
        ansatz = RealAmplitudes(num_qubits=n_qubits, reps=reps, entanglement=entanglement).decompose()
        
    print(f"[Ansatz] RealAmplitudes | qubits={n_qubits} | reps={reps} | entanglement={entanglement}")
    print(f"  Trainable parameters: {ansatz.num_parameters}")
    return ansatz


def build_vqc_circuit(feature_map, ansatz):
    """
    Compose the full VQC circuit = Feature Map + Ansatz.
    """
    qc = feature_map.compose(ansatz)
    qc.name = "VQC Circuit"
    print(f"\n[VQC Circuit] Total parameters: {qc.num_parameters}")
    print(f"  Feature params : {feature_map.num_parameters}")
    print(f"  Ansatz params  : {ansatz.num_parameters}")
    return qc


def build_reuploading_circuit(n_qubits: int = N_QUBITS, layers: int = 2, entanglement: str = 'full'):
    """
    Data Re-uploading Architecture:
    [Feature Map Layer -> Ansatz Layer] repeated `layers` times.
    
    In Data Re-uploading QML, embedding the classical data multiple times interspersed
    with trainable variational layers dramatically boosts circuit expressivity and non-linearity.
    """
    qc = QuantumCircuit(n_qubits)
    for layer in range(layers):
        fm = build_feature_map(n_qubits, reps=1, entanglement=entanglement)
        ans = build_ansatz(n_qubits, reps=1, entanglement=entanglement)
        qc = qc.compose(fm).compose(ans)
    qc.name = f"Data Re-uploading VQC ({layers} layers)"
    print(f"\n[Data Re-uploading Circuit] {layers} layers | qubits={n_qubits} | entanglement={entanglement}")
    print(f"  Total parameters: {qc.num_parameters}")
    return qc


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def draw_circuit(style='text', save_path=None):
    """
    Draw the full VQC circuit.

    Args:
        style : 'text' (ASCII), 'mpl' (matplotlib), 'latex'
        save_path : if given, save figure to this path (mpl only)
    """
    fm  = build_feature_map()
    ans = build_ansatz()
    qc  = build_vqc_circuit(fm, ans)

    if style == 'text':
        print(qc.decompose().draw('text'))
    elif style == 'mpl':
        import matplotlib.pyplot as plt
        fig = qc.decompose().draw('mpl', fold=40)
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    return qc


def count_parameters(n_qubits=N_QUBITS, fm_reps=FEATURE_REPS, ans_reps=ANSATZ_REPS):
    """Print parameter breakdown for the VQC."""
    # ZZFeatureMap: n_qubits input params (not trainable, these are DATA)
    # RealAmplitudes: n_qubits * (reps + 1) trainable params
    trainable = n_qubits * (ans_reps + 1)
    print(f"\nParameter count (n_qubits={n_qubits}):")
    print(f"  ZZFeatureMap input params : {n_qubits}   (data, not trained)")
    print(f"  RealAmplitudes trainable  : {trainable}")
    print(f"  Total trainable weights   : {trainable}")
    return trainable
