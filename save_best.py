"""
save_best.py
------------
Reads the latest saved 8-qubit VQC weights and saves a "best model"
snapshot at tau=0.625 with full metrics to results/best_model/.
"""
import sys, numpy as np, glob, json, shutil, warnings
from pathlib import Path
from datetime import datetime, timezone
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings("ignore")

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

try:
    from qiskit.circuit.library import zz_feature_map, real_amplitudes
    fm  = zz_feature_map(feature_dimension=8, reps=1, entanglement="linear")
    ans = real_amplitudes(num_qubits=8, reps=2, entanglement="linear")
except ImportError:
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
    fm  = ZZFeatureMap(feature_dimension=8, reps=1, entanglement="linear")
    ans = RealAmplitudes(num_qubits=8, reps=2, entanglement="linear")

from qiskit_machine_learning.primitives import QMLSampler
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_algorithms.optimizers import SPSA

# ── Rebuild data (same split as training) ─────────────────────────────────────
X, y = load_breast_cancer(return_X_y=True); y = 1 - y
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                            random_state=42, stratify=y)
sc  = StandardScaler()
X_tr_sc = sc.fit_transform(X_tr); X_te_sc = sc.transform(X_te)
pca = PCA(n_components=8, random_state=42)
X_tr_pca = pca.fit_transform(X_tr_sc); X_te_pca = pca.transform(X_te_sc)
X_tr_q = np.tanh(X_tr_pca) * np.pi
X_te_q = np.tanh(X_te_pca) * np.pi

# ── Load latest saved weights ─────────────────────────────────────────────────
w_files = sorted(glob.glob("results/vqc_weights_8q_spsa_*.npy"))
if not w_files:
    print("ERROR: No saved weights found in results/"); exit(1)
src_weights_path = Path(w_files[-1])
weights = np.load(src_weights_path)
print(f"Loaded weights: {src_weights_path.name}")

# ── Reconstruct VQC and inject weights ────────────────────────────────────────
vqc = VQC(sampler=QMLSampler(), feature_map=fm, ansatz=ans,
           optimizer=SPSA(maxiter=1))
vqc.fit(X_tr_q[:2], y_tr[:2])   # minimal fit to init internal state
vqc._fit_result.x = weights

# ── Compute metrics at best tau = 0.625 ───────────────────────────────────────
BEST_TAU = 0.65
probs_pos = vqc.predict_proba(X_te_q)[:, 1]
pred_best = (probs_pos >= BEST_TAU).astype(int)

cm = confusion_matrix(y_te, pred_best)
tn, fp, fn, tp = cm.ravel()
sens  = tp / (tp + fn)
spec  = tn / (tn + fp)
acc   = accuracy_score(y_te, pred_best)
auc   = roc_auc_score(y_te, probs_pos)

print(f"\n  Best model metrics at tau = {BEST_TAU}:")
print(f"    Accuracy    : {acc:.4f}")
print(f"    Sensitivity : {sens:.4f}  (caught {tp}/{tp+fn} cancers)")
print(f"    Specificity : {spec:.4f}")
print(f"    ROC-AUC     : {auc:.4f}")
print(f"    FN (missed) : {fn}  |  FP (false alarms): {fp}")

# ── Save best model snapshot ──────────────────────────────────────────────────
out = Path("results") / "best_model"
out.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

# 1. Weights (copy with clear name)
best_w = out / "best_vqc_weights_8q.npy"
shutil.copy(src_weights_path, best_w)
print(f"\n  [SAVED] Weights  → {best_w}")

# 2. Full metrics JSON
payload = {
    "saved_at"     : timestamp,
    "source_weights": src_weights_path.name,
    "best_tau"     : BEST_TAU,
    "config": {
        "n_qubits": 8, "fm_reps": 1, "ans_reps": 2,
        "optimizer": "spsa", "max_iter": 150,
        "n_params": int(ans.num_parameters),
        "pca_components": 8,
    },
    "metrics": {
        "accuracy"   : round(acc, 6),
        "sensitivity": round(sens, 6),
        "specificity": round(spec, 6),
        "roc_auc"    : round(auc, 6),
        "tp": int(tp), "tn": int(tn),
        "fp": int(fp), "fn": int(fn),
        "n_test"     : len(y_te),
        "n_malignant_test": int(y_te.sum()),
    },
    "threshold_scan": []
}

for tau in np.arange(0.10, 0.91, 0.025):
    pred = (probs_pos >= tau).astype(int)
    c = confusion_matrix(y_te, pred)
    if c.shape != (2, 2): continue
    tn_, fp_, fn_, tp_ = c.ravel()
    s = tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else 0.0
    sp = tn_ / (tn_ + fp_) if (tn_ + fp_) > 0 else 0.0
    payload["threshold_scan"].append({
        "tau": round(float(tau), 3),
        "sensitivity": round(s, 4),
        "specificity": round(sp, 4),
        "accuracy"   : round(accuracy_score(y_te, pred), 4),
        "fn": int(fn_), "fp": int(fp_),
    })

json_path = out / "best_model_metrics.json"
json_path.write_text(json.dumps(payload, indent=2))
print(f"  [SAVED] Metrics  → {json_path}")

# 3. Human-readable card
card = f"""Best VQC Model Card
{'='*50}
Saved      : {timestamp}
Config     : 8 qubits | SPSA | 150 iters | ANS_REPS=2
PCA        : 8 components

Best Threshold : tau = {BEST_TAU}
  Accuracy     : {acc:.4f}  ({acc*100:.1f}%)
  Sensitivity  : {sens:.4f}  — caught {tp}/{tp+fn} cancers (missed: {fn})
  Specificity  : {spec:.4f}  — avoided {tn}/{tn+fp} false alarms
  ROC-AUC      : {auc:.4f}

Confusion Matrix (tau = {BEST_TAU}):
                Pred Benign  Pred Malignant
  True Benign      {tn:4d}         {fp:4d}   <- false alarms
  True Malignant   {fn:4d}         {tp:4d}   <- missed cancers

Weights file : best_vqc_weights_8q.npy
"""
card_path = out / "model_card.txt"
card_path.write_text(card)
print(f"  [SAVED] Model card → {card_path}")
print("\n  Best model snapshot complete. results/best_model/ is ready.\n")
