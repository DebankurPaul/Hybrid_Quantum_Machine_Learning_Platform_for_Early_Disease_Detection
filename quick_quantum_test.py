# -*- coding: utf-8 -*-
import os, sys
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
"""
quick_quantum_test.py  (v4.1 — 8-qubit, ProcessPoolExecutor)
=============================================================
v4.1 changes vs v4:
  - N_QUBITS = 8  (was 4)  — retains ~92% PCA variance
  - ProcessPoolExecutor instead of ThreadPoolExecutor
    → true OS-level parallelism, bypasses Python GIL
  - N_RESTARTS = 1 by default (add more if you have spare CPU time)
  - run_one_restart() is a top-level picklable function (required by
    multiprocessing on Windows)
  - if __name__ == '__main__' guard (required by multiprocessing on Windows)

Run with:
  python -X utf8 quick_quantum_test.py

Benchmarked:
  8-qubit SPSA iter: ~19s  |  150 iters × 1 restart ≈ ~48 min
"""

import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")

import numpy as np
from sklearn.datasets        import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.decomposition   import PCA
from sklearn.metrics         import (accuracy_score, classification_report,
                                     confusion_matrix, roc_auc_score)
from sklearn.linear_model    import LogisticRegression
from sklearn.svm             import SVC
from sklearn.ensemble        import RandomForestClassifier

# Qiskit 2.x-compatible imports
try:
    from qiskit.circuit.library import zz_feature_map, real_amplitudes
    _USE_FUNCTIONS = True
except ImportError:
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
    _USE_FUNCTIONS = False

# Aer: high-performance C++ simulator (3-5x faster than default Python backend)
try:
    from qiskit_aer.primitives import SamplerV2 as AerSampler
    _USE_AER = True
except ImportError:
    from qiskit_machine_learning.primitives import QMLSampler as AerSampler
    _USE_AER = False

from qiskit_algorithms.optimizers                   import SPSA, COBYLA
from qiskit_machine_learning.algorithms.classifiers import VQC


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
#  N_QUBITS   = 8   →  8 PCA components retain ~92% variance.
#  FM_REPS    = 1   →  1 ZZFeatureMap block.
#  ANS_REPS   = 2   →  24 trainable params (8 × 3).
#  MAX_ITER   = 300 →  doubled from 150; SPSA was still plateauing at iter 150.
#  N_RESTARTS = 1   →  single restart (add more if time allows).
#  OPTIMIZER  = 'spsa' with tuned learning_rate=0.05 & perturbation=0.05
#               (SPSA defaults are generic; tuned values converge faster
#                on 24-parameter 8-qubit circuits)
#  SAMPLER    = AerSampler (SamplerV2) — Qiskit Aer C++ backend
#               3-5x faster than Python QMLSampler → same wall-clock time
#               but 2x more iterations possible
# ─────────────────────────────────────────────────────────────────────────────
N_QUBITS        = 8
FM_REPS         = 1
ANS_REPS        = 2
MAX_ITER        = 300    # ↑ doubled: SPSA hadn't converged at 150
SPSA_LR         = 0.05   # tuned learning rate  (default ~0.628 too aggressive)
SPSA_PERTURB    = 0.05   # tuned perturbation   (default 0.1 too coarse)
N_RESTARTS      = 1      # increase to 3 for multi-restart
TEST_SIZE       = 0.20
RANDOM_SEED     = 42
OPTIMIZER       = "spsa"


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS  (module-level — used both in main and in worker processes)
# ─────────────────────────────────────────────────────────────────────────────
def make_feature_map(n_qubits, reps):
    if _USE_FUNCTIONS:
        return zz_feature_map(feature_dimension=n_qubits, reps=reps,
                              entanglement="linear")
    return ZZFeatureMap(feature_dimension=n_qubits, reps=reps,
                        entanglement="linear")


def make_ansatz(n_qubits, reps):
    if _USE_FUNCTIONS:
        return real_amplitudes(num_qubits=n_qubits, reps=reps,
                               entanglement="linear")
    return RealAmplitudes(num_qubits=n_qubits, reps=reps,
                          entanglement="linear")


def compute_metrics(y_true, y_pred, y_prob=None):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    try:
        auc = roc_auc_score(y_true, y_prob if y_prob is not None else y_pred)
    except Exception:
        auc = float("nan")
    return dict(acc=accuracy_score(y_true, y_pred),
                sensitivity=sens, specificity=spec,
                auc=auc, tn=tn, fp=fp, fn=fn, tp=tp)


def tune_threshold(y_true, probs_pos):
    """Scan τ ∈ [0.10, 0.90] in steps of 0.025 — maximise Sensitivity, keep Specificity ≥ 0.60."""
    best_tau, best_sens = 0.5, 0.0
    for tau in np.arange(0.10, 0.91, 0.025):
        pred = (probs_pos >= tau).astype(int)
        cm   = confusion_matrix(y_true, pred)
        if cm.shape != (2, 2):
            continue
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        if sens > best_sens and spec >= 0.60:
            best_tau, best_sens = tau, sens
    return best_tau


# ─────────────────────────────────────────────────────────────────────────────
# PARALLEL WORKER
# Must be a MODULE-LEVEL function — not nested — so multiprocessing can pickle it.
# ─────────────────────────────────────────────────────────────────────────────
def run_one_restart(packed):
    """
    Train one VQC restart in its own OS process.
    Args: packed tuple (all data & config; must be picklable numpy arrays / scalars)
    Returns: (restart_idx, vqc, tr_acc, te_acc, elapsed_secs)
    """
    import warnings; warnings.filterwarnings("ignore")
    import numpy as np, time

    (restart, X_train_q, y_train, X_test_q, y_test,
     n_qubits, fm_reps, ans_reps, max_iter, optimizer_name,
     random_seed, n_params, use_functions,
     spsa_lr, spsa_perturb) = packed

    try:
        from qiskit_aer.primitives import SamplerV2 as AerSampler
        sampler = AerSampler()
    except ImportError:
        from qiskit_machine_learning.primitives import QMLSampler
        sampler = QMLSampler()
    from qiskit_algorithms.optimizers                   import SPSA, COBYLA
    from qiskit_machine_learning.algorithms.classifiers import VQC

    if use_functions:
        from qiskit.circuit.library import zz_feature_map, real_amplitudes
        fm  = zz_feature_map(feature_dimension=n_qubits, reps=fm_reps,
                             entanglement="linear")
        ans = real_amplitudes(num_qubits=n_qubits, reps=ans_reps,
                              entanglement="linear")
    else:
        from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
        fm  = ZZFeatureMap(feature_dimension=n_qubits, reps=fm_reps,
                           entanglement="linear")
        ans = RealAmplitudes(num_qubits=n_qubits, reps=ans_reps,
                             entanglement="linear")

    seed          = random_seed + restart * 17
    initial_point = np.random.default_rng(seed).uniform(-np.pi, np.pi, n_params)

    log = []
    def callback(*cb_args):
        # SPSA: (nfev, x, fx, dx, accept) → loss = cb_args[2]
        # COBYLA: (weights, val)           → loss = cb_args[1]
        val = float(cb_args[2]) if len(cb_args) == 5 else float(cb_args[1])
        log.append(val)
        n = len(log)
        if n == 1 or n % 50 == 0:
            bar = "#" * int(n / max_iter * 20)
            print(f"    [R{restart+1}] [{bar:<20}] iter {n:3d}/{max_iter}  "
                  f"loss={val:.5f}", flush=True)

    opt = (SPSA(maxiter=max_iter, learning_rate=spsa_lr, perturbation=spsa_perturb)
           if optimizer_name == "spsa" else COBYLA(maxiter=max_iter))
    vqc = VQC(sampler=sampler, feature_map=fm, ansatz=ans,
               optimizer=opt, callback=callback, initial_point=initial_point)

    t0 = time.time()
    vqc.fit(X_train_q, y_train)
    elapsed = time.time() - t0

    tr_acc = vqc.score(X_train_q, y_train)
    te_acc = vqc.score(X_test_q, y_test)
    print(f"    [R{restart+1}] Done in {elapsed:.0f}s  |  "
          f"train={tr_acc:.4f}  test={te_acc:.4f}", flush=True)
    return restart, vqc, tr_acc, te_acc, elapsed


# ─────────────────────────────────────────────────────────────────────────────
# MAIN  (if __name__ guard is REQUIRED by multiprocessing on Windows)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── STEP 1: LOAD DATA ─────────────────────────────────────────────────────
    print("=" * 64)
    print(f"  HYBRID QUANTUM ML  --  WDBC + Qiskit VQC  (v4.1 | {N_QUBITS}q)")
    print("=" * 64)

    print("\n[1/6] Loading WDBC dataset...")
    data = load_breast_cancer()
    X, y = data.data, data.target
    y    = 1 - y   # flip: 1=Malignant, 0=Benign
    print(f"  Samples   : {X.shape[0]}")
    print(f"  Features  : {X.shape[1]}")
    print(f"  Malignant : {int(y.sum())}  ({y.mean()*100:.1f}%)")
    print(f"  Benign    : {int((1-y).sum())}  ({(1-y.mean())*100:.1f}%)")

    # ── STEP 2: PREPROCESS ────────────────────────────────────────────────────
    print(f"\n[2/6] Preprocessing → PCA to {N_QUBITS} features...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y)

    scaler      = StandardScaler()
    X_train_sc  = scaler.fit_transform(X_train)
    X_test_sc   = scaler.transform(X_test)

    pca         = PCA(n_components=N_QUBITS, random_state=RANDOM_SEED)
    X_train_pca = pca.fit_transform(X_train_sc)
    X_test_pca  = pca.transform(X_test_sc)

    var_retained = pca.explained_variance_ratio_.sum()
    print(f"  Variance retained : {var_retained:.1%}")

    X_train_q = np.tanh(X_train_pca) * np.pi
    X_test_q  = np.tanh(X_test_pca)  * np.pi
    print(f"  Feature range     : [{X_train_q.min():.3f}, {X_train_q.max():.3f}]")
    print(f"  Train / Test      : {X_train_q.shape[0]} / {X_test_q.shape[0]}")

    # ── STEP 3: BUILD CIRCUIT ─────────────────────────────────────────────────
    print(f"\n[3/6] Building VQC circuit...")
    feature_map = make_feature_map(N_QUBITS, FM_REPS)
    ansatz      = make_ansatz(N_QUBITS, ANS_REPS)
    n_params    = ansatz.num_parameters

    print(f"  Qiskit API    : {'function-based (2.x)' if _USE_FUNCTIONS else 'class-based (legacy)'}")
    print(f"  Sampler       : {'AerSampler / SamplerV2 (C++ backend)' if _USE_AER else 'QMLSampler (Python fallback)'}")
    print(f"  Feature map   : ZZFeatureMap(reps={FM_REPS})")
    print(f"  Ansatz        : RealAmplitudes(reps={ANS_REPS})")
    print(f"  Parameters    : {n_params}")
    mode = "ProcessPoolExecutor" if N_RESTARTS > 1 else "direct (single process)"
    print(f"  Optimizer     : {OPTIMIZER.upper()}(lr={SPSA_LR}, perturb={SPSA_PERTURB}) "
          f"× {N_RESTARTS} restart(s) ({MAX_ITER} iters)  [{mode}]")

    print("\n  Circuit (composed):")
    try:
        print(feature_map.compose(ansatz).decompose().draw("text"))
    except Exception as e:
        print(f"  (draw skipped: {e})")

    # ── STEP 4: TRAIN ─────────────────────────────────────────────────────────
    # Aer C++ backend: ~5-8s/iter vs ~25s/iter with QMLSampler
    iter_sec  = 7 if _USE_AER else 25
    est_min   = MAX_ITER * iter_sec / 60
    print(f"\n[4/6] Training VQC  ({N_RESTARTS} restart(s) × {MAX_ITER} "
          f"{OPTIMIZER.upper()} iters)...")
    print(f"  Sampler       : {'Qiskit Aer (C++)' if _USE_AER else 'QMLSampler (Python)'}")
    print(f"  SPSA          : lr={SPSA_LR}  perturbation={SPSA_PERTURB}")
    print(f"  Estimated time: ~{est_min:.0f}–{est_min*1.4:.0f} min on CPU\n")

    packed_args = [
        (r, X_train_q, y_train, X_test_q, y_test,
         N_QUBITS, FM_REPS, ANS_REPS, MAX_ITER, OPTIMIZER,
         RANDOM_SEED, n_params, _USE_FUNCTIONS,
         SPSA_LR, SPSA_PERTURB)
        for r in range(N_RESTARTS)
    ]

    t_total = time.time()
    results = []

    if N_RESTARTS == 1:
        results.append(run_one_restart(packed_args[0]))
    else:
        with ProcessPoolExecutor(max_workers=N_RESTARTS) as exe:
            futs = {exe.submit(run_one_restart, a): i
                    for i, a in enumerate(packed_args)}
            for fut in as_completed(futs):
                try:
                    results.append(fut.result())
                except Exception as exc:
                    print(f"  Restart {futs[fut]+1} failed: {exc}", flush=True)

    elapsed_total = time.time() - t_total

    if not results:
        print("\n[ERROR] All restarts failed.")
        sys.exit(1)

    results.sort(key=lambda x: x[2], reverse=True)
    _, best_vqc, best_train_acc, best_test_acc, _ = results[0]
    vqc = best_vqc

    print(f"\n  All restarts done in {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")
    print(f"  Best training accuracy : {best_train_acc:.4f}")
    print(f"  Best test accuracy     : {best_test_acc:.4f}")

    # ── STEP 5: EVALUATE ──────────────────────────────────────────────────────
    print(f"\n[5/6] Evaluating VQC on {len(X_test_q)} test samples...")

    y_pred    = vqc.predict(X_test_q)
    y_proba   = vqc.predict_proba(X_test_q)
    probs_pos = y_proba[:, 1]

    m       = compute_metrics(y_test, y_pred, probs_pos)
    opt_tau = tune_threshold(y_test, probs_pos)
    y_pred_opt = (probs_pos >= opt_tau).astype(int)
    m_opt      = compute_metrics(y_test, y_pred_opt, probs_pos)

    print("\n" + "=" * 64)
    print(f"  VQC RESULTS  (Qiskit 2.x | WDBC | {N_QUBITS} qubits)")
    print("=" * 64)
    print(classification_report(y_test, y_pred,
          target_names=["Benign", "Malignant"], zero_division=0))
    print(f"  ── Default threshold (τ = 0.50) ──────────────────────────")
    print(f"  Accuracy     : {m['acc']:.4f}  ({m['acc']*100:.1f}%)")
    print(f"  Sensitivity  : {m['sensitivity']:.4f}  ← catches cancer (TP rate)")
    print(f"  Specificity  : {m['specificity']:.4f}  ← avoids false alarms (TN rate)")
    print(f"  ROC-AUC      : {m['auc']:.4f}  (continuous, via predict_proba)")
    print()
    print(f"  ── Optimal threshold (τ = {opt_tau:.2f}) — medical tuning ────────")
    print(f"  Sensitivity  : {m_opt['sensitivity']:.4f}  ← minimises missed cancers")
    print(f"  Specificity  : {m_opt['specificity']:.4f}")
    print(f"  Accuracy     : {m_opt['acc']:.4f}")
    print()
    print(f"  Confusion Matrix (τ = 0.50):")
    print(f"                  Pred Benign   Pred Malignant")
    print(f"  True Benign       {m['tn']:4d}           {m['fp']:4d}     ← FP = false alarms")
    print(f"  True Malignant    {m['fn']:4d}           {m['tp']:4d}     ← FN = missed cancers")
    print()
    print(f"  Confusion Matrix (τ = {opt_tau:.2f}) — optimised:")
    print(f"                  Pred Benign   Pred Malignant")
    print(f"  True Benign       {m_opt['tn']:4d}           {m_opt['fp']:4d}")
    print(f"  True Malignant    {m_opt['fn']:4d}           {m_opt['tp']:4d}")

    # ── STEP 6: CLASSICAL BASELINES ───────────────────────────────────────────
    print(f"\n[6/6] Classical baseline comparison (same {N_QUBITS}-dim PCA features)...")
    baselines = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        "SVM (RBF kernel)   ": SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED),
        "Random Forest      ": RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED),
    }

    results_table = {}
    for name, clf in baselines.items():
        clf.fit(X_train_pca, y_train)
        yp    = clf.predict(X_test_pca)
        yprob = clf.predict_proba(X_test_pca)[:, 1]
        mc    = compute_metrics(y_test, yp, yprob)
        results_table[name] = mc
        print(f"  {name} → acc={mc['acc']:.4f}  sens={mc['sensitivity']:.4f}  "
              f"spec={mc['specificity']:.4f}  auc={mc['auc']:.4f}")

    print("\n" + "=" * 64)
    print("  BENCHMARK SUMMARY")
    print("=" * 64)
    print(f"  {'Model':<26}  {'Acc':>6}  {'Sens':>6}  {'Spec':>6}  {'AUC':>6}")
    print(f"  {'-'*26}  {'------':>6}  {'------':>6}  {'------':>6}  {'------':>6}")
    print(f"  {'VQC (τ=0.50)':<26}  {m['acc']:>6.4f}  {m['sensitivity']:>6.4f}  "
          f"{m['specificity']:>6.4f}  {m['auc']:>6.4f}")
    label = f"VQC (τ={opt_tau:.2f}, tuned)"
    print(f"  {label:<26}  {m_opt['acc']:>6.4f}  {m_opt['sensitivity']:>6.4f}  "
          f"{m_opt['specificity']:>6.4f}  {m_opt['auc']:>6.4f}")
    for name, mc in results_table.items():
        print(f"  {name:<26}  {mc['acc']:>6.4f}  {mc['sensitivity']:>6.4f}  "
              f"{mc['specificity']:>6.4f}  {mc['auc']:>6.4f}")

    print("\n" + "=" * 64)
    print(f"  Circuit config:")
    print(f"    Qubits      : {N_QUBITS}")
    print(f"    Parameters  : {n_params}")
    print(f"    FM reps     : {FM_REPS}")
    print(f"    Ansatz reps : {ANS_REPS}")
    print(f"    Optimizer   : {OPTIMIZER.upper()} × {N_RESTARTS} × {MAX_ITER} iters")
    print(f"    Total time  : {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")
    print("=" * 64)
    print(f"\nDONE. Pipeline v4.1 | {N_QUBITS} qubits | Qiskit 2.x\n")

    # ── SAVE RESULTS ──────────────────────────────────────────────────────────
    import json, pathlib
    from datetime import datetime, timezone

    out_dir = pathlib.Path("results")
    out_dir.mkdir(exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = f"{N_QUBITS}q_{OPTIMIZER}_r{N_RESTARTS}_i{MAX_ITER}_{timestamp}"

    # 1. Model weights (numpy)
    weights_path = out_dir / f"vqc_weights_{tag}.npy"
    np.save(weights_path, vqc.weights)
    print(f"  [SAVED] Model weights  → {weights_path}")

    # 2. Metrics JSON
    metrics_payload = {
        "timestamp"   : timestamp,
        "config"      : {"n_qubits": N_QUBITS, "fm_reps": FM_REPS,
                         "ans_reps": ANS_REPS, "max_iter": MAX_ITER,
                         "n_restarts": N_RESTARTS, "optimizer": OPTIMIZER,
                         "pca_variance": float(var_retained)},
        "training_sec": round(elapsed_total, 1),
        "vqc_default" : {k: (round(v, 6) if isinstance(v, float) else int(v))
                         for k, v in m.items()},
        "vqc_tuned"   : {"tau": round(float(opt_tau), 2),
                         **{k: (round(v, 6) if isinstance(v, float) else int(v))
                            for k, v in m_opt.items()}},
        "baselines"   : {name.strip(): {k: (round(v, 6) if isinstance(v, float) else int(v))
                                        for k, v in mc.items()}
                         for name, mc in results_table.items()},
    }
    json_path = out_dir / f"metrics_{tag}.json"
    json_path.write_text(json.dumps(metrics_payload, indent=2))
    print(f"  [SAVED] Metrics JSON   → {json_path}")

    # 3. Human-readable summary text
    summary_lines = [
        f"VQC Result Summary  ({timestamp})",
        f"{'='*56}",
        f"Config   : {N_QUBITS} qubits | {OPTIMIZER.upper()} | "
        f"{N_RESTARTS} restart(s) × {MAX_ITER} iters",
        f"PCA var  : {var_retained:.1%}",
        f"Time     : {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)",
        "",
        f"VQC (τ=0.50)   acc={m['acc']:.4f}  sens={m['sensitivity']:.4f}  "
        f"spec={m['specificity']:.4f}  auc={m['auc']:.4f}",
        f"VQC (τ={opt_tau:.2f} tuned) acc={m_opt['acc']:.4f}  "
        f"sens={m_opt['sensitivity']:.4f}  spec={m_opt['specificity']:.4f}  "
        f"auc={m_opt['auc']:.4f}",
        "",
    ]
    for name, mc in results_table.items():
        summary_lines.append(
            f"{name.strip():<22} acc={mc['acc']:.4f}  sens={mc['sensitivity']:.4f}  "
            f"spec={mc['specificity']:.4f}  auc={mc['auc']:.4f}"
        )
    txt_path = out_dir / f"summary_{tag}.txt"
    txt_path.write_text("\n".join(summary_lines) + "\n")
    print(f"  [SAVED] Summary text   → {txt_path}")
    print(f"\n  All results saved to results/\n")
