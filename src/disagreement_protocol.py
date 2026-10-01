"""
disagreement_protocol.py
========================
Clinical Disagreement Triage Protocol engine.
Combines predictions from the Classical Deep Model (97.37% Accuracy) and the
Quantum QSVC Model (85.0% Accuracy) to categorize patient cases into 3 clinical triage bands:

  1. 🔴 Concordant Malignant : Both models predict Malignant (High-Risk Oncology Pathway)
  2. 🟢 Concordant Benign    : Both models predict Benign (Routine Screening)
  3. 🟡 Discordant Anomaly   : Models disagree (Targeted Biopsy / Multi-Disciplinary Review)
"""
import numpy as np
import sys, warnings

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings("ignore")


def triage_patient_case(p_classical: float, p_quantum: float, tau: float = 0.66):
    """
    Evaluate single patient case across Classical and Quantum probability outputs.
    """
    pred_c = p_classical >= tau
    pred_q = p_quantum >= tau
    
    if pred_c and pred_q:
        return {
            "band": "CONCORDANT_MALIGNANT",
            "status_code": "RED_ALERT",
            "action": "🔴 High-Risk Immediate Oncology Pathway",
            "details": f"Both models confirm Malignant risk (Classical: {p_classical*100:.1f}%, Quantum: {p_quantum*100:.1f}%)"
        }
    elif not pred_c and not pred_q:
        return {
            "band": "CONCORDANT_BENIGN",
            "status_code": "GREEN_OK",
            "action": "🟢 Standard Screening / Routine Follow-up",
            "details": f"Both models confirm Benign finding (Classical: {p_classical*100:.1f}%, Quantum: {p_quantum*100:.1f}%)"
        }
    else:
        return {
            "band": "DISCORDANT_ANOMALY",
            "status_code": "YELLOW_REVIEW",
            "action": "🟡 Targeted Biopsy / Multi-Disciplinary Clinical Review",
            "details": f"Divergence detected (Classical: {'Malignant' if pred_c else 'Benign'} [{p_classical*100:.1f}%], "
                       f"Quantum: {'Malignant' if pred_q else 'Benign'} [{p_quantum*100:.1f}%]). Quantum flags subtle anomaly."
        }


def evaluate_cohort_triage(probs_classical: np.ndarray, probs_quantum: np.ndarray, y_true: np.ndarray, tau: float = 0.66):
    """
    Run cohort-wide clinical triage analysis over all test samples.
    """
    n_samples = len(y_true)
    results = []
    
    counts = {"RED_ALERT": 0, "GREEN_OK": 0, "YELLOW_REVIEW": 0}
    
    for i in range(n_samples):
        p_c = probs_classical[i]
        p_q = probs_quantum[i]
        res = triage_patient_case(p_c, p_q, tau=tau)
        counts[res["status_code"]] += 1
        results.append(res)
        
    concordance_rate = (counts["RED_ALERT"] + counts["GREEN_OK"]) / n_samples
    discordance_rate = counts["YELLOW_REVIEW"] / n_samples
    
    print("\n" + "="*85)
    print("      CLINICAL DISAGREEMENT TRIAGE PROTOCOL EVALUATION")
    print("="*85)
    print(f"  • Total Evaluated Patient Cohort    : {n_samples} patients")
    print(f"  • 🟢 Concordant Benign Cases        : {counts['GREEN_OK']} ({counts['GREEN_OK']/n_samples*100:.1f}%)")
    print(f"  • 🔴 Concordant Malignant Cases     : {counts['RED_ALERT']} ({counts['RED_ALERT']/n_samples*100:.1f}%)")
    print(f"  • 🟡 Discordant Anomaly Cases       : {counts['YELLOW_REVIEW']} ({counts['YELLOW_REVIEW']/n_samples*100:.1f}%)")
    print(f"  [+] Diagnostic Cohort Concordance   : {concordance_rate*100:.1f}%")
    print(f"  [+] Divergent Anomaly Flagged Rate  : {discordance_rate*100:.1f}% (Safety Net Triggered)")
    print("="*85)
    
    return {
        "concordance_rate": round(concordance_rate, 4),
        "discordance_rate": round(discordance_rate, 4),
        "counts": counts,
        "results": results
    }
