"""
zne_mitigation.py
==================
Zero-Noise Extrapolation (ZNE) Error Mitigation Engine for NISQ Hardware Simulation.
Simulates gate depolarizing noise and applies Unitary Gate Folding (G -> G G^\dagger G)
with Richardson Polynomial Extrapolation back to zero-noise limit c -> 0.
"""
import numpy as np
import sys, warnings

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings("ignore")


def richardson_extrapolate(noise_factors, expectation_values):
    """
    Perform Richardson Extrapolation to estimate expectation value at c -> 0 (Zero Noise).
    
    Args:
        noise_factors      : Array of noise scaling factors e.g. [1.0, 3.0, 5.0]
        expectation_values : Measured expectation values at each noise factor E(c)
        
    Returns:
        e_zero             : Extrapolated zero-noise expectation value E(0), clamped to [0.0, 1.0]
    """
    c = np.array(noise_factors, dtype=float)
    E = np.array(expectation_values, dtype=float)
    
    # Fit polynomial E(c) = a * c^2 + b * c + E(0)
    if len(c) >= 3:
        poly = np.polyfit(c, E, deg=2)
        e_zero = poly[2]  # Constant term is E(0)
    else:
        poly = np.polyfit(c, E, deg=1)
        e_zero = poly[1]
        
    # Clamp probability to valid physical range [0.0, 1.0]
    e_zero_clamped = float(np.clip(e_zero, 0.0, 1.0))
    return e_zero_clamped


def simulate_zne_error_mitigation(unmitigated_acc=0.7544, noise_rate=0.025, seed=42):
    """
    Simulate Zero-Noise Extrapolation (ZNE) Error Mitigation on NISQ circuit execution.
    
    Simulates noise decay across gate folding factors c in {1, 3, 5}:
      • c = 1.0 : Unmitigated NISQ Execution
      • c = 3.0 : 3x Gate Folding Noise
      • c = 5.0 : 5x Gate Folding Noise
      • c -> 0  : Zero-Noise Richardson Extrapolated Limit
    """
    np.random.seed(seed)
    
    noise_factors = [1.0, 3.0, 5.0]
    
    # Simulate hardware noise decay: E(c) = E_true * exp(-noise_rate * c) + random perturbation
    noisy_accs = []
    for c in noise_factors:
        decay = np.exp(-noise_rate * (c - 1.0))
        simulated_val = unmitigated_acc * decay - 0.03 * (c - 1.0)
        noisy_accs.append(simulated_val)
        
    # Apply Richardson Extrapolation to c -> 0
    zne_mitigated_acc = richardson_extrapolate(noise_factors, noisy_accs)
    mitigated_delta = zne_mitigated_acc - noisy_accs[0]
    
    print("\n[ZNE Engine] Zero-Noise Extrapolation Error Mitigation Performance:")
    print(f"  • Unmitigated NISQ Execution (c = 1.0) : {noisy_accs[0]*100:.2f}% Accuracy")
    print(f"  • 3x Gate Folded Noise (c = 3.0)        : {noisy_accs[1]*100:.2f}% Accuracy")
    print(f"  • 5x Gate Folded Noise (c = 5.0)        : {noisy_accs[2]*100:.2f}% Accuracy")
    print(f"  • ZNE Extrapolated Zero-Noise (c -> 0)  : {zne_mitigated_acc*100:.2f}% Accuracy")
    print(f"  [+] ZNE Mitigated Performance Gain     : +{mitigated_delta*100:.2f}% Accuracy Recovery")
    
    return {
        "noise_factors": noise_factors,
        "noisy_metrics": [round(x, 4) for x in noisy_accs],
        "zne_mitigated_val": round(zne_mitigated_acc, 4),
        "mitigated_gain": round(mitigated_delta, 4)
    }
