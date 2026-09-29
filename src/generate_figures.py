import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')

def plot_radar(title, c_metrics, q_metrics, out_path, q_label="8-Qubit Quantum"):
    # Metrics to display
    categories = ['Accuracy', 'Sensitivity (Recall)', 'Specificity', 'ROC-AUC']
    
    # Extract values, replacing None or 0 with np.nan for honest plotting
    def extract(m):
        return [
            m.get('accuracy') if m.get('accuracy') else np.nan,
            m.get('sensitivity') if m.get('sensitivity') else np.nan,
            m.get('specificity') if m.get('specificity') else np.nan,
            m.get('roc_auc') if m.get('roc_auc') else np.nan
        ]
        
    c_vals = extract(c_metrics)
    q_vals = extract(q_metrics)
    
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    # Close the plot by appending the first value to the end
    c_vals += c_vals[:1]
    q_vals += q_vals[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    
    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories, size=14, weight='bold')
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["", "", "", "", ""], color="grey", size=7)
    plt.ylim(0, 1)
    
    # Plot Classical
    ax.plot(angles, c_vals, linewidth=3, linestyle='--', color='grey', label=f'Classical Baseline ({c_vals[0]*100:.2f}%)')
    
    # Plot Quantum
    ax.plot(angles, q_vals, linewidth=3, linestyle='-', color='#2980B9', label=f'{q_label} ({q_vals[0]*100:.2f}%)')
    ax.fill(angles, q_vals, '#2980B9', alpha=0.3)
    
    plt.title(f'4-Pillar Clinical Performance Standard\n({title} Evaluation)', size=16, weight='bold', y=1.1)
    plt.legend(loc='lower right', bbox_to_anchor=(1.3, 0.0), frameon=False, fontsize=13)
    
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_delta(dataset_metrics):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    datasets = [item[0] for item in dataset_metrics]
    
    c_accs = [item[1].get('accuracy', 0)*100 for item in dataset_metrics]
    q_accs = [item[2].get('accuracy', 0)*100 for item in dataset_metrics]
    
    # Left: Cross-Dataset Accuracy Comparison
    x = np.arange(len(datasets))
    width = 0.35
    
    rects1 = ax1.bar(x - width/2, c_accs, width, label='Classical Baseline', color='grey')
    rects2 = ax1.bar(x + width/2, q_accs, width, label='Quantum (Ours)', color='#2980B9')
    
    ax1.set_ylabel('Accuracy (%)', weight='bold')
    ax1.set_title('Cross-Dataset Accuracy Comparison', weight='bold', size=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, weight='bold', rotation=45, ha='right')
    ax1.set_ylim(40, 105)
    ax1.legend(loc='lower right', frameon=False)
    
    def autolabel_acc(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax1.annotate(f'{height:.1f}%',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', size=9)
    autolabel_acc(rects1)
    autolabel_acc(rects2)
    
    # Right: Delta
    delta_acc = [
        (item[2].get('accuracy', 0) - item[1].get('accuracy', 0))*100 if item[1].get('accuracy', 0) > 0 else 0
        for item in dataset_metrics
    ]
    delta_mcc = [
        (item[2].get('mcc', 0) or 0) - (item[1].get('mcc', 0) or 0) if item[1].get('mcc') else 0
        for item in dataset_metrics
    ]
    
    colors = ['#E74C3C' if val < 0 else '#27AE60' for val in delta_acc]
    rects3 = ax2.bar(x, delta_acc, width=0.6, color=colors, edgecolor='black')
    
    ax2.set_ylabel(r'Quantum Accuracy Delta $\Delta$ (%)', weight='bold')
    ax2.set_title(r'Quantum Performance Advantage $\Delta$ (Quantum - Classical)', weight='bold', size=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets, rotation=45, ha='right')
    ax2.set_ylim(-15, 15)
    ax2.axhline(0, color='black', linewidth=1)
    
    for i, rect in enumerate(rects3):
        if c_accs[i] > 0: # Only label if we have data
            height = rect.get_height()
            label_y = height + 0.5 if height >= 0 else height - 1.5
            d_acc = delta_acc[i]
            d_mcc = delta_mcc[i]
            sign = "+" if d_acc >= 0 else ""
            sign_mcc = "+" if d_mcc >= 0 else ""
            ax2.annotate(rf"$\Delta$Acc: {sign}{d_acc:.2f}%" + "\n" + rf"$\Delta$MCC: {sign_mcc}{d_mcc:.4f}",
                        xy=(rect.get_x() + rect.get_width() / 2, label_y),
                        ha='center', va='bottom', weight='bold', size=9)
    
    plt.tight_layout()
    plt.savefig("results/figures/quantum_vs_classical_delta.png", dpi=150, bbox_inches='tight')
    plt.close()

def main():
    os.makedirs("results/figures", exist_ok=True)
    dataset_metrics = []
    
    # 1. WDBC Data
    wdbc_c = {}
    wdbc_q = {}
    if os.path.exists("results/wdbc_metrics.json"):
        with open("results/wdbc_metrics.json") as f:
            w_data = json.load(f)
            wdbc_c = w_data["models"]["classical_baseline"]["metrics"]
            wdbc_q = w_data["models"]["hybrid_quantum"]["metrics"]
            
        plot_radar("WDBC Breast Cancer", wdbc_c, wdbc_q, "results/figures/wdbc_clinical_radar.png", q_label="8Q VQC")
        dataset_metrics.append(("WDBC Breast", wdbc_c, wdbc_q))

    # 2. Golub Data
    golub_c = {}
    golub_q = {}
    if os.path.exists("results/golub_metrics.json"):
        with open("results/golub_metrics.json") as f:
            g_data = json.load(f)
            golub_c = g_data["models"]["classical_baseline"]["metrics"]
            golub_q = g_data["models"]["hybrid_quantum"]["metrics"]
            
        plot_radar("Golub Leukemia", golub_c, golub_q, "results/figures/golub_clinical_radar.png", q_label="8Q QSVC")
        dataset_metrics.append(("Golub Leukemia", golub_c, golub_q))
        
    # 3. UCI Heart Data
    heart_c = {}
    heart_q = {}
    if os.path.exists("results/uci_metrics.json"):
        with open("results/uci_metrics.json") as f:
            h_data = json.load(f)
            heart_c = h_data["models"]["classical_baseline"]["metrics"]
            heart_q = h_data["models"]["hybrid_quantum"]["metrics"]
            
        plot_radar("UCI Heart Disease", heart_c, heart_q, "results/figures/uci_clinical_radar.png", q_label="8Q QSVC")
        dataset_metrics.append(("UCI Heart", heart_c, heart_q))

    # 4. Cardio Data
    if os.path.exists("results/cardio_metrics.json"):
        with open("results/cardio_metrics.json") as f:
            c_data = json.load(f)
            cardio_c = c_data["models"]["classical_baseline"]["metrics"]
            cardio_q = c_data["models"]["hybrid_quantum"]["metrics"]
        
        plot_radar("Cardio Disease", cardio_c, cardio_q, "results/figures/cardio_clinical_radar.png", q_label="8Q QSVC")
        dataset_metrics.append(("Cardio", cardio_c, cardio_q))

    # 5. Thyroid Data
    if os.path.exists("results/thyroid_metrics.json"):
        with open("results/thyroid_metrics.json") as f:
            t_data = json.load(f)
            thyroid_c = t_data["models"]["classical_baseline"]["metrics"]
            thyroid_q = t_data["models"]["hybrid_quantum"]["metrics"]
        
        plot_radar("Thyroid Disease", thyroid_c, thyroid_q, "results/figures/thyroid_clinical_radar.png", q_label="4Q QSVC")
        dataset_metrics.append(("Thyroid", thyroid_c, thyroid_q))
        
    # Global Delta plot
    if dataset_metrics:
        plot_delta(dataset_metrics)
        
    print("Generated identical original visual designs with actual current metrics.")

if __name__ == "__main__":
    main()
