"""
ch1_robustness.py

Refactored to be read-only. This script now loads exported tensors from the
canonical benchmark (scripts/benchmark_byzantine.py) instead of rebuilding
the simulation internally.

It evaluates the raw signed cosine exactly as dc_fltrust_aggregate does.
"""
import sys, math, warnings, os, glob, pickle, subprocess
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, confusion_matrix

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

N_SEEDS = 3
EXPORT_DIR = cfg.MODELS_DIR

def get_exported_file(seed, round_num=1):
    return EXPORT_DIR / f"exported_tensors_seed_{seed}_round_{round_num}.pkl"

def generate_tensors_if_missing():
    # If the exported tensors are missing, we use the benchmark as the canonical producer.
    # We only need round 1 to get the initial gradient trajectory.
    missing_seeds = [s for s in range(N_SEEDS) if not get_exported_file(s, 1).exists()]
    if not missing_seeds:
        print("All exported tensors found.")
        return

    print(f"Generating exported tensors for {len(missing_seeds)} seeds via benchmark...")
    for s in missing_seeds:
        print(f"  Running benchmark for seed {s}...")
        cmd = [
            "python", "scripts/benchmark_byzantine.py",
            "--mode", "dc_fltrust",
            "--attack-mode", "latent_inversion",
            "--rounds", "1",
            "--seed", str(s),
            "--export-tensors"
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        
def cos_delta(d1, d2):
    t1 = torch.cat([v.flatten() for v in d1.values()])
    t2 = torch.cat([v.flatten() for v in d2.values()])
    if t1.norm() == 0 or t2.norm() == 0: return 0.0
    return F.cosine_similarity(t1.unsqueeze(0), t2.unsqueeze(0)).item()

def main():
    generate_tensors_if_missing()
    
    records = []
    
    for seed in range(N_SEEDS):
        file_path = get_exported_file(seed, 1)
        if not file_path.exists():
            print(f"Missing file for seed {seed}, skipping...")
            continue
            
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        r_ae_delta = data['root_ae_delta']
        c_ae_deltas = data['client_ae_deltas']
        roles = data['roles']
        
        for idx, (ae_d, role) in enumerate(zip(c_ae_deltas, roles)):
            is_byz = (role == 'byzantine')
            # Compute exact signed cosine as used in dc_fltrust_aggregate
            raw_cosine = cos_delta(r_ae_delta, ae_d)
            # ReLU is used for aggregation weights, but raw signed is used for classification
            relu_cosine = max(0.0, raw_cosine)
            
            records.append({
                'seed': seed,
                'client': idx,
                'is_byz': is_byz,
                'ch1_raw_cosine': raw_cosine,
                'ch1_relu_cosine': relu_cosine
            })
            
    df = pd.DataFrame(records)
    
    # --- UNIFIED SCORE SEMANTICS ---
    # For the latent inversion threat, the Byzantine attacker is MORE aligned
    # with the root reference than an honest client (Byzantine ch1 > Honest ch1).
    #
    # We define the anomaly score such that HIGHER score = MORE LIKELY BYZANTINE.
    # Therefore, the anomaly score is simply the ch1_relu_cosine itself.
    #
    # The decision rule is: 
    #   if anomaly_score > threshold => Byzantine
    
    anomaly_score = df['ch1_relu_cosine']
    is_byz_label  = df['is_byz'].astype(int)
    
    hon_scores = anomaly_score[is_byz_label == 0]
    byz_scores = anomaly_score[is_byz_label == 1]
    
    print("\n--- STATISTICS ---")
    print(f"Honest (N={len(hon_scores)}): Mean={hon_scores.mean():.4f}, Std={hon_scores.std():.4f}, Min={hon_scores.min():.4f}, Max={hon_scores.max():.4f}, Median={hon_scores.median():.4f}")
    print(f"Byzantine (N={len(byz_scores)}): Mean={byz_scores.mean():.4f}, Std={byz_scores.std():.4f}, Min={byz_scores.min():.4f}, Max={byz_scores.max():.4f}, Median={byz_scores.median():.4f}")
    
    def bhattacharyya(a, b, bins=50):
        lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
        if lo == hi: return 1.0 if a.mean() == b.mean() else 0.0
        ha, _ = np.histogram(a, bins=bins, range=(lo, hi), density=True)
        hb, _ = np.histogram(b, bins=bins, range=(lo, hi), density=True)
        return np.sum(np.sqrt(ha * hb)) * ((hi - lo) / bins)

    bc = bhattacharyya(hon_scores, byz_scores)
    print(f"Bhattacharyya: {bc:.4f}")
    
    auc_roc = roc_auc_score(is_byz_label, anomaly_score)
    precision, recall, _ = precision_recall_curve(is_byz_label, anomaly_score)
    auc_pr = auc(recall, precision)
    print(f"ROC-AUC: {auc_roc:.4f}, PR-AUC: {auc_pr:.4f}")

    thresholds = np.linspace(0, 1.0, 101)
    results = []
    
    for t in thresholds:
        # Explicit scoring convention: Higher anomaly score -> Flagged as Byzantine
        pred_byzantine = (anomaly_score > t).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(is_byz_label, pred_byzantine, labels=[0, 1]).ravel()
        acc = (tp + tn) / len(is_byz_label)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        bal_acc = (rec + (1 - fpr)) / 2
        j = rec - fpr
        
        results.append({
            'Threshold': t,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1': f1,
            'Balanced_Accuracy': bal_acc,
            'FPR': fpr,
            'FNR': fnr,
            'J': j
        })
        
    res_df = pd.DataFrame(results)
    best_j = res_df.loc[res_df['J'].idxmax()]
    
    print("\n--- BEST THRESHOLDS ---")
    print(f"Max Youden's J: t={best_j['Threshold']:.4f} (J={best_j['J']:.4f}, F1={best_j['F1']:.4f})")
    
    print("\n--- DETAILED TABLE (0.25 to 0.55) ---")
    print(res_df[(res_df['Threshold'] >= 0.25) & (res_df['Threshold'] <= 0.55)].to_string(index=False))

if __name__ == "__main__":
    main()
