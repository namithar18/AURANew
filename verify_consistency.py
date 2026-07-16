import sys, pickle, torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

def cos_delta(d1, d2):
    t1 = torch.cat([v.flatten() for v in d1.values()])
    t2 = torch.cat([v.flatten() for v in d2.values()])
    if t1.norm() == 0 or t2.norm() == 0: return 0.0
    return F.cosine_similarity(t1.unsqueeze(0), t2.unsqueeze(0)).item()

def main():
    pkl_file = Path(cfg.MODELS_DIR) / "exported_tensors_seed_0_round_12.pkl"
    if not pkl_file.exists():
        print(f"File {pkl_file} not found.")
        return

    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    r_ae_delta = data['root_ae_delta']
    c_ae_deltas = data['client_ae_deltas']
    roles = data['roles']

    records = []
    print("--- OFFLINE TENSOR VERIFICATION ---")
    for idx, (ae_d, role) in enumerate(zip(c_ae_deltas, roles)):
        raw_cos = cos_delta(r_ae_delta, ae_d)
        relu_cos = max(0.0, raw_cos)
        is_byz = (role == 'byzantine')
        records.append({
            'client': idx,
            'role': role,
            'is_byz': is_byz,
            'raw_cos': raw_cos,
            'ch1': relu_cos
        })
        print(f"Offline Client {idx} | Role: {role} | Raw: {raw_cos:.6f} | ReLU: {relu_cos:.6f}")

    df = pd.DataFrame(records)
    
    anomaly_score = df['ch1']
    is_byz_label = df['is_byz'].astype(int)

    hon_scores = anomaly_score[is_byz_label == 0]
    byz_scores = anomaly_score[is_byz_label == 1]

    auc_roc = roc_auc_score(is_byz_label, anomaly_score)
    precision, recall, _ = precision_recall_curve(is_byz_label, anomaly_score)
    auc_pr = auc(recall, precision)

    def bhattacharyya(a, b, bins=50):
        lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
        if lo == hi: return 1.0 if a.mean() == b.mean() else 0.0
        ha, _ = np.histogram(a, bins=bins, range=(lo, hi), density=True)
        hb, _ = np.histogram(b, bins=bins, range=(lo, hi), density=True)
        return np.sum(np.sqrt(ha * hb)) * ((hi - lo) / bins)

    bc = bhattacharyya(hon_scores, byz_scores)
    
    if len(hon_scores) > 1 and len(byz_scores) > 1:
        s_pool = np.sqrt(((len(hon_scores)-1)*hon_scores.std()**2 + (len(byz_scores)-1)*byz_scores.std()**2)/(len(hon_scores)+len(byz_scores)-2))
        cd = (hon_scores.mean() - byz_scores.mean()) / s_pool if s_pool > 0 else 0
    else:
        cd = 0

    print(f"ROC-AUC: {auc_roc:.4f}")
    print(f"PR-AUC: {auc_pr:.4f}")
    print(f"Bhattacharyya: {bc:.4f}")
    print(f"Cohen's d: {cd:.4f}")

    thresholds = np.linspace(0, 1.0, 101)
    best_j = -1
    best_t = 0
    for t in thresholds:
        yp = (anomaly_score > t).astype(int)
        tn, fp, fn, tp = confusion_matrix(is_byz_label, yp, labels=[0, 1]).ravel()
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        j = rec - fpr
        if j > best_j:
            best_j = j
            best_t = t
            
    print(f"Best Threshold (Youden's J): {best_t:.4f}")

if __name__ == "__main__":
    main()
