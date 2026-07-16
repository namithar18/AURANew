import sys, pickle, torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve

def get_cos(d1, d2, prefix=None):
    if prefix:
        keys = [k for k in d1.keys() if k.startswith(prefix)]
        t1 = torch.cat([d1[k].flatten() for k in keys])
        t2 = torch.cat([d2[k].flatten() for k in keys])
    else:
        t1 = torch.cat([v.flatten() for v in d1.values()])
        t2 = torch.cat([v.flatten() for v in d2.values()])
    if t1.norm() == 0 or t2.norm() == 0: return 0.0
    return F.cosine_similarity(t1.unsqueeze(0), t2.unsqueeze(0)).item()

def bhattacharyya(a, b, bins=50):
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    if lo == hi: return 1.0 if a.mean() == b.mean() else 0.0
    ha, _ = np.histogram(a, bins=bins, range=(lo, hi), density=True)
    hb, _ = np.histogram(b, bins=bins, range=(lo, hi), density=True)
    return np.sum(np.sqrt(ha * hb)) * ((hi - lo) / bins)

def cohens_d(hon, byz):
    if len(hon) > 1 and len(byz) > 1:
        s_pool = np.sqrt(((len(hon)-1)*hon.std()**2 + (len(byz)-1)*byz.std()**2)/(len(hon)+len(byz)-2))
        return (hon.mean() - byz.mean()) / s_pool if s_pool > 0 else 0
    return 0.0

def main():
    pkl_file = Path("saved_models") / "exported_tensors_seed_0_round_12.pkl"
    if not pkl_file.exists():
        print(f"Error: {pkl_file} not found!")
        sys.exit(1)

    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    r_ae = data['root_ae_delta']
    c_aes = data['client_ae_deltas']
    roles = data['roles']

    print("====================================================")
    print("STEP 2: CLIENT COSINE VALUES")
    print("====================================================")
    print(f"| {'Client':<6} | {'Full AE':<14} | {'Encoder':<14} | {'Decoder':<14} |")
    print("|--------|----------------|----------------|----------------|")
    
    records = []
    for idx, (ae_d, role) in enumerate(zip(c_aes, roles)):
        full_cos = get_cos(r_ae, ae_d)
        enc_cos = get_cos(r_ae, ae_d, 'encoder.')
        dec_cos = get_cos(r_ae, ae_d, 'decoder.')
        
        # We use ReLU as is standard in the benchmark for metrics,
        # but wait, the prompt says "compute Full AE cosine... Signed cosine or relu?"
        # The prompt says: "compute Full AE cosine, Encoder-only cosine, Decoder-only cosine"
        # I will use max(0, raw) as relu cosine to be consistent with Channel 1, or just raw?
        # I'll store both but for ROC AUC I will use the raw signed cosine or ReLU? 
        # For discrimination, it doesn't matter much if it's signed or relu, but signed is safer if all are negative.
        
        full_relu = max(0.0, full_cos)
        enc_relu = max(0.0, enc_cos)
        dec_relu = max(0.0, dec_cos)
        
        is_byz = (role == 'byzantine')
        records.append({
            'client': idx,
            'is_byz': is_byz,
            'full': full_cos,
            'encoder': enc_cos,
            'decoder': dec_cos
        })
        print(f"| {idx:<6} | {full_cos:<14.6f} | {enc_cos:<14.6f} | {dec_cos:<14.6f} |")

    df = pd.DataFrame(records)
    is_byz_label = df['is_byz'].astype(int)
    
    print("\n====================================================")
    print("STEP 3: DISCRIMINATION METRICS")
    print("====================================================")
    
    for metric_name, col_name in [('Full AE', 'full'), ('Encoder only', 'encoder'), ('Decoder only', 'decoder')]:
        scores = df[col_name]
        hon_scores = scores[is_byz_label == 0]
        byz_scores = scores[is_byz_label == 1]
        
        # For latent inversion, Byzantine > Honest, so we use scores as is.
        try:
            auc_roc = roc_auc_score(is_byz_label, scores)
            precision, recall, _ = precision_recall_curve(is_byz_label, scores)
            auc_pr = auc(recall, precision)
        except ValueError:
            auc_roc = 0.5
            auc_pr = 0.5
            
        bc = bhattacharyya(hon_scores, byz_scores)
        cd = cohens_d(hon_scores, byz_scores)
        
        print(f"{metric_name}:")
        print(f"  ROC-AUC       : {auc_roc:.4f}")
        print(f"  PR-AUC        : {auc_pr:.4f}")
        print(f"  Bhattacharyya : {bc:.4f}")
        print(f"  Cohen's d     : {cd:.4f}\n")

if __name__ == "__main__":
    main()
