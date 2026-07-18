import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import joblib
from pathlib import Path

AURA_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AURA_ROOT))

import config as cfg
from aura.data_loader import CICIDSDataLoader
from aura.split_manager import get_canonical_split
from aura.models import FlowAutoencoder, AttackHead
from scripts.benchmark_byzantine import generate_client_data, _run_local_training_dual

def flatten(d):
    return torch.cat([v.flatten() for v in d.values()]).cpu()

def cos(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

class SnapshotAdam(torch.optim.Adam):
    def __init__(self, params, ae_model, base_state, target_steps, *args, **kwargs):
        super().__init__(params, *args, **kwargs)
        self.ae_model = ae_model
        self.base_state = base_state
        self.target_steps = target_steps
        self.step_count = 0
        self.snapshots = {}
        
    def step(self, closure=None):
        loss = super().step(closure)
        self.step_count += 1
        
        if self.step_count in self.target_steps:
            current_state = self.ae_model.state_dict()
            self.snapshots[self.step_count] = flatten({k: current_state[k].clone() - self.base_state[k] for k in current_state})
            
        return loss

def compute_metrics(honest_vals, byz_vals):
    if not honest_vals:
        return 0, 0, 0, 0
    h = np.array(honest_vals)
    b = np.array(byz_vals)
    mean_h, std_h = np.mean(h), np.std(h)
    mean_b, std_b = np.mean(b), np.std(b)
    sep = mean_h - mean_b
    
    y_true = np.array([1]*len(h) + [0]*len(b))
    y_score = np.concatenate([h, b])
    
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = 0.5
        
    return mean_h, mean_b, std_h, std_b, sep, auc

def main():
    loader = CICIDSDataLoader()
    scaler_path = os.path.join(cfg.MODELS_DIR, "scaler.joblib")
    scaler = joblib.load(scaler_path)
    
    all_windows = list(loader.stream_graphs(scaler))
    calib, train, test, attack = get_canonical_split(all_windows, test_fraction=0.20)
    
    def get_benign(windows):
        flows = []
        for g, l in windows:
            if (l == 0).any():
                flows.append(g['edge_attr'][l==0])
        return torch.cat(flows) if flows else torch.empty(0)
    
    root_data = get_benign(calib)[:cfg.FLTRUST_ROOT_SAMPLES]
    
    bundle_path = os.path.join(cfg.MODELS_DIR, "aura_bundle.pth")
    saved_state = torch.load(bundle_path, map_location='cpu', weights_only=True)
    global_ae = {k.replace('autoencoder.', ''): v for k, v in saved_state.items() if k.startswith('autoencoder.')}
    global_head = AttackHead().state_dict()
    
    # SERVER ROOT
    root_ae = FlowAutoencoder()
    root_ae.load_state_dict(global_ae)
    root_ae.eval()
    with torch.no_grad():
        recon, _ = root_ae(root_data)
        mse = F.mse_loss(recon, root_data, reduction='none').mean(dim=1)
    mask = mse < cfg.CH2_MSE_SPLIT_THRESHOLD
    filtered_root = root_data[mask]
    
    root_ae.train()
    root_opt = torch.optim.Adam(root_ae.parameters(), lr=1e-3)
    
    actual_bs = min(cfg.AE_BATCH_SIZE, len(filtered_root))
    root_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(filtered_root),
        batch_size=actual_bs, shuffle=True
    )
    root_step = 0
    for (b,) in root_loader:
        root_opt.zero_grad()
        r, _ = root_ae(b)
        loss = F.mse_loss(r, b)
        loss.backward()
        root_opt.step()
        root_step += 1
        
    root_final_delta = flatten({k: root_ae.state_dict()[k].clone() - global_ae[k] for k in global_ae})
    print(f"Server root executed {root_step} optimizer steps.")

    # CLIENTS
    num_clients = 5
    byz_ratio = 0.2
    target_steps = [1, 2, 4, 8, 16, 32, 64]
    
    clients_info = []
    
    for i in range(num_clients):
        is_byz = i < int(num_clients * byz_ratio)
        c_train, _ = generate_client_data(i, is_byz, False, num_clients)
        
        cae = FlowAutoencoder()
        cae.load_state_dict(global_ae)
        chead = AttackHead()
        chead.load_state_dict(global_head, strict=False)
        
        # We need the final step as well, which _run_local_training_dual computes implicitly.
        c_opt = SnapshotAdam(cae.parameters(), cae, global_ae, set(target_steps), lr=1e-3)
        h_opt = torch.optim.Adam(chead.parameters(), lr=1e-3)
        
        ae_d, _, _, _, _ = _run_local_training_dual(
            cae, chead, c_train, c_opt, h_opt,
            global_ae, global_head,
            cfg.CH2_MSE_SPLIT_THRESHOLD,
            head_epochs=3, batch_size=256
        )
        
        clients_info.append({
            'idx': i,
            'is_byz': is_byz,
            'final_delta': flatten(ae_d),
            'opt': c_opt,
            'final_step': c_opt.step_count
        })
        
    # Analyze
    print("\n" + "="*80)
    print("TASK 3 & 4 - Compute CH1 Through Time")
    print("================================================================================")
    
    print("HONEST CLIENTS (Cosine with Root):")
    h_clients = [c for c in clients_info if not c['is_byz']]
    header = "Client | " + " | ".join(f"Step {s:<2}" for s in target_steps) + " | Final"
    print(header)
    print("-" * len(header))
    for c in h_clients:
        row = [f"C{c['idx']:<4}"]
        for s in target_steps:
            if s in c['opt'].snapshots:
                row.append(f"{cos(root_final_delta, c['opt'].snapshots[s]):>7.4f}")
            else:
                row.append("   -   ")
        row.append(f"{cos(root_final_delta, c['final_delta']):>7.4f}")
        print(" | ".join(row))
        
    print("\nBYZANTINE CLIENT (Cosine with Root):")
    b_clients = [c for c in clients_info if c['is_byz']]
    print(header)
    print("-" * len(header))
    for c in b_clients:
        row = [f"C{c['idx']:<4}"]
        for s in target_steps:
            if s in c['opt'].snapshots:
                row.append(f"{cos(root_final_delta, c['opt'].snapshots[s]):>7.4f}")
            else:
                row.append("   -   ")
        row.append(f"{cos(root_final_delta, c['final_delta']):>7.4f}")
        print(" | ".join(row))

    print("\n" + "="*80)
    print("TASK 5 - Separation Curve")
    print("================================================================================")
    
    results = []
    # Add target steps
    for s in target_steps:
        h_vals = [cos(root_final_delta, c['opt'].snapshots[s]) for c in h_clients if s in c['opt'].snapshots]
        b_vals = [cos(root_final_delta, c['opt'].snapshots[s]) for c in b_clients if s in c['opt'].snapshots]
        if h_vals and b_vals:
            mh, mb, sh, sb, sep, auc = compute_metrics(h_vals, b_vals)
            results.append((f"Step {s}", mh, mb, sh, sb, sep, auc))
            
    # Add final
    h_vals = [cos(root_final_delta, c['final_delta']) for c in h_clients]
    b_vals = [cos(root_final_delta, c['final_delta']) for c in b_clients]
    mh, mb, sh, sb, sep, auc = compute_metrics(h_vals, b_vals)
    results.append(("Final (Cum)", mh, mb, sh, sb, sep, auc))
    
    # Sort by separation
    results.sort(key=lambda x: x[5], reverse=True)
    
    print(f"{'Checkpoint':<15} | {'Honest Mean':<12} | {'Byz Mean':<10} | {'Sep':<8} | {'AUC':<7} | {'Honest Std'}")
    print("-" * 80)
    for name, mh, mb, sh, sb, sep, auc in results:
        print(f"{name:<15} | {mh:>12.4f} | {mb:>10.4f} | {sep:>8.4f} | {auc:>7.4f} | {sh:>10.4f}")
        

if __name__ == '__main__':
    main()
