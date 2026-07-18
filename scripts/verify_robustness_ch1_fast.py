import os
import sys
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
from scripts.benchmark_byzantine import generate_client_data
from scripts.experiments.byzantine_deception_experiment import _run_latent_inversion_byzantine

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

def main():
    print("Loading data...")
    loader = CICIDSDataLoader()
    scaler = joblib.load(os.path.join(cfg.MODELS_DIR, "scaler.joblib"))
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
    
    target_steps = [8, 16, 32]
    seeds = list(range(10))
    
    results = {
        'Step 8': {'hon': [], 'byz': []},
        'Step 16': {'hon': [], 'byz': []},
        'Step 32': {'hon': [], 'byz': []},
        'Final': {'hon': [], 'byz': []}
    }
    
    for seed in seeds:
        print(f"\n--- Running Seed {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Server Root
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
        for (b,) in root_loader:
            root_opt.zero_grad()
            r, _ = root_ae(b)
            loss = F.mse_loss(r, b)
            loss.backward()
            root_opt.step()
            
        root_final_delta = flatten({k: root_ae.state_dict()[k].clone() - global_ae[k] for k in global_ae})
        
        num_clients = 5
        byz_ratio = 0.2
        
        for i in range(num_clients):
            is_byz = (i == 0) # Client 0 is Byzantine
            c_train, _ = generate_client_data(i, is_byz, False, num_clients)
            
            cae = FlowAutoencoder()
            cae.load_state_dict(global_ae)
            chead = AttackHead()
            chead.load_state_dict(global_head, strict=False)
            
            if is_byz:
                # Latent inversion attacker perfectly mimics the server trajectory 
                # (1 full batch step on benign data)
                ae_opt = torch.optim.Adam(cae.parameters(), lr=1e-3)
                head_opt = torch.optim.Adam(chead.parameters(), lr=1e-3)
                ae_d, _, _, _, _ = _run_latent_inversion_byzantine(
                    cae, chead, c_train, ae_opt, head_opt,
                    global_ae, global_head, cfg.CH2_MSE_SPLIT_THRESHOLD, head_epochs=3
                )
                flat_d = flatten(ae_d)
                # Byz delta is the same across all "steps" because it only takes 1 step
                for step_name in results.keys():
                    results[step_name]['byz'].append(cos(root_final_delta, flat_d))
            else:
                # Honest client runs for 1 epoch (~400 steps)
                c_opt = SnapshotAdam(cae.parameters(), cae, global_ae, set(target_steps), lr=1e-3)
                h_opt = torch.optim.Adam(chead.parameters(), lr=1e-3)
                
                from aura.local_training import run_two_pass_local_training
                run_two_pass_local_training(
                    cae, chead, c_train, c_opt, h_opt,
                    mse_threshold=cfg.CH2_MSE_SPLIT_THRESHOLD,
                    head_epochs=3, batch_size=256
                )
                
                final_d = flatten({k: cae.state_dict()[k].clone() - global_ae[k] for k in global_ae})
                
                if 8 in c_opt.snapshots: results['Step 8']['hon'].append(cos(root_final_delta, c_opt.snapshots[8]))
                if 16 in c_opt.snapshots: results['Step 16']['hon'].append(cos(root_final_delta, c_opt.snapshots[16]))
                if 32 in c_opt.snapshots: results['Step 32']['hon'].append(cos(root_final_delta, c_opt.snapshots[32]))
                results['Final']['hon'].append(cos(root_final_delta, final_d))

    print("\n" + "="*80)
    print("TASK 3 & 4 - Aggregate Statistics & Comparison")
    print("================================================================================")
    
    from sklearn.metrics import roc_auc_score
    
    summary = []
    
    for step_name, data in results.items():
        hon = np.array(data['hon'])
        byz = np.array(data['byz'])
        
        mean_hon = np.mean(hon)
        std_hon = np.std(hon)
        mean_byz = np.mean(byz)
        sep = mean_hon - mean_byz
        
        y_true = np.array([1]*len(byz) + [0]*len(hon))
        y_score = np.concatenate([byz, hon])
        try:
            auc = roc_auc_score(y_true, y_score)
        except:
            auc = 0.5
            
        # Threshold logic: if score < mean_score, flag as byzantine (FLTrust logic)
        # But wait! For Latent Inversion, Byzantine has HIGHER score than honest!
        # If we use strict FLTrust logic (flag < threshold), we flag honest clients!
        # Let's compute False Positives (Honest flagged) and False Negatives (Byz missed)
        mean_score = np.mean(y_score)
        flagged = y_score < mean_score
        
        fp = np.sum(flagged[len(byz):]) / len(seeds) # average FP per round (max 4)
        fn = np.sum(~flagged[:len(byz)]) / len(seeds) # average FN per round (max 1)
        
        summary.append({
            'name': step_name, 'sep': sep, 'auc': auc, 'fp': fp, 'fn': fn,
            'mean_hon': mean_hon, 'mean_byz': mean_byz, 'std_hon': std_hon
        })
        
    summary.sort(key=lambda x: x['auc'], reverse=True)
    
    print(f"{'Checkpoint':<10} | {'Honest Mean':<12} | {'Byz Mean':<10} | {'Sep':<8} | {'AUC':<7} | {'FP':<4} | {'FN':<4} | {'Honest Std'}")
    print("-" * 85)
    for s in summary:
        print(f"{s['name']:<10} | {s['mean_hon']:>12.4f} | {s['mean_byz']:>10.4f} | {s['sep']:>8.4f} | {s['auc']:>7.4f} | {s['fp']:>4.1f} | {s['fn']:>4.1f} | {s['std_hon']:>10.4f}")

if __name__ == '__main__':
    main()
