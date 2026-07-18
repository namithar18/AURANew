import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import joblib
from pathlib import Path
from sklearn.decomposition import PCA

AURA_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AURA_ROOT))

import config as cfg
from aura.data_loader import CICIDSDataLoader
from aura.split_manager import get_canonical_split
from aura.models import FlowAutoencoder, AttackHead, AURAModelBundle
from scripts.benchmark_byzantine import generate_client_data, _run_local_training_dual

def flatten(d):
    return torch.cat([v.flatten() for v in d.values()]).cpu()

def cos(a, b):
    # a and b are flattened 1D tensors
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

class SnapshotAdam(torch.optim.Adam):
    def __init__(self, params, ae_model, base_state, target_steps, *args, **kwargs):
        super().__init__(params, *args, **kwargs)
        self.ae_model = ae_model
        self.base_state = base_state
        self.target_steps = target_steps
        self.step_count = 0
        self.snapshots = {}
        self.step_deltas = []
        self.prev_state = {k: v.clone() for k, v in ae_model.state_dict().items()}

    def step(self, closure=None):
        loss = super().step(closure)
        self.step_count += 1
        
        current_state = self.ae_model.state_dict()
        
        step_delta = {k: current_state[k].clone() - self.prev_state[k] for k in current_state}
        self.step_deltas.append(flatten(step_delta))
        self.prev_state = {k: v.clone() for k, v in current_state.items()}
        
        if self.step_count in self.target_steps:
            self.snapshots[self.step_count] = flatten({k: current_state[k].clone() - self.base_state[k] for k in current_state})
            
        return loss

def compute_metrics(honest_vals, byz_vals):
    if not honest_vals:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    h = np.array(honest_vals)
    b = np.array(byz_vals)
    mean_h, std_h = np.mean(h), np.std(h)
    mean_b, std_b = np.mean(b), np.std(b)
    sep = mean_h - mean_b
    
    # Bhattacharyya
    var_h = np.var(h) + 1e-8
    var_b = np.var(b) + 1e-8
    db = 0.25 * np.log(0.25 * ((var_h/var_b) + (var_b/var_h) + 2)) + \
         0.25 * ((mean_h - mean_b)**2 / (var_h + var_b))
         
    # ROC-AUC
    y_true = np.array([1]*len(h) + [0]*len(b))
    y_score = np.concatenate([h, b])
    
    from sklearn.metrics import roc_auc_score, roc_curve
    try:
        auc = roc_auc_score(y_true, y_score)
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        
        # Youden's J
        J = tpr - fpr
        best_idx = np.argmax(J)
        opt_thresh = thresholds[best_idx]
        best_fpr = fpr[best_idx]
        best_fnr = 1 - tpr[best_idx]
    except Exception:
        auc = 0.5
        best_fpr = 0.0
        best_fnr = 0.0
        opt_thresh = 0.0
        
    return mean_h, mean_b, std_h, np.min(h), np.max(h), db, auc, best_fpr, best_fnr, opt_thresh

def main():
    # Setup data
    loader = CICIDSDataLoader()
    scaler_path = os.path.join(cfg.MODELS_DIR, "scaler.joblib")
    scaler = joblib.load(scaler_path)
    
    all_windows = list(loader.stream_graphs(scaler))
    calib, train, test, attack = get_canonical_split(all_windows, test_fraction=0.20)
    
    # Root dataset
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
    
    # AttackHead is missing from saved bundle, initialize randomly
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
    target_steps = {1, 2, 4, 8, 16, 32, 64}
    root_opt = SnapshotAdam(root_ae.parameters(), root_ae, global_ae, target_steps, lr=1e-3)
    
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
        
    root_final_delta = {k: root_ae.state_dict()[k].clone() - global_ae[k] for k in global_ae}
    root_total_steps = root_opt.step_count
    
    print(f"Root steps: {root_total_steps}")

    # CLIENTS
    num_clients = 5
    byz_ratio = 0.2
    
    clients_info = []
    
    for i in range(num_clients):
        is_byz = i < int(num_clients * byz_ratio)
        c_train, _ = generate_client_data(i, is_byz, False, num_clients)
        
        cae = FlowAutoencoder()
        cae.load_state_dict(global_ae)
        chead = AttackHead()
        chead.load_state_dict(global_head, strict=False)
        
        c_opt = SnapshotAdam(cae.parameters(), cae, global_ae, target_steps, lr=1e-3)
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
            'final_delta': ae_d,
            'opt': c_opt
        })
        
    # --- VARIANTS ---
    results = {}
    
    def add_result(variant, c_idx, val):
        if variant not in results:
            results[variant] = {'honest': [], 'byz': []}
        if clients_info[c_idx]['is_byz']:
            results[variant]['byz'].append(val)
        else:
            results[variant]['honest'].append(val)
            
    # Helper extractors
    def ext_enc(d): return torch.cat([v.flatten() for k,v in d.items() if k.startswith('encoder')]).cpu()
    def ext_dec(d): return torch.cat([v.flatten() for k,v in d.items() if k.startswith('decoder')]).cpu()
    
    root_flat = flatten(root_final_delta)
    root_enc = ext_enc(root_final_delta)
    root_dec = ext_dec(root_final_delta)
    
    for c in clients_info:
        c_flat = flatten(c['final_delta'])
        
        # V1
        add_result("V1 - Baseline", c['idx'], cos(root_flat, c_flat))
        
        # V2
        add_result("V2 - Encoder Only", c['idx'], cos(root_enc, ext_enc(c['final_delta'])))
        
        # V3
        add_result("V3 - Decoder Only", c['idx'], cos(root_dec, ext_dec(c['final_delta'])))
        
        # V4
        r_n = root_flat / (torch.norm(root_flat) + 1e-8)
        c_n = c_flat / (torch.norm(c_flat) + 1e-8)
        add_result("V4 - Normalized", c['idx'], cos(r_n, c_n))
        
        # V5
        if root_total_steps in c['opt'].snapshots and root_total_steps in root_opt.snapshots:
            add_result("V5 - Per-Step Delta", c['idx'], cos(root_opt.snapshots[root_total_steps], c['opt'].snapshots[root_total_steps]))
        elif root_total_steps in c['opt'].snapshots:
            add_result("V5 - Per-Step Delta", c['idx'], cos(root_flat, c['opt'].snapshots[root_total_steps]))
            
        # V6
        if root_opt.step_deltas and c['opt'].step_deltas:
            r_mean_step = torch.stack(root_opt.step_deltas).mean(dim=0)
            c_mean_step = torch.stack(c['opt'].step_deltas).mean(dim=0)
            add_result("V6 - Mean Step Delta", c['idx'], cos(r_mean_step, c_mean_step))
            
        # V7
        if root_opt.step_deltas and c['opt'].step_deltas:
            add_result("V7 - First Batch Only", c['idx'], cos(root_opt.step_deltas[0], c['opt'].step_deltas[0]))
            
        # V8
        if len(root_opt.step_deltas) >= 8 and len(c['opt'].step_deltas) >= 8:
            r_last8 = torch.stack(root_opt.step_deltas[-8:]).sum(dim=0)
            c_last8 = torch.stack(c['opt'].step_deltas[-8:]).sum(dim=0)
            add_result("V8 - Last 8 Step Window", c['idx'], cos(r_last8, c_last8))
            
        # V9
        layer_cosines = []
        layer_weights = []
        for k in root_final_delta:
            r_l = root_final_delta[k].flatten().cpu()
            c_l = c['final_delta'][k].flatten().cpu()
            layer_cos = cos(r_l, c_l)
            w = r_l.numel()
            layer_cosines.append(layer_cos * w)
            layer_weights.append(w)
        weighted_cos = sum(layer_cosines) / sum(layer_weights)
        add_result("V9 - Layerwise Weighted", c['idx'], weighted_cos)
        
        # V10
        add_result("V10 - Sign Cosine", c['idx'], cos(torch.sign(root_flat), torch.sign(c_flat)))
        
        # V11
        add_result("V11 - Absolute Cosine", c['idx'], cos(torch.abs(root_flat), torch.abs(c_flat)))
        
        # V14
        r_lat = root_final_delta['encoder.6.weight'].flatten().cpu()
        c_lat = c['final_delta']['encoder.6.weight'].flatten().cpu()
        add_result("V14 - Latent Encoder", c['idx'], cos(r_lat, c_lat))
        
        # V15
        for ts in [1, 2, 4, 8, 16, 32, 64]:
            if ts in c['opt'].snapshots:
                r_snap = root_opt.snapshots.get(ts, root_flat)
                add_result(f"V15 - Trajectory Step {ts:<2}", c['idx'], cos(r_snap, c['opt'].snapshots[ts]))
                
    # V12
    honest_flats = [flatten(c['final_delta']).numpy() for c in clients_info if not c['is_byz']]
    if len(honest_flats) > 0:
        pca = PCA(n_components=min(10, len(honest_flats)))
        pca.fit(honest_flats)
        r_pca = torch.tensor(pca.transform([root_flat.numpy()])[0])
        for c in clients_info:
            c_pca = torch.tensor(pca.transform([flatten(c['final_delta']).numpy()])[0])
            add_result("V12 - PCA (Honest Fit)", c['idx'], cos(r_pca, c_pca))
            
    # V13
    if len(honest_flats) > 1:
        h_matrix = np.stack(honest_flats)
        var = np.var(h_matrix, axis=0) + 1e-8
        r_white = root_flat / torch.tensor(np.sqrt(var), dtype=torch.float32)
        for c in clients_info:
            c_white = flatten(c['final_delta']) / torch.tensor(np.sqrt(var), dtype=torch.float32)
            add_result("V13 - Diagonal Whitening", c['idx'], cos(r_white, c_white))
            
    print("\n=========================================================================================================")
    print("FINAL RANKING")
    print("=========================================================================================================")
    print(f"{'Variant':<30} | {'Honest':<8} | {'Byzantine':<9} | {'Sep':<7} | {'AUC':<7} | {'Recommended'}")
    print("-" * 105)
    
    summary = []
    for var, vals in results.items():
        if not vals['honest'] or not vals['byz']:
            continue
        mh, mb, stdh, minh, maxh, db, auc, fpr, fnr, thresh = compute_metrics(vals['honest'], vals['byz'])
        sep = mh - mb
        summary.append((var, mh, mb, sep, auc))
        
    summary.sort(key=lambda x: x[3], reverse=True)
    
    for var, mh, mb, sep, auc in summary:
        rec = "*" if auc == 1.0 and sep > 0.05 else ""
        print(f"{var:<30} | {mh:>8.4f} | {mb:>9.4f} | {sep:>7.4f} | {auc:>7.4f} | {rec}")

if __name__ == '__main__':
    main()
