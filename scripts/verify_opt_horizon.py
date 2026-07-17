import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

AURA_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AURA_ROOT))

import config as cfg
from aura.data_loader import CICIDSDataLoader
from aura.split_manager import get_canonical_split
from aura.models import FlowAutoencoder, AttackHead
import joblib

def flatten(d):
    return torch.cat([v.flatten() for v in d.values()]).cpu()

def cos(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

def main():
    loader = CICIDSDataLoader()
    scaler_path = os.path.join(cfg.MODELS_DIR, "scaler.joblib")
    scaler = joblib.load(scaler_path)
    
    all_windows = list(loader.stream_graphs(scaler))
    calib, train, test, attack = get_canonical_split(all_windows, test_fraction=0.20)
    
    def get_benign_flows(windows):
        if not windows: return torch.empty(0)
        flows = []
        for g, l in windows:
            b_mask = l == 0
            if b_mask.any():
                flows.append(g['edge_attr'][b_mask])
        if flows:
            return torch.cat(flows)
        return torch.empty(0)

    # Server Root Data
    root_data = get_benign_flows(calib)[:cfg.FLTRUST_ROOT_SAMPLES]
    
    # Client 1 Data
    train_for_clients = train[len(calib):]
    all_train_flows = torch.cat([g['edge_attr'] for g, l in train_for_clients])
    torch.manual_seed(0)
    all_train_flows = all_train_flows[torch.randperm(len(all_train_flows))]
    client_pool = all_train_flows[cfg.FLTRUST_ROOT_SAMPLES:]
    
    per_client = len(client_pool) // 5
    start = 1 * per_client
    end = start + per_client
    train_d = client_pool[start:end]
    val_size = max(1, len(train_d) // 5)
    c_data = train_d[val_size:]
    
    bundle_path = os.path.join(cfg.MODELS_DIR, "aura_bundle.pth")
    state = torch.load(bundle_path, map_location='cpu', weights_only=True)
    ae_state = {k.replace('autoencoder.', ''): v for k, v in state.items() if k.startswith('autoencoder.')}
    
    print("============================================================")
    print("TASK 1 - Count Exact Optimizer Steps")
    print("============================================================")
    
    # --- Server Root ---
    root_ae = FlowAutoencoder()
    root_ae.load_state_dict(ae_state)
    root_ae.eval()
    with torch.no_grad():
        recon, _ = root_ae(root_data)
        mse = F.mse_loss(recon, root_data, reduction='none').mean(dim=1)
    mask = mse < cfg.CH2_MSE_SPLIT_THRESHOLD
    root_filtered = root_data[mask]
    
    root_ae.train()
    root_opt = torch.optim.Adam(root_ae.parameters(), lr=1e-3)
    root_bs = min(256, len(root_filtered))
    root_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(root_filtered),
        batch_size=root_bs, shuffle=True
    )
    
    root_steps = 0
    for (b,) in root_loader:
        root_opt.zero_grad()
        r, _ = root_ae(b)
        loss = F.mse_loss(r, b)
        loss.backward()
        root_opt.step()
        root_steps += 1
        
    root_delta_dict = {k: root_ae.state_dict()[k].clone() - ae_state[k] for k in ae_state}
    root_flat = flatten(root_delta_dict)
    
    print("Server Root")
    print(f"* dataset size:           {len(root_filtered)}")
    print(f"* batch size:             {root_bs}")
    print(f"* number of batches:      {len(root_loader)}")
    print(f"* optimizer.step() calls: {root_steps}")
    print(f"* backward() calls:       {root_steps}")
    print()
    
    # --- Honest Client ---
    cae = FlowAutoencoder()
    cae.load_state_dict(ae_state)
    cae.eval()
    with torch.no_grad():
        recon, _ = cae(c_data)
        mse = F.mse_loss(recon, c_data, reduction='none').mean(dim=1)
    mask = mse < cfg.CH2_MSE_SPLIT_THRESHOLD
    c_filtered = c_data[mask]
    
    cae.train()
    c_opt = torch.optim.Adam(cae.parameters(), lr=1e-3)
    c_bs = min(256, len(c_filtered))
    c_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(c_filtered),
        batch_size=c_bs, shuffle=True
    )
    
    print("Honest Client")
    print(f"* dataset size:           {len(c_filtered)}")
    print(f"* batch size:             {c_bs}")
    print(f"* number of batches:      {len(c_loader)}")
    print(f"* optimizer.step() calls: {len(c_loader)}")
    print(f"* backward() calls:       {len(c_loader)}")
    
    # --- Trajectory Snapshots ---
    snapshots = {}
    target_steps = {1, 2, 4, 8, 16, 32, 64}
    
    step = 0
    for (b,) in c_loader:
        c_opt.zero_grad()
        r, _ = cae(b)
        loss = F.mse_loss(r, b)
        loss.backward()
        c_opt.step()
        step += 1
        
        if step in target_steps or step == len(c_loader):
            c_delta = {k: cae.state_dict()[k].clone() - ae_state[k] for k in ae_state}
            name = str(step) if step != len(c_loader) else "Final"
            snapshots[name] = flatten(c_delta)

    print("\n" + "="*60)
    print("TASK 3 - Compare Against Root")
    print("="*60)
    print(f"{'Client Step':<12} | {'Cosine with Root':>15}")
    print("-" * 30)
    
    max_cos = -2.0
    max_step = None
    
    # Preserve order
    keys_order = ['1', '2', '4', '8', '16', '32', '64', 'Final']
    for k in keys_order:
        if k in snapshots:
            c = cos(root_flat, snapshots[k])
            print(f"{k:<12} | {c:>15.4f}")
            if c > max_cos:
                max_cos = c
                max_step = k
                
    print("\n" + "="*60)
    print("TASK 5 - Determine the Peak")
    print("="*60)
    print(f"Maximum cosine: {max_cos:.4f}")
    print(f"Occurred after: {max_step} optimizer steps")

if __name__ == '__main__':
    main()
