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
from aura.local_training import run_two_pass_local_training

def flatten(d):
    return torch.cat([v.flatten() for v in d.values()]).cpu()

def flatten_keys(d, prefix):
    return torch.cat([v.flatten() for k, v in d.items() if k.startswith(prefix)]).cpu()

def cos(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

def main():
    loader = CICIDSDataLoader()
    scaler_path = os.path.join(cfg.MODELS_DIR, "scaler.joblib")
    scaler = joblib.load(scaler_path)
    
    all_windows = list(loader.stream_graphs(scaler))
    calib, train, test, attack = get_canonical_split(all_windows, test_fraction=0.20)
    
    # Need to isolate benign flows
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

    # Root A: calib_windows
    root_a = get_benign_flows(calib)[:cfg.FLTRUST_ROOT_SAMPLES]
    
    # Honest client
    train_for_clients = train[len(calib):]
    all_train_flows = torch.cat([g['edge_attr'] for g, l in train_for_clients])
    torch.manual_seed(0)
    all_train_flows = all_train_flows[torch.randperm(len(all_train_flows))]
    client_pool = all_train_flows[cfg.FLTRUST_ROOT_SAMPLES:]
    
    # Extract Client 1
    per_client = len(client_pool) // 5
    start = 1 * per_client
    end = start + per_client
    train_d = client_pool[start:end]
    val_size = max(1, len(train_d) // 5)
    c_data = train_d[val_size:]
    
    bundle_path = os.path.join(cfg.MODELS_DIR, "aura_bundle.pth")
    state = torch.load(bundle_path, map_location='cpu', weights_only=True)
    ae_state = {k.replace('autoencoder.', ''): v for k, v in state.items() if k.startswith('autoencoder.')}
    head_state = {k.replace('attack_head.', ''): v for k, v in state.items() if k.startswith('attack_head.')}
    
    # Train Root
    root_ae = FlowAutoencoder()
    root_ae.load_state_dict(ae_state)
    root_ae.eval()
    with torch.no_grad():
        recon, _ = root_ae(root_a)
        mse = F.mse_loss(recon, root_a, reduction='none').mean(dim=1)
    mask = mse < cfg.CH2_MSE_SPLIT_THRESHOLD
    root_filtered = root_a[mask]
    
    root_ae.train()
    root_opt = torch.optim.Adam(root_ae.parameters(), lr=1e-3)
    root_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(root_filtered),
        batch_size=256, shuffle=True
    )
    for (b,) in root_loader:
        root_opt.zero_grad()
        r, _ = root_ae(b)
        loss = F.mse_loss(r, b)
        loss.backward()
        root_opt.step()
        
    root_delta = {k: root_ae.state_dict()[k].clone() - ae_state[k] for k in ae_state}
    
    # Train Client
    cae = FlowAutoencoder()
    cae.load_state_dict(ae_state)
    chead = AttackHead()
    chead.load_state_dict(head_state, strict=False)
    c_opt = torch.optim.Adam(cae.parameters(), lr=1e-3)
    h_opt = torch.optim.Adam(chead.parameters(), lr=1e-3)
    
    run_two_pass_local_training(
        cae, chead, c_data, c_opt, h_opt,
        mse_threshold=cfg.CH2_MSE_SPLIT_THRESHOLD,
        head_epochs=3, batch_size=256
    )
    client_delta = {k: cae.state_dict()[k].clone() - ae_state[k] for k in ae_state}
    
    # Representations
    root_enc = flatten_keys(root_delta, 'encoder')
    c_enc = flatten_keys(client_delta, 'encoder')
    
    root_dec = flatten_keys(root_delta, 'decoder')
    c_dec = flatten_keys(client_delta, 'decoder')
    
    root_full = flatten(root_delta)
    c_full = flatten(client_delta)
    
    root_norm = root_full / (torch.norm(root_full) + 1e-8)
    c_norm = c_full / (torch.norm(c_full) + 1e-8)
    
    print("============================================================")
    print("TASK 6 - Representation Audit")
    print("============================================================")
    print(f"Encoder-only params: {cos(root_enc, c_enc):.4f}")
    print(f"Decoder-only params: {cos(root_dec, c_dec):.4f}")
    print(f"Full AE (Raw):       {cos(root_full, c_full):.4f}")
    print(f"Full AE (Normed):    {cos(root_norm, c_norm):.4f}")

if __name__ == '__main__':
    main()
