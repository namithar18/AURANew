import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import math

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

def cos(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

def main():
    loader = CICIDSDataLoader()
    scaler_path = os.path.join(cfg.MODELS_DIR, "scaler.joblib")
    scaler = joblib.load(scaler_path)
    
    # Load canonical split
    all_windows = list(loader.stream_graphs(scaler))
    calib, train, test, attack = get_canonical_split(all_windows, test_fraction=0.20)
    train = train[len(calib):]
    all_train_flows = torch.cat([g['edge_attr'] for g, l in train])
    
    # Use fixed seed for exact reproducibility matching the benchmark
    torch.manual_seed(0)
    all_train_flows = all_train_flows[torch.randperm(len(all_train_flows))]
    
    root_size = cfg.FLTRUST_ROOT_SAMPLES
    client_pool = all_train_flows[root_size:]
    
    def get_client_data(client_idx, num_clients=5):
        per_client = len(client_pool) // num_clients
        start = client_idx * per_client
        end = start + per_client
        train_d = client_pool[start:end]
        val_size = max(1, len(train_d) // 5)
        return train_d[val_size:] # train split only
    
    # Extract root data and apply pass-0
    root_data = all_train_flows[:root_size]
    
    bundle_path = os.path.join(cfg.MODELS_DIR, "aura_bundle.pth")
    state = torch.load(bundle_path, map_location='cpu', weights_only=True)
    
    ae_state = {k.replace('autoencoder.', ''): v for k, v in state.items() if k.startswith('autoencoder.')}
    head_state = {k.replace('attack_head.', ''): v for k, v in state.items() if k.startswith('attack_head.')}
    
    # Server Root Delta Generation
    root_ae = FlowAutoencoder()
    root_ae.load_state_dict(ae_state)
    root_ae.eval()
    with torch.no_grad():
        recon, _ = root_ae(root_data)
        mse_per_flow = F.mse_loss(recon, root_data, reduction='none').mean(dim=1)
    
    root_mask = mse_per_flow < cfg.CH2_MSE_SPLIT_THRESHOLD
    root_filtered = root_data[root_mask]
    
    root_ae.train()
    root_opt = torch.optim.Adam(root_ae.parameters(), lr=1e-3)
    root_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(root_filtered), batch_size=256, shuffle=True)
    for (b,) in root_loader:
        root_opt.zero_grad()
        r, _ = root_ae(b)
        loss = F.mse_loss(r, b)
        loss.backward()
        root_opt.step()
        
    root_delta = {k: root_ae.state_dict()[k].clone() - ae_state[k] for k in ae_state}
    root_flat = flatten(root_delta)
    
    # Honest Client Generation
    client_deltas = {}
    for i in range(1, 5): # 1, 2, 3, 4 are honest
        client_data = get_client_data(i)
        
        cae = FlowAutoencoder()
        cae.load_state_dict(ae_state)
        chead = AttackHead()
        chead.load_state_dict(head_state, strict=False)
        
        c_opt = torch.optim.Adam(cae.parameters(), lr=1e-3)
        h_opt = torch.optim.Adam(chead.parameters(), lr=1e-3)
        
        # Run standard local training
        run_two_pass_local_training(
            cae, chead, client_data, c_opt, h_opt,
            mse_threshold=cfg.CH2_MSE_SPLIT_THRESHOLD,
            head_epochs=3, batch_size=256
        )
        
        c_delta = {k: cae.state_dict()[k].clone() - ae_state[k] for k in ae_state}
        client_deltas[i] = flatten(c_delta)
        
    print("="*60)
    print("TASK 2 - Honest Pairwise Cosines")
    print("="*60)
    pairs = [(1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
    honest_cosines = []
    for c1, c2 in pairs:
        c = cos(client_deltas[c1], client_deltas[c2])
        honest_cosines.append(c)
        r = max(0.0, c)
        print(f"Client {c1} <-> Client {c2}: signed = {c:.4f} | relu = {r:.4f}")
        
    print("\n" + "="*60)
    print("TASK 3 - Compare Against Root")
    print("="*60)
    root_cosines = []
    for i in range(1, 5):
        c = cos(client_deltas[i], root_flat)
        root_cosines.append(c)
        r = max(0.0, c)
        print(f"Client {i} <-> Root: signed = {c:.4f} | relu = {r:.4f}")
        
    avg_honest = sum(honest_cosines) / len(honest_cosines)
    avg_root = sum(root_cosines) / len(root_cosines)
    
    print(f"\nAverage Honest <-> Honest: {avg_honest:.4f}")
    print(f"Average Honest <-> Root:   {avg_root:.4f}")

if __name__ == '__main__':
    main()
