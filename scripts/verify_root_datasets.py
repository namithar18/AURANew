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

def cos(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

def train_ae_only(dataset, ae_state):
    # Pass 0 Filter
    ae = FlowAutoencoder()
    ae.load_state_dict(ae_state)
    ae.eval()
    with torch.no_grad():
        recon, _ = ae(dataset)
        mse = F.mse_loss(recon, dataset, reduction='none').mean(dim=1)
    mask = mse < cfg.CH2_MSE_SPLIT_THRESHOLD
    filtered = dataset[mask]
    
    # Train
    ae.train()
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(filtered),
        batch_size=256, shuffle=True
    )
    for (b,) in loader:
        opt.zero_grad()
        r, _ = ae(b)
        loss = F.mse_loss(r, b)
        loss.backward()
        opt.step()
        
    return flatten({k: ae.state_dict()[k].clone() - ae_state[k] for k in ae_state})

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
    root_a_pool = get_benign_flows(calib)
    
    # Honest clients pool (train without calib)
    train_for_clients = train[len(calib):]
    all_train_flows = torch.cat([g['edge_attr'] for g, l in train_for_clients])
    torch.manual_seed(0)
    all_train_flows = all_train_flows[torch.randperm(len(all_train_flows))]
    
    client_pool = all_train_flows[cfg.FLTRUST_ROOT_SAMPLES:] # exactly as benchmark
    
    # Root B: Union of honest client partitions
    root_b_pool = get_benign_flows(train_for_clients) # wait, client_pool is a subset of this
    root_b_pool = client_pool # just take the client pool directly (it has attacks but pass0 drops high mse)
    # wait, the prompt says "Random benign subset".
    # I should explicitly filter labels == 0 if possible, but client_pool is unsupervised.
    # We can just trust Pass-0 to drop attacks, or we can filter it. Let's use get_benign_flows on train_for_clients.
    # Actually, the prompt says: "Random benign subset sampled from the UNION of all honest client training partitions."
    # Client partitions are slices of client_pool. I will extract all benign flows from client_pool.
    # Wait, client_pool doesn't have labels attached in this flat tensor. 
    # Let me just get benign flows directly from train_for_clients.
    b_flows_train = get_benign_flows(train_for_clients)
    torch.manual_seed(1)
    b_flows_train = b_flows_train[torch.randperm(len(b_flows_train))]
    root_b_pool = b_flows_train
    
    # Root C: COMPLETE benign dataset
    b_flows_all = get_benign_flows(all_windows)
    torch.manual_seed(2)
    b_flows_all = b_flows_all[torch.randperm(len(b_flows_all))]
    root_c_pool = b_flows_all

    root_size = cfg.FLTRUST_ROOT_SAMPLES
    
    root_a = root_a_pool[:root_size]
    root_b = root_b_pool[:root_size]
    root_c = root_c_pool[:root_size]
    
    bundle_path = os.path.join(cfg.MODELS_DIR, "aura_bundle.pth")
    state = torch.load(bundle_path, map_location='cpu', weights_only=True)
    ae_state = {k.replace('autoencoder.', ''): v for k, v in state.items() if k.startswith('autoencoder.')}
    head_state = {k.replace('attack_head.', ''): v for k, v in state.items() if k.startswith('attack_head.')}
    
    # Train Roots
    grad_A = train_ae_only(root_a, ae_state)
    grad_B = train_ae_only(root_b, ae_state)
    grad_C = train_ae_only(root_c, ae_state)
    
    # Train Clients (using exactly the same logic as previous test)
    client_deltas = {}
    for i in range(1, 5):
        per_client = len(client_pool) // 5
        start = i * per_client
        end = start + per_client
        train_d = client_pool[start:end]
        val_size = max(1, len(train_d) // 5)
        c_data = train_d[val_size:]
        
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
        c_delta = {k: cae.state_dict()[k].clone() - ae_state[k] for k in ae_state}
        client_deltas[i] = flatten(c_delta)

    print("="*60)
    print("TASK 3 - Compute Honest Alignment")
    print("="*60)
    def print_alignments(name, root_grad):
        cosines = []
        for i in range(1, 5):
            c = cos(client_deltas[i], root_grad)
            cosines.append(c)
        avg = sum(cosines)/4
        print(f"Average Honest <-> {name}: {avg:.4f}")
        return avg

    avg_A = print_alignments("Root A (calib_windows)", grad_A)
    avg_B = print_alignments("Root B (Client Union)", grad_B)
    avg_C = print_alignments("Root C (Complete Dataset)", grad_C)
    
    print("\n" + "="*60)
    print("TASK 4 - Compare Against Honest Cluster")
    print("="*60)
    honest_cluster_cos = 0.6979
    print(f"Honest Cluster Reference: {honest_cluster_cos:.4f}")
    print(f"Distance to Root A:       {abs(honest_cluster_cos - avg_A):.4f}")
    print(f"Distance to Root B:       {abs(honest_cluster_cos - avg_B):.4f}")
    print(f"Distance to Root C:       {abs(honest_cluster_cos - avg_C):.4f}")

if __name__ == '__main__':
    main()
