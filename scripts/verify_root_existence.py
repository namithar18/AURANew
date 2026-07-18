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

    # Base initial weights
    bundle_path = os.path.join(cfg.MODELS_DIR, "aura_bundle.pth")
    state = torch.load(bundle_path, map_location='cpu', weights_only=True)
    ae_state = {k.replace('autoencoder.', ''): v for k, v in state.items() if k.startswith('autoencoder.')}
    head_state = {k.replace('attack_head.', ''): v for k, v in state.items() if k.startswith('attack_head.')}
    
    def train_model(data, name):
        cae = FlowAutoencoder()
        cae.load_state_dict(ae_state)
        chead = AttackHead()
        chead.load_state_dict(head_state, strict=False)
        c_opt = torch.optim.Adam(cae.parameters(), lr=1e-3)
        h_opt = torch.optim.Adam(chead.parameters(), lr=1e-3)
        
        run_two_pass_local_training(
            cae, chead, data, c_opt, h_opt,
            mse_threshold=cfg.CH2_MSE_SPLIT_THRESHOLD,
            head_epochs=3, batch_size=256
        )
        return {k: cae.state_dict()[k].clone() - ae_state[k] for k in ae_state}
    
    # ── Client Data Preparation ──
    train_for_clients = train[len(calib):]
    all_train_flows = torch.cat([g['edge_attr'] for g, l in train_for_clients])
    torch.manual_seed(0)
    all_train_flows = all_train_flows[torch.randperm(len(all_train_flows))]
    client_pool = all_train_flows[cfg.FLTRUST_ROOT_SAMPLES:]
    
    client_deltas = {}
    for i in range(1, 5):
        per_client = len(client_pool) // 5
        start = i * per_client
        end = start + per_client
        train_d = client_pool[start:end]
        val_size = max(1, len(train_d) // 5)
        c_data = train_d[val_size:]
        client_deltas[i] = flatten(train_model(c_data, f"Client {i}"))
    
    print("============================================================")
    print("TASK 1 - Reproduce the Honest Cluster")
    print("============================================================")
    honest_cosines = []
    print("Honest <-> Honest Cosine Matrix:")
    for i in range(1, 5):
        row = []
        for j in range(1, 5):
            if i == j:
                row.append(1.0)
            else:
                c = cos(client_deltas[i], client_deltas[j])
                row.append(c)
                if i < j:
                    honest_cosines.append(c)
        print(f"Client {i}: " + " | ".join([f"{v:.4f}" for v in row]))
        
    avg_h = np.mean(honest_cosines)
    std_h = np.std(honest_cosines)
    min_h = np.min(honest_cosines)
    max_h = np.max(honest_cosines)
    print(f"\nAverage:            {avg_h:.4f}")
    print(f"Standard deviation: {std_h:.4f}")
    print(f"Minimum:            {min_h:.4f}")
    print(f"Maximum:            {max_h:.4f}")

    # ── Root Candidate Generation ──
    # Root A: Current benchmark root (calib)
    root_a_data = get_benign_flows(calib)[:cfg.FLTRUST_ROOT_SAMPLES]
    rae = FlowAutoencoder()
    rae.load_state_dict(ae_state)
    rae.eval()
    with torch.no_grad():
        recon, _ = rae(root_a_data)
        mse = F.mse_loss(recon, root_a_data, reduction='none').mean(dim=1)
    mask = mse < cfg.CH2_MSE_SPLIT_THRESHOLD
    root_a_filtered = root_a_data[mask]
    rae.train()
    ropt = torch.optim.Adam(rae.parameters(), lr=1e-3)
    rloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(root_a_filtered),
        batch_size=min(256, len(root_a_filtered)), shuffle=True
    )
    for (b,) in rloader:
        ropt.zero_grad()
        r, _ = rae(b)
        loss = F.mse_loss(r, b)
        loss.backward()
        ropt.step()
    root_a_delta = flatten({k: rae.state_dict()[k].clone() - ae_state[k] for k in ae_state})
    
    # Prepare pool for Root B, C, D, E (benign flows from all train)
    benign_pool = get_benign_flows(train_for_clients)
    
    def generate_candidate(seed):
        torch.manual_seed(seed)
        shuffled = benign_pool[torch.randperm(len(benign_pool))]
        subset = shuffled[:cfg.FLTRUST_ROOT_SAMPLES]
        return flatten(train_model(subset, f"Root_Seed_{seed}"))
        
    root_b_delta = generate_candidate(10)
    root_c_delta = generate_candidate(20)
    root_d_delta = generate_candidate(30)
    root_e_delta = generate_candidate(40)
    
    roots = {
        "Root A (calib)   ": root_a_delta,
        "Root B (seed=10) ": root_b_delta,
        "Root C (seed=20) ": root_c_delta,
        "Root D (seed=30) ": root_d_delta,
        "Root E (seed=40) ": root_e_delta,
    }
    
    print("\n" + "="*60)
    print("TASK 3 & 4 & 5 - Compute Root-to-Honest Matrix & Variance")
    print("============================================================")
    print(f"{'Root Name':<17} | C1     | C2     | C3     | C4     | Average")
    print("-" * 65)
    
    root_avgs = []
    for name, delta in roots.items():
        cosines = [cos(delta, client_deltas[i]) for i in range(1, 5)]
        avg = np.mean(cosines)
        root_avgs.append(avg)
        c_strs = [f"{c:.4f}" for c in cosines]
        print(f"{name:<17} | {' | '.join(c_strs)} | {avg:.4f}")
        
    print(f"\nVariance across root averages: {np.var(root_avgs):.6f}")

if __name__ == '__main__':
    main()
