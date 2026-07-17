import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# Add project root to path
AURA_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AURA_ROOT))

import config as cfg
from aura.data_loader import CICIDSDataLoader
from aura.split_manager import get_canonical_split
from aura.models import FlowAutoencoder, AttackHead
import joblib

def main():
    # Load Data
    loader = CICIDSDataLoader()
    scaler_path = os.path.join(cfg.MODELS_DIR, "scaler.joblib")
    scaler = joblib.load(scaler_path)
    all_windows = list(loader.stream_graphs(scaler))
    calib, train, test, attack = get_canonical_split(all_windows, test_fraction=0.20)
    train = train[len(calib):]
    all_train_flows = torch.cat([g['edge_attr'] for g, l in train])
    all_train_flows = all_train_flows[torch.randperm(len(all_train_flows))]
    
    # Simulate a client dataset
    client_flows = all_train_flows[2000:4000] # 2000 flows
    
    # Load Model
    bundle_path = os.path.join(cfg.MODELS_DIR, "aura_bundle.pth")
    state = torch.load(bundle_path, map_location='cpu', weights_only=True)
    
    # Need to extract AE state dict
    ae_state = {k.replace('autoencoder.', ''): v for k, v in state.items() if k.startswith('autoencoder.')}
    
    ae = FlowAutoencoder()
    ae.load_state_dict(ae_state)
    ae.eval()
    
    # Pass 0 Filtering
    with torch.no_grad():
        recon, _ = ae(client_flows)
        mse_per_flow = F.mse_loss(recon, client_flows, reduction='none').mean(dim=1)
    
    mse_threshold = cfg.CH2_MSE_SPLIT_THRESHOLD
    benign_mask = mse_per_flow < mse_threshold
    
    kept_flows = client_flows[benign_mask]
    discarded_flows = client_flows[~benign_mask]
    
    kept_mse = mse_per_flow[benign_mask]
    discarded_mse = mse_per_flow[~benign_mask]
    
    print("="*50)
    print("TASK 2 - Runtime Statistics")
    print("="*50)
    print(f"Initial flow count:    {len(client_flows)}")
    print(f"Kept flow count:       {len(kept_flows)}")
    print(f"Discarded flow count:  {len(discarded_flows)}")
    print(f"Percentage discarded:  {100.0 * len(discarded_flows) / len(client_flows):.2f}%")
    print(f"Mean MSE (kept):       {kept_mse.mean().item():.6f}")
    if len(discarded_mse) > 0:
        print(f"Mean MSE (discarded):  {discarded_mse.mean().item():.6f}")
    print(f"Median MSE:            {mse_per_flow.median().item():.6f}")
    print(f"P75 threshold:         {mse_threshold:.6f}")
    print(f"Minimum MSE:           {mse_per_flow.min().item():.6f}")
    print(f"Maximum MSE:           {mse_per_flow.max().item():.6f}")
    
    def get_gradient(flows_dataset):
        if len(flows_dataset) == 0:
            return None
        model = FlowAutoencoder()
        model.load_state_dict(ae_state)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(flows_dataset),
            batch_size=256, shuffle=True
        )
        
        for (batch,) in loader:
            opt.zero_grad()
            rec, _ = model(batch)
            loss = F.mse_loss(rec, batch)
            loss.backward()
            opt.step()
            
        return {k: model.state_dict()[k].clone() - ae_state[k] for k in ae_state}
    
    def flatten(d):
        return torch.cat([v.flatten() for v in d.values()])
    
    grad_A = get_gradient(kept_flows)
    grad_B = get_gradient(discarded_flows)
    grad_C = get_gradient(client_flows)
    
    A_flat = flatten(grad_A)
    B_flat = flatten(grad_B)
    C_flat = flatten(grad_C)
    
    print("\n" + "="*50)
    print("TASK 3 - Gradient Contribution")
    print("="*50)
    print(f"Gradient A (Kept flows):       norm = {A_flat.norm().item():.6f}, samples = {len(kept_flows)}")
    print(f"Gradient B (Discarded flows):  norm = {B_flat.norm().item():.6f}, samples = {len(discarded_flows)}")
    print(f"Gradient C (All benign flows): norm = {C_flat.norm().item():.6f}, samples = {len(client_flows)}")
    
    print("\n" + "="*50)
    print("TASK 4 - Direction Analysis")
    print("="*50)
    def cos(a, b):
        return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    
    print(f"A vs C: {cos(A_flat, C_flat):.4f}")
    print(f"B vs C: {cos(B_flat, C_flat):.4f}")
    print(f"A vs B: {cos(A_flat, B_flat):.4f}")
    
    print("\n" + "="*50)
    print("TASK 5 - Magnitude Analysis")
    print("="*50)
    print(f"Norm A (Kept):       {A_flat.norm().item():.6f}")
    print(f"Norm B (Discarded):  {B_flat.norm().item():.6f}")
    print(f"Relative norm B / A: {B_flat.norm().item() / A_flat.norm().item():.2f}x")
    
if __name__ == '__main__':
    main()
