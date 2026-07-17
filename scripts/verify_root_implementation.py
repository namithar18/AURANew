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

def flatten_keys(d, prefix):
    return torch.cat([v.flatten() for k, v in d.items() if k.startswith(prefix)]).cpu()

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
    c_data_full = train_d[val_size:]
    
    # TASK 2 - Extract 2000 flows for root
    root_data = c_data_full[:2000]
    client_data = c_data_full[2000:]
    
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

    # TASK 3 - Build Two Updates
    root_delta = train_model(root_data, "Root")
    client_delta = train_model(client_data, "Client")
    
    # TASK 4 - Compare Cosines
    r_enc = flatten_keys(root_delta, 'encoder')
    c_enc = flatten_keys(client_delta, 'encoder')
    r_dec = flatten_keys(root_delta, 'decoder')
    c_dec = flatten_keys(client_delta, 'decoder')
    r_full = flatten(root_delta)
    c_full = flatten(client_delta)
    
    print("============================================================")
    print("TASK 4 - Compare Cosines")
    print("============================================================")
    print(f"Representation\tCosine")
    print(f"Encoder       \t{cos(r_enc, c_enc):.4f}")
    print(f"Decoder       \t{cos(r_dec, c_dec):.4f}")
    print(f"Full          \t{cos(r_full, c_full):.4f}")
    
    cos_full = cos(r_full, c_full)
    
    print("\n" + "="*60)
    print("TASK 5 - Compare Against Historical Results")
    print("============================================================")
    print(f"New experiment cosine: {cos_full:.4f}")
    if cos_full > 0.5:
        print("Most closely resembles: Historical Honest -> Honest (~0.70)")
    else:
        print("Most closely resembles: Current benchmark Root -> Client (~0.22)")
        
if __name__ == '__main__':
    main()
