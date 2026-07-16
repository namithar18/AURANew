import torch
import torch.nn.functional as F
import numpy as np
import config as cfg
from aura.data_loader import CICIDSDataLoader
from aura.split_manager import get_canonical_split
from aura.models import AURAModelBundle

def get_cos(d1, d2):
    t1 = torch.cat([v.flatten() for v in d1.values()])
    t2 = torch.cat([v.flatten() for v in d2.values()])
    if t1.norm() == 0 or t2.norm() == 0: return 0.0
    return F.cosine_similarity(t1.unsqueeze(0), t2.unsqueeze(0)).item()

def get_delta(model, base_w):
    return {k: model.state_dict()[k].clone() - base_w[k] for k in base_w}

# Load data
print("Loading data...")
loader = CICIDSDataLoader()
scaler = loader.fit_scaler()
all_windows = list(loader.stream_graphs(scaler))
calib_windows, train_windows, test_windows, _ = get_canonical_split(all_windows, test_fraction=0.20)

train_windows = train_windows[len(calib_windows):]

# Server Root Dataset (first 2000 benign flows of calib)
server_flows = []
for graph, labels in calib_windows:
    f = graph['edge_attr']
    b_mask = labels == 0
    if b_mask.any():
        server_flows.append(f[b_mask])
    if sum(len(x) for x in server_flows) >= 2000:
        break
root_data = torch.cat(server_flows)[:2000]

# Client Data (random slice of train_windows)
all_train_f = []
all_train_l = []
for graph, labels in train_windows:
    all_train_f.append(graph['edge_attr'])
    all_train_l.append(labels)

all_train_flows = torch.cat(all_train_f)
all_train_labels = torch.cat(all_train_l)

perm = torch.randperm(len(all_train_flows))
all_train_flows = all_train_flows[perm]
all_train_labels = all_train_labels[perm]

client_data = all_train_flows[:6000].clone()
client_labels = all_train_labels[:6000].clone()

# Pretrained model
global_model = AURAModelBundle()
try:
    global_model.load_state_dict(torch.load("saved_models/aura_bundle.pth", map_location='cpu'))
except Exception:
    pass

base_w = {k: v.clone() for k, v in global_model.autoencoder.state_dict().items()}

def train_ae(data, epochs, batch_size, base_w):
    model = global_model.autoencoder.__class__()
    model.load_state_dict(base_w)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        actual_bs = len(data) if batch_size == -1 else batch_size
        dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data), batch_size=actual_bs, shuffle=True)
        for (b,) in dl:
            opt.zero_grad()
            recon, _ = model(b)
            loss = F.mse_loss(recon, b)
            loss.backward()
            opt.step()
    return get_delta(model, base_w)

def train_byzantine(data_in, base_w):
    data = data_in.clone()
    n_attack = int(len(data) * 0.8)
    feature_dim = data.shape[1]
    attack_rows = torch.rand(n_attack, feature_dim)
    attack_rows[:, :16]  = torch.rand(n_attack, 16) * 0.5 + 0.5
    attack_rows[:, 16:32] = torch.rand(n_attack, 16) * 0.4 + 0.6
    attack_rows[:, 32:]  = torch.rand(n_attack, feature_dim - 32) * 0.6 + 0.4
    data[:n_attack] = attack_rows
    
    model = global_model.autoencoder.__class__()
    model.load_state_dict(base_w)
    model.eval()
    with torch.no_grad():
        recon, _ = model(data)
        mse = F.mse_loss(recon, data, reduction='none').mean(dim=1)
    model.train()
    benign_mask = mse < cfg.CH2_MSE_SPLIT_THRESHOLD
    b_flows = data[benign_mask]
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    if len(b_flows) > 0:
        opt.zero_grad()
        recon_b, _ = model(b_flows)
        loss = F.mse_loss(recon_b, b_flows)
        loss.backward()
        opt.step()
    return get_delta(model, base_w)

# Exp 4: AE Pollution
print("\n--- Experiment 4: AE Pollution ---")
global_model.autoencoder.eval()
with torch.no_grad():
    recon, _ = global_model.autoencoder(client_data)
    mse = F.mse_loss(recon, client_data, reduction='none').mean(dim=1)
benign_mask = mse < cfg.CH2_MSE_SPLIT_THRESHOLD
filtered_labels = client_labels[benign_mask]
n_attack_survived = (filtered_labels == 1).sum().item()
n_total_survived = len(filtered_labels)
print(f"Total survived: {n_total_survived}")
print(f"Attack survived: {n_attack_survived} ({n_attack_survived/n_total_survived*100:.2f}%)")

client_b_flows = client_data[benign_mask]
server_1step = train_ae(root_data, epochs=1, batch_size=-1, base_w=base_w)
byz_delta = train_byzantine(client_data, base_w=base_w)

print("\n--- Experiment 1: Server Optimizer Steps ---")
client_honest = train_ae(client_b_flows, epochs=1, batch_size=256, base_w=base_w)
for steps in [1, 2, 5, 10, 20, 24]:
    server_n = train_ae(root_data, epochs=steps, batch_size=-1, base_w=base_w)
    cos = get_cos(server_n, client_honest)
    print(f"Server {steps} steps -> Honest Cosine = {cos:.4f}")

print("\n--- Experiment 2: Vary Batch Size (1 Epoch Server) ---")
for bs in [-1, 1024, 512, 256]:
    server_bs = train_ae(root_data, epochs=1, batch_size=bs, base_w=base_w)
    cos = get_cos(server_bs, client_honest)
    print(f"Server BS={bs if bs != -1 else 'full'} -> Honest Cosine = {cos:.4f}")

print("\n--- Experiment 3: Client Epochs (against 1-step server) ---")
for ep in [1, 2, 5, 10, 20]:
    c_ep = train_ae(client_b_flows, epochs=ep, batch_size=256, base_w=base_w)
    cos = get_cos(server_1step, c_ep)
    print(f"Client {ep} epochs -> Honest Cosine = {cos:.4f}")

print("\n--- Experiment 5: Server Strategies vs Client ---")
server_A = server_1step
server_B = train_ae(root_data, epochs=24, batch_size=-1, base_w=base_w)
server_C = train_ae(root_data, epochs=1, batch_size=256, base_w=base_w)

for name, s_delta in [("A (1 full step)", server_A), ("B (24 full steps)", server_B), ("C (matched bs=256)", server_C)]:
    h_cos = get_cos(s_delta, client_honest)
    b_cos = get_cos(s_delta, byz_delta)
    print(f"Strategy {name}: Honest={h_cos:.4f}, Byz={b_cos:.4f}")
