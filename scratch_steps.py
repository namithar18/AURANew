import math
import torch
import torch.nn.functional as F
import config as cfg
from aura.data_loader import CICIDSDataLoader
from aura.split_manager import get_canonical_split
from aura.models import AURAModelBundle

loader = CICIDSDataLoader()
scaler = loader.fit_scaler()
all_windows = list(loader.stream_graphs(scaler))
calib_windows, train_windows, test_windows, _ = get_canonical_split(all_windows, test_fraction=0.20)
train_windows = train_windows[len(calib_windows):]

# Server root data
server_flows = []
for graph, labels in calib_windows:
    f = graph['edge_attr']
    b_mask = labels == 0
    if b_mask.any():
        server_flows.append(f[b_mask])
    if sum(len(x) for x in server_flows) >= cfg.FLTRUST_ROOT_SAMPLES:
        break
root_data = torch.cat(server_flows)[:cfg.FLTRUST_ROOT_SAMPLES]

# Client data (one partition)
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

# Measure benign flows that client AE trains on
global_model = AURAModelBundle()
try:
    global_model.load_state_dict(torch.load("saved_models/aura_bundle.pth", map_location='cpu'))
except Exception:
    pass

global_model.autoencoder.eval()
with torch.no_grad():
    recon, _ = global_model.autoencoder(client_data)
    mse = F.mse_loss(recon, client_data, reduction='none').mean(dim=1)
benign_mask = mse < cfg.CH2_MSE_SPLIT_THRESHOLD
n_benign = benign_mask.sum().item()

client_ae_steps = math.ceil(n_benign / cfg.AE_BATCH_SIZE)
server_target_steps = math.ceil(len(root_data) / cfg.AE_BATCH_SIZE)

print(f"FL_LOCAL_EPOCHS:       {cfg.FL_LOCAL_EPOCHS}")
print(f"AE_BATCH_SIZE:         {cfg.AE_BATCH_SIZE}")
print(f"root_data size:        {len(root_data)}")
print(f"client benign flows:   {n_benign}")
print(f"Client AE steps/round: ceil({n_benign}/{cfg.AE_BATCH_SIZE}) = {client_ae_steps}")
print(f"Server steps target:   ceil({len(root_data)}/{cfg.AE_BATCH_SIZE}) = {server_target_steps}")
print(f"NOTE: AE has 1 DataLoader pass per _run_local_training_dual call. FL_LOCAL_EPOCHS only controls AttackHead.")
