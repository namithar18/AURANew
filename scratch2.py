import torch
import torch.nn as nn
import torch.nn.functional as F

class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(47, 32), nn.ReLU(), nn.Linear(32, 16))
        self.decoder = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 47))
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

torch.manual_seed(42)
X = torch.rand(2000, 47) * 0.2  # Benign
X_client = torch.rand(6000, 47) * 0.2  # Benign

def train(model, data, epochs, batch_size):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data), batch_size=batch_size, shuffle=True)
        for (b,) in loader:
            opt.zero_grad()
            recon, _ = model(b)
            loss = F.mse_loss(recon, b)
            loss.backward()
            opt.step()
    return {k: model.state_dict()[k].clone() for k in model.state_dict()}

def get_cos(d1, d2):
    t1 = torch.cat([v.flatten() for v in d1.values()])
    t2 = torch.cat([v.flatten() for v in d2.values()])
    if t1.norm() == 0 or t2.norm() == 0: return 0.0
    return F.cosine_similarity(t1.unsqueeze(0), t2.unsqueeze(0)).item()

base_model = AE()
# "Pretrain" to convergence on benign data
opt = torch.optim.Adam(base_model.parameters(), lr=1e-3)
for _ in range(100):
    recon, _ = base_model(X)
    loss = F.mse_loss(recon, X)
    opt.zero_grad()
    loss.backward()
    opt.step()

base_w = {k: v.clone() for k, v in base_model.state_dict().items()}

server = AE()
server.load_state_dict(base_w)
server_d = train(server, X, epochs=1, batch_size=len(X))
server_delta = {k: server_d[k] - base_w[k] for k in base_w}

for ep in [1, 2]:
    client = AE()
    client.load_state_dict(base_w)
    client_d = train(client, X_client, epochs=ep, batch_size=256)
    client_delta = {k: client_d[k] - base_w[k] for k in base_w}
    print(f"Cosine after pretraining: 1-step server vs {ep} epoch client (bs=256): {get_cos(server_delta, client_delta):.4f}")

    client_1step = AE()
    client_1step.load_state_dict(base_w)
    client_d_1step = train(client_1step, X_client, epochs=1, batch_size=len(X_client))
    client_delta_1step = {k: client_d_1step[k] - base_w[k] for k in base_w}
    print(f"Cosine after pretraining: 1-step server vs 1-step client (bs=all): {get_cos(server_delta, client_delta_1step):.4f}")
