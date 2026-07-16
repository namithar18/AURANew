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
X = torch.randn(2000, 47)
X_client = torch.randn(6000, 47)

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
    return F.cosine_similarity(t1.unsqueeze(0), t2.unsqueeze(0)).item()

base_model = AE()
base_w = {k: v.clone() for k, v in base_model.state_dict().items()}

server = AE()
server.load_state_dict(base_w)
server_d = train(server, X, epochs=1, batch_size=len(X))
server_delta = {k: server_d[k] - base_w[k] for k in base_w}

for ep in [1, 3, 5, 10, 20]:
    client = AE()
    client.load_state_dict(base_w)
    client_d = train(client, X_client, epochs=ep, batch_size=256)
    client_delta = {k: client_d[k] - base_w[k] for k in base_w}
    print(f"Cosine 1-step server vs {ep} epoch client: {get_cos(server_delta, client_delta):.4f}")
