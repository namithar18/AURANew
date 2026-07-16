"""
offline_crosscheck.py
Loads the exact tensors saved by dc_fltrust_aggregate during the benchmark
and recomputes cosine_similarity independently.

The cosine reported here must match the benchmark's printed signed cosine.
If it does not, the flattening or dict iteration order diverges between
the two call sites.
"""
import pickle, sys
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

pkl = Path("debug_deltas.pkl")
if not pkl.exists():
    print("ERROR: debug_deltas.pkl not found. Run the benchmark first (1 round).")
    sys.exit(1)

with open(pkl, 'rb') as f:
    data = pickle.load(f)

root_delta   = data['root_ae_delta']
client_delta = data['client1_ae_delta']

print("="*50)
print("Offline cross-check: debug_deltas.pkl")
print("="*50)
print(f"Root  keys ({len(root_delta)}):   {list(root_delta.keys())}")
print(f"Client keys ({len(client_delta)}): {list(client_delta.keys())}")
print(f"Keys identical: {list(root_delta.keys()) == list(client_delta.keys())}")

root_flat   = torch.cat([v.flatten() for v in root_delta.values()])
client_flat = torch.cat([v.flatten() for v in client_delta.values()])

print(f"\nRoot  flatten length: {root_flat.numel()}")
print(f"Client flatten length: {client_flat.numel()}")
print(f"Root  norm:   {root_flat.norm():.6f}")
print(f"Client norm:  {client_flat.norm():.6f}")

offline_cosine = F.cosine_similarity(root_flat.unsqueeze(0), client_flat.unsqueeze(0)).item()
print(f"\nOffline cosine (torch.cosine_similarity): {offline_cosine:.6f}")
print(f"\nRoot   first 10 vals: {root_flat[:10].tolist()}")
print(f"Client first 10 vals: {client_flat[:10].tolist()}")

print("\n" + "="*50)
print("Compare this cosine against '[CH1 CHECK] Client 1: raw_signed=...' in benchmark output.")
print("If they match, the benchmark is computing over the correct tensors.")
