import sys, pickle, torch
import torch.nn.functional as F
from pathlib import Path

def get_cos(d1, d2):
    t1 = torch.cat([v.flatten() for v in d1.values()])
    t2 = torch.cat([v.flatten() for v in d2.values()])
    if t1.norm() == 0 or t2.norm() == 0: return 0.0
    return F.cosine_similarity(t1.unsqueeze(0), t2.unsqueeze(0)).item()

def main():
    print("STEP 3: Loading exported tensors and dynamic benchmark metrics offline...")
    pkl_file = Path("saved_models") / "exported_tensors_seed_0_round_12.pkl"
    if not pkl_file.exists():
        print(f"Error: {pkl_file} not found! Please run STEP 1 (benchmark_byzantine.py --export-tensors).")
        sys.exit(1)

    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    r_ae = data['root_ae_delta']
    c_aes = data['client_ae_deltas']
    benchmark_ch1 = data['metadata']['benchmark_ch1']

    print("\nSTEP 4 & 5: Verifying Consistency\n")
    print("| Client | Benchmark ch1 | Offline ch1 | Match |")
    print("|---|---|---|---|")

    all_matched = True
    for idx, ae_d in enumerate(c_aes):
        raw_cos = get_cos(r_ae, ae_d)
        relu_cos = max(0.0, raw_cos)
        bench_val = benchmark_ch1[idx]
        
        diff = abs(relu_cos - bench_val)
        is_match = diff < 1e-6
        
        match_str = "Yes" if is_match else "No"
        print(f"| {idx} | {bench_val:.6f} | {relu_cos:.6f} | {match_str} |")
        
        if not is_match:
            print(f"\nDIVERGENCE DETECTED at Client {idx}!")
            print(f"Benchmark: {bench_val:.6f} != Offline: {relu_cos:.6f} (Diff: {diff})")
            all_matched = False
            break

    if all_matched:
        print("\nChannel 1 measurement verified.")

if __name__ == "__main__":
    main()
