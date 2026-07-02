#!/usr/bin/env python3
"""
scripts/benchmark_byzantine.py
==============================
Implements the 5.1.5 FLTrust Byzantine Benchmark (Hypothesis mapping: H2).

Runs a comparative evaluation of:
  - FedAvg (no defense)
  - Krum (distance-based exclusion)
  - FLTrust (cosine similarity against server root dataset)

Under various Byzantine attack ratios (10%, 20%, 30%, 40%).
Also includes the rare-client contribution preservation experiment.
"""

import sys
import os
import logging
import warnings
from pathlib import Path

# ── Silence ALL noise before any imports ─────────────────────────────────────
# 1. Suppress Python warnings (Flower deprecation etc.)
warnings.filterwarnings("ignore")

# 2. Suppress Ray's own C-level stderr output (the access violation traces)
os.environ["RAY_DISABLE_IMPORT_WARNING"]   = "1"
os.environ["RAY_LOG_TO_STDERR"]            = "0"
os.environ["RAY_LOG_LEVEL"]               = "ERROR"
os.environ["GLOG_minloglevel"]            = "3"   # suppress glog (Ray C++ layer)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg
import torch
import flwr as fl
from typing import Dict, List, Tuple
from flwr.server.strategy import FedAvg

# ── Configure logging: only show our benchmark messages ──────────────────────
logging.basicConfig(
    level=logging.WARNING,                          # global floor: WARNING and above
    format="%(message)s",                           # clean format — no timestamps for bulk output
)
logger = logging.getLogger("byz_bench")
logger.setLevel(logging.INFO)                       # our logger stays at INFO

# Silence every noisy third-party logger explicitly
for noisy in ["flwr", "flwr.server", "flwr.simulation", "ray", "urllib3",
              "asyncio", "grpc", "tensorflow", "torch"]:
    logging.getLogger(noisy).setLevel(logging.ERROR)

from aura.fl_server import KrumFedAURA, KrumOnlyStrategy
from aura.fl_client import AURAFlowerClient
from aura.data_loader import CICIDSDataLoader, load_client_subpartition
from aura.attack_injector import _benign_profile

# ── Global scaler — fitted once, shared across all experiments ───────────────
_loader = CICIDSDataLoader()
try:
    _shared_scaler = _loader.fit_scaler()
    logger.info("✓ Dataset scaler fitted.")
except Exception as e:
    logger.warning(f"Scaler fallback: {e}")
    _shared_scaler = None


# ─────────────────────────────────────────────────────────────────────────────
# Data Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_client_data(
    client_idx: int,
    is_byzantine: bool,
    is_rare: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n_samples = 200

    train_data, val_data = None, None
    if _shared_scaler is not None:
        try:
            # Use sub-partition: clients 0-4 get first half,
            # clients 5-9 get second half of same org partition
            # → 10 genuinely non-overlapping datasets
            train_data, val_data = load_client_subpartition(
                client_idx=client_idx,
                scaler=_shared_scaler,
            )
            if len(train_data) > n_samples:
                train_data = train_data[:n_samples]
            if len(val_data) > max(1, n_samples // 5):
                val_data = val_data[:n_samples // 5]
        except Exception as e:
            logger.warning(f"Sub-partition failed for client {client_idx}: {e}")

    if train_data is None:
        _train_np  = _benign_profile(n_samples, cfg.FEATURE_DIM)
        _val_np    = _benign_profile(max(1, n_samples // 5), cfg.FEATURE_DIM)
        train_data = torch.tensor(_train_np, dtype=torch.float32)
        val_data   = torch.tensor(_val_np,   dtype=torch.float32)

    if is_rare:
        train_data = train_data + 0.15

    if is_byzantine:
        ddos_profile = cfg.ATTACK_CORRUPTION_PROFILES.get("ddos", {})
        feat_map     = cfg.FEATURE_INDEX_MAP
        n_attack     = int(len(train_data) * 0.8)
        attack_rows  = train_data[:n_attack].clone()
        for feat_name, (lo, hi) in ddos_profile.items():
            if feat_name in feat_map:
                col_idx = feat_map[feat_name]
                attack_rows[:, col_idx] = torch.rand(n_attack) * (hi - lo) + lo
        train_data[:n_attack] = attack_rows

    return train_data, val_data


# ─────────────────────────────────────────────────────────────────────────────
# Experiment Runner
# ─────────────────────────────────────────────────────────────────────────────

# Collect results here for the final summary table
_results: list = []

def run_experiment(
    strategy_name:  str,
    num_clients:    int,
    byzantine_ratio: float,
    rare_client:    bool = False,
):
    num_byzantine = int(num_clients * byzantine_ratio)
    roles = ["benign"] * num_clients
    for i in range(num_byzantine):
        roles[i] = "byzantine"
    if rare_client and "benign" in roles:
        roles[-1] = "rare"

    rare_idx      = roles.index("rare") if "rare" in roles else None
    byzantine_idx = [i for i, r in enumerate(roles) if r == "byzantine"]

    def client_fn(cid: str) -> fl.client.Client:
        idx  = int(cid)
        role = roles[idx]
        train_data, val_data = generate_client_data(
            idx,
            is_byzantine=(role == "byzantine"),
            is_rare=(role == "rare"),
        )
        client = AURAFlowerClient(f"client_{cid}", train_data, val_data)
    
        # Load pre-trained weights so process_incoming_batch uses a
        # calibrated AE, not random weights — filtering is meaningless otherwise
        _pretrained = Path(cfg.MODELS_DIR) / "global_model.pth"
        if _pretrained.exists():
            try:
                state = torch.load(_pretrained, map_location="cpu")
                client.model.load_state_dict(state)
                client.model.eval()
            except Exception as e:
                logger.warning(f"Could not load pre-trained weights: {e}")
    
        # Simulate time window before FL round
        if role == "byzantine":
            # Adversary bypasses RT checks — poisons buffer directly
            client._healthy_buffer.append(train_data.cpu())
        else:
            # Honest node: RT inference filters healthy flows into buffer
            for start in range(0, len(train_data), cfg.AE_BATCH_SIZE):
                batch = train_data[start:start + cfg.AE_BATCH_SIZE]
                client.process_incoming_batch(batch)
    
        return client.to_client()

    if strategy_name == "FedAvg":
        strategy = FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=num_clients,
            min_available_clients=num_clients,
        )
    elif strategy_name == "Krum":
        strategy = KrumOnlyStrategy(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=num_clients,
            min_available_clients=num_clients,
        )
    else:
        # FLTrust uses KrumFedAURA
        strategy = KrumFedAURA(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=num_clients,
            min_available_clients=num_clients,
        )

    # Run the simulation — suppress its stdout chatter via logging already set
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.FL_NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 4, "num_gpus": 0.0},
    )

    # ── Extract the metrics we actually care about ────────────────────────
    metrics = history.metrics_distributed_fit if hasattr(history, "metrics_distributed_fit") else {}

    # Trust scores per round (FLTrust/Krum only)
    trust_by_round = dict(metrics.get("trust_scores", []))

    # Which clients were flagged (FLTrust/Krum only)
    flagged_by_round = dict(metrics.get("fltrust_flagged_indices", []))

    # Final round values
    final_round      = cfg.FL_NUM_ROUNDS
    final_trust      = trust_by_round.get(final_round, [])
    final_flagged    = flagged_by_round.get(final_round, [])

    byzantine_flagged = [i for i in byzantine_idx if i in final_flagged]
    rare_flagged      = rare_idx in final_flagged if rare_idx is not None else False

    _results.append({
        "strategy":         strategy_name,
        "byz_ratio":        byzantine_ratio,
        "rare_client":      rare_client,
        "byzantine_idx":    byzantine_idx,
        "rare_idx":         rare_idx,
        "final_trust":      final_trust,
        "flagged":          final_flagged,
        "byzantine_caught": len(byzantine_flagged),
        "byzantine_total":  len(byzantine_idx),
        "rare_flagged":     rare_flagged,
    })

    # ── Print ONE clean summary block per experiment ──────────────────────
    tag = f"{strategy_name:8s} | byz={int(byzantine_ratio*100):2d}%"
    if rare_client:
        tag += " | rare=✓"

    if strategy_name == "FedAvg":
        # FedAvg has no trust scores — just confirm it ran
        print(f"  {tag} | no defense (FedAvg baseline)")
    else:
        if final_trust:
            byz_scores  = [f"{final_trust[i]:.3f}" for i in byzantine_idx if i < len(final_trust)]
            rare_score  = f"{final_trust[rare_idx]:.3f}" if rare_idx is not None and rare_idx < len(final_trust) else "N/A"
            other_avg   = sum(
                final_trust[i] for i in range(len(final_trust))
                if i not in byzantine_idx and i != rare_idx
            ) / max(1, len(final_trust) - len(byzantine_idx) - (1 if rare_idx else 0))

            caught_str  = f"{len(byzantine_flagged)}/{len(byzantine_idx)} caught"
            rare_str    = f"rare {'FLAGGED ⚠' if rare_flagged else 'preserved ✓'}" if rare_client else ""

            print(
                f"  {tag} | "
                f"byz_trust={','.join(byz_scores)} "
                f"benign_avg={other_avg:.3f} "
                f"| {caught_str}"
                + (f" | {rare_str}" if rare_str else "")
            )
        else:
            print(f"  {tag} | (no trust scores returned)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "═"*72)
    print("  AURA — 5.1.5 FLTrust Byzantine Benchmark  (Hypothesis H2)")
    print("═"*72)

    num_clients = 10

    # ── Phase 1: Byzantine ratio sweep ───────────────────────────────────
    print("\n[Phase 1]  Ratio sweep: FedAvg vs FLTrust\n")
    for ratio in [0.1, 0.2, 0.3, 0.4]:
        run_experiment("FedAvg",   num_clients, ratio)
        run_experiment("FLTrust",  num_clients, ratio)
        print()   # blank line between ratio groups

    # ── Phase 2: Rare client preservation ────────────────────────────────
    print("[Phase 2]  Rare-client preservation: Krum vs FLTrust (byz=10%)\n")
    run_experiment("Krum",    num_clients, byzantine_ratio=0.1, rare_client=True)
    run_experiment("FLTrust", num_clients, byzantine_ratio=0.1, rare_client=True)

    # ── Final summary table ───────────────────────────────────────────────
    print("\n" + "═"*72)
    print("  SUMMARY")
    print("═"*72)
    print(f"  {'Strategy':<10} {'Byz%':<6} {'Rare':<6} {'Caught':<14} {'Rare Flagged'}")
    print("  " + "-"*60)
    for r in _results:
        if r["strategy"] == "FedAvg":
            caught_str = "N/A (no defense)"
            rare_str   = "N/A"
        else:
            caught_str = f"{r['byzantine_caught']}/{r['byzantine_total']}"
            rare_str   = ("FLAGGED ⚠" if r["rare_flagged"] else "preserved ✓") if r["rare_client"] else "-"
        print(
            f"  {r['strategy']:<10} "
            f"{int(r['byz_ratio']*100):<6} "
            f"{'yes' if r['rare_client'] else 'no':<6} "
            f"{caught_str:<14} "
            f"{rare_str}"
        )
    print("═"*72 + "\n")


if __name__ == "__main__":
    main()