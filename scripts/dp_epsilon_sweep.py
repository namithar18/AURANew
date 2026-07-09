"""
dp_epsilon_sweep.py — Tier 2.1 Privacy Evaluation (H4)
========================================================
Sweeps cfg.DP_NOISE_SWEEP, and for each noise multiplier sigma:

  1. Runs a real multi-round FLTrust federation (5 org clients, via the
     actual AURAFlowerClient.fit() path — same code run.py/run_fl.py use,
     NOT a reimplemented training loop) at that sigma.
  2. Records per-client, per-round dp_epsilon / dp_epsilon_head (both
     channels are DP-wrapped separately in fl_client.py — tracked separately
     here, never averaged into one number).
  3. Records the FLTrust trust-score distribution for that sigma (the
     DP-rotates-gradients-into-Byzantine-looking-territory question from
     the roadmap doc, section 3.4).
  4. Evaluates final AE-only detection F1 on the held-out canonical test
     split (reuses benchmark_ablation.run_mode_a / compute_metrics directly
     — does not reimplement the metric).
  5. Saves the sigma-trained AE checkpoint and invokes mia_attack.py against
     it (subprocess, since mia_attack.py reads a fixed checkpoint path by
     design — see NOTE below).
  6. Invokes gradient_inversion_attack.py — see NOTE, its output is NOT
     currently sigma-sensitive; flagged explicitly rather than silently
     reported as if it were.

KNOWN LIMITATIONS (intentionally not hidden — read before citing results)
---------------------------------------------------------------------------
* Per-round epsilon, not cumulative epsilon. Opacus's PrivacyEngine is
  re-created fresh every round in fl_client.py (required — see comment
  there about clean hook attachment after global-weight loading). That
  means epsilon here is "cost of one round's local training," not the
  end-to-end privacy budget spent by a client across all rounds it
  participated in. Composing across rounds would need the PrivacyEngine
  to persist across rounds, which conflicts with the existing weight-
  reload design. Stated as a limitation in the output JSON, not silently
  glossed over.

* gradient_inversion_attack.py currently builds its own fresh
  _StandaloneAE/_StandaloneHead with random weights internally — it has
  no --sigma flag and does not load any saved checkpoint. Running it here
  documents that gap for the record (its own output is annotated
  "not_sigma_sensitive": true) rather than fabricating a sigma-dependent
  number. Fixing it to accept a checkpoint path is a separate, small
  follow-up task, not done here without your sign-off.

USAGE
-----
    python scripts/dp_epsilon_sweep.py                  # full sweep, all sigmas
    python scripts/dp_epsilon_sweep.py --quick           # 2 rounds, 1 org client subset, skip attacks
    python scripts/dp_epsilon_sweep.py --sigma 1.0       # single sigma, for debugging
    python scripts/dp_epsilon_sweep.py --skip-attacks    # FL + trust + F1 only, no MIA/inversion
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from flwr.common import FitIns, ndarrays_to_parameters, parameters_to_ndarrays

AURA_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AURA_ROOT))

import config as cfg
from aura.data_loader import CICIDSDataLoader, load_client_partition, _ORG_NAMES
from aura.models import AURAModelBundle
from aura.fl_client import AURAFlowerClient, model_to_ndarrays, ndarrays_to_model
from aura.fl_server import fltrust_aggregate, _build_root_dataset

# Reuse the ablation script's evaluation machinery rather than reimplementing
# reconstruction-error-based F1 from scratch.
from scripts.benchmark_ablation import (
    collect_test_windows, calibrate_ae_threshold, run_mode_a, compute_metrics,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger("dp_sweep")

RESULTS_DIR = AURA_ROOT / "results"
SWEEP_MODELS_DIR = AURA_ROOT / "saved_models" / "dp_sweep"
MIA_TARGET_PATH = AURA_ROOT / "saved_models" / "autoencoder_best.pth"


# ─────────────────────────────────────────────────────────────────────────
# One full FLTrust federation run at a fixed sigma
# ─────────────────────────────────────────────────────────────────────────

def run_federation_at_sigma(sigma: float, n_rounds: int, n_clients: int,
                             scaler, device: str) -> dict:
    """
    Manually drives AURAFlowerClient.fit() + fltrust_aggregate() for
    n_rounds — bypassing the Flower gRPC/simulation harness (consistent
    with how benchmark_byzantine.py already treats these as local-process
    experiments), so we can read dp_epsilon / dp_epsilon_head directly off
    each client object after every round instead of threading it through
    Flower's metrics plumbing.
    """
    cfg.DP_NOISE_MULTIPLIER = sigma  # the actual dial for this sweep point
    logger.info(f"── sigma={sigma}  (DP {'ON' if sigma > 0 else 'OFF — control point'}) ──")

    org_ids = [f"org_{name}_1" for name in _ORG_NAMES[:n_clients]]
    clients = []
    for cid in org_ids:
        train_data, val_data = load_client_partition(cid, scaler=scaler)
        clients.append(AURAFlowerClient(
            client_id=cid, train_data=train_data, val_data=val_data, device=device,
        ))

    global_model = AURAModelBundle().to(device)
    root_data = _build_root_dataset()

    round_records = []
    for rnd in range(1, n_rounds + 1):
        global_arrays = model_to_ndarrays(global_model)
        fit_ins = FitIns(
            parameters=ndarrays_to_parameters(global_arrays),
            config={"local_epochs": cfg.FL_LOCAL_EPOCHS, "round": rnd},
        )

        client_updates, per_client_eps = [], []
        for client in clients:
            fit_res = client.fit(fit_ins)
            client_updates.append(parameters_to_ndarrays(fit_res.parameters))
            per_client_eps.append({
                "client_id": client.client_id,
                "dp_epsilon_ae": client.last_epsilon,
                "dp_epsilon_head": client.last_epsilon_head,
            })

        aggregated, trust_scores, flagged = fltrust_aggregate(
            global_model=global_model,
            client_updates=client_updates,
            root_data=root_data,
            server_lr=cfg.FLTRUST_SERVER_LR,
            min_trust=cfg.FLTRUST_MIN_TRUST_SCORE,
        )
        with torch.no_grad():
            for p, arr in zip(global_model.parameters(), aggregated):
                p.copy_(torch.tensor(arr, device=device))

        round_records.append({
            "round": rnd,
            "per_client_epsilon": per_client_eps,
            "trust_scores": [round(t, 4) for t in trust_scores],
            "flagged_indices": flagged,
        })
        logger.info(
            f"  round {rnd}/{n_rounds}  trust={[round(t,3) for t in trust_scores]}  "
            f"flagged={flagged}  "
            f"eps_ae(client0)={per_client_eps[0]['dp_epsilon_ae']:.4f}"
        )

    return {"sigma": sigma, "rounds": round_records, "final_model": global_model}


# ─────────────────────────────────────────────────────────────────────────
# Detection quality at the resulting global model
# ─────────────────────────────────────────────────────────────────────────

def evaluate_f1(global_model: AURAModelBundle, calibration_windows: list,
                test_windows: list, device: str) -> dict:
    ae = global_model.autoencoder
    ae.eval()
    threshold, _, _ = calibrate_ae_threshold(ae, calibration_windows, torch.device(device))
    y_true, y_pred, y_score = run_mode_a(ae, test_windows, threshold, torch.device(device))
    return compute_metrics(y_true, y_pred, y_score)


# ─────────────────────────────────────────────────────────────────────────
# Privacy attacks against this sigma's checkpoint
# ─────────────────────────────────────────────────────────────────────────

def run_privacy_attacks(sigma: float, global_model: AURAModelBundle) -> dict:
    SWEEP_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    sigma_tag = f"sigma_{sigma:.2f}"
    archive_path = SWEEP_MODELS_DIR / f"{sigma_tag}_autoencoder.pth"
    torch.save(global_model.autoencoder.state_dict(), archive_path)

    # mia_attack.py reads a fixed path (saved_models/autoencoder_best.pth) —
    # point it at this sigma's checkpoint by overwriting that path. Back up
    # whatever was there first so a real run.py-trained checkpoint isn't lost.
    backup_path = None
    if MIA_TARGET_PATH.exists():
        backup_path = MIA_TARGET_PATH.with_suffix(".pth.sweep_backup")
        shutil.copy2(MIA_TARGET_PATH, backup_path)
    shutil.copy2(archive_path, MIA_TARGET_PATH)

    mia_result = {"ran": False}
    try:
        proc = subprocess.run(
            [sys.executable, "aura_attacks/mia_attack.py"],
            cwd=str(AURA_ROOT), capture_output=True, text=True, timeout=600,
        )
        mia_result = {"ran": True, "returncode": proc.returncode,
                       "stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-1000:]}
    except Exception as e:
        mia_result = {"ran": False, "error": str(e)}
    finally:
        if backup_path is not None:
            shutil.copy2(backup_path, MIA_TARGET_PATH)
            backup_path.unlink()

    inversion_result = {"ran": False, "not_sigma_sensitive": True,
                         "note": ("gradient_inversion_attack.py builds its own random-weight "
                                  "standalone model internally; it has no --sigma flag and does "
                                  "not consume this checkpoint. Result recorded for completeness "
                                  "but is NOT comparable across sigma values until the script "
                                  "is extended to accept a trained checkpoint.")}
    try:
        proc = subprocess.run(
            [sys.executable, "aura_attacks/gradient_inversion_attack.py",
             "--steps", "50", "--restarts", "1"],  # cheap pass; full run is expensive x5
            cwd=str(AURA_ROOT), capture_output=True, text=True, timeout=600,
        )
        inversion_result.update({"ran": True, "returncode": proc.returncode,
                                  "stdout_tail": proc.stdout[-2000:]})
    except Exception as e:
        inversion_result.update({"ran": False, "error": str(e)})

    return {"mia": mia_result, "gradient_inversion": inversion_result,
            "archived_checkpoint": str(archive_path)}


# ─────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                     help="2 rounds, skip attacks — fast wiring sanity pass")
    ap.add_argument("--sigma", type=float, default=None,
                     help="run a single sigma value instead of the full DP_NOISE_SWEEP")
    ap.add_argument("--rounds", type=int, default=None,
                     help="override FL rounds per sigma (default: cfg.FL_NUM_ROUNDS, or 2 with --quick)")
    ap.add_argument("--n-clients", type=int, default=5)
    ap.add_argument("--skip-attacks", action="store_true",
                     help="run FL + trust + F1 only, skip MIA/gradient-inversion subprocess calls")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sigmas = [args.sigma] if args.sigma is not None else list(cfg.DP_NOISE_SWEEP)
    n_rounds = args.rounds or (2 if args.quick else cfg.FL_NUM_ROUNDS)
    skip_attacks = args.skip_attacks or args.quick

    logger.info("Fitting shared scaler once (reused across all sigma runs + eval) …")
    loader = CICIDSDataLoader()
    scaler = loader.fit_scaler()

    logger.info("Collecting canonical test split once (independent of sigma) …")
    calibration_windows, test_windows = collect_test_windows(loader, scaler)

    original_sigma = cfg.DP_NOISE_MULTIPLIER
    sweep_results = []
    try:
        for sigma in sigmas:
            t0 = time.time()
            fed = run_federation_at_sigma(sigma, n_rounds, args.n_clients, scaler, device)
            metrics = evaluate_f1(fed["final_model"], calibration_windows, test_windows, device)
            logger.info(f"  sigma={sigma}  F1={metrics['F1']}  FPR={metrics['FPR']}")

            entry = {
                "sigma": sigma,
                "n_rounds": n_rounds,
                "detection_metrics": metrics,
                "round_records": fed["rounds"],
                "elapsed_sec": round(time.time() - t0, 1),
            }
            if not skip_attacks:
                entry["privacy_attacks"] = run_privacy_attacks(sigma, fed["final_model"])
            sweep_results.append(entry)
    finally:
        cfg.DP_NOISE_MULTIPLIER = original_sigma  # restore, don't leak sweep state

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "dp_epsilon_sweep.json"
    payload = {
        "limitations": [
            "epsilon is per-round local-training budget; NOT composed across rounds "
            "(PrivacyEngine is recreated fresh each round in fl_client.py by design).",
            "gradient_inversion_attack.py results are not sigma-sensitive in the "
            "current codebase — see privacy_attacks.gradient_inversion.note per entry.",
        ],
        "sweep": sweep_results,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    logger.info(f"Wrote {out_path}")

    print("\n=== DP EPSILON SWEEP SUMMARY ===")
    print(f"{'sigma':>6} | {'F1':>7} | {'FPR':>7} | {'eps_ae(last round, client0)':>28}")
    for e in sweep_results:
        last_round = e["round_records"][-1]
        eps0 = last_round["per_client_epsilon"][0]["dp_epsilon_ae"]
        print(f"{e['sigma']:>6} | {e['detection_metrics']['F1']:>7} | "
              f"{e['detection_metrics']['FPR']:>7} | {eps0:>28.4f}")


if __name__ == "__main__":
    main()
