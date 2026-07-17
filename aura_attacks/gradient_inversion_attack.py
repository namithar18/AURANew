"""
gradient_inversion_attack.py  —  Deep Leakage from Gradients (DLG) against AURA
=================================================================================
Target  : AURA's dual-channel Layer 1
            Channel 1 — FlowAutoencoder   47->[32,24]->16->[24,32]->47
            Channel 2 — AttackHead        16 (z) -> 8 -> 1
Reference: Zhu et al. 2019, "Deep Leakage from Gradients"
Dataset : NF-UNSW-NB15-v3

HOW TO RUN (standalone, no server needed):
    python aura_attacks/gradient_inversion_attack.py

Why this needed its own file
-----------------------------
mitm_simulation.py's docstring always referenced this script for the
gradient-inversion half of its EAVESDROP mode, but it never existed as a
standalone file — the inversion logic was one small inline helper. This
version is the real thing: multi-restart DLG optimisation, on BOTH channels,
plus a third attack the dual-channel architecture specifically opens up.

Three attacks
-------------
1. CHANNEL 1 DIRECT   — invert the AE's gradient straight to the 47-dim flow.
                        Classic DLG. Ceiling case: full raw-feature leakage.

2. CHANNEL 2 DIRECT   — invert the AttackHead's gradient to the 16-dim latent
                        z. Cannot recover raw features this way — the head
                        never sees x, only z — so on its own this is a
                        narrower leak.

3. CHAINED (channel 2 -> encoder inversion)   *** the one that matters ***
   The FlowAutoencoder's `encoder` submodule is part of the shared GLOBAL
   model — every federation participant holds its weights, gradient access
   or not. So an attacker who only compromised channel 2 doesn't have to
   stop at recovering z:
     step A : invert the AttackHead gradient -> recovered z_hat  (channel 2)
     step B : holding the PUBLIC encoder weights fixed, optimise a dummy x
               so that encoder(dummy_x) matches z_hat            (no gradient
               needed here at all — just the public forward function)
   If step B gets dummy_x close to the true flow, then channel 2 alone
   was enough to reconstruct channel-1-grade information. This is the
   question worth reporting: do the two channels leak independently, or
   does one channel's leak compound through the other via the shared model?

Standalone by design: no `aura.fl_server` / `config` imports (see
mia_attack.py / mitm_simulation.py for why). Keep architecture mirrors in
sync with aura/models.py if FlowAutoencoder / AttackHead change shape.
"""

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

AURA_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AURA_ROOT))

FEATURE_DIM  = 47
ENCODER_DIMS = [32, 24]
LATENT_DIM   = 16
DECODER_DIMS = [24, 32]




# ─────────────────────────────────────────────────────────────────────────────
# Attack 1 & 2 — direct DLG gradient matching (Zhu et al. 2019)
# ─────────────────────────────────────────────────────────────────────────────

def dlg_invert(
    model:        nn.Module,
    true_grad:    list,
    dummy_shape:  tuple,
    loss_fn,             # callable(model, dummy) -> scalar loss, matching how true_grad was produced
    steps:        int = 300,
    restarts:     int = 3,
    lr:           float = 0.1,
) -> torch.Tensor:
    """
    Generic DLG: optimise a dummy input so that the gradient it produces
    matches `true_grad` as closely as possible. Runs several random restarts
    (a real attacker doesn't stop at the first unlucky initialisation) and
    keeps whichever converged to the lowest gradient-matching loss.
    """
    best_dummy, best_loss = None, float("inf")

    for r in range(restarts):
        torch.manual_seed(1000 + r)
        dummy = torch.randn(dummy_shape, requires_grad=True)
        opt = torch.optim.Adam([dummy], lr=lr)

        for _ in range(steps):
            def closure():
                opt.zero_grad()
                model.zero_grad()
                loss = loss_fn(model, dummy)
                dummy_grad = torch.autograd.grad(loss, list(model.parameters()), create_graph=True)
                grad_diff = sum(((a - b) ** 2).sum() for a, b in zip(dummy_grad, true_grad))
                grad_diff.backward()
                return grad_diff
            opt.step(closure)

        # Recompute the final grad-matching loss once more (cheap) to rank this restart.
        model.zero_grad()
        loss = loss_fn(model, dummy)
        dummy_grad = torch.autograd.grad(loss, list(model.parameters()), create_graph=True)
        grad_diff = sum(((a - b) ** 2).sum() for a, b in zip(dummy_grad, true_grad)).item()

        if grad_diff < best_loss:
            best_loss = grad_diff
            best_dummy = dummy.detach().clone()

    return best_dummy


def invert_ae_direct(ae: nn.Module, true_grad: list, batch_size: int, steps: int, restarts: int) -> torch.Tensor:
    """Channel 1 direct: dummy x -> MSE(model(x), x) -> match true AE gradient."""
    def loss_fn(model, dummy):
        # AURA's FlowAutoencoder returns (recon, z). We optimize against recon.
        recon, _ = model(dummy)
        return nn.MSELoss()(recon, dummy)
    return dlg_invert(ae, true_grad, (batch_size, FEATURE_DIM), loss_fn, steps, restarts)


def invert_head_direct(head: nn.Module, true_grad: list, batch_size: int, steps: int, restarts: int,
                        pseudo_label_value: float = 1.0) -> torch.Tensor:
    """Channel 2 direct: dummy z -> BCE(head(z), label) -> match true head gradient."""
    def loss_fn(model, dummy):
        preds = model(dummy).squeeze(-1)
        labels = torch.full_like(preds, pseudo_label_value)
        return F.binary_cross_entropy(preds, labels)
    return dlg_invert(head, true_grad, (batch_size, LATENT_DIM), loss_fn, steps, restarts)


# ─────────────────────────────────────────────────────────────────────────────
# Attack 3 — chained: channel-2 z leak -> public encoder inversion -> x_hat
# ─────────────────────────────────────────────────────────────────────────────

def invert_encoder(ae: nn.Module, target_z: torch.Tensor, steps: int = 300, restarts: int = 3, lr: float = 0.05) -> torch.Tensor:
    """
    No gradient needed here — the encoder's weights are public (they're part
    of the shared global model everyone in the federation holds). Just
    optimise a dummy x directly against the encoder's forward output.
    """
    best_dummy, best_loss = None, float("inf")
    for p in ae.parameters():
        p.requires_grad_(False)

    for r in range(restarts):
        torch.manual_seed(2000 + r)
        dummy = torch.randn(target_z.shape[0], FEATURE_DIM, requires_grad=True)
        opt = torch.optim.Adam([dummy], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            pred_z = ae.encoder(dummy)
            loss = F.mse_loss(pred_z, target_z)
            loss.backward()
            opt.step()
        final_loss = F.mse_loss(ae.encoder(dummy), target_z).item()
        if final_loss < best_loss:
            best_loss = final_loss
            best_dummy = dummy.detach().clone()

    for p in ae.parameters():
        p.requires_grad_(True)
    return best_dummy


def chained_attack(ae: nn.Module, head: nn.Module, true_head_grad: list, batch_size: int,
                    steps: int, restarts: int) -> torch.Tensor:
    """Full chain: channel 2 gradient -> z_hat -> public encoder inversion -> x_hat."""
    z_hat = invert_head_direct(head, true_head_grad, batch_size, steps, restarts)
    x_hat = invert_encoder(ae, z_hat, steps=steps, restarts=restarts)
    return z_hat, x_hat


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def _report(label: str, recon: torch.Tensor, truth: torch.Tensor, leak_cutoff: float = 0.7):
    mse = ((recon - truth) ** 2).mean().item()
    cos = F.cosine_similarity(recon.flatten(), truth.flatten(), dim=0).item()
    verdict = "LEAKED" if cos > leak_cutoff else "protected"
    print(f"    {label:38s} MSE={mse:.4f}  cosine_sim={cos:.4f}  {verdict}")
    return cos


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-clients",  type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=1, help="flows per client gradient")
    ap.add_argument("--steps",      type=int, default=200)
    ap.add_argument("--restarts",   type=int, default=3)
    args = ap.parse_args()

    from aura.models import FlowAutoencoder, AttackHead
    import pickle

    print("\n" + "="*72)
    print("  AURA — Gradient Inversion (DLG), dual-channel")
    print("="*72)
    print(f"  {args.restarts} random restarts x {args.steps} steps per inversion, "
          f"batch_size={args.batch_size}\n")
          
    export_path = AURA_ROOT / "saved_models" / "exported_tensors_seed_0_round_1.pkl"
    if not export_path.exists():
        raise FileNotFoundError(f"Missing benchmark checkpoint: {export_path}\nRun benchmark_byzantine with --export-tensors.")
        
    print(f"  Loading canonical exported tensors from round 1...")
    with open(export_path, 'rb') as f:
        data = pickle.load(f)

    global_ae = FlowAutoencoder()
    global_ae.load_state_dict(data['global_ae_weights'])
    global_ae.eval()
    
    global_head = AttackHead()
    global_head.load_state_dict(data['global_head_weights'])
    global_head.eval()

    c_ae_deltas = data['client_ae_deltas']
    c_head_deltas = data['client_head_deltas']
    roles = data['roles']

    ch1_leaks, ch2_leaks, chain_leaks = [], [], []

    # The canonical dataset loader to get true client batch for MSE evaluation
    from aura.data_loader import load_client_partition
    org_names_all = ["hospital", "bank", "university", "isp", "retail"]

    for i in range(min(args.n_clients, len(c_ae_deltas))):
        name = org_names_all[i]
        print(f"  -- org_{name} ({roles[i]}) --")
        
        # Ground truth data batch for reference (first batch of their train partition)
        client_id = f"org_{name}_1"
        X_train, _ = load_client_partition(client_id)
        true_data = X_train[:args.batch_size].clone()
        with torch.no_grad():
            _, true_z = global_ae(true_data)

        # In DP-SGD, delta = W_{t+1} - W_t = -lr * grad. 
        # We use -delta as the gradient target for DLG matching.
        ae_grad = [-c_ae_deltas[i][k].cpu() for k, _ in global_ae.named_parameters()]
        head_grad = [-c_head_deltas[i][k].cpu() for k, _ in global_head.named_parameters()]

        x_hat_direct = invert_ae_direct(copy.deepcopy(global_ae), ae_grad,
                                        args.batch_size, args.steps, args.restarts)
        ch1_leaks.append(_report("Channel 1 direct  (grad -> x)", x_hat_direct, true_data))

        z_hat = invert_head_direct(copy.deepcopy(global_head), head_grad,
                                    args.batch_size, args.steps, args.restarts)
        ch2_leaks.append(_report("Channel 2 direct  (grad -> z)", z_hat, true_z))

        z_hat_chain, x_hat_chain = chained_attack(copy.deepcopy(global_ae), copy.deepcopy(global_head),
                                                  head_grad, args.batch_size, args.steps, args.restarts)
        chain_leaks.append(_report("Chained (ch2 grad -> z -> x)", x_hat_chain, true_data))
        print()

    def _avg(xs):
        return sum(xs) / len(xs)

    print("-"*72)
    print("  Summary (mean cosine similarity across all clients)")
    print("-"*72)
    print(f"    Channel 1 direct   (raw feature leak)         : {_avg(ch1_leaks):.4f}")
    print(f"    Channel 2 direct   (latent z leak only)       : {_avg(ch2_leaks):.4f}")
    print(f"    Chained ch2->encoder (raw feature leak, via ch2 only) : {_avg(chain_leaks):.4f}")
    print()
    gap = _avg(ch1_leaks) - _avg(chain_leaks)
    if gap < 0.15:
        print("  FINDING: the chained attack recovers raw features almost as well as")
        print("  attacking channel 1 directly. Channel 2 is NOT an independently weaker")
        print("  attack surface -- via the public encoder, it compounds into a")
        print("  channel-1-grade leak. Any privacy defence (DP-SGD noise, secure agg)")
        print("  applied to channel 1 alone would be bypassed this way; it needs to")
        print("  cover channel 2's gradient too.")
    else:
        print("  FINDING: the chained attack recovers raw features substantially worse")
        print("  than attacking channel 1 directly -- channel 2 leakage does not fully")
        print("  compound in this setup. Still nonzero: treat channel 2's gradient with")
        print("  the same protection as channel 1's, just note the gap in the writeup.")
    print("="*72 + "\n")