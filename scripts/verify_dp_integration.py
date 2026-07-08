"""
scripts/verify_dp_integration.py — Opacus DP-SGD Integration Verification
==========================================================================

Verifies that Differential Privacy (Opacus DP-SGD) is correctly integrated
into the AURA federated learning client without breaking the training loop.

Checklist verified:
  [1] Opacus imports successfully
  [2] FlowAutoencoder passes ModuleValidator (no incompatible layers)
  [3] PrivacyEngine attaches to AE optimizer without errors
  [4] Training loop completes with DP-wrapped model
  [5] Epsilon is reported as a finite positive number
  [6] Model output shapes are preserved (47→16→47)
  [7] DP-trained model produces different weights than init (training actually ran)

Run:
    python3 scripts/verify_dp_integration.py

PASS criterion: all 7 checks pass, epsilon is finite and positive.
"""

import sys
import logging
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PASS_COUNT = 0
FAIL_COUNT = 0

def check(name: str, condition: bool, detail: str = ""):
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"  ✅ {name}")
    else:
        FAIL_COUNT += 1
        print(f"  ❌ {name}")
    if detail:
        print(f"     {detail}")


def main():
    global PASS_COUNT, FAIL_COUNT
    print("=" * 65)
    print("  AURA — Opacus DP-SGD Integration Verification")
    print("=" * 65)
    print()

    # ── Check 1: Opacus imports ──────────────────────────────────────────
    print("[1/7] Opacus imports...")
    try:
        from opacus import PrivacyEngine
        from opacus.validators import ModuleValidator
        check("Opacus imports", True, f"PrivacyEngine v{__import__('opacus').__version__}")
    except ImportError as e:
        check("Opacus imports", False, str(e))
        print("\n⛔ FATAL: Opacus not installed. Cannot continue.")
        return

    # ── Check 2: ModuleValidator ─────────────────────────────────────────
    print("\n[2/7] FlowAutoencoder Opacus compatibility...")
    from aura.models import FlowAutoencoder
    ae = FlowAutoencoder()
    errors = ModuleValidator.validate(ae, strict=False)
    check("ModuleValidator.validate() — no errors", len(errors) == 0,
          f"{len(errors)} errors" if errors else "AE compatible as-is")

    # ── Check 3: PrivacyEngine attaches ──────────────────────────────────
    print("\n[3/7] PrivacyEngine attachment...")
    import config as cfg

    n_samples = 200
    feature_dim = cfg.FEATURE_DIM
    train_data = torch.randn(n_samples, feature_dim)
    dataset = torch.utils.data.TensorDataset(train_data)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.AE_BATCH_SIZE, shuffle=True
    )

    ae_dp = FlowAutoencoder()
    optimizer = torch.optim.Adam(ae_dp.parameters(), lr=cfg.AE_LEARNING_RATE)

    # Save initial weights for comparison
    init_weights = [p.clone().detach() for p in ae_dp.parameters()]

    try:
        privacy_engine = PrivacyEngine()
        ae_dp, optimizer, loader = privacy_engine.make_private(
            module=ae_dp,
            optimizer=optimizer,
            data_loader=loader,
            noise_multiplier=cfg.DP_NOISE_MULTIPLIER,
            max_grad_norm=cfg.DP_MAX_GRAD_NORM,
        )
        check("PrivacyEngine.make_private()", True,
              f"σ={cfg.DP_NOISE_MULTIPLIER}, C={cfg.DP_MAX_GRAD_NORM}")
    except Exception as e:
        check("PrivacyEngine.make_private()", False, str(e))
        print("\n⛔ FATAL: PrivacyEngine attachment failed. Cannot continue.")
        return

    # ── Check 4: Training loop ───────────────────────────────────────────
    print("\n[4/7] DP-SGD training loop (3 epochs)...")
    try:
        n_epochs = 3
        last_loss = 0.0
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                optimizer.zero_grad()
                x_hat, z = ae_dp(batch)
                loss = F.mse_loss(x_hat, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            last_loss = epoch_loss / max(len(loader), 1)

        check("Training loop completed", True,
              f"final_loss={last_loss:.6f} over {n_epochs} epochs")
    except Exception as e:
        check("Training loop completed", False, str(e))
        print("\n⛔ FATAL: Training loop failed. Cannot continue.")
        return

    # ── Check 5: Epsilon reporting ───────────────────────────────────────
    print("\n[5/7] Epsilon reporting...")
    try:
        epsilon = privacy_engine.get_epsilon(delta=cfg.DP_DELTA)
        is_valid = (
            isinstance(epsilon, float)
            and np.isfinite(epsilon)
            and epsilon > 0
        )
        check("Epsilon is finite positive", is_valid,
              f"ε={epsilon:.4f}, δ={cfg.DP_DELTA}")
    except Exception as e:
        check("Epsilon is finite positive", False, str(e))
        epsilon = None

    # ── Check 6: Output shapes preserved ─────────────────────────────────
    print("\n[6/7] Model output shape preservation...")
    try:
        test_input = torch.randn(8, feature_dim)
        x_hat, z = ae_dp(test_input)
        shapes_ok = (
            x_hat.shape == (8, feature_dim)
            and z.shape == (8, cfg.LATENT_DIM)
        )
        check("Output shapes preserved", shapes_ok,
              f"input={test_input.shape} → latent={z.shape} → recon={x_hat.shape}")
    except Exception as e:
        check("Output shapes preserved", False, str(e))

    # ── Check 7: Weights actually changed ────────────────────────────────
    print("\n[7/7] Weights changed during training...")
    # Get the underlying module's parameters (unwrap GradSampleModule)
    unwrapped = ae_dp._module if hasattr(ae_dp, '_module') else ae_dp
    current_weights = [p.clone().detach() for p in unwrapped.parameters()]
    any_changed = any(
        not torch.equal(init_w, curr_w)
        for init_w, curr_w in zip(init_weights, current_weights)
    )
    check("Weights changed from init", any_changed,
          "Training actually modified the model")

    # ── Summary ──────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    total = PASS_COUNT + FAIL_COUNT
    if FAIL_COUNT == 0:
        print(f"  ALL {total} CHECKS PASSED ✅")
        print(f"  Opacus DP-SGD integration verified successfully.")
        if epsilon is not None:
            print(f"  Reported privacy budget: ε={epsilon:.4f}, δ={cfg.DP_DELTA}")
            print(f"  Target ε: {cfg.DP_TARGET_EPSILON}")
    else:
        print(f"  {FAIL_COUNT}/{total} CHECKS FAILED ❌")
        print(f"  DP integration has issues — see above for details.")
    print("=" * 65)

    return FAIL_COUNT == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
