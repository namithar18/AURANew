# Forensic Audit: Channel 1 (AE) Collapse

I have thoroughly investigated the Channel 1 generation and comparison pipelines. I did not modify any code. Here are the answers to your forensic questions:

### 1. Does the server compute exactly the same mathematical object as the clients?
**Yes.** Both the server and the clients calculate the delta as `ae.state_dict()[k] - global_weights[k]`. The subtraction order and tensors are identical.

### 2. Is the server optimizer identical to the client optimizer?
**Yes.** Both use `torch.optim.Adam` with `lr=1e-3`. There is no weight decay, scheduling, or gradient clipping. 
*Note on DP:* While Differential Privacy is supported for standard FL, the `dc_fltrust` benchmark bypasses `fl_client.fit()` and calls `_run_local_training_dual` directly, meaning DP is strictly **OFF** for both the server and clients during this benchmark.

### 3. Does the client perform one optimization step while the server performs one optimization step?
**NO. This is a massive discrepancy.**
* **Server:** In `benchmark_byzantine.py`, the server calls `_run_local_training_dual` with `batch_size=-1`. This forces `actual_bs = len(benign_flows)`. The DataLoader yields exactly one batch, resulting in **exactly 1 Adam step**.
* **Honest Clients:** The clients call `_run_local_training_dual` without specifying `batch_size`, which defaults to `256`. For ~6,000 benign flows, the client's DataLoader yields ~24 batches, resulting in **~24 Adam steps** per epoch.

### 4. Compare the exact datasets.
* **Server Root Dataset:** ~2,000 samples. It is **100% pure benign** (filtered by `labels == 0` from the `calib_windows`). Chronologically, it is the first 2,000 flows.
* **Client Local Dataset:** ~6,000 samples. It is a non-overlapping slice of globally shuffled `train_windows` containing a natural mixture of benign and attack flows. Crucially, because `CH2_MSE_SPLIT_THRESHOLD` is currently too strict (filtering out 84% of real attacks), those attacks survive the `benign_mask` and **pollute the honest client's AE training data**.

### 5. Verify that the server and client compute the AE loss identically.
**Yes.** Both use `F.mse_loss(recon, batch)` which computes the mean squared error over the batch.

### 6. Verify delta extraction.
**Yes.** Both compute `new_weights - old_weights` across the exact same tensors in the same order.

### 7. Investigate optimizer trajectory.
I ran a scratch script simulating 1-step vs multi-step Adam trajectories in a pretrained flat basin.
While 1-step vs 24-step trajectories on identical data distributions maintained a cosine of `~0.63`, the combination of **24 Adam steps** taking place on a **polluted dataset** (due to the strict threshold) forces the honest client trajectory to wander away from the server's 1-step pure-benign reference.

### 8. Verify pretrained convergence / The Byzantine Mystery
You noted the attacker achieved a cosine of `0.43–0.49` while honest clients collapsed to `0.02`. Why?
I investigated `_run_latent_inversion_byzantine`. Unlike `_run_local_training_dual`, the attacker's AE training loop is manually implemented **without a DataLoader** (lines 66-72). The attacker computes the loss on the entire `benign_flows` tensor at once and takes **exactly 1 Adam step**. 
By accidentally hardcoding a full-batch update, the attacker perfectly mimicked the server's 1-step trajectory mechanics, resulting in high cosine similarity, while honest clients took 24 steps and collapsed!

---

## Ranked List of Causes

1. **Optimizer Step Discrepancy (Confidence: 99%)**
   * **Evidence:** The server uses `batch_size=-1` (1 Adam step), while honest clients use `batch_size=256` (~24 Adam steps). The Byzantine attacker accidentally manually implements a 1-step update, which explains why the attacker aligns (`~0.49`) and honest clients do not (`~0.02`).
   * **Affected files:** `scripts/benchmark_byzantine.py`

2. **Dataset Pollution due to Strict Threshold (Confidence: 85%)**
   * **Evidence:** The server root data is 100% pure benign. Honest clients use the AE `benign_mask` to filter their data. Because the MSE threshold is too strict, 84% of real attacks slip through the mask and pollute the honest client's AE training, pulling the 24-step trajectory further away from the server's pure reference.
   * **Affected files:** `aura/config.py` (`CH2_MSE_SPLIT_THRESHOLD`)

3. **Chronological Distribution Shift (Confidence: 30%)**
   * **Evidence:** The server takes the first 2,000 flows chronologically, while clients get slices of the globally shuffled remaining dataset.

---

## Recommended Implementation Change

The server's fundamental role in FLTrust is to compute a reference direction that accurately mimics an honest client's update. Therefore, the server **must** execute the exact same number of optimization steps as the clients.

**Change:** In `scripts/benchmark_byzantine.py` (around line 472), remove `batch_size=-1` from the server's `_run_local_training_dual` call so it defaults to `256`, matching the clients.
