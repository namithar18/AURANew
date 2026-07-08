# AURA — Project Context Document
**Autonomous Unified Resilience Architecture**
Byzantine-Robust Federated Dual-Layer Network Intrusion Detection and Response

**Document version:** Post-Audit Baseline
**GitHub:** SenseiSuraj24/TRINETRA
**Roadmap version:** v3.0
**Dataset:** NF-UNSW-NB15-v3 (switched from CICIDS2017 — CICIDS2017 lacks real IPs needed for graph construction)

---

## 1. What AURA Is

AURA is a research-grade federated learning Network Detection and Response (NDR) system. It combines:

- **Layer 1:** Flow Autoencoder (AE) — statistical anomaly detection on NetFlow features
- **Layer 2:** GraphSAGE (AuraSTGNN) — topological validation on network communication graphs
- **Layer 3:** EMA + Temporal Accumulator — persistence detection for low-and-slow attacks
- **Layer 4:** FLTrust — Byzantine-robust federated aggregation
- **Layer 5:** Three-tier HITL Response Engine — LOG / THROTTLE / HUMAN-AUTHORIZED ISOLATION

The central research question: does combining unsupervised statistical anomaly detection, graph-topological validation, adaptive temporal escalation, and Byzantine-robust federated aggregation produce superior detection performance and operational robustness compared to any of these approaches independently?

---

## 2. Architecture — Actual Implemented State

### 2.1 Flow Autoencoder (AE)
- **File:** `aura/models.py` — class `FlowAutoencoder`
- **Architecture:** `47 → 32 → 24 → 16 → 24 → 32 → 47` (symmetric bottleneck)
- **Feature dimension:** 47 (not 78 — all doc references to 78 have been updated)
- **Training:** Benign-only traffic, IsolationForest sanitisation applied before fitting
- **Loss function:** Pure MSE (`F.mse_loss`) — contrastive term removed (see §4.1)
- **Anomaly score:** Per-sample reconstruction error
- **Threshold:** EMA-adjusted UCL — sigma multiplier and K values set empirically (see §5)
- **Status:** Fully implemented, contrastive term cleanly removed

### 2.2 GraphSAGE (AuraSTGNN)
- **File:** `aura/models.py` — classes `SAGEConv`, `AuraSTGNN`
- **Architecture:** Input 47 → SAGEConv → 32 → 16 → 1 (sigmoid output)
- **Message passing:** Manual `torch.scatter_add_` — no torch_geometric dependency
- **Aggregation:** Sparse mean aggregation in `sparse_mean_aggregate()`
- **Topology:** Synthetic 20-node network (`NUM_SYNTHETIC_NODES = 20` in config.py)
- **Scope:** GraphSAGE is strictly local per organization — only AE is federated globally
- **Status:** Fully implemented

### 2.3 EMA + Temporal Accumulator
- **File:** `aura/detector.py`
- **Window:** 5-minute sliding temporal accumulator (`TEMPORAL_WINDOW_SECONDS = 300`)
- **Escalation logic:**
  - LOW → MEDIUM: 3 LOW alerts within window
  - LOW/MEDIUM → HIGH: 5 LOWs or 3 MEDIUMs within window
- **K_CONSECUTIVE_READINGS:** Set empirically to 1 (see §5 — K=5 collapses recall on short-burst attacks)
- **EMA_SIGMA_MULTIPLIER:** Set empirically to 1.0 (see §5)
- **Status:** Fully implemented

### 2.4 FLTrust Aggregation
- **File:** `aura/fl_server.py` — function `fltrust_aggregate`
- **Mechanism:** Cosine trust scoring vs. server root gradient, ReLU to discard negative similarities, normalized weighted aggregation
- **Root dataset:** Server holds clean representative benign traffic — FLTrust correctness conditioned on this
- **Byzantine injection:** `aura_attacks/mitm_simulation.py` — **delegated to separate team member, do not touch**
- **Opacus DP:** Absent — planned for Tier 2.1, not yet implemented
- **Status:** Fully implemented, Byzantine adversarial validation delegated

### 2.5 Response Engine
- **Files:** `aura/response_engine.py`, `aura/policy_engine.py`
- **Tiers:** LOG (LOW) / THROTTLE (MEDIUM) / HITL isolation (HIGH)
- **HITL gate:** Real stdin human authorization gate implemented
- **Windows limitation:** Actual bash enforcement (`iptables`/`tc`) is mocked on Windows — returns `[SIMULATED]` string. Real enforcement requires Linux deployment.
- **Status:** Partially implemented — logic correct, network enforcement platform-dependent

### 2.6 Audit Trail
- **File:** `aura/blockchain.py`
- **Current state:** Ganache/Web3 with SHA-256 hashing — ABI methods are `registerModel` and `verifyHash` (not `storeRecord`/`getRecord` as some older docs say)
- **Planned:** Merkle tree replacement — not yet implemented
- **Fallback:** Writes to `.jsonl` log file if Ganache not running
- **Status:** Partial — Merkle tree migration is a pre-submission requirement (Tier 3)

---

## 3. Bugs Found and Fixed (Full Audit Trail)

### 3.1 CRITICAL — Gradient Inversion Dimensional Mismatch
**File:** `aura_attacks/mitm_simulation.py` — `invert_gradient()`
**Bug:** `true_grad` computed from batch of 64 samples, but `dummy` initialized as `torch.randn(1, FEATURE_DIM)` — single-sample gradient optimized against 64-sample averaged gradient. Mathematically cannot converge. All previous "poor reconstruction fidelity" results were artifacts of a broken attack, not evidence of privacy.
**Fix:** Change dummy initialization to match batch size: `torch.randn(N, FEATURE_DIM)` where N matches the batch used to compute `true_grad`.
**Status:** **Delegated to team member handling mitm_simulation.py** — must be verified with positive control (MSE < 0.1 on unprotected model) before any privacy claim is made in the paper.
**Impact:** No privacy claim (H4) is valid until positive control passes.

### 3.2 CRITICAL — MinMaxScaler Data Leakage
**File:** `aura/data_loader.py` — `fit_scaler()` method
**Bug:** `scaler.fit(X_clean)` called on all benign rows including future test set rows. Global min/max of test distribution leaked into training normalization. All out-of-distribution evaluation numbers were optimistic.
**Fix:** Restructured `fit_scaler` to accept `train_indices` parameter. Scaler now fits only on training rows. Fallback to chronological 80/20 if no indices provided.
**Verification:** `scripts/verify_no_leakage.py` — confirmed 1 feature has test values outside fitted range (correct — proves scaler didn't see test distribution).
**Impact:** All evaluation numbers produced before this fix are invalid. Full pipeline re-run required.

```python
# Fixed signature:
def fit_scaler(self, train_indices=None) -> MinMaxScaler:
    # ... fits only on X_clean[train_indices] or first 80% if None
```

### 3.3 CRITICAL — Train/Test Split Inconsistency
**Files:** `train.py` vs `benchmark_ablation.py`
**Bug:** Two completely different split methods on different data granularities:
- `train.py`: random shuffle on individual flow edges, capped at 50 windows
- `benchmark_ablation.py`: stratified chronological split on graph windows

These produced entirely different test sets — not even the same data granularity.

**Fix:** Created `aura/split_manager.py` — single source of truth for all splits.
- `get_canonical_split(all_windows, test_fraction=0.20)` returns consistent train/test window indices
- Persisted to `splits/canonical_split.npz`
- Both `train.py` and `benchmark_ablation.py` now import and use this

**Additional fix in train.py:** Removed 50-window cap. AE now trains on all benign flows from train windows (80% of ~39,423 windows), not just first 50.

**Verification output:**
```
Train index arrays identical    : PASS
Test  index arrays identical    : PASS
Index overlap (must be 0)       : 0
PASS — zero overlap, identical split across both calls.
```

### 3.4 HIGH — Contrastive Loss Term: Unit Mismatch + Zero Value
**File:** `aura/models.py` — `reconstruction_loss()`
**Bug (unit mismatch):** MSE term uses squared L2 distance; contrastive term used linear L2 (`torch.norm(..., p=2)`). Incompatible units caused unstable gradient scaling.
**Fix applied:** Squared the contrastive distance term: `dist = torch.sum((z - z_neg)**2, dim=1)`

**Bug (zero value / no contribution):** After fix, diagnostic showed contrastive term collapsed to 0.000000 after epoch 10 in original code. After unit fix, term remained active throughout training.

**Diagnostic result (`scripts/verify_contrastive_value.py`):**
```
ae_contrastive: Separation 9.73x,  AUROC 0.9648
ae_mse_only:    Separation 11.60x, AUROC 0.9692
AUROC difference: -0.0044
```
MSE-only was marginally better on every metric. DECISION: **Branch A — Remove contrastive term.**

**Final fix:** Contrastive term removed entirely from `reconstruction_loss()`. AE now trains on pure MSE only.

```python
# Current loss function (post-fix):
def reconstruction_loss(self, x, x_hat):
    # Contrastive term removed after diagnostic — see scripts/verify_contrastive_value.py
    # MSE-only produces superior separation (AUROC 0.9692 vs 0.9648)
    l_recon = F.mse_loss(x_hat, x)
    return l_recon
```

### 3.5 HIGH — MSE Threshold Collapse
**File:** `config.py`
**Bug:** `MSE_THRESHOLD_HIGH = 0.0432` was below the 99th percentile of normal benign traffic (P99 = 0.1293). Normal traffic was triggering HIGH alerts — three-tier response architecture was functionally collapsed.

**Root cause:** Thresholds were set before scaler leakage fix. Clean scaler produces different reconstruction error distribution.

**Fix:** Dynamic threshold calibration pipeline implemented:
- `calibrate_thresholds.py` computes P90/P99 of benign MSE distribution and writes to `calibration_results.json`
- `config.py` loads from JSON via `load_ae_thresholds()` — no manual editing required
- Fallback to hardcoded values if JSON missing

**Current calibrated values (from clean scaler run):**
```
MSE_THRESHOLD_MEDIUM = 0.0193   # P90 of benign MSE distribution
MSE_THRESHOLD_HIGH   = 0.1293   # P99 of benign MSE distribution
MSE_THRESHOLD_HIGH_CONSERVATIVE = 0.1590  # P99.5 — use if FPR is high
```

**Pipeline order (mandatory):**
```
train.py → calibrate_thresholds.py → benchmark_ablation.py → everything else
```

### 3.6 MEDIUM — Silent Exception Handlers
**Files:** `scripts/train_explainer.py`, `aura/detector.py`, `aura/fl_server.py`
**Bug:** Three silent exception handlers that swallow errors and continue with garbage state:
1. AE bundle load failure → continues with randomly initialized AE
2. Explainer failure → silently returns "Normal" for all labels
3. FL server root dataset failure → trains on synthetic Gaussian noise

**Fix:** All three converted to `raise RuntimeError(...)` with descriptive messages.

### 3.7 LOW — Config Values Mismatched from Empirically Better Values
**File:** `config.py`
**Bug:** Roadmap claimed K=5 and sigma=3σ. Actual empirically better values are K=1 and sigma=1.0 because NF-UNSW-NB15 attacks are short-burst and don't sustain across K=5 consecutive windows.

**Resolution:** Config updated to match empirically chosen values. Parameter sweep table required for paper to justify selection (see §5).

### 3.8 LOW — Feature Dimension Documentation
**Bug:** All comments, docstrings, and markdown referenced 78-feature vectors. Actual `FEATURE_DIM = 47` in config.py.
**Fix:** All occurrences updated to 47 throughout codebase.

---

## 4. Key Diagnostic Scripts (All in `scripts/`)

| Script | Purpose | Status |
|---|---|---|
| `verify_loss_units.py` | Checks l_recon vs l_cont magnitude balance across training | Complete — used to drive Branch A decision |
| `verify_contrastive_value.py` | Side-by-side AE comparison: contrastive vs MSE-only AUROC | Complete — confirms contrastive removal |
| `verify_no_leakage.py` | Confirms scaler fitted on train split only | Complete — PASS |
| `verify_split_consistency.py` | Confirms canonical split is identical across all callers | Complete — PASS, zero overlap |

---

## 5. Current Ablation Results (Pre-Final — Awaiting Full Retrain)

These numbers were produced after scaler fix and split_manager wiring, but before the 50-window cap removal in train.py. A full retrain is currently in progress. Treat these as directionally correct but not final paper numbers.

**Parameters:** `--ema-sigma 1.0 --k-consecutive 1 --ae-percentile 95`

| Mode | Precision | Recall | F1 | Macro-F1 | FPR | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| A: AE Only | 0.584570 | 0.738963 | 0.652762 | 0.810500 | 0.041983 | 0.948915 | 0.501605 |
| B: GraphSAGE Only | 0.991930 | 0.894807 | 0.940868 | 0.968195 | 0.000582 | 0.999467 | 0.992767 |
| C: AE+GraphSAGE (no EMA) | 0.990841 | 0.703379 | 0.822723 | 0.905373 | 0.000520 | 0.967177 | 0.863919 |
| D: Full (AE+GraphSAGE+EMA) | 0.991905 | 0.892120 | 0.939370 | 0.967393 | 0.000582 | 0.998581 | 0.991086 |

**Key findings from this table:**

1. **AE blind spot confirmed empirically:** B recall (0.895) vs C recall (0.703) = 19.2 percentage point gap. GraphSAGE alone detects more attacks than AE+GraphSAGE sequential. The AE gatekeeper is blocking flows that GraphSAGE would have caught — this is H1's blind spot quantified on clean data.

2. **EMA does real work:** C recall 0.703 → D recall 0.892 = 18.9 point recovery. EMA persistence detection is recovering nearly all the recall the sequential pipeline lost.

3. **FPR near-zero:** B, C, D all achieve FPR ≈ 0.0006. GraphSAGE is extremely conservative — only flags nodes with overwhelming topological evidence.

4. **Parallel fusion argument strengthened:** The B vs C gap is the empirical motivation for Tier 2.5 (parallel fusion). This gap should narrow significantly in a parallel configuration — that result would be a concrete architectural contribution.

**Parameter sweep required before final submission:** Sweep K ∈ {1,2,3,5} × sigma ∈ {1.0,1.5,2.0,3.0} on Mode D, report as supplementary table justifying K=1/sigma=1.0 selection empirically.

---

## 6. Scripts With Independent Splits — Still Need split_manager Wiring

These were identified in Task 5 of the split_manager audit. Fix in priority order:

| Script | Priority | Reason |
|---|---|---|
| `scripts/benchmark_byzantine.py` L83 | HIGH | H2 evidence — Byzantine benchmark split must match ablation |
| `aura/fl_client.py` L515 | HIGH | FL client local training partition affects federation results |
| `scripts/train_explainer.py` | MEDIUM | Secondary component but needs consistent data |
| `scripts/verify_contrastive_value.py` | LOW | Diagnostic only, decision already made |
| `scripts/verify_no_leakage.py` | DO NOT TOUCH | Intentionally standalone — wiring to split_manager would be circular |

---

## 7. Mandatory Pre-Submission Checklist (From Roadmap v3.0)

### Tier 1 — Blocks Submission
- [ ] Formal threat model section in paper (defined in roadmap §6 — needs to appear in paper text)
- [ ] Per-attack-category metrics on NF-UNSW-NB15 attack types individually
- [ ] Statistical significance testing on all metric comparisons (paired t-test or Wilcoxon, bootstrap CIs, n≥10 runs)
- [ ] Ablation study A/B/C/D — **in progress, awaiting clean retrain**
- [ ] FLTrust Byzantine benchmark vs FedAvg/Krum at 10/20/30/40% Byzantine ratios — **delegated**
- [ ] MSE threshold recalibration — **COMPLETE** (dynamic pipeline implemented)
- [ ] Merkle tree replacing Ganache — **not started**
- [ ] split_manager wiring to benchmark_byzantine.py and fl_client.py — **not started**

### Tier 2 — Required for Competitive Venues
- [ ] MIA (shadow model attack) — code exists in `aura_attacks/mia_attack.py`, positive/negative controls not yet run
- [ ] Gradient inversion (DLG) — dimensional mismatch fix **delegated**, positive control must pass before privacy claim
- [ ] Opacus DP integration — **absent**, not yet implemented
- [ ] Watts-Strogatz structural generalization experiment
- [ ] Behavioral generalization experiment (unseen node roles)
- [ ] UNSW-NB15 cross-dataset validation (train on one, test on other without retraining)
- [ ] Response engine evaluation metrics (escalation latency, false escalation rate, operator workload)
- [ ] Adversarial evasion benchmark (feature perturbation, reconstruction-min, mimicry) — 3 targeted attacks against AE
- [ ] Parameter sweep table K × sigma — **not yet run, required before locking paper numbers**

### Tier 2.5 — Parallel Fusion Experiment
- [ ] Sequential vs parallel AE+GraphSAGE fusion — **strongly motivated by current ablation (19.2pt recall gap)**. Compare F1, FPR, Recall, inference latency, memory footprint, detection rate on evasion-resistant flows.

### Tier 3 — Required Implementation Changes
- [ ] Blockchain → Merkle tree migration (see roadmap §5.4.1 for exact specification)
- [ ] FastAPI + React migration (Streamlit crashed at nationals — demo stability requirement)
- [ ] Traffic obfuscation robustness evaluation

---

## 8. Research Hypotheses (From Roadmap v3.0)

| Hypothesis | Claim | Validation Method | Current Status |
|---|---|---|---|
| H0 | Dual-layer architecture superior to components independently | Ablation study A/B/C/D | In progress — awaiting clean retrain |
| H1 | AE+GraphSAGE jointly achieves lower FPR than AE alone at equivalent recall | Ablation configurations A-E | Partially validated — FPR claim holds, recall equivalence claim weakened by blind spot |
| H2 | FLTrust preserves rare-client gradients under 10-40% Byzantine conditions | FLTrust vs FedAvg/Krum benchmark | Delegated |
| H3 | EMA detects low-and-slow attacks at higher recall than static threshold | EMA vs static threshold experiment | Supported by D vs C recall gap (18.9 points) |
| H4 | DP reduces MIA success rate to ~0.5 | MIA + gradient inversion + DP sweep | Blocked pending gradient inversion fix |

---

## 9. Novelty Positioning

**SecureTrust-FL (Alshammari et al., Sci Rep, June 2026) — closest overlapping paper:**
Published 19 June 2026. Overlaps on: trust-weighted FL, DP noise sweep, FGSM eval, multi-dataset NIDS. Does NOT overlap on: dual-layer AE+GraphSAGE, actual Byzantine injection testing, temporal persistence (EMA), MIA/gradient inversion privacy evaluation, parallel vs sequential fusion question.

**What this means for the paper:** The positioning sentence "first to combine trust-aware FL, DP, and adversarial eval for NIDS" is no longer available. Do not use it. Frame contributions through H0-H4 empirical structure — what your ablation study proves that theirs doesn't attempt. Their Byzantine robustness is future work in their own limitations section. Yours (H2) is a claimed and tested contribution.

**Do not claim:** "No prior system combines all these properties."
**Do claim:** "We empirically demonstrate via ablation that [specific quantified improvement], and validate Byzantine robustness against actual adversarial clients under [specific injection conditions] — neither of which SecureTrust-FL evaluates."

---

## 10. Pipeline Execution Order

Every time you run experiments, follow this exact order. Running out of order produces inconsistent results.

```
1. python train.py
      ↓ (produces autoencoder_best.pth, gnn_best.pth)
2. python calibrate_thresholds.py
      ↓ (writes calibration_results.json — thresholds auto-loaded by config.py)
3. python scripts/benchmark_ablation.py --ema-sigma 1.0 --k-consecutive 1 --ae-percentile 95
      ↓ (primary H0/H1 evidence)
4. python scripts/benchmark_byzantine.py
      ↓ (H2 evidence — wire to split_manager first)
5. python scripts/train_explainer.py
      ↓ (secondary component)
6. FL execution: fl_server.py + fl_client.py
      ↓ (federation results)
```

---

## 11. What Is Delegated / Do Not Touch

- **`aura_attacks/mitm_simulation.py`** — Byzantine injection and gradient inversion fix — delegated to separate team member. The dimensional mismatch fix (dummy batch size must match true_grad batch size) must be verified with a positive control achieving MSE < 0.1 on an unprotected model before any privacy claim is made. Do not run or modify this file.

---

## 12. Known Limitations (Must Appear in Paper)

Per roadmap §8 — reviewers who find unlisted limitations view it as dishonesty; reviewers who find honestly listed ones view it as rigor.

- **AE gatekeeper blind spot:** Sequential pipeline — if AE threshold not exceeded, GraphSAGE never evaluates the flow. Quantified: 19.2 recall points lost (B vs C). Mitigation: Tier 2.5 parallel fusion.
- **Synthetic network topology:** GraphSAGE trained on 20-node synthetic topology. Structural memorization risk unresolved until Watts-Strogatz (Tier 2.2).
- **No deep packet inspection:** Flow-based architecture. Payload-level attacks invisible. Fundamental boundary.
- **Honest majority assumption:** FLTrust fails under >50% Byzantine collusion.
- **No cryptographic aggregator privacy:** Aggregator sees individual client gradients in plaintext. DP provides statistical not cryptographic privacy.
- **Privacy claim conditioned:** H4 claim must not appear in paper until MIA positive control passes and gradient inversion dimensional mismatch is fixed and verified.
- **Windows simulation only:** Response engine network enforcement mocked on Windows. Real enforcement requires Linux.
- **DP-robustness tradeoff:** High DP noise may cause legitimate clients to receive low FLTrust trust scores — quantify in Tier 2.1.

---

## 13. File Reference Map

| Component | Primary Files |
|---|---|
| AE + GraphSAGE models | `aura/models.py` |
| Data loading + scaler | `aura/data_loader.py` |
| Detector + EMA + escalation | `aura/detector.py` |
| FLTrust aggregation | `aura/fl_server.py` |
| FL client local training | `aura/fl_client.py` |
| Response + policy engine | `aura/response_engine.py`, `aura/policy_engine.py` |
| Blockchain audit trail | `aura/blockchain.py` |
| Canonical split | `aura/split_manager.py` → `splits/canonical_split.npz` |
| Dynamic thresholds | `calibrate_thresholds.py` → `calibration_results.json` → `config.py` |
| Ablation study | `scripts/benchmark_ablation.py` |
| Byzantine benchmark | `scripts/benchmark_byzantine.py` |
| MIA attack | `aura_attacks/mia_attack.py` |
| Byzantine injection | `aura_attacks/mitm_simulation.py` (**delegated**) |
| Attack injector | `aura/attack_injector.py` |
| Diagnostic scripts | `scripts/verify_*.py` |
