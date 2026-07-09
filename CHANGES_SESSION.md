# AURA — Full Session Code Changes Report
## Branch: `Watts-Strogatz-tests`
## Commits: `d1a9155` → `cb05b57`

---

## Table of Contents
1. [Phase 1 — Refactoring & Feature Work](#phase-1)
2. [Phase 2 — PR #35 Merge Conflict Resolution](#phase-2)
3. [Hardcode Audit — Final Status](#hardcode-audit)
4. [Commit Summary](#commit-summary)

---

<a name="phase-1"></a>
## Phase 1 — Refactoring & Feature Work (4 commits)

---

### 1.1 Commit `d1a9155` — Centralize Split Fractions into `config.py`

#### Why it was required
Before this commit, the train/test split fraction (`0.20`) and calibration fraction
(`0.10`) were **hardcoded as numeric literals in five separate scripts**. This meant:

- Changing the split required editing 5 files by hand
- No guarantee all files used the same number
- `aura/split_manager.py` used a relative `Path(__file__)` expression instead of the
  `config.py`-controlled `BASE_DIR`, making the split save path fragile

#### Files changed and what changed

**[`config.py`](file:///c:/Users/SURAJ/Desktop/ISFCR-1/AURA/config.py)**
```python
# ADDED
SPLITS_DIR           = BASE_DIR / "splits"   # canonical save directory
TEST_SPLIT_FRACTION  = 0.20                  # single source of truth
CALIB_SPLIT_FRACTION = 0.10                  # single source of truth
```
**Why:** All scripts importing `config.py` now reference the same value. Change one constant, the entire pipeline updates.

---

**[`aura/split_manager.py`](file:///c:/Users/SURAJ/Desktop/ISFCR-1/AURA/aura/split_manager.py)**
```python
# BEFORE
_SPLIT_DIR  = Path(__file__).resolve().parent.parent / "splits"   # fragile
def get_canonical_split(all_windows, test_fraction: float = 0.20, calib_fraction: float = 0.10):

# AFTER
_SPLIT_FILE = cfg.SPLITS_DIR / "canonical_split.npz"             # config-driven
def get_canonical_split(all_windows, test_fraction: float = cfg.TEST_SPLIT_FRACTION,
                        calib_fraction: float = cfg.CALIB_SPLIT_FRACTION):
```
**Impact:** The split save location and default fractions now come from the same authority as every other script. Previously, a researcher who moved the project directory could silently save splits to a different path.

---

**[`train.py`](file:///c:/Users/SURAJ/Desktop/ISFCR-1/AURA/train.py)**
```python
# BEFORE
n_val = int(len(all_benign) * 0.20)

# AFTER
n_val = int(len(all_benign) * cfg.TEST_SPLIT_FRACTION)
```
**Why it matters:** The val split inside `train.py` now exactly mirrors the canonical test fraction. Previously, if `TEST_SPLIT_FRACTION` was changed during an experiment, the AE val boundary would silently diverge from the test boundary.

---

**[`calibrate_thresholds.py`](file:///c:/Users/SURAJ/Desktop/ISFCR-1/AURA/calibrate_thresholds.py)**
```python
# BEFORE
_, train_windows, _ = get_canonical_split(all_windows, test_fraction=0.20)

# AFTER
_, train_windows, _ = get_canonical_split(all_windows, test_fraction=cfg.TEST_SPLIT_FRACTION)
```
**Impact:** Threshold calibration and ablation now guaranteed to use the same train/test boundary. Previously, if one used 0.20 and the other silently had a different default, thresholds would be calibrated on data that the evaluation also scored.

---

**[`scripts/benchmark_ablation.py`](file:///c:/Users/SURAJ/Desktop/ISFCR-1/AURA/scripts/benchmark_ablation.py)**
**[`scripts/ablation_sweep.py`](file:///c:/Users/SURAJ/Desktop/ISFCR-1/AURA/scripts/ablation_sweep.py)**
**[`aura_attacks/ablation_sweep.py`](file:///c:/Users/SURAJ/Desktop/ISFCR-1/AURA/aura_attacks/ablation_sweep.py)**
```python
# BEFORE (argparse default)
parser.add_argument("--test-fraction", type=float, default=0.20)

# AFTER
parser.add_argument("--test-fraction", type=float, default=cfg.TEST_SPLIT_FRACTION)
```
**Impact:** CLI users who run the benchmark without specifying `--test-fraction` now get the config value, not a hardcoded `0.20` that could diverge from the actual split stored in `splits/canonical_split.npz`.

---

### 1.2 Commit `cce88a4` — Section 3.5 HITL Response Engine Benchmark

#### Why it was required
The paper's Section 3.5 describes the Three-Tier HITL Response Engine as a **contribution** but had:
- No latency numbers → cannot claim "real-time" operation
- No False Escalation Rate → cannot claim the response layer doesn't amplify detector FPR
- No operator workload metric → cannot claim the design is operationally viable

Without these, the response engine is a design description, not a scientific result.

#### Files changed and what changed

**[`config.py`](file:///c:/Users/SURAJ/Desktop/ISFCR-1/AURA/config.py)** — 6 new constants added
```python
# Every number from Section 3.5 table, now a named constant
HITL_LOW_TO_MEDIUM_THRESHOLD  = 3    # "3 LOWs → MEDIUM"
HITL_MEDIUM_TO_HIGH_THRESHOLD = 3    # "3 MEDIUMs → HIGH"
HITL_LOW_TO_HIGH_THRESHOLD    = 5    # "5 LOWs → HIGH"
HITL_TIMEOUT_SEC              = 30   # DEGRADED tier trigger
HITL_APPROVAL_RATE            = 0.85 # simulated operator (benchmark only)
RESPONSE_DEDUP_WINDOW_SEC     = 30   # was hardcoded 30 in response_engine.py
```
**Why:** The Section 3.5 table values must not appear as magic numbers anywhere in code. Changing them in `config.py` now propagates to both the live inference engine and the benchmark.

---

**[`aura/response_engine.py`](file:///c:/Users/SURAJ/Desktop/ISFCR-1/AURA/aura/response_engine.py)** — 2 changes
```python
# BEFORE (hardcoded magic number)
self._dedup_window_sec = 30

# AFTER
self._dedup_window_sec = cfg.RESPONSE_DEDUP_WINDOW_SEC

# NEW METHOD ADDED
def act_with_latency(self, event: AnomalyEvent) -> tuple:
    """Returns (records, latency_ms) using monotonic clock."""
    t0 = time.monotonic()
    records = self.act(event)
    return records, (time.monotonic() - t0) * 1000.0
```
**Why monotonic clock:** `time.time()` can jump backward (NTP sync). `time.monotonic()` guarantees non-decreasing values — essential for accurate latency measurement in the benchmark.

---

**[`scripts/benchmark_hitl_response.py`](file:///c:/Users/SURAJ/Desktop/ISFCR-1/AURA/scripts/benchmark_hitl_response.py)** — NEW file (~400 lines)

| Class | Purpose |
|-------|---------|
| `HITLSimulator` | Replaces blocking `input()` with seeded `random.Random`. Same seed = identical benchmark every run. |
| `BenchmarkResponseEngine` | Subclass suppressing disk I/O during benchmark; intercepts HIGH events for simulated HITL. |
| `classify_response_tier()` | Maps engine output to `NONE/LOW/MEDIUM/HIGH/DEGRADED`. |
| `ResponseBenchmark` | Orchestrator: loads only `test_windows` from split_manager, never train data. |

**Metrics produced:**

| Metric | Pass Criterion |
|--------|---------------|
| Escalation Latency P95 | < 100 ms |
| False Escalation Rate | < 10% |
| HITL calls/hour | < 10/hr |

**Dataset usage:** Only `test_windows` from `get_canonical_split()` — same 20% partition as `benchmark_ablation.py`. Zero training data in evaluation.

---

### 1.3 Commit `16f7843` — Fix Hardcoded HITL Thresholds in `detector.py`

#### Why it was required
The previous commit added `HITL_LOW_TO_HIGH_THRESHOLD = 5` to `config.py`, but the **live inference engine** still had the magic number `5` directly in its `if` statement. Changing the constant in config would have had zero effect on actual escalation behaviour.

**[`aura/detector.py`](file:///c:/Users/SURAJ/Desktop/ISFCR-1/AURA/aura/detector.py)**
```python
# BEFORE — magic numbers in production logic
if low_count >= 5 or medium_count >= 3:
    candidate = AlertSeverity.HIGH
elif low_count >= 3:
    candidate = AlertSeverity.MEDIUM

# AFTER — config-driven
if (
    low_count    >= cfg.HITL_LOW_TO_HIGH_THRESHOLD       # was 5
    or medium_count >= cfg.HITL_MEDIUM_TO_HIGH_THRESHOLD # was 3
):
    candidate = AlertSeverity.HIGH
elif low_count >= cfg.HITL_LOW_TO_MEDIUM_THRESHOLD:      # was 3
    candidate = AlertSeverity.MEDIUM
```
**Impact:** The same threshold constants now control:
1. Live inference engine (`detector.py`)
2. Benchmark evaluation (`benchmark_hitl_response.py`)
3. Docstring documentation in `config.py`

---

---

<a name="phase-2"></a>
## Phase 2 — PR #35 Merge Conflict Resolution (Commit `cb05b57`)

**5 files had merge conflicts.** Each decision is documented below with the rationale.

---

### 2.1 `.gitignore`

| Side | Content |
|------|---------|
| HEAD (ours) | Nothing |
| main | `scratch/` |

**Resolution:** Added `scratch/` from main — purely additive, no logical conflict.

**Impact:** Scratch/temp files will now be ignored by git. No code change.

---

### 2.2 `aura_attacks/ablation_sweep.py` and `scripts/ablation_sweep.py`

| Side | Content |
|------|---------|
| HEAD (ours) | `default=cfg.TEST_SPLIT_FRACTION` |
| main | `default=0.20` (hardcoded) |

**Resolution:** Kept our `cfg.TEST_SPLIT_FRACTION`.

**Why:** This was the core refactoring from Commit `d1a9155`. Accepting main's `0.20` would have immediately undone that work and reintroduced the hardcoded magic number into the CLI default. A researcher running the sweep with a different `TEST_SPLIT_FRACTION` in config would get unexpected results.

---

### 2.3 `scripts/benchmark_ablation.py`

| Side | Content |
|------|---------|
| HEAD (ours) | `get_canonical_split()` — uses split_manager's registered partition |
| main | Manual arithmetic slice + hardcoded `0.10` calib fraction |

```python
# main's version (REJECTED):
test_start = int(total * (1.0 - test_fraction))
train_windows = all_windows[:test_start]
test_windows = all_windows[test_start:]
n_calib = max(5, int(len(train_windows) * 0.10))   # hardcoded 0.10

# Our version (KEPT):
calibration_windows, train_windows, test_windows = get_canonical_split(
    all_windows, test_fraction=test_fraction
)
```

**Why:** The `get_canonical_split()` function:
- Saves split indices to `splits/canonical_split.npz` so every script uses the **identical partition** on every run
- Uses `cfg.CALIB_SPLIT_FRACTION` instead of `0.10`
- Is stratified (preserves attack ratio in both splits)

Main's manual slice was not reproducible (no persisted indices) and introduced a hardcoded `0.10`.

---

### 2.4 `train.py` — Two Conflict Blocks

#### Block 1: Where `attack_graphs_for_gnn` is built

| Side | Content |
|------|---------|
| HEAD (ours) | Built `attack_graphs_for_gnn` during the `all_windows` streaming loop (before the split) |
| main | Clean stream loop with no GNN collection; GNN collection moved to after split |

**Resolution:** Adopted main's cleaner stream loop. The `attack_graphs_for_gnn` build is now done *after* `train_windows` is known.

**Why main's design is better:**
- Our version accidentally collected test-partition graphs for GNN training (the cap of 100 happened to exclude them but was not a guarantee)
- Main's version explicitly iterates `train_windows` — making it impossible for test data to appear in GNN training, no matter how the split changes

#### Block 2: Val split fraction and GNN cap

| Side | Content |
|------|---------|
| HEAD (ours) | `cfg.TEST_SPLIT_FRACTION` |
| main | Hardcoded `0.20` + `attack_graphs_for_gnn` build from `train_windows` |

**Resolution:** Combined: took main's `attack_graphs_for_gnn` placement (correct design) + kept our `cfg.TEST_SPLIT_FRACTION` (no magic numbers).

**Bonus fix:** The hardcoded caps `100` and `200` in Phase 2 and Phase 4 were also eliminated:
```python
# config.py — ADDED
GNN_ATTACK_GRAPH_CAP = 100   # Phase 2 and Phase 4 cap, config-driven

# train.py Phase 2 — BEFORE
if len(attack_graphs_for_gnn) >= 100:

# train.py Phase 2 — AFTER
if len(attack_graphs_for_gnn) >= cfg.GNN_ATTACK_GRAPH_CAP:

# train.py Phase 4 — BEFORE
if len(attack_graphs_for_gnn) >= 200:

# train.py Phase 4 — AFTER
if len(attack_graphs_for_gnn) >= cfg.GNN_ATTACK_GRAPH_CAP * 2:
```

---

### 2.5 `scripts/benchmark_byzantine.py` — Most Complex Conflict

The file had **multiple overlapping issues** from the merge:

#### Issue A — Corrupt duplicate `run_experiment` stub (lines 303-341)
Main's DualFL branch added a new `run_experiment` with `mode` and `num_rounds` parameters, but the git merge left a broken stub that referenced `seed` without it being in the parameter list. This was a merge artifact — an incomplete function body.

**Resolution:** Removed the corrupt stub entirely.

#### Issue B — Real `run_experiment` missing `mode` and `num_rounds`
The complete `run_experiment` function (from our branch) had only `seed` but not the new parameters that main added.

**Resolution:** Merged both: the real function now accepts all four optional params:
```python
def run_experiment(
    strategy_name:   str,
    num_clients:     int,
    byzantine_ratio: float,
    rare_client:     bool = False,
    mode:            str = "single_channel",         # NEW from main
    num_rounds:      int = cfg.FL_NUM_ROUNDS,        # NEW — config-driven
    seed:            int = None,
):
```
**Why `cfg.FL_NUM_ROUNDS` not a hardcoded `10`:** Main had `default=10` in argparse. FL_NUM_ROUNDS in config is `3`. The config constant is the source of truth.

#### Issue C — `main()` conflict block 1 (header + Experiment 1)
Our side had: duplicate header prints + Experiment 1 loop (itself duplicated).
Main's side had: argparse setup.

**Resolution:** Combined both — argparse CLI from main, deduped header, Experiment 1 loop from ours. All `run_experiment` calls pass `args.mode`, `args.rounds`, `args.seed`:
```python
# argparse defaults are config-driven, not hardcoded:
parser.add_argument("--rounds", type=int, default=cfg.FL_NUM_ROUNDS)
```

#### Issue D — `main()` conflict block 2 (Experiment 2 vs DC-FLTrust)
Our side: Experiment 2 — Rare Client Preservation (Krum vs FLTrust).
Main's side: DC-FLTrust Deception Experiment with 5 clients / 20% Byzantine.

**Resolution:** Both experiments kept. Main's DC-FLTrust added as **Experiment 3**:
```
Experiment 1: Byzantine Ratio Sweep (FedAvg vs Krum vs FLTrust)
Experiment 2: Rare Client Preservation (Krum vs FLTrust)     ← from ours
Experiment 3: DC-FLTrust Deception Experiment                ← from main
```

#### Issue E — Duplicate code artifacts
The merge left duplicate `roles=` assignments, duplicate `clients` list+loop, and duplicate `generate_client_data` call arguments. All removed.

#### Issue F — Hardcoded `num_rounds = 2` inside `run_experiment` body
Main introduced this as a local variable, overriding the parameter. Removed so the parameter actually controls the rounds.

---

---

<a name="hardcode-audit"></a>
## Hardcode Audit — Final Status

### Files scanned

| File | Status | Notes |
|------|--------|-------|
| `config.py` | ✅ CLEAN | `0.20`, `0.10` are **definitions** — correct |
| `train.py` | ✅ CLEAN | All caps use `cfg.GNN_ATTACK_GRAPH_CAP` |
| `calibrate_thresholds.py` | ✅ CLEAN | `cfg.TEST_SPLIT_FRACTION` |
| `aura/split_manager.py` | ✅ CLEAN | `cfg.SPLITS_DIR`, `cfg.TEST/CALIB_SPLIT_FRACTION` |
| `aura/detector.py` | ✅ CLEAN | `cfg.HITL_LOW/MEDIUM/HIGH_TO_*` |
| `aura/response_engine.py` | ✅ CLEAN | `cfg.RESPONSE_DEDUP_WINDOW_SEC` |
| `scripts/benchmark_ablation.py` | ✅ CLEAN | `get_canonical_split()`, `cfg.TEST_SPLIT_FRACTION` |
| `scripts/ablation_sweep.py` | ✅ CLEAN | `cfg.TEST_SPLIT_FRACTION` |
| `scripts/benchmark_hitl_response.py` | ✅ CLEAN | All params from `cfg.*` |
| `scripts/benchmark_byzantine.py` | ✅ CLEAN | `cfg.FL_NUM_ROUNDS`, no magic 2/10/5/0.2 |
| `aura_attacks/ablation_sweep.py` | ✅ CLEAN | `cfg.TEST_SPLIT_FRACTION` |

### All constants now centralised in `config.py`

| Category | Constants |
|----------|-----------|
| Paths | `BASE_DIR`, `MODELS_DIR`, `LOGS_DIR`, `SPLITS_DIR`, `CONTRACTS_DIR` |
| Data split | `TEST_SPLIT_FRACTION`, `CALIB_SPLIT_FRACTION`, `DATA_LOAD_FRACTION` |
| GNN training | `GNN_ATTACK_GRAPH_CAP`, `GNN_EPOCHS`, `GNN_LEARNING_RATE` |
| HITL escalation | `HITL_LOW_TO_MEDIUM_THRESHOLD`, `HITL_LOW_TO_HIGH_THRESHOLD`, `HITL_MEDIUM_TO_HIGH_THRESHOLD`, `HITL_TIMEOUT_SEC` |
| HITL benchmark | `HITL_APPROVAL_RATE` |
| Response engine | `RESPONSE_DEDUP_WINDOW_SEC`, `CONFIDENCE_LOW_THRESHOLD`, `CONFIDENCE_MED_THRESHOLD` |
| EMA detector | `EMA_ALPHA`, `EMA_SIGMA_MULTIPLIER`, `EMA_WARMUP_BATCHES`, `K_CONSECUTIVE_READINGS`, `TEMPORAL_WINDOW_SECONDS` |
| FL | `FL_NUM_ROUNDS`, `FL_MIN_CLIENTS`, `FL_LOCAL_EPOCHS`, `FLTRUST_SERVER_LR`, `FLTRUST_MIN_TRUST_SCORE` |

### Acceptable numeric literals (not policy values)
| Location | Value | Why acceptable |
|----------|-------|----------------|
| `np.percentile(..., 50/95/99)` | `50, 95, 99` | Standard statistical percentile values |
| `neural layer widths` | e.g. `64, 128` | Architecture constants, versioned via model bundle |
| `GNN_ATTACK_GRAPH_CAP * 2` | `2` | Multiplier, not an independent threshold |

---

<a name="commit-summary"></a>
## Commit Summary

| Commit | Type | Description |
|--------|------|-------------|
| `d1a9155` | refactor | Centralise split fractions and paths into config.py |
| `cce88a4` | feat | Section 3.5 HITL Response Engine Benchmark (new script) |
| `16f7843` | fix | Replace hardcoded HITL thresholds in detector.py |
| `c466534` | docs | CHANGES_SESSION.md — hardcode audit doc |
| `cb05b57` | fix(merge) | Resolve all PR #35 merge conflicts |

**Total files changed across session:** 12
**New files created:** `scripts/benchmark_hitl_response.py`, `CHANGES_SESSION.md`
**Hardcoded magic numbers eliminated:** `0.20` ×5, `0.10` ×2, `30` (dedup), `3/5/3` (HITL thresholds), `100` ×1, `200` ×1, `2` (FL rounds) ×1
