---
title: "Institutional Meta-Ensemble Training Implementation Plan"
design_ref: "docs/maestro/plans/2026-04-09-meta-ensemble-training-design.md"
created: "2026-04-09T14:15:00Z"
status: "draft"
total_phases: 4
estimated_files: 3
task_complexity: "complex"
---

# Institutional Meta-Ensemble Training Implementation Plan

## Plan Overview

- **Total phases**: 4
- **Agents involved**: `data_engineer`, `coder`, `performance_engineer`, `code_reviewer`
- **Estimated effort**: High. Requires a fundamental re-architecture of the labeling and training loops to support multi-stage meta-labeling and TCN integration.

## Dependency Graph

```text
[Phase 1: Meta-Labeling Targets (data_engineer)]
           |
[Phase 2: Ensemble & TCN Integration (coder)]
           |
[Phase 3: TCN Optimization (performance_engineer)]
           |
[Phase 4: Final Reliability Audit (code_reviewer)]
```

## Execution Strategy

| Stage | Phases | Execution | Agent Count | Notes |
|-------|--------|-----------|-------------|-------|
| 1     | Phase 1 | Sequential | 1 | Foundation: Targets & Data |
| 2     | Phase 2 | Sequential | 1 | Core: Refactor Trainer |
| 3     | Phase 3 | Sequential | 1 | Domain: Deep Learning Tune |
| 4     | Phase 4 | Sequential | 1 | Quality: E2E Audit |

## Phase 1: Meta-Labeling Target Generation

### Objective
Refactor `financial_engineer.py` to support two-stage labeling: (1) Trend identification and (2) Meta-labeling (profitability of that trend).

### Agent: `data_engineer`
### Parallel: No

### Files to Modify
- `core/models/training/financial_engineer.py` — Add `get_meta_labels()` method.

### Implementation Details
- Implement a primary trend identification function (using a simple SMA cross or RSI breakout).
- Implement `get_meta_labels(df, signals)`: returns 1 if the primary signal hit PT before SL using the Triple Barrier Method, 0 otherwise.
- Ensure the primary threshold is tunable to maintain sufficient sample size (>200).

### Validation
- Unit test `get_meta_labels` with a known profitable vs. losing segment.
- Verify signal density for 10-stock universe.

### Dependencies
- Blocked by: None
- Blocks: [2]

---

## Phase 2: Ensemble & TCN Integration

### Objective
Refactor `trainer.py` to implement the two-stage meta-labeling CPCV loop and integrate the TCN model.

### Agent: `coder`
### Parallel: No

### Files to Modify
- `core/models/training/trainer.py` — Complete overhaul of `train_advanced_ensemble`.

### Implementation Details
- **Stage 1**: Train/Use primary trend identifier.
- **Stage 2**: Use `CombinatorialPurgedKFold` to train the Meta-Ensemble (XGB, LGB, CAT, TCN) on the success of Stage 1 signals.
- **Non-Negative Stacking**: Implement `scipy.optimize.nnls` or a constrained meta-learner to ensure weights ≥ 0.
- **TCN Integration**: Properly format sequences `(batch, 64, 13)` for the TCN and train it in every CPCV fold.

### Validation
- `python3 -m py_compile core/models/training/trainer.py`.
- Run training for a single ticker and verify `weights.json` has only positive coefficients.

### Dependencies
- Blocked by: [1]
- Blocks: [3]

---

## Phase 3: TCN Parameter Optimization

### Objective
Optimize TCN hyperparameters and training settings to ensure convergence and respect hardware limits across the 10-stock universe.

### Agent: `performance_engineer`
### Parallel: No

### Files to Modify
- `core/models/tcn/system.py` — Tune focal loss parameters and dropout.
- `core/models/training/trainer.py` — Adjust TCN epochs/batch sizes.

### Implementation Details
- Tune `gamma` and `alpha` in `focal_loss`.
- Implement learning rate scheduling for the TCN to prevent overshooting.
- Verify GPU memory usage doesn't overflow during 28-fold parallelizable training.

### Validation
- Report on training convergence (loss curves) and hardware metrics.

### Dependencies
- Blocked by: [2]
- Blocks: [4]

---

## Phase 4: Final Reliability Audit

### Objective
Perform a 100% audit of the new meta-labeling pipeline to ensure "Toyota engine" reliability and zero data leakage.

### Agent: `code_reviewer`
### Parallel: No

### Files to Modify
- None (Analysis only)

### Implementation Details
- Audit the CPCV purging/embargo logic in the two-stage setup.
- Verify the "Train All Universe" loop handles individual stock failures gracefully.
- Confirm the new metrics (AUC > 0.60 goal) are accurately calculated and logged.

### Validation
- Final approval report.

### Dependencies
- Blocked by: [3]
- Blocks: None

---

## File Inventory

| # | File | Phase | Purpose |
|---|------|-------|---------|
| 1 | `core/models/training/financial_engineer.py` | 1 | Target engineering |
| 2 | `core/models/training/trainer.py` | 2, 3 | Core training logic |
| 3 | `core/models/tcn/system.py` | 3 | TCN architecture |

## Risk Classification

| Phase | Risk | Rationale |
|-------|------|-----------|
| 1 | MEDIUM | Critical for signal quality; requires careful tuning of the primary signal. |
| 2 | HIGH | Fundamental re-architecture; highest chance of introducing bugs or leakage. |
| 3 | MEDIUM | Performance-heavy; risk of slow training or out-of-memory errors. |
| 4 | LOW | Verification only. |

## Execution Profile

```text
Execution Profile:
- Total phases: 4
- Parallelizable phases: 0
- Sequential-only phases: 4
- Estimated parallel wall time: 4 phases length
- Estimated sequential wall time: 4 phases length

Note: Native subagents currently run without user approval gates.
All tool calls are auto-approved without user confirmation.
```

| Phase | Agent | Model | Est. Input | Est. Output | Est. Cost |
|-------|-------|-------|-----------|------------|----------|
| 1 | data_engineer | Flash | 1500 | 500 | ~$0.00 |
| 2 | coder | Pro | 4000 | 1500 | ~$0.10 |
| 3 | performance_engineer | Flash | 2000 | 400 | ~$0.00 |
| 4 | code_reviewer | Pro | 6000 | 600 | ~$0.08 |
| **Total** | | | **13500** | **3000** | **~$0.18** |
