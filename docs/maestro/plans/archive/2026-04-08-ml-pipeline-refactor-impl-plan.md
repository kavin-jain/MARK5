---
title: "ML Pipeline Refactor & Institutional Validation Implementation Plan"
design_ref: "docs/maestro/plans/2026-04-08-ml-pipeline-refactor-design.md"
created: "2026-04-08T00:00:00Z"
status: "draft"
total_phases: 5
estimated_files: 5
task_complexity: "complex"
---

# ML Pipeline Refactor & Institutional Validation Implementation Plan

## Plan Overview

- **Total phases**: 5
- **Agents involved**: `data_engineer`, `coder`, `performance_engineer`, `code_reviewer`
- **Estimated effort**: Complex, requiring deep feature decoupling, statistical model rewrites, and 100% file audits.

## Dependency Graph

```text
[Phase 1: Feature Decoupling]
           |
[Phase 2: Small-Data CPCV]
           |
[Phase 3: TCN Engine & DSR]
           |
    +------+------+
    |             |
[Phase 4]     [Phase 5]
(Perf)        (Review)
```

## Execution Strategy

| Stage | Phases | Execution | Agent Count | Notes |
|-------|--------|-----------|-------------|-------|
| 1     | Phase 1 | Sequential | 1 | Foundation (Feature Engineering) |
| 2     | Phase 2 | Sequential | 1 | Foundation (CPCV Splits) |
| 3     | Phase 3 | Sequential | 1 | Core Domain & Integration |
| 4     | Phase 4, 5 | Parallel | 2 | Quality & Validation |

## Phase 1: Feature Engineering Decoupling

### Objective
Decouple the feature engineering logic from the model state to ensure a pristine read-only boundary for the 60-minute interval data.

### Agent: `data_engineer`
### Parallel: No

### Files to Modify
- `core/models/features.py` — Refactor to strictly expose standardized tensors without data leakage.
- `core/models/tcn/features.py` — Deprecate or isolate legacy feature generation specific to the TCN if redundant.

### Implementation Details
Ensure all feature engineering functions act as pure transformations (`data_df -> torch.Tensor`).

### Validation
- Run unit tests to verify identical tensor outputs.
- Lint and type-check the isolated functions.

### Dependencies
- Blocked by: None
- Blocks: [2]

---

## Phase 2: Small-Data CPCV Implementation

### Objective
Implement a Small-Data Aware Combinatorial Purged Cross-Validation (CPCV) splitter to maximize data retention while mathematically preventing leakage.

### Agent: `data_engineer`
### Parallel: No

### Files to Modify
- `core/models/training/cpcv.py` — Add the tailored purge/embargo limits logic for 60-minute frequencies.

### Implementation Details
Create `generate_cpcv_splits` that yields memory-efficient index masks rather than duplicated data tensors.

### Validation
- Programmatic assertions verifying zero overlap between train/embargo/test indices.

### Dependencies
- Blocked by: [1]
- Blocks: [3]

---

## Phase 3: TCN Engine Refactor & DSR

### Objective
Refactor the training engine to calculate Deflated Sharpe Ratio (DSR), integrate the CPCV splits, isolate unwanted files, and implement MLflow/WandB tracking with Shadow Deployment routing.

### Agent: `coder`
### Parallel: No

### Files to Modify
- `core/models/training/trainer.py` — Integrate CPCV splits and DSR validation gate.
- `core/models/training/engine.py` — Add MLflow tracking hooks and Shadow Deployment logic.
- *Unwanted legacy files* — Isolate safely into deprecated folders or rename out of the active path.

### Implementation Details
Implement the DSR statistical penalty to reject overfitted models. Route models failing DSR strictly to a shadow MLflow tracking environment.

### Validation
- Execute a full training dry-run over a mocked dataset.
- Verify MLflow logging and DSR gate branching.

### Dependencies
- Blocked by: [2]
- Blocks: [4, 5]

---

## Phase 4: Performance Validation

### Objective
Profile the refactored training loop and CPCV generation to ensure no pipeline bottlenecks are introduced.

### Agent: `performance_engineer`
### Parallel: Yes

### Files to Modify
- None (Analysis only)

### Implementation Details
Profile CPU/GPU utilization during CPCV generation and TCN training.

### Validation
- Output profiling traces and performance verification report.

### Dependencies
- Blocked by: [3]
- Blocks: None

---

## Phase 5: Code Review & Final Audit

### Objective
Rigorously review every file in `@core/models/` to guarantee zero bugs, institutional accuracy, and confirm no data leakage exists.

### Agent: `code_reviewer`
### Parallel: Yes

### Files to Modify
- None (Analysis only)

### Implementation Details
Review all changes made in Phases 1-3. Assert mathematical purity of the DSR and CPCV implementations.

### Validation
- Final block/approve report with any critical findings.

### Dependencies
- Blocked by: [3]
- Blocks: None

---

## File Inventory

| # | File | Phase | Purpose |
|---|------|-------|---------|
| 1 | `core/models/features.py` | 1 | Decoupled feature logic |
| 2 | `core/models/tcn/features.py` | 1 | Deprecate/isolate |
| 3 | `core/models/training/cpcv.py` | 2 | Small-Data CPCV logic |
| 4 | `core/models/training/trainer.py` | 3 | TCN training loop & DSR |
| 5 | `core/models/training/engine.py` | 3 | MLflow tracking & Shadow Mode |

## Risk Classification

| Phase | Risk | Rationale |
|-------|------|-----------|
| 1     | MEDIUM | Core logic transformation; requires careful regression testing. |
| 2     | HIGH | Directly controls data leakage prevention; mathematically sensitive. |
| 3     | HIGH | Complex state branching and tracking integration. |
| 4     | LOW | Read-only profiling. |
| 5     | LOW | Read-only code review. |

## Execution Profile

```text
Execution Profile:
- Total phases: 5
- Parallelizable phases: 2 (in 1 batch)
- Sequential-only phases: 3
- Estimated parallel wall time: 4 phases length
- Estimated sequential wall time: 5 phases length
```

| Phase | Agent | Model | Est. Input | Est. Output | Est. Cost |
|-------|-------|-------|-----------|------------|----------|
| 1 | data_engineer | Flash | 1500 | 500 | ~$0.00 |
| 2 | data_engineer | Flash | 1000 | 300 | ~$0.00 |
| 3 | coder | Pro | 3000 | 1000 | ~$0.07 |
| 4 | performance_engineer | Flash | 2500 | 200 | ~$0.00 |
| 5 | code_reviewer | Pro | 5000 | 500 | ~$0.07 |
| **Total** | | | **13000** | **2500** | **~$0.14** |
