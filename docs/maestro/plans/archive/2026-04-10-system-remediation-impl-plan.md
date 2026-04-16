---
title: "Institutional System Remediation Implementation Plan"
design_ref: "docs/maestro/plans/2026-04-10-system-remediation-design.md"
created: "2026-04-10T10:15:00Z"
status: "approved"
total_phases: 5
estimated_files: 5
task_complexity: "complex"
---

# Institutional System Remediation Implementation Plan

## Plan Overview

- **Total phases**: 5
- **Agents involved**: `data_engineer`, `coder`, `refactor`, `tester`, `code_reviewer`
- **Estimated effort**: Complex. Involves a complete data-source overhaul, system-wide centralization, and a mandatory 10-iteration automated stress-testing cycle.

## Dependency Graph

```text
[Phase 1: Data Power-Train (data_engineer)]
           |
[Phase 2: Unified Interface (coder)]
           |
[Phase 3: Redundancy Purge (refactor)]
           |
[Phase 4: 10-Cycle Stress Test (tester)]
           |
[Phase 5: Institutional Audit (code_reviewer)]
```

## Execution Strategy

| Stage | Phases | Execution | Agent Count | Notes |
|-------|--------|-----------|-------------|-------|
| 1     | Phase 1 | Sequential | 1 | Foundation: ISE API & Pipeline Refactor |
| 2     | Phase 2 | Sequential | 1 | Integration: Dashboard & Trainer Refactor |
| 3     | Phase 3 | Sequential | 1 | Cleanup: Purging Redundancy |
| 4     | Phase 4 | Sequential | 1 | Quality: Automated 10-cycle stress tests |
| 5     | Phase 5 | Sequential | 1 | Quality: Final E2E Audit |

## Phase 1: Data Power-Train (ISE API & Master Pipeline)

### Objective
Implement the feature-complete `ISEAdapter` mapping to all 14 documented endpoints and refactor `DataPipeline` to act as the master data router.

### Agent: `data_engineer`
### Parallel: No

### Files to Create
- `core/data/adapters/ise_adapter.py` — High-fidelity mapping of ISE documentation (OHLCV, News, Analyst Recs, etc.).

### Files to Modify
- `core/data/data_pipeline.py` — Refactor to use `ISEAdapter` and enforce zero-synthetic data guards.

### Implementation Details
- Standardize all ISE responses into consistent Pandas DataFrames.
- Implement hard assertions (`assert not is_synthetic`) in the pipeline entry points.

### Validation
- `python3 core/data/data_pipeline.py` (Verify data flow for 3 tickers).
- Unit tests for each of the 14 ISE endpoints.

### Dependencies
- Blocked by: None
- Blocks: [2]

---

## Phase 2: Unified Interface & Centralization

### Objective
Fix naming mismatches in `dashboard.py`, add the ISE Intelligence menu, and refactor `MARK5MLTrainer` to consume the master pipeline.

### Agent: `coder`
### Parallel: No

### Files to Modify
- `dashboard.py` — Fix `backtesting_menu` bug, add ISE numeric sub-menu, and restore the ASCII banner.
- `core/models/training/trainer.py` — Remove internal `DataProvider` calls; use `DataPipeline` exclusively.
- `core/models/predictor.py` — Update inference loader to use centralized data router.

### Implementation Details
- Implement `ise_intelligence_menu()` with options for News, Bulk News, and Active Stocks.
- Use Dependency Injection to pass `DataPipeline` to the core engines.

### Validation
- `python3 dashboard.py` (Verify navigation loop).
- Verify successful model training using centralized data.

### Dependencies
- Blocked by: [1]
- Blocks: [3]

---

## Phase 3: Aggressive Redundancy Purge

### Objective
Identify and permanently delete all redundant scripts and fetchers to achieve a "factory pristine" repository state.

### Agent: `refactor`
### Parallel: No

### Files to Modify
- N/A (Broad deletion across `deprecated/scripts/` and duplicate modules).

### Implementation Details
- Delete `deprecated/scripts/retrain_universe.py`, `retrain_all.py`, etc.
- Verify zero internal references before deletion.

### Validation
- Ensure `pytest` passes after purge.
- Confirm repository file count reduction.

### Dependencies
- Blocked by: [2]
- Blocks: [4]

---

## Phase 4: 10-Cycle Automated Stress Test

### Objective
Develop a robust test harness and run it for 10 full iterations to guarantee "Toyota engine" reliability.

### Agent: `tester`
### Parallel: No

### Files to Create
- `tests/dashboard_stress_test.py` — Automated script simulating user input for every menu/feature.

### Implementation Details
- The test suite must simulate: Training, Backtesting (Single/All), Paper Trading Status, and ISE Intelligence.
- Catch and report any UI corruption or data leakage.
- Run exactly 10 full cycles.

### Validation
- Report output: "10/10 cycles successful."

### Dependencies
- Blocked by: [3]
- Blocks: [5]

---

## Phase 5: Institutional Audit & Archival

### Objective
Perform a final 100% End-to-End audit of the system logic and non-synthetic data flows.

### Agent: `code_reviewer`
### Parallel: No

### Files to Modify
- None (Analysis only).

### Implementation Details
- Deep audit of the `DataPipeline` guards.
- Verify "Toyota engine" reliability in the dashboard loop.

### Validation
- Final block/approve report.

### Dependencies
- Blocked by: [4]
- Blocks: None

---

## File Inventory

| # | File | Phase | Purpose |
|---|------|-------|---------|
| 1 | `core/data/adapters/ise_adapter.py` | 1 | Full ISE API Integration |
| 2 | `core/data/data_pipeline.py` | 1 | Master Data Router |
| 3 | `dashboard.py` | 2 | Unified Hub (Rich Menu) |
| 4 | `core/models/training/trainer.py` | 2 | Centralized Training Logic |
| 5 | `tests/dashboard_stress_test.py` | 4 | 10-Cycle Quality Gate |

## Risk Classification

| Phase | Risk | Rationale |
|-------|------|-----------|
| 1 | HIGH | Complex mapping of 14 external endpoints; API stability risk. |
| 2 | MEDIUM | Circular import risks during centralization. |
| 3 | MEDIUM | Aggressive deletion risks losing experimental logic. |
| 4 | LOW | Automated verification only. |
| 5 | LOW | Read-only audit. |

## Execution Profile

```text
Execution Profile:
- Total phases: 5
- Parallelizable phases: 0
- Sequential-only phases: 5
- Estimated parallel wall time: 5 phases length
- Estimated sequential wall time: 5 phases length

Note: Native subagents currently run in YOLO mode.
All tool calls are auto-approved without user confirmation.
```

| Phase | Agent | Model | Est. Input | Est. Output | Est. Cost |
|-------|-------|-------|-----------|------------|----------|
| 1 | data_engineer | Flash | 2000 | 800 | ~$0.01 |
| 2 | coder | Pro | 4000 | 1200 | ~$0.09 |
| 3 | refactor | Pro | 8000 | 400 | ~$0.10 |
| 4 | tester | Pro | 5000 | 800 | ~$0.08 |
| 5 | code_reviewer | Pro | 6000 | 600 | ~$0.08 |
| **Total** | | | **25000** | **3800** | **~$0.36** |
