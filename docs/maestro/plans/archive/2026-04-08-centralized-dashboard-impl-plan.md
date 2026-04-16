---
title: "Centralized CLI Dashboard & Pipeline Consolidation Implementation Plan"
design_ref: "docs/maestro/plans/2026-04-08-centralized-dashboard-design.md"
created: "2026-04-08T00:00:00Z"
status: "draft"
total_phases: 5
estimated_files: 3
task_complexity: "complex"
---

# Centralized CLI Dashboard & Pipeline Consolidation Implementation Plan

## Plan Overview

- **Total phases**: 5
- **Agents involved**: `coder`, `data_engineer`, `refactor`, `code_reviewer`
- **Estimated effort**: Complex. Requires building a new entry point, aggressively removing old scripts, enforcing zero-synthetic data, and writing robust background threads.

## Dependency Graph

```text
[Phase 1: CLI Entry Point (coder)]
           |
    +------+------+
    |             |             |
[Phase 2]     [Phase 3]     [Phase 4]
(data_eng)     (coder)      (refactor)
    |             |             |
    +------+------+-------------+
           |
[Phase 5: Code Review]
```

## Execution Strategy

| Stage | Phases | Execution | Agent Count | Notes |
|-------|--------|-----------|-------------|-------|
| 1     | Phase 1 | Sequential | 1 | Foundation: Setup main `dashboard.py` |
| 2     | Phase 2, 3, 4 | Parallel | 3 | Core logic: Db/Data, Background Threads, Redundancy Purge |
| 3     | Phase 5 | Sequential | 1 | Quality: E2E Review |

## Phase 1: Scaffolding the Rich CLI Entry Point

### Objective
Create the single monolithic entry point `dashboard.py` (or `main.py`) powered by Typer and Rich, integrating subcommands for train, backtest, and paper-trade.

### Agent: `coder`
### Parallel: No

### Files to Create
- `dashboard.py` — The unified CLI router and Rich UI rendering logic.

### Implementation Details
Setup Typer commands. Initialize Rich console. Create placeholder commands that import from the core logic (`data_pipeline.py`, `predictor.py`, `trainer.py`).

### Validation
- `python dashboard.py --help`
- Run basic linting.

### Dependencies
- Blocked by: None
- Blocks: [2, 3, 4]

---

## Phase 2: Data Pipeline Strict Validation & Database Upgrades

### Objective
Refactor `core/data/data_pipeline.py` to assert non-synthetic data, and upgrade `core/infrastructure/database_manager.py` to persist stats.

### Agent: `data_engineer`
### Parallel: Yes

### Files to Modify
- `core/data/data_pipeline.py` — Add assertions blocking synthetic data flows.
- `core/infrastructure/database_manager.py` — Add tables/schemas for detailed stats logging.

### Implementation Details
Ensure any synthetic data flag or stub logic immediately raises an exception. Update the DB manager to handle high-frequency writes during paper trading.

### Validation
- Unit tests validating the synthetic data block.

### Dependencies
- Blocked by: [1]
- Blocks: [5]

---

## Phase 3: Background Auto-Retraining Threading

### Objective
Implement background auto-training jobs that hook into `trainer.py` and `predictor.py` natively from `dashboard.py` without blocking the UI.

### Agent: `coder`
### Parallel: Yes

### Files to Modify
- `dashboard.py` — Add Python `threading` implementation for auto-training jobs and live UI status updates.

### Implementation Details
Integrate the logic from `check_models_ready` to gracefully pause and prompt the user. Trigger `trigger_background_training` in a separate thread.

### Validation
- Compile checks. Run simulated prompt in CLI.

### Dependencies
- Blocked by: [1]
- Blocks: [5]

---

## Phase 4: Redundancy Purge and Import Consolidation

### Objective
Aggressively identify and remove standalone root scripts performing independent tasks. Centralize imports inside the `core/` structure.

### Agent: `refactor`
### Parallel: Yes

### Files to Modify
- N/A (Broad deletion and import path fixing across the repo)

### Implementation Details
Delete any scripts that overlap with the new `dashboard.py` entry point. Fix all internal imports across `core/data`, `core/models`, and `core/trading`.

### Validation
- Run full codebase linting and `pytest`.

### Dependencies
- Blocked by: [1]
- Blocks: [5]

---

## Phase 5: Code Review & Final Audit

### Objective
Verify "Toyota engine" reliability, ensuring no loose scripts remain, and validating non-synthetic data flows end-to-end.

### Agent: `code_reviewer`
### Parallel: No

### Files to Modify
- None (Analysis only)

### Implementation Details
Review changes from Phases 1-4. Validate that `dashboard.py` handles thread completion gracefully and that the codebase is significantly cleaner without redundant files.

### Validation
- Audit report output.

### Dependencies
- Blocked by: [2, 3, 4]
- Blocks: None

---

## File Inventory

| # | File | Phase | Purpose |
|---|------|-------|---------|
| 1 | `dashboard.py` | 1, 3 | Core entry point |
| 2 | `core/data/data_pipeline.py` | 2 | Anti-synthetic data guards |
| 3 | `core/infrastructure/database_manager.py` | 2 | Detailed stats DB |

## Risk Classification

| Phase | Risk | Rationale |
|-------|------|-----------|
| 1     | LOW | Straightforward CLI scaffolding. |
| 2     | MEDIUM | Strict data validation could break existing test suites relying on synthetic stubs. |
| 3     | HIGH | Threading combined with complex ML/DB interactions requires careful state management. |
| 4     | HIGH | Aggressive file deletion and import changes risk broad compilation errors. |
| 5     | LOW | Read-only audit. |

## Execution Profile

```text
Execution Profile:
- Total phases: 5
- Parallelizable phases: 3 (in 1 batch)
- Sequential-only phases: 2
- Estimated parallel wall time: 3 phases length
- Estimated sequential wall time: 5 phases length

Note: Native parallel execution currently runs agents in autonomous mode.
All tool calls are auto-approved without user confirmation.
```

| Phase | Agent | Model | Est. Input | Est. Output | Est. Cost |
|-------|-------|-------|-----------|------------|----------|
| 1 | coder | Pro | 1000 | 500 | ~$0.03 |
| 2 | data_engineer | Flash | 2000 | 300 | ~$0.01 |
| 3 | coder | Pro | 3000 | 600 | ~$0.05 |
| 4 | refactor | Pro | 8000 | 400 | ~$0.10 |
| 5 | code_reviewer | Pro | 5000 | 500 | ~$0.07 |
| **Total** | | | **19000** | **2300** | **~$0.26** |
