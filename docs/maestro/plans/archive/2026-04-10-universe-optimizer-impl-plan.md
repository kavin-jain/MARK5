---
title: "Smart Universe Optimizer Implementation Plan"
design_ref: "docs/maestro/plans/2026-04-10-universe-optimizer-design.md"
created: "2026-04-10T12:15:00Z"
status: "draft"
total_phases: 4
estimated_files: 3
task_complexity: "complex"
---

# Smart Universe Optimizer Implementation Plan

## Plan Overview

- **Total phases**: 4
- **Agents involved**: `data_engineer`, `coder`, `tester`, `code_reviewer`
- **Estimated effort**: Complex. Building a standalone subprocess-based optimizer module that fetches data, trains missing models, backtests, filters for strict profitability (≥15% return), and updates a dynamic `config/universe.json`. 

## Dependency Graph

```text
[Phase 1: Optimizer Module (data_engineer)]
           |
[Phase 2: Dashboard Integration (coder)]
           |
[Phase 3: End-to-End Testing (tester)]
           |
[Phase 4: Final Quality Audit (code_reviewer)]
```

## Execution Strategy

| Stage | Phases | Execution | Agent Count | Notes |
|-------|--------|-----------|-------------|-------|
| 1     | Phase 1 | Sequential | 1 | Foundation: UniverseOptimizer class |
| 2     | Phase 2 | Sequential | 1 | Integration: Dashboard Menu & Dynamic Load |
| 3     | Phase 3 | Sequential | 1 | Quality: Automated Test |
| 4     | Phase 4 | Sequential | 1 | Quality: Code Review |

## Phase 1: Optimizer Module Development

### Objective
Create `core/optimization/universe_optimizer.py` that fetches a large candidate universe, trains models in subprocesses (to avoid OOM), backtests them, and filters for elite performance.

### Agent: `data_engineer`
### Parallel: No

### Files to Create
- `core/optimization/universe_optimizer.py` — The core logic for screening, training, and selecting the active universe.

### Files to Modify
- None

### Implementation Details
- `get_candidate_universe()`: Fetches the top 150 most active stocks from the `ISEAdapter`.
- `optimize_universe()`: 
    - Loops over candidates.
    - Uses `subprocess.run([sys.executable, "core/models/training/trainer.py", "--symbols", ticker])` to train missing/stale models cleanly.
    - Runs `RobustBacktester` on each for 365 days.
    - Filters: `Total Return % >= 15.0` AND `Sharpe Ratio > 0.0`.
    - Writes winning symbols to `config/universe.json`.

### Validation
- `python -m py_compile core/optimization/universe_optimizer.py`

### Dependencies
- Blocked by: None
- Blocks: [2]

---

## Phase 2: Dashboard UI Integration & Dynamic Load

### Objective
Refactor `dashboard.py` and `core/data/data_pipeline.py` to use `config/universe.json` instead of a hardcoded list, and add the optimizer to the UI.

### Agent: `coder`
### Parallel: No

### Files to Create
- None

### Files to Modify
- `dashboard.py` — Add "Smart Universe Optimizer" to the menu, and dynamically load `_DEFAULT_UNIVERSE`.
- `core/data/data_pipeline.py` — Helper functions if needed to read `config/universe.json`.

### Implementation Details
- In `dashboard.py`, replace `_DEFAULT_UNIVERSE = [...]` with a function that loads `config/universe.json` and falls back to a safe default if missing.
- Add an option to the Data Management menu (or main menu) to "Run Smart Universe Optimizer".
- When selected, use `subprocess.run([sys.executable, "-c", "from core.optimization.universe_optimizer import UniverseOptimizer; UniverseOptimizer().optimize_universe()"])` to launch the optimizer.
- Add `rich` spinners or stream the subprocess stdout to the console so the user sees progress.

### Validation
- `python -m py_compile dashboard.py`

### Dependencies
- Blocked by: [1]
- Blocks: [3]

---

## Phase 3: End-to-End Testing

### Objective
Develop a test script to verify the optimizer runs without crashing and correctly filters a small subset of stocks.

### Agent: `tester`
### Parallel: No

### Files to Create
- `tests/test_universe_optimizer.py` — Automated verification script.

### Implementation Details
- Write a script that instantiates `UniverseOptimizer` and overrides the candidate list to 3-5 specific symbols.
- Verify it writes a valid JSON file.
- Ensure no OOM errors occur during the subprocess training calls.

### Validation
- `pytest tests/test_universe_optimizer.py`

### Dependencies
- Blocked by: [2]
- Blocks: [4]

---

## Phase 4: Final Quality Audit

### Objective
Verify the "Toyota engine" reliability, ensuring the optimizer correctly isolates memory, uses real data, and handles errors gracefully.

### Agent: `code_reviewer`
### Parallel: No

### Files to Modify
- None (Analysis only)

### Implementation Details
- Review `universe_optimizer.py` for subprocess usage and exception handling.
- Review `dashboard.py` dynamic loading logic.

### Validation
- Final block/approve audit report.

### Dependencies
- Blocked by: [3]
- Blocks: None

---

## File Inventory

| # | File | Phase | Purpose |
|---|------|-------|---------|
| 1 | `core/optimization/universe_optimizer.py` | 1 | Core optimization logic |
| 2 | `dashboard.py` | 2 | UI Integration |
| 3 | `tests/test_universe_optimizer.py` | 3 | Verification |

## Risk Classification

| Phase | Risk | Rationale |
|-------|------|-----------|
| 1 | HIGH | Spawning subprocesses in a loop requires very careful standard out/error handling and process management. |
| 2 | MEDIUM | Refactoring the global `_DEFAULT_UNIVERSE` might break downstream references if not caught. |
| 3 | LOW | Testing only. |
| 4 | LOW | Read-only audit. |

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
| 1 | data_engineer | Pro | 2000 | 1000 | ~$0.06 |
| 2 | coder | Pro | 3000 | 800 | ~$0.06 |
| 3 | tester | Pro | 2500 | 500 | ~$0.04 |
| 4 | code_reviewer | Pro | 4000 | 500 | ~$0.06 |
| **Total** | | | **11500** | **2800** | **~$0.22** |
