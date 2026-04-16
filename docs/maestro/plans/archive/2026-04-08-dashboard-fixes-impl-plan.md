---
title: "Dashboard Bug Fixes & All-Stock Backtest Feature Implementation Plan"
design_ref: "docs/maestro/plans/2026-04-08-dashboard-fixes-design.md"
created: "2026-04-08T17:15:00Z"
status: "draft"
total_phases: 4
estimated_files: 1
task_complexity: "complex"
---

# Dashboard Bug Fixes & All-Stock Backtest Feature Implementation Plan

## Plan Overview

- **Total phases**: 4
- **Agents involved**: `coder`, `tester`, `code_reviewer`
- **Estimated effort**: Complex. Refactoring logger setup to suppress external noise, adding an iterative universe backtest, and conducting rigorous interactive testing of all menus.

## Dependency Graph

```text
[Phase 1: Logger Architecture (coder)]
           |
[Phase 2: All-Stock Backtest (coder)]
           |
[Phase 3: Interactive Testing (tester)]
           |
[Phase 4: Code Review (code_reviewer)]
```

## Execution Strategy

| Stage | Phases | Execution | Agent Count | Notes |
|-------|--------|-----------|-------------|-------|
| 1     | Phase 1 | Sequential | 1 | Foundation: Logger redirection |
| 2     | Phase 2 | Sequential | 1 | Domain: Feature implementation |
| 3     | Phase 3 | Sequential | 1 | Quality: End-user interactive tests |
| 4     | Phase 4 | Sequential | 1 | Quality: Architecture and code audit |

## Phase 1: Clean Logger Architecture

### Objective
Reconfigure all Python logging in `dashboard.py` to route exclusively to `logs/system.log` and suppress verbose output from third-party libraries (TF, LightGBM) to preserve the Rich UI.

### Agent: `coder`
### Parallel: No

### Files to Modify
- `dashboard.py` — Add logging setup logic at the very top of the script.

### Implementation Details
- Ensure the `logs/` directory exists.
- Configure `logging.basicConfig` to use a `FileHandler` targeting `logs/system.log`.
- Set environment variables (`os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'`, etc.) before importing heavy ML libraries.
- Silence specific loggers (`logging.getLogger("lightgbm").setLevel(logging.ERROR)`, etc.).

### Validation
- `python3 dashboard.py` and ensure the startup banner prints without any prior logging artifacts.

### Dependencies
- Blocked by: None
- Blocks: [2]

---

## Phase 2: All-Stock Backtest Feature

### Objective
Implement the "All Stock Backtest" logic under the backtesting menu, utilizing Rich progress bars and summary tables.

### Agent: `coder`
### Parallel: No

### Files to Modify
- `dashboard.py` — Update `backtesting_menu()`.

### Implementation Details
- In `backtesting_menu()`, add an option "2. Run All Universe Backtest" (shift "Back" to option 3).
- Iterate through a default universe of symbols. For each symbol, fetch data, generate signals, and run `RobustBacktester.run_simulation()`.
- Use `with Progress(transient=True) as progress:` to track iteration over the tickers.
- Catch and handle missing model or missing data exceptions per ticker gracefully.
- Collect key metrics (e.g., Total Return, Sharpe Ratio, Win Rate) per ticker.
- After the loop finishes, render a Rich `Table` showing the summarized performance of the entire universe.

### Validation
- `python3 -m py_compile dashboard.py`.
- Run a dry-run or mock test of the All-Stock Backtest to ensure the Rich table renders correctly.

### Dependencies
- Blocked by: [1]
- Blocks: [3]

---

## Phase 3: Interactive End-User Testing

### Objective
Test every single menu option in `dashboard.py` to ensure absolute UI cleanliness and functionality.

### Agent: `tester`
### Parallel: No

### Files to Modify
- None (Analysis/testing only via `run_shell_command`).

### Implementation Details
- Run `dashboard.py` interactively or using an expect script/Python `subprocess` to simulate navigating through:
  - Training -> Single Ticker
  - Training -> All Universe
  - Backtesting -> Single
  - Backtesting -> All
  - Paper Trading
  - System Status
  - Data Management -> Daily Data / Build Matrix
- Validate that no external logging or errors corrupt the stdout.

### Validation
- Detailed test report documenting which flows were verified.

### Dependencies
- Blocked by: [2]
- Blocks: [4]

---

## Phase 4: Final Code Review & Audit

### Objective
Verify the "Toyota engine" reliability, ensuring the "smart algo trader" aesthetic is perfectly maintained without leaks.

### Agent: `code_reviewer`
### Parallel: No

### Files to Modify
- None (Analysis only)

### Implementation Details
- Review `dashboard.py` for code quality, strict adherence to the logging rules, and robust error handling in the new backtest loop.
- Confirm the architectural constraints set in the design phase were met.

### Validation
- Final block/approve audit report.

### Dependencies
- Blocked by: [3]
- Blocks: None

---

## File Inventory

| # | File | Phase | Purpose |
|---|------|-------|---------|
| 1 | `dashboard.py` | 1, 2 | Centralized Dashboard CLI |

## Risk Classification

| Phase | Risk | Rationale |
|-------|------|-----------|
| 1 | HIGH | Suppressing third-party libraries globally can be tricky and depends on the specific import order and environment variables. |
| 2 | MEDIUM | Long-running sequential loop requires solid exception handling to avoid crashing midway. |
| 3 | LOW | Read-only/execution testing. |
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
| 1 | coder | Pro | 1500 | 500 | ~$0.03 |
| 2 | coder | Pro | 2500 | 1000 | ~$0.07 |
| 3 | tester | Pro | 4000 | 500 | ~$0.06 |
| 4 | code_reviewer | Pro | 5000 | 500 | ~$0.07 |
| **Total** | | | **13000** | **2500** | **~$0.23** |