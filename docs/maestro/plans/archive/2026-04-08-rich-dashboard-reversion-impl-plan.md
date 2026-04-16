---
title: "Rich Menu Dashboard Reversion Implementation Plan"
design_ref: "docs/maestro/plans/2026-04-08-rich-dashboard-reversion-design.md"
created: "2026-04-08T16:45:00Z"
status: "draft"
total_phases: 3
estimated_files: 1
task_complexity: "medium"
---

# Rich Menu Dashboard Reversion Implementation Plan

## Plan Overview

- **Total phases**: 3
- **Agents involved**: `coder`, `code_reviewer`
- **Estimated effort**: Moderate refactor. Replacing the event-driven TUI with a procedural loop while ensuring core logic is correctly re-wired.

## Dependency Graph

```text
[Phase 1: Base Menu Loop (coder)]
           |
[Phase 2: Sub-menus & Feature Integration (coder)]
           |
[Phase 3: Final Audit (code_reviewer)]
```

## Execution Strategy

| Stage | Phases | Execution | Agent Count | Notes |
|-------|--------|-----------|-------------|-------|
| 1     | Phase 1 | Sequential | 1 | Foundation: Loop & Banner |
| 2     | Phase 2 | Sequential | 1 | Domain: Feature wiring |
| 3     | Phase 3 | Sequential | 1 | Quality Gate |

## Phase 1: Base Menu Loop & Banner Restoration

### Objective
Remove all Textual dependencies and classes from `dashboard.py`. Restore the "MARK5" ASCII banner and implement the top-level `while True` loop with numeric input.

### Agent: `coder`
### Parallel: No

### Files to Modify
- `dashboard.py` — Replace `Mark5App` and `Screen` classes with a main loop.

### Implementation Details
- Remove `textual` imports.
- Re-implement `show_banner()` as the primary visual.
- Create a `main_menu()` function that prints options (1-5) and uses `IntPrompt.ask` for selection.
- Implement the loop that clears the console (`os.system('cls' if os.name == 'nt' else 'clear')`) before rendering.

### Validation
- `python3 dashboard.py` launches and shows the banner and top-level menu.
- `python3 -m py_compile dashboard.py` passes.

### Dependencies
- Blocked by: None
- Blocks: [2]

---

## Phase 2: Sub-menus & Feature Integration

### Objective
Implement sub-menus for Training and wire up Backtesting, Paper Trading, and Status functionality using synchronous logic and Rich Progress feedback.

### Agent: `coder`
### Parallel: No

### Files to Modify
- `dashboard.py` — Add sub-menu logic and feature calls.

### Implementation Details
- **Training Sub-menu**: 1. Single Ticker, 2. All Universe, 3. Back.
- **Progress Feedback**: Use `with Progress() as progress:` for `train_model` and `run_simulation` to maintain the user's requirement for a beautiful UI.
- **Data Preservation**: Ensure `DataPipeline`, `MARK5Predictor`, and `MARK5MLTrainer` are called with the same parameters as the TUI version.
- **Status Display**: Re-implement the `status` command using a Rich `Table` inside a `Panel`.

### Validation
- Run a training session for a single ticker and verify the progress bar and completion message.
- Run a backtest and verify the result table is displayed correctly.

### Dependencies
- Blocked by: [1]
- Blocks: [3]

---

## Phase 3: Final Audit & Polish

### Objective
Verify the end-to-end flow of the menu system, ensure visual consistency with the "beautiful Rich" requirement, and confirm no regressions in the core ML logic.

### Agent: `code_reviewer`
### Parallel: No

### Files to Modify
- None (Analysis only)

### Implementation Details
- Verify all numeric paths lead to the correct functions.
- Check error handling: ensure invalid inputs or model failures don't crash the loop.
- Review the aesthetic quality of the terminal output.

### Validation
- Audit report output.

### Dependencies
- Blocked by: [2]
- Blocks: None

---

## File Inventory

| # | File | Phase | Purpose |
|---|------|-------|---------|
| 1 | `dashboard.py` | 1, 2 | Centralized Dashboard CLI |

## Risk Classification

| Phase | Risk | Rationale |
|-------|------|-----------|
| 1 | MEDIUM | Procedural refactor of the entry point; requires careful preservation of imports. |
| 2 | MEDIUM | Integrating synchronous ML logic with Rich Progress requires correct task management. |
| 3 | LOW | Read-only verification. |

## Execution Profile

```text
Execution Profile:
- Total phases: 3
- Parallelizable phases: 0
- Sequential-only phases: 3
- Estimated parallel wall time: 3 phases length
- Estimated sequential wall time: 3 phases length

Note: Native subagents currently run without user approval gates.
All tool calls are auto-approved without user confirmation.
```

| Phase | Agent | Model | Est. Input | Est. Output | Est. Cost |
|-------|-------|-------|-----------|------------|----------|
| 1 | coder | Pro | 1500 | 600 | ~$0.04 |
| 2 | coder | Pro | 2500 | 1000 | ~$0.07 |
| 3 | code_reviewer | Pro | 4000 | 500 | ~$0.06 |
| **Total** | | | **8000** | **2100** | **~$0.17** |
