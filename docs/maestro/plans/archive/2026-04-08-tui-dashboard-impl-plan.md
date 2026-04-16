---
title: "Aesthetic Trader TUI Dashboard Refactor Implementation Plan"
design_ref: "docs/maestro/plans/2026-04-08-tui-dashboard-design.md"
created: "2026-04-08T16:15:00Z"
status: "draft"
total_phases: 4
estimated_files: 1
task_complexity: "complex"
---

# Aesthetic Trader TUI Dashboard Refactor Implementation Plan

## Plan Overview

- **Total phases**: 4
- **Agents involved**: `coder`, `design_system_engineer`, `code_reviewer`
- **Estimated effort**: Complex refactor of the main application entry point to transition from sequential CLI to event-driven TUI.

## Dependency Graph

```text
[Phase 1: TUI Scaffolding (coder)]
           |
[Phase 2: Custom Widgets & CSS (design_system_engineer)]
           |
[Phase 3: Core Logic Integration (coder)]
           |
[Phase 4: Final Review & Audit (code_reviewer)]
```

## Execution Strategy

| Stage | Phases | Execution | Agent Count | Notes |
|-------|--------|-----------|-------------|-------|
| 1     | Phase 1 | Sequential | 1 | Foundation: Textual app structure |
| 2     | Phase 2 | Sequential | 1 | Styling: Tokyo Night theme & Charts |
| 3     | Phase 3 | Sequential | 1 | Integration: Sync to Async logic |
| 4     | Phase 4 | Sequential | 1 | Quality Gate |

## Phase 1: TUI Scaffolding & Layout

### Objective
Scaffold the `Textual` application structure in `dashboard.py`, defining the 3-column Grid layout and the main navigation screens.

### Agent: `coder`
### Parallel: No

### Files to Modify
- `dashboard.py` — Replace Typer logic with a `textual.app.App` subclass.

### Implementation Details
- Initialize `Mark5App(App)`.
- Define the `compose` method with placeholder widgets for `Sidebar`, `MainContent`, and `FooterLog`.
- Implement basic `Screen` switching logic for Dashboard, Training, and Backtest views.

### Validation
- `python dashboard.py` launches a blank 3-zone layout.
- Linting checks.

### Dependencies
- Blocked by: None
- Blocks: [2, 3]

---

## Phase 2: Custom Widgets, Sparklines & Charts

### Objective
Implement the "Aesthetic Trader" look using custom CSS and specialized widgets like Sparklines and Plotext candlestick charts.

### Agent: `design_system_engineer`
### Parallel: No

### Files to Modify
- `dashboard.py` — Implement widget classes and CSS.

### Implementation Details
- Create `WatchlistSidebar` with `Sparkline` indicators.
- Create `CandlestickChart` widget using `textual-plotext`.
- Apply "Tokyo Night" theme (Deep blues, neon accents) via Textual CSS.
- Implement `RichLog` at the bottom for system events.

### Validation
- Visual inspection of the dashboard UI.
- Verify chart rendering with sample data.

### Dependencies
- Blocked by: [1]
- Blocks: [3]

---

## Phase 3: Core Logic Integration & Async Workers

### Objective
Map existing `dashboard.py` logic (training, backtesting, status) into Textual `run_worker` tasks to ensure non-blocking UI.

### Agent: `coder`
### Parallel: No

### Files to Modify
- `dashboard.py` — Connect sidebar buttons to core backend modules.

### Implementation Details
- Wrap `MARK5MLTrainer.train_model()` in an async background worker.
- Connect Backtest form inputs to `RobustBacktester`.
- Implement real-time progress updates in the TUI using messages from workers.
- Ensure the "Zero Synthetic Data" guards are preserved.

### Validation
- Trigger a training task and verify UI remains responsive.
- Run a backtest and verify results display in a `DataTable`.

### Dependencies
- Blocked by: [2]
- Blocks: [4]

---

## Phase 4: Final Review & Audit

### Objective
Verify end-to-end reliability, UI responsiveness, and compliance with the "Aesthetic Trader" requirements.

### Agent: `code_reviewer`
### Parallel: No

### Files to Modify
- None (Analysis only)

### Implementation Details
- Audit the async/sync bridge for potential deadlocks.
- Verify zero synthetic data flow end-to-end.
- Check for any leftover Typer redundancy.

### Validation
- Audit report output.

### Dependencies
- Blocked by: [3]
- Blocks: None

---

## File Inventory

| # | File | Phase | Purpose |
|---|------|-------|---------|
| 1 | `dashboard.py` | 1, 2, 3 | Main TUI Application |

## Risk Classification

| Phase | Risk | Rationale |
|-------|------|-----------|
| 1 | MEDIUM | Changing the entry point logic is a major breaking change for existing CLI users. |
| 2 | LOW | Purely visual/widget implementation. |
| 3 | HIGH | Complexity of threading/async integration with existing synchronous ML code. |
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
| 1 | coder | Pro | 1500 | 600 | ~$0.04 |
| 2 | design_system_engineer | Flash | 2000 | 800 | ~$0.01 |
| 3 | coder | Pro | 3000 | 1000 | ~$0.07 |
| 4 | code_reviewer | Pro | 4000 | 500 | ~$0.06 |
| **Total** | | | **10500** | **2900** | **~$0.18** |
