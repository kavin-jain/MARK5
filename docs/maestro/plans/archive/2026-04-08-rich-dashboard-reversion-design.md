---
title: "Rich Menu Dashboard Reversion"
created: "2026-04-08T16:30:00Z"
status: "draft"
authors: ["TechLead", "User"]
type: "design"
design_depth: "deep"
task_complexity: "medium"
---

# Rich Menu Dashboard Reversion Design Document

## Problem Statement

The recently implemented Textual TUI for `dashboard.py` does not align with the user's aesthetic preferences and workflow. The user prefers a simpler, more classic "Rich" terminal interface that uses numeric menu inputs (e.g., 1. Train, 2. Backtest) rather than complex TUI widgets and screen switching. There is a specific requirement to restore the "MARK5" ASCII banner at startup and provide a beautiful, well-formatted console experience using Rich's layout and rendering capabilities. The system must still centralize all core features (Training, Backtesting, Paper Trading, Status) into this single entry point, ensuring a "diamond solid" and reliable "Toyota engine" performance without the overhead of a full-screen event-driven TUI.

## Requirements

### Functional Requirements

1. **REQ-1**: Implement a persistent `while True` loop in `dashboard.py` that serves as the main navigation engine.
2. **REQ-2**: Restore and display the large "MARK5" ASCII banner at every application startup and menu refresh.
3. **REQ-3**: Implement a numeric menu system (1-5) for top-level categories: Training, Backtesting, Paper Trading, System Status, and Exit.
4. **REQ-4**: Implement sub-menus for complex tasks, specifically for Training (Single Ticker, All Universe).
5. **REQ-5**: Use `rich.progress.Progress` to provide visual feedback for long-running operations (Training, Backtesting) since they will now be executed synchronously.

### Non-Functional Requirements

1. **REQ-6**: Aesthetic Consistency: All output must be beautifully formatted using Rich `Panels`, `Tables`, and `Columns`.
2. **REQ-7**: Reliability: Maintain all existing "diamond solid" logic, such as zero-synthetic data guards and model registry checks.

### Constraints

- Revert from the `textual` library back to a core `rich` implementation.
- Avoid complex event loops where possible to maximize reliability and simplicity.

## Approach

### Selected Approach

**Rich Terminal Menu Loop**

We will refactor `dashboard.py` by removing all Textual code and replacing it with a main loop driven by `rich.console`. The loop will clear the terminal, display the "MARK5" banner and a formatted menu, and then wait for numeric input.

[Refactor to standard Python main loop] — *[Chosen to satisfy the preference for numeric input menus and a simpler, more predictable console interaction model.]* Traces To: REQ-1, REQ-3

[Restore "MARK5" Banner] — *[Addresses the specific visual request for the startup branding.]* Traces To: REQ-2

[Synchronous Execution with Rich Progress] *(considered: Background threads with a UI — rejected because it adds complexity back into the console flow)* — *[Ensures absolute 'Toyota engine' reliability by performing tasks in a clear, sequential manner while providing high-quality visual feedback via progress bars.]* Traces To: REQ-5

[Categorized Hierarchy for Menus] — *[Chosen to provide a clean organization of features without overwhelming the user with too many top-level options.]* Traces To: REQ-4

### Alternatives Considered

#### Interactive Menu Screen
- **Description**: Use `rich-menu` or a custom selection list to pick options.
- **Pros**: Slightly more "UI" feel than typing numbers.
- **Cons**: Numbers were explicitly requested.
- **Rejected Because**: The user explicitly asked for numeric input ("1.backtest 2.train something like that").

### Decision Matrix

| Criterion | Weight | Rich Menu Loop | Interactive Menu Screen |
|-----------|--------|----------------|-------------------------|
| User Preference Alignment | 50% | 5: Direct match for numeric inputs | 3: Selection vs typing |
| Implementation Simplicity | 30% | 5: Standard while-loop | 4: External library dependency |
| Visual Polish | 20% | 5: High (Rich native) | 5: High |
| **Weighted Total** | | **5.0** | **3.7** |

## Architecture

### Component Diagram

```text
[User Keyboard Input (Numeric)]
       |
[dashboard.py (Rich Main Loop)]
       |
       |-- [Banner Renderer] (ASCII Art)
       |-- [Menu Switcher] (Logic for selection)
       |
       +--> [Feature Modules]
                |
                +--> [MARK5MLTrainer] (Synchronous + Progress Bar)
                +--> [RobustBacktester] (Synchronous + Result Table)
                +--> [MARK5Predictor] (Inference)
                +--> [System Status] (Rich Table display)
```

### Data Flow

1. The app starts, clears the screen, and renders the ASCII banner.
2. The `Main Menu` is displayed as a Rich `Table` within a `Panel`.
3. The user enters a number (e.g., `1` for Training).
4. The `Menu Switcher` identifies the choice and displays a `Sub-menu` (e.g., `1.1 Single`, `1.2 All`).
5. Upon final selection, the corresponding function is called.
6. Long-running tasks use `Progress()` to show a live bar.
7. Results are rendered as a Rich `Table`.
8. The app pauses for a "Press Enter to continue" before looping back to the Main Menu.

### Key Interfaces

```python
def main_menu_loop():
    # while True loop with screen clearing and input prompts
    pass

def run_training_workflow(single: bool):
    # Handles ticker input and calls trainer.train_model() within Progress context
    pass
```

[Synchronous sequential flow] — *[Chosen to eliminate the race conditions and complexity of async UI events, ensuring 'diamond solid' execution paths.]* Traces To: REQ-1

[Rich Progress context managers] — *[Used to maintain high-quality visual feedback while the main thread performs blocking operations.]* Traces To: REQ-5

## Agent Team

| Phase | Agent(s) | Parallel | Deliverables |
|-------|----------|----------|--------------|
| 1     | coder    | No       | Refactoring `dashboard.py`: removing Textual, restoring the banner, and implementing the main menu loop. |
| 2     | coder    | No       | Implementing sub-menus and wiring up core features (Training, Backtesting, Trade) with Rich Progress feedback. |
| 3     | code_reviewer | No | Verifying the menu navigation, visual polish of the Rich output, and ensuring all core logic remains 'diamond solid'. |

[Assigning `coder` to all implementation phases] — *[Required because the refactor back to Rich is a procedural overhaul of the main file, requiring consistent logic across the loop and feature integration.]* Traces To: REQ-1, REQ-3
[Dedicated Review] — *[Ensures the user's high standards for 'beautiful' and 'perfect' UI are met before final archival.]* Traces To: REQ-6

## Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| Feature Omission | MEDIUM | LOW | The `coder` will perform a direct audit of the current `dashboard.py` to ensure every feature (Backtest, Paper Trade, etc.) is mapped to a menu item. |
| Regression in Model Safety | HIGH | LOW | All existing guards (zero-synthetic data assertions, registry checks) must be preserved in the refactored functions. |
| User Interface Clutter | LOW | MEDIUM | Use consistent spacing, clear separators, and limited color palettes (e.g., cyan/white) to maintain the "beautiful" Rich aesthetic. Traces To: REQ-6 |
| Logic Errors in Loops | MEDIUM | MEDIUM | Rigorous testing of the 'Go Back' and 'Exit' paths within the nested menu structure to prevent infinite loops or crashes. Traces To: REQ-1 |

## Success Criteria

1. **Classic Rich Menu UI**: `dashboard.py` launches with the "MARK5" banner and a clear numeric menu (1-5).
2. **Intuitive Numeric Navigation**: All system features (Train Single/All, Backtest, Trade, Status) are accessible via numbers and sub-menus.
3. **High-Quality Visual Feedback**: Long-running tasks display accurate Rich progress bars and completion tables.
4. **Preserved Logic**: Zero data leakage, no synthetic data usage, and all "diamond solid" pipeline achievements from previous sessions remain active.
5. **Clean Codebase**: All `textual` dependencies and related TUI code are purged from `dashboard.py`.
