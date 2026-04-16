---
title: "Aesthetic Trader TUI Dashboard Refactor"
created: "2026-04-08T16:00:00Z"
status: "draft"
authors: ["TechLead", "User"]
type: "design"
design_depth: "deep"
task_complexity: "complex"
---

# Aesthetic Trader TUI Dashboard Refactor Design Document

## Problem Statement

The current system entry point, `dashboard.py`, is a command-driven CLI that requires users to remember specific arguments and flags (e.g., `--ticker`, `--days`). While functional, it lacks the intuitive and aesthetic appeal of a modern trading dashboard. There is a need for a centralized, interactive Terminal User Interface (TUI) that provides a "well-oiled toyota engine" experience—reliable, efficient, and visually polished. The user specifically requests an "Aesthetic Trader" look with menu-driven navigation, candlestick charts, real-time status indicators (heat-maps, sparklines), and sub-menus for core tasks like training (single/all), backtesting, and paper trading. The goal is to consolidate all system features into a single, high-fidelity TUI that remains responsive during long-running background tasks.

## Requirements

### Functional Requirements

1. **REQ-1**: Develop a full-screen TUI using `Textual` that replaces the current Typer-based CLI logic.
2. **REQ-2**: Implement a 3-zone Grid layout:
   - **Left Sidebar**: Watchlist navigation and Portfolio status.
   - **Center Main**: Dynamic view switching between Charts (plotext), Training console, and Backtest stats.
   - **Bottom Strip**: A persistent `RichLog` for real-time signals and system events.
3. **REQ-3**: Integrate `textual-plotext` to render candlestick charts for tickers.
4. **REQ-4**: Implement menu-driven navigation via the sidebar to trigger core functions:
   - **Training**: Sub-menu for "Train Single Ticker" and "Train All Universe".
   - **Backtesting**: Interactive ticker and date range selection.
   - **Paper Trading**: Start/Stop control with live status.
5. **REQ-5**: Background execution: Long-running ML tasks (training, signal generation) must run in async background workers to prevent the UI from freezing.

### Non-Functional Requirements

1. **REQ-6**: Aesthetic Design: Use a "Developer Aesthetic" theme (e.g., Tokyo Night or Catppuccin Mocha).
2. **REQ-7**: Reliability: The system must handle missing models or data by prompting the user within the TUI rather than crashing.

### Constraints

- Must run in a standard terminal environment.
- Use existing core modules (trainer, predictor, backtester) without modification where possible.

## Approach

### Selected Approach

**Full Textual Dashboard App**

We will completely refactor `dashboard.py` into a `textual.app.App` subclass. The application will leverage a `Grid` layout with a custom CSS for the "Tokyo Night" aesthetic. We will use `Screens` to manage the different modes (Dashboard, Training, Backtesting) and `RichLog` for real-time output.

[Refactor to `Textual App`] — *[This is the most direct path to the "Aesthetic Trader" look requested, allowing for complex layouts and event-driven interaction that Typer cannot provide.]* Traces To: REQ-1, REQ-2, REQ-6

[Async Background Workers] *(considered: Synchronous calls with spinners — rejected because it would freeze the full-screen TUI layout during multi-minute training sessions)* — *[Using Textual's `run_worker` ensures the dashboard remains interactive while heavy ML tasks compute in the background.]* Traces To: REQ-5

[Sidebar for Navigation] *(considered: Command palette — rejected because the user explicitly preferred a Sidebar-based sub-menu structure)* — *[Provides a clear, persistent navigation hierarchy for switching between backtesting, training, and trading modes.]* Traces To: REQ-4

### Alternatives Considered

#### Hybrid TUI/CLI Wrapper
- **Description**: Maintain the current `typer` subcommands for automation/scripting, but make the default action launch the full Textual TUI.
- **Pros**: Interactive UI for humans, CLI for automation.
- **Cons**: Codebase bloat from maintaining two interaction models.
- **Rejected Because**: Approach 1 provides a cleaner, dedicated entry point for the centralized system vision.

### Decision Matrix

| Criterion | Weight | Approach 1 (Full TUI) | Approach 2 (Hybrid) |
|-----------|--------|-----------------------|---------------------|
| Aesthetic Finish | 40% | 5: Pure TUI optimization | 4: Mixed entry points |
| UI Responsiveness | 30% | 5: Pure async model | 3: Risk of CLI blocking |
| Ease of Navigation | 30% | 5: Dedicated sidebar | 4: Sub-menu over args |
| **Weighted Total** | | **5.0** | **3.7** |

## Architecture

### Component Diagram

```text
[Terminal Window (TUI)]
       |
[Textual App (Mark5App)]
       |-- [CSS Theme: Tokyo Night]
       |-- [Layout: 3-Column Grid]
       |
       |-- [Sidebar (Watchlist/Navigation)]
       |-- [MainDisplay (Charts/Stats/Training)]
       |-- [BottomLog (RichLog events)]
       |
       +--> [Async Workers]
                |
                +--> [MARK5MLTrainer] (Training jobs)
                +--> [RobustBacktester] (Backtest simulations)
                +--> [MARK5Predictor] (Signal generation)
```

### Data Flow

The `Mark5App` acts as the event orchestrator. When a user selects a ticker in the **Sidebar**, the **MainDisplay** fetches data via `DataPipeline` and renders a candlestick chart using `textual-plotext`. Triggering "Train" or "Backtest" dispatches an **Async Worker**. The worker communicates progress back to the main thread via messages, which update the **BottomLog** and relevant status indicators (Sparklines/Tables) in real-time. All backend interactions (fetching data, loading models) are causal and non-synthetic.

### Key Interfaces

```python
class Mark5App(App):
    def on_mount(self) -> None:
        # Initialize grid and zones
        pass

    async def action_switch_mode(self, mode: str) -> None:
        # Switch CenterMain view (Backtest / Train / Trade)
        pass

    def run_training_worker(self, ticker: str) -> None:
        # Dispatches trainer.train_model() to background
        pass
```

[Grid Layout with CSS] — *[Enables the requested docking of widgets into specific zones (Sidebar, Main, Bottom) with precise pixel/cell control.]*

[Switchable Screens/Views] *(considered: A single flat screen — rejected because it would clutter the interface with unrelated widgets)* — *[Ensures the user only sees relevant data for the current task (e.g., Backtest stats vs. Training logs).]* Traces To: REQ-4

## Agent Team

| Phase | Agent(s) | Parallel | Deliverables |
|-------|----------|----------|--------------|
| 1     | coder    | No       | Scaffolding the `Textual` application structure, CSS themes (Tokyo Night), and the 3-zone Grid layout. |
| 2     | design_system_engineer | No | Implementing custom widgets: `RichLog` for system events, `Sparklines` for the watchlist, and `plotext` chart wrappers. |
| 3     | coder    | No       | Integrating core logic (Trainer, Predictor, Backtester) into async workers and wiring up the Sidebar menu actions. |
| 4     | code_reviewer | No | Verifying UI responsiveness during workers, checking for proper import centralization, and ensuring zero synthetic data flows. |

[Assigning `coder` to TUI scaffolding] — *[Required because the transition from sequential CLI to event-driven TUI is a fundamental code structure change.]* Traces To: REQ-1
[Assigning `design_system_engineer`] — *[Required to achieve the specific "Aesthetic Trader" look, ensuring widgets like Sparklines and Plotext charts are perfectly styled and integrated.]* Traces To: REQ-2, REQ-3, REQ-6

## Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| UI Blocking | HIGH | MEDIUM | Strictly enforce `run_worker` for any logic that exceeds 100ms. Use thread-safe status updates to refresh the UI. Traces To: REQ-5 |
| Async/Sync Conflicts | MEDIUM | HIGH | Existing ML code is synchronous. Use `run_in_executor` or Textual's threading support to wrap calls to `trainer.py` and `backtester.py`. |
| Dependency Issues | LOW | LOW | Verify `textual`, `plotext`, and `textual-plotext` are available in the project environment before starting. |
| Navigation Complexity | MEDIUM | MEDIUM | Use a formal `Screen` management system in Textual to keep the Center-Main area code modular and maintainable. Traces To: REQ-4 |

## Success Criteria

1. **Full-Screen Interactive TUI**: `dashboard.py` launches a full-screen application with a clear Sidebar, Chart area, and Event Log.
2. **Menu-Driven Workflow**: Core tasks (Training, Backtesting, Paper Trading) are triggered via sidebar buttons and sub-menus rather than CLI flags.
3. **Aesthetic Trader Visualization**: Candlestick charts (plotext) and live status indicators (sparklines) are rendered with a high-fidelity theme (Tokyo Night).
4. **Responsive UI**: The TUI remains perfectly responsive (not frozen) while a model is training or a backtest is running in the background.
5. **No Synthetic Data**: The entire workflow is verified to use 100% real historical data from Kite/DataPipeline.
