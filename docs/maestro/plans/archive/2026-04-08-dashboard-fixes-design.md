---
title: "Dashboard Bug Fixes & All-Stock Backtest Feature"
created: "2026-04-08T17:00:00Z"
status: "draft"
authors: ["TechLead", "User"]
type: "design"
design_depth: "deep"
task_complexity: "complex"
---

# Dashboard Bug Fixes & All-Stock Backtest Feature Design Document

## Problem Statement

The current `dashboard.py` has critical bugs, primarily logging output bleeding into the standard output, which corrupts the beautifully structured Rich terminal UI. Furthermore, some core end-user features are missing, such as an "All Stock Backtest" which would allow a trader to evaluate models across the entire trading universe sequentially. To achieve a "smart Python algo trader" standard, the system needs to be rigorously tested function by function to ensure "Toyota engine" reliability, while maintaining an aesthetically pleasing, pristine terminal interface without extraneous print statements or log traces interrupting the flow.

## Requirements

### Functional Requirements

1. **REQ-1**: Reconfigure Python's root `logging` mechanism inside `dashboard.py` (and potentially across `core/`) to output strictly to a file (e.g., `logs/system.log`) instead of `sys.stdout` or `sys.stderr`.
2. **REQ-2**: Implement an "All Stock Backtest" feature that iterates through the entire universe (like "Train All Universe" does), aggregating performance metrics and rendering a final summary table in Rich.
3. **REQ-3**: Conduct manual, step-by-step verification of every single menu option in `dashboard.py` (Train Single, Train All, Run Backtest, View Status, Data Management) to ensure they work smoothly without crashing or corrupting the UI.

### Non-Functional Requirements

1. **REQ-4**: A "Pristine UI" standard where no external library or core module prints unstructured text to the console.
2. **REQ-5**: Maintain the synchronous execution flow designed in the previous session, utilizing `Rich` progress bars for long-running ML tasks.

### Constraints

- Must not use synthetic data.
- Must ensure compatibility with the existing `MARK5DatabaseManager` and `ModelVersionManager` infrastructure.

## Approach

### Selected Approach

**Clean Architecture & Feature Expansion**

We will configure the `logging` module at the very top of `dashboard.py` (and suppress warnings from external libraries like `lightgbm`, `xgboost`, or `tensorflow`) to write exclusively to `logs/system.log`. This ensures the terminal remains pristine for the Rich UI. We will then add the "All Stock Backtest" option under the Backtesting menu, iterating through the universe sequentially and accumulating metrics into a single summary table. Finally, we will run manual checks through the `run_shell_command` on a test script or directly testing the dashboard functions to ensure end-to-end "Toyota engine" reliability.

[Route all logging to file] *(considered: Quick Patch — rejected because it ignores root causes of UI corruption)* — *[Chosen to permanently solve the problem of background logs breaking the Rich layout, ensuring a "smart algo trader" aesthetic.]* Traces To: REQ-1, REQ-4

[Sequential Aggregation for All-Stock Backtest] — *[Chosen to safely iterate over the universe without hitting memory limits or requiring complex multiprocessing, while providing a satisfying final summary table.]* Traces To: REQ-2

[Manual Step-by-step Verification] — *[Crucial to fulfilling the request to "go through every feature and function... test them hardly" and ensure "positive no bugs" across the UI.]* Traces To: REQ-3

### Alternatives Considered

#### Quick Patch
- **Description**: Quickly patch the logger to suppress output and just add the All-Stock Backtest without heavily verifying the rest of the file.
- **Pros**: Fast to implement.
- **Cons**: Does not fix root cause of logger corrupting UI, risk of silent failures.
- **Rejected Because**: The user explicitly requested an end-user level of reliability and "test them hardly".

### Decision Matrix

| Criterion | Weight | Approach 1 (Clean Arch) | Approach 2 (Quick Patch) |
|-----------|--------|-------------------------|--------------------------|
| UI Cleanliness | 40% | 5: File routing guarantees no stdout corruption | 2: Hacks only hide some output |
| Feature Completeness | 30% | 5: Implements requested 'All Stock' and tests everything | 3: Only adds the feature without testing |
| "Toyota" Reliability | 30% | 5: Systematic testing across all menus | 2: High risk of silent failures |
| **Weighted Total** | | **5.0** | **2.3** |

## Architecture

### Component Diagram

```text
[User Keyboard Input]
       |
[dashboard.py (Rich Main Loop)]
       |
       |-- [Logger Configuration] ---> [logs/system.log (FileHandler)]
       |
       |-- [Menu Switcher]
       |
       +--> [MARK5MLTrainer] (Synchronous + Progress Bar)
       +--> [RobustBacktester]
                |-- [Single Ticker Backtest]
                |-- [All Universe Backtest (Sequential Aggregation)] ---> [Rich Summary Table]
       +--> [MARK5Predictor]
```

### Data Flow

1. `dashboard.py` initializes, immediately configuring the root Python logger and third-party libraries (e.g., TF, XGBoost, LightGBM) to route ALL INFO, WARNING, and ERROR logs to `logs/system.log`. Nothing prints to `sys.stdout` except Rich components.
2. The user navigates to Backtesting and selects "All Universe Backtest".
3. A progress bar initializes for the total number of tickers.
4. For each ticker, the backtest runs synchronously. The metrics (e.g., Sharpe, AUC, Total Return) are collected into a dictionary.
5. After all tickers complete, a comprehensive Rich `Table` is rendered to the screen summarizing the universe's performance.

### Key Interfaces

```python
def setup_logging():
    # Configure root logger with FileHandler and set external loggers to ERROR/CRITICAL
    pass

def backtest_universe():
    # Iterates over all tickers, runs single backtest, aggregates results
    pass
```

[Route external loggers to file] — *[Chosen to ensure "smart algo trader" visual standards where no random warnings break the UI panels.]* Traces To: REQ-1

## Agent Team

| Phase | Agent(s) | Parallel | Deliverables |
|-------|----------|----------|--------------|
| 1     | coder    | No       | Reconfiguring all logging in `dashboard.py` and across core modules to redirect exclusively to `logs/system.log`. Suppressing third-party library output. |
| 2     | coder    | No       | Implementing the "All Stock Backtest" logic in `dashboard.py` under the backtesting menu, utilizing Rich progress bars and summary tables. |
| 3     | tester   | No       | Interactively testing every menu option in the dashboard (Train Single, Train All, Backtest Single, Backtest All, Data Management, Status) by running dry-runs or specific mock scripts to verify absolute UI cleanliness and functionality. |
| 4     | code_reviewer | No  | Verifying the "Toyota engine" reliability, ensuring the "smart algo trader" aesthetic is perfectly maintained without leaks. |

## Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| External Lib Output | HIGH | HIGH | Libraries like LightGBM, CatBoost, or TensorFlow often bypass standard Python `logging`. The `coder` must explicitly set environment variables (`TF_CPP_MIN_LOG_LEVEL`) and pass `verbose=False` parameters in `trainer.py` to ensure pristine terminal output. Traces To: REQ-4 |
| Universe Backtest Timeout | LOW | MEDIUM | Running an all-stock backtest sequentially may take time. Use `with Progress(transient=True)` so the UI explicitly tracks completion time and doesn't appear frozen. Traces To: REQ-5 |
| Missing Models | MEDIUM | LOW | The all-stock backtest must gracefully skip tickers that don't have active models instead of crashing the summary table. Traces To: REQ-2 |

## Success Criteria

1. **Pristine Rich Interface**: When `dashboard.py` is executed, no third-party warnings (e.g., TensorFlow, LightGBM) or Python `INFO/WARNING` logs print to the terminal, preserving the aesthetic layout.
2. **Dedicated Log File**: All system logs are written strictly to a file like `logs/system.log` for backend debugging.
3. **All Stock Backtest Implemented**: Under the Backtest menu, users can select an "All Universe" option that iterates over the defined ticker symbols, compiles the metrics (e.g., Sharpe Ratio, Total Returns), and outputs a clean Rich table.
4. **Resilient Operation**: Missing models during an all-stock backtest are skipped gracefully, tracked, and reported in the final summary.
5. **Verified Functionality**: Every menu option in `dashboard.py` has been explicitly run and verified to function seamlessly like a "well oiled Toyota engine".
