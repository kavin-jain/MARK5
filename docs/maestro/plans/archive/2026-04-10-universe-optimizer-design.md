---
title: "Smart Universe Optimizer"
created: "2026-04-10T12:00:00Z"
status: "draft"
authors: ["TechLead", "User"]
type: "design"
design_depth: "deep"
task_complexity: "complex"
---

# Smart Universe Optimizer Design Document

## Problem Statement

The MARK5 system currently operates on a hardcoded universe of 10 stocks. To achieve the "holy shit, that's done" standard, the system must be intelligent enough to dynamically test a large universe of highly liquid stocks (e.g., top 150 NSE most active), train models if missing, execute historical backtests, and strictly select only those that prove mathematically profitable (≥ 15-20% annual returns, positive Sharpe ratios, high accuracy). This "Smart Universe Optimizer" must be integrated cleanly into the `dashboard.py` interface. Due to the immense computational and memory overhead of training and backtesting up to 150 deep-learning TCN models, running this in the main dashboard thread will cause Out-Of-Memory (OOM) crashes. The system requires a bulletproof, decoupled execution model that ensures the dashboard remains a reliable "Toyota engine" while the optimizer runs autonomously in the background, finally outputting a dynamically updatable `universe.json` configuration file.

## Requirements

### Functional Requirements

1. **REQ-1**: Create a dedicated `core/optimization/universe_optimizer.py` that handles fetching a dynamic universe of highly liquid NSE stocks (e.g., using `ISEAdapter` or a predefined Nifty Midcap 150 list).
2. **REQ-2**: The Optimizer must programmatically invoke `trainer.py` (via isolated subprocesses) for any ticker missing a model.
3. **REQ-3**: The Optimizer must programmatically invoke `RobustBacktester` on the universe over a set historical period (e.g., 365 days).
4. **REQ-4**: Define hard filtering thresholds: Minimum Annual Return ≥ 15%, Sharpe Ratio > 0.0, and positive trade count.
5. **REQ-5**: Save the filtered "elite" tickers to `config/universe.json` (or database), completely replacing the hardcoded `_DEFAULT_UNIVERSE` in `dashboard.py`.
6. **REQ-6**: Add a new "Smart Universe Optimizer" menu option in `dashboard.py` (e.g., inside the Training or Data Management module) to kick off this background job safely without crashing the dashboard.

### Non-Functional Requirements

1. **REQ-7**: Reliability ("Toyota Engine"): Training up to 150 models continuously WILL crash a single Python process due to TensorFlow/LightGBM memory leaks. The optimizer must wrap model training in a rigorous, crash-resilient `subprocess.run` loop.
2. **REQ-8**: Completeness ("Holy Shit, That's Done"): The solution must be end-to-end, with robust documentation, proper tests, and bulletproof logging indicating exact reasons for stock acceptance/rejection.

### Constraints

- Zero synthetic data usage.
- Strict adherence to the `rich` UI aesthetics in `dashboard.py` while the optimizer runs via an external log viewer or status indicator.

## Approach

### Selected Approach

**Subprocess Optimizer**

We will develop a standalone `UniverseOptimizer` module (`core/optimization/universe_optimizer.py`). The dashboard will have a new menu item that calls this module in a detached `subprocess.run` to guarantee the TUI never hangs and the heavy training load doesn't crash the Python interpreter running the dashboard. We will then refactor `dashboard.py` and `DataPipeline` to load `_DEFAULT_UNIVERSE` dynamically from a JSON file (e.g., `config/universe.json`) rather than a hardcoded list.

[Dynamically load `_DEFAULT_UNIVERSE` from JSON] — *[This approach was chosen to ensure the "elite" universe picked by the optimizer propagates immediately to all dashboard functions like "Train All" or "Backtest Universe".]* Traces To: REQ-5

[Subprocess Orchestration for Training] *(considered: in-memory `threading` — rejected because training 150 models consecutively causes massive TensorFlow memory leaks and OOM crashes)* — *[Ensures absolute 'Toyota' reliability, isolating heavy ML compute from the core dashboard application.]* Traces To: REQ-2, REQ-7

[Hard Filtering Thresholds (Sharpe > 0, Return > 15%)] — *[Mathematically enforces your exact profitability mandates ("profitable, gives atleast 15 to 20% annual return, positive sharpe") before a stock is allowed into the active trading universe.]* Traces To: REQ-4

### Alternatives Considered

#### In-Memory Threaded Optimizer
- **Description**: Run the 150-stock scan and backtesting using Python's `threading` within `dashboard.py`, tracking live with Rich Progress.
- **Pros**: Beautiful live progress bars directly in the UI.
- **Cons**: Extremely high risk of out-of-memory (OOM) crashes from training 150 heavy TCN models in a single process, potentially killing the dashboard.
- **Rejected Because**: It fails the "Toyota Reliability" requirement.

### Decision Matrix

| Criterion | Weight | Approach 1: Subprocess | Approach 2: In-Memory |
|-----------|--------|------------------------|-----------------------|
| **Toyota Reliability (Crash Prevention)** | 50% | 5: Complete memory isolation | 1: Almost guaranteed to OOM |
| **Data Propagation to Dash** | 30% | 5: File-based handoff | 5: Direct memory |
| **Developer Ergonomics (Rich UI)** | 20% | 4: Subprocess streaming logs | 5: Live Rich integration |
| **Weighted Total** | | **4.8** | **2.6** |

## Architecture

### Component Diagram

```text
[dashboard.py (Rich Hub)] -> [Smart Universe Optimizer Menu]
                               | (Calls via subprocess to avoid RAM crashes)
[core.optimization.universe_optimizer.py]
  |
  |-- 1. Fetch Candidate Stocks (Nifty 150 / Active NSE via ISEAdapter)
  |-- 2. Train Missing Models via `subprocess.run(trainer.py --symbols X)`
  |-- 3. Run Backtest (RobustBacktester) for X days (e.g., 365)
  |-- 4. Apply Filters: Annual Return >= 15%, Sharpe > 0, Trades > 0
  |-- 5. Save winning symbols to `config/universe.json`
  |
[config/universe.json] <----- (Used by dashboard.py as `_DEFAULT_UNIVERSE`)
```

### Data Flow

When the user triggers the "Smart Universe Optimizer", `dashboard.py` runs `universe_optimizer.py` as an isolated subprocess and streams the output directly to the Rich console or a log viewer. The optimizer pulls a hardcoded list of highly liquid Indian stocks (e.g., Nifty Midcap 150 components) or uses the `ISEAdapter` to query active stocks. For each stock, it checks the `RobustModelRegistry`. If a model is missing or stale, it spawns `trainer.py` in a separate process (to completely free GPU/RAM after completion). It then backtests the stock for the past year. If the metrics meet the strict requirements (Return >= 15%, Sharpe > 0), the ticker is appended to a list. Finally, this "elite" list is written to `config/universe.json`, permanently overriding the 10 hardcoded stocks in the `dashboard.py` memory.

### Key Interfaces

```python
class UniverseOptimizer:
    def get_candidate_universe(self) -> List[str]:
        # Returns Nifty 150 or top liquid NSE stocks
        pass
        
    def optimize_universe(self, min_return_pct: float = 15.0, min_sharpe: float = 0.5):
        # The main runner logic that trains, backtests, filters, and saves.
        pass

def load_dynamic_universe() -> List[str]:
    # In dashboard.py: reads config/universe.json on boot.
    pass
```

[Isolated Subprocess Training] — *[Guarantees the system does not crash halfway through a multi-hour 150-stock scan.]* Traces To: REQ-2, REQ-7

## Agent Team

| Phase | Agent(s) | Parallel | Deliverables |
|-------|----------|----------|--------------|
| 1     | data_engineer | No       | `core/optimization/universe_optimizer.py` implementation, fetching highly liquid candidate stocks from `ISEAdapter` or predefined lists. |
| 2     | coder    | No       | Wiring the `UniverseOptimizer` to `dashboard.py` (menu UI, subprocess runner). Updating `dashboard.py` to use `config/universe.json`. |
| 3     | tester   | No       | Developing `tests/test_universe_optimizer.py` and running the full optimization loop on a small subset (e.g., 5 stocks) to prove it functions flawlessly without OOM errors. |
| 4     | code_reviewer | No  | Final "Holy Shit, That's Done" check: Reviewing code quality, architecture, missing edge cases, docstrings, and zero-synthetic data compliance. |

## Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| **TensorFlow OOM (Out-of-Memory)** | HIGH | HIGH | Training hundreds of models sequentially causes TensorFlow context leaks. We mitigate this by completely isolating the training call `MARK5MLTrainer.train_model()` inside a `subprocess.run()`, ensuring the OS reclaims GPU/RAM after every stock. |
| **API Rate Limits / Bans** | HIGH | MEDIUM | Scanning 150+ stocks requires heavy OHLCV and ISE data fetching. The `data_engineer` must ensure the adapter uses `time.sleep` or exponential backoff to avoid Kite Connect or Indian API bans. |
| **Zero Stocks Passing Filter** | MEDIUM | LOW | If the market is in a deep bear regime, 0 stocks might achieve >15% annual return. The system will gracefully fall back to the last known good universe or a hardcoded default (e.g., RELIANCE, TCS) if the active list is empty. |

## Success Criteria

1. **Fully Automated Stock Screener ("Universe Optimizer")**: The dashboard features a new "Smart Universe Optimizer" that automatically pulls a large candidate list (Nifty Midcap 150 / Active NSE), backtests/trains them, and outputs a filtered list of only the best stocks.
2. **Mathematically Profitable Output**: The resulting `config/universe.json` (or similar config location) contains only stocks that have proven over the backtest period to yield ≥ 15-20% Annual Returns and a positive Sharpe Ratio.
3. **Bulletproof Execution**: The heavy multi-hour optimization runs flawlessly, leveraging `subprocess` to completely avoid OOM crashes or UI freezes, allowing the user to view progress tracking in the terminal.
4. **Dynamic Dashboard Integration**: The `dashboard.py` `_DEFAULT_UNIVERSE` variable is no longer hardcoded to 10 stocks. It reads directly from the optimized list on boot. If the list is missing, it falls back gracefully to a safe default.
5. **No Broken Pipes or Dangling Logic**: Every function, import, and module involved in this process is documented, cleanly structured, and tested. The final product is polished to "holy shit, that's done" standards, completely ready for end-user execution.
