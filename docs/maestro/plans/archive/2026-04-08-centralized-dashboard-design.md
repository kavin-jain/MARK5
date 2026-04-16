---
title: "Centralized CLI Dashboard & Pipeline Consolidation"
created: "2026-04-08T00:00:00Z"
status: "draft"
authors: ["TechLead", "User"]
type: "design"
design_depth: "deep"
task_complexity: "complex"
---

# Centralized CLI Dashboard & Pipeline Consolidation Design Document

## Problem Statement

The current system has grown organically, resulting in redundant scripts performing similar tasks across `core/data`, `core/models`, and `core/trading`. A major centralization effort is needed to establish a "Toyota engine"-like reliability—where every component from data fetching (A) to prediction and paper trading (Z) is integrated into a unified, seamless pipeline. A CLI-rich dashboard is required to surface functionality (backtesting, paper trading, and detailed stats) through a single entry point, minimizing human error and context switching. Additionally, the system must feature auto-retraining capabilities—intelligently detecting missing models and initiating training using purely real (non-synthetic) data, logging all actions persistently via the existing Database Manager. Finally, deprecated and unwanted files must be purged to maintain a pristine, trustworthy repository.

## Requirements

### Functional Requirements

1. **REQ-1**: Develop a unified CLI entry point (`dashboard.py` or `main.py`) powered by `Rich` that surfaces subcommands for `backtest`, `paper-trade`, `train`, and `status`.
2. **REQ-2**: Implement intelligent auto-retraining: the system must auto-detect missing models via the Model Registry and trigger background training jobs without freezing the CLI.
3. **REQ-3**: The pipeline must integrate seamlessly from data fetching (`core/data/data_pipeline.py`) to inference (`core/models/predictor.py`), guaranteeing reliable end-to-end execution.
4. **REQ-4**: Persistently log detailed backtesting and paper trading stats to a database (via `MARK5DatabaseManager`) so runs can be resumed or verified.

### Non-Functional Requirements

1. **REQ-5**: Extreme Reliability ("Toyota Engine"): All errors must fail gracefully with descriptive prompts rather than hard crashing.
2. **REQ-6**: Zero synthetic data: Enforce strict checks that prohibit the use of synthetic data across the entire pipeline.
3. **REQ-7**: Eliminate redundancy: Ensure proper Python imports and remove duplicate standalone scripts/folders.

### Constraints

- The dashboard must run smoothly on a local workstation environment.
- Must be a monolithic architecture avoiding distributed overhead (like network ports/servers) unless necessary.

## Approach

### Selected Approach

**Centralized Rich CLI Monolith**

We will implement a single `main.py` entry point powered by `Rich` and `Typer` that imports the core backend (`core/data`, `core/models`, `core/trading`). The dashboard will execute long-running tasks like auto-training inside background threads with DB-backed state management, while redundant standalone scripts are aggressively purged.

[Single `main.py` with `Rich`] *(considered: Hybrid FastAPI server — rejected because it adds network port overhead which violates your 'Local Workstation' preference; Modular CLI Package — rejected because it fragments commands, preventing a unified dashboard feel)* — *[Chosen to provide a beautiful, consolidated dashboard interface that seamlessly centralizes backtesting, paper-trading, and detailed stats without breaking the monolithic simplicity.]* Traces To: REQ-1, REQ-3, REQ-7

[Background Threads for Training] *(considered: Blocking UI during execution — rejected because it freezes the dashboard; Async task queues via Redis/Celery — rejected because it violates the 'monolithic local execution' constraint)* — *[Ensures the dashboard stays responsive and 'Toyota reliable', allowing users to view logs while auto-training safely happens in the background.]* Traces To: REQ-2, REQ-5

[Graceful Fallback & Prompting for Missing Data] *(considered: Hard crash on missing data — rejected because it is brittle; Silent auto-fetching — rejected because it hides state from the trader)* — *[Chosen to ensure extreme reliability, notifying the user via the rich dashboard to manually approve data fetches or retraining to prevent trading on stale or zero data.]* Traces To: REQ-5

### Alternatives Considered

#### Modular CLI Package
- **Description**: Splitting tools into multiple distinct CLI commands (e.g., `mark5-train`).
- **Pros**: Excellent separation of concerns.
- **Cons**: Slower to switch contexts, lacks a single "dashboard" feel.
- **Rejected Because**: You explicitly requested a centralized "A to Z" dashboard rather than isolated CLI scripts.

#### Hybrid Web Backend + Thin CLI
- **Description**: Expose the pipeline as a local FastAPI server. The CLI acts as a thin HTTP client to trigger tasks.
- **Pros**: True background execution, easy web UI later.
- **Cons**: Over-engineered for a local workstation, port management overhead.
- **Rejected Because**: It adds unnecessary port management and architectural bloat for local execution.

### Decision Matrix

| Criterion | Weight | App 1: Monolithic CLI | App 2: Modular CLI | App 3: Web + Thin CLI |
|-----------|--------|-----------------------|--------------------|-----------------------|
| Centralization & No Redundancy | 30% | 5: Perfect single entry point | 3: Still separate commands | 4: Centralized server |
| "Toyota Engine" Reliability | 30% | 5: Direct DB access, less moving parts | 5: Simple native execution | 2: Network/port dependencies |
| Developer Ergonomics (Local) | 20% | 5: Rich dashboard is highly ergonomic | 3: Standard terminal output | 2: Requires running a background server |
| Ease of Auto-Retraining | 20% | 4: Needs threading inside CLI | 2: Harder across distinct commands | 5: Easy via async web workers |
| **Weighted Total** | | **4.8** | **3.4** | **3.2** |

## Architecture

### Component Diagram

```text
[User Terminal / Local Workstation]
       | (Subcommands: train, backtest, paper-trade)
[Dashboard.py (Typer + Rich GUI)]
       | (Direct Imports)
       +---> [core.data.data_pipeline] (Fetches non-synthetic data A-to-Z)
       |
       +---> [core.models.features / predictor] (Inference engine, auto-detects missing models)
       |
       +---> [core.models.training.trainer] (Auto-retraining via background threading)
       |
       +---> [core.trading.simulator] (Backtesting & Paper Trading engine)
       |
[MARK5DatabaseManager (SQLite)] (Persistent storage of trades, metrics, logs, run states)
```

### Data Flow

When a user launches `dashboard.py paper-trade`, the system checks `MARK5DatabaseManager` and the Model Registry for active models. If models are missing, the UI gracefully prompts the user to auto-retrain. If accepted, `trainer.py` runs in a background thread while the main thread uses `Rich` live layouts to stream logs. During trading/backtesting, data is fetched via `data_pipeline.py` (strictly real, non-synthetic data) and passed through `features.py` to `predictor.py`. All results (backtest stats, paper trades) are written back to SQLite, which the dashboard queries to display real-time terminal tables.

### Key Interfaces

```python
def check_models_ready(ticker: str) -> bool:
    # Verifies real models exist via registry; blocks synthetic fallbacks.
    pass

def trigger_background_training(ticker: str) -> None:
    # Dispatches training job to a background thread and streams progress.
    pass
```

[Synchronous `Rich` UI with background threading] — *[Chosen to satisfy the constraint of having a single monolithic CLI entry point while preventing the application from freezing during intensive tasks like backtesting or model training.]* Traces To: REQ-1, REQ-2

## Agent Team

| Phase | Agent(s) | Parallel | Deliverables |
|-------|----------|----------|--------------|
| 1     | coder    | No       | Scaffolding `main.py` entry point (Typer/Rich CLI interface). Integrating core logic imports and identifying standalone scripts for deletion. |
| 2     | data_engineer | No | Refactoring data pipelines (e.g., `data_pipeline.py`) to strictly enforce zero synthetic data policies and adapt the DB Manager for detailed stat persistence. |
| 3     | coder    | No       | Implementing background auto-training jobs that hook into `trainer.py` and `predictor.py`, safely failing/prompting on missing models. |
| 4     | code_reviewer | No | Verifying "Toyota engine" reliability, ensuring no loose scripts remain, and validating non-synthetic data flows. |

[Assigning `coder` to CLI generation] — *[Required because this task focuses predominantly on building a comprehensive user interface, routing arguments via Typer, and rendering Rich tables.]* Traces To: REQ-1
[Assigning `data_engineer` to database enhancements and pipeline checks] — *[Critical for meeting the rigid anti-synthetic data constraints and ensuring backtesting/paper trading results are reliably serialized to disk.]* Traces To: REQ-4, REQ-6

## Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| Import Breaking | HIGH | HIGH | The `coder` will carefully update imports system-wide as standalone scripts are deprecated. Thorough compilation checks (`python -m py_compile`) will run. Traces To: REQ-7 |
| Accidental Synthetic Data Usage | HIGH | MEDIUM | The `data_engineer` will enforce strict validation blocks (`assert not synthetic`) inside `data_pipeline.py`. Failing this check triggers a graceful prompt on the CLI. Traces To: REQ-6 |
| Main Thread Blocking | MEDIUM | HIGH | Long-running training jobs executed inside the CLI could freeze the Rich display. This is mitigated by isolating auto-training and backtesting logic inside Python `threading` or `multiprocessing` pools. Traces To: REQ-2 |
| Database Locking | LOW | MEDIUM | Concurrent writes from backtesting or model logging could trigger SQLite locks. Mitigated by using connection pooling/timeouts inside `MARK5DatabaseManager`. Traces To: REQ-4 |

[Mitigating Thread Blocking] *(considered: AsyncIO event loops — rejected because existing `trainer.py` and ML logic is synchronous and CPU-bound)* — *[Threads or separate processes are the most robust way to run heavy synchronous ML tasks behind a responsive UI.]* Traces To: REQ-1, REQ-2

## Success Criteria

1. **A Single Unified Entry Point**: A `dashboard.py` (or `main.py`) powered by Typer and Rich exists, allowing terminal commands like `backtest`, `paper-trade`, and `train` to execute fluidly.
2. **Zero Synthetic Data**: The entire A to Z pipeline mathematically guarantees that all data queried, processed, or modeled is authentic historical market data, failing gracefully otherwise.
3. **No Duplicate Logic**: Unnecessary scripts and scattered folders are completely removed or consolidated into the appropriate `core/` modules.
4. **Smart Auto-Retraining**: When attempting inference or backtesting on missing ticker models, the dashboard gracefully halts and prompts the user to launch a background training thread without crashing the UI.
5. **Persistent History**: All backtesting statistics and paper trades are fully integrated with the persistent `MARK5DatabaseManager`.
