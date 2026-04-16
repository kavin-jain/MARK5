---
title: "Institutional System Remediation & ISE API Integration"
created: "2026-04-10T10:00:00Z"
status: "approved"
authors: ["TechLead", "User"]
type: "design"
design_depth: "deep"
task_complexity: "complex"
---

# Institutional System Remediation & ISE API Integration Design Document

## Problem Statement

The current MARK5 system is in a fragmented and unstable state, suffering from "broken everything"—including naming mismatches in the primary dashboard (`backtesting_menu` vs `backtest_menu`), redundant data-fetching logic duplicated between the `MARK5MLTrainer` and the `DataPipeline`, and a lack of high-fidelity market intelligence from the Indian Stock Exchange. These architectural gaps result in brittle execution where pipelines fail and imports break. To achieve "Toyota engine" reliability and a professional "A to Z" package, the system requires total centralization of its data and execution layers, the full integration of the Indian Stock Exchange API (OHLCV, News, Analyst Recs, etc.), and a rigorous 10-iteration automated stress-testing cycle to guarantee a bug-free end-user experience.

## Requirements

### Functional Requirements

1. **REQ-1 (Unified Interface)**: Refactor `dashboard.py` to fix all naming mismatches and consolidate all system entry points (Train, Backtest, Trade, Status, ISE Intelligence) into a single, numeric-driven Rich menu loop.
2. **REQ-2 (ISE Power-Train)**: Implement a complete `ISEAdapter` mapping to ALL documented endpoints (Trending, 52-Week High/Low, Most Active, Mutual Funds, Price Shockers, Commodities, Analyst Recs, Historical Data/Stats).
3. **REQ-3 (ISE Market Intelligence)**: Add a dedicated "ISE Intelligence" menu to the dashboard for high-fidelity extraction of News, Bulk News, and Fundamental data.
4. **REQ-4 (Total Centralization)**: Refactor `MARK5MLTrainer` and other modules to strictly route all market data requests through the central `DataPipeline`, eliminating direct `DataProvider` calls.
5. **REQ-5 (Aggressive Purge)**: Identify and permanently delete all redundant experimental scripts and duplicate logic artifacts identified during grounding.

### Non-Functional Requirements

1. **REQ-6 (Toyota Reliability)**: Ensure the system is "well-oiled" and trustworthy, handling all missing models, data gaps, or API errors with graceful user prompts.
2. **REQ-7 (Iterative Perfection)**: Develop an automated `tests/dashboard_health_check.py` that simulates every menu/feature path and run it for 10 full cycles to assert perfection.

### Constraints

- Strictly prohibited: Synthetic data usage.
- Must run in a standard terminal environment using the `rich` library.

## Approach

### Selected Approach

**The Unified "A to Z" Power-Train**

We will execute a comprehensive structural refactor that establishes the `DataPipeline` as the project's master data router. All redundant fetchers will be deleted, and the dashboard will be rebuilt as a numeric-entry control center. A dedicated `ISE Intelligence` menu is added, and a 10-cycle automated stress test harness is built to verify every code path.

[Refactor to Master `DataPipeline`] — *[This approach was chosen to enforce a single source of truth for market data, ensuring that if a data fix is applied in the pipeline, it propagates to training, prediction, and the dashboard simultaneously.]* Traces To: REQ-4

[Implement Full ISE documentation suite] — *[Provides the necessary non-price alpha needed for high-conviction institutional-level trading.]* Traces To: REQ-2, REQ-3

[10-Iteration Automated Stress Test] — *[Guarantees "Toyota engine" reliability by mathematically verifying that every bug fixed in cycle 1 stays fixed in cycle 10.]* Traces To: REQ-7

[Aggressive Purge of Redundancy] — *[Addresses the request for "no redundancy" by ruthlessly removing technical debt and loose scripts.]* Traces To: REQ-5

### Alternatives Considered

#### Approach 2: The Plug-and-Play Hub
- **Description**: Focuses on fixing the dashboard bugs and integrating the ISE API as a separate service layer. Consolidation of the `MARK5MLTrainer` data logic is done via a "wrapper" rather than a full internal refactor.
- **Pros**: Faster to deliver the new features.
- **Cons**: Leaves some internal redundancy in the "A to Z" pipeline. Less rigorous validation.
- **Rejected Because**: Does not achieve the requested "no redundancy" and "well-oiled" reliability goal.

### Decision Matrix

| Criterion | Weight | Approach 1 (Unified) | Approach 2 (Hub) |
|-----------|--------|----------------------|---------------------------|
| **Toyota Reliability** | 40% | 5: Automated 10-cycle stress tests ensure perfection. | 3: Manual testing has gaps. |
| **No Redundancy** | 30% | 5: Total centralization and aggressive purge. | 2: Wrappers leave legacy debt. |
| **Feature Richness** | 20% | 5: Full ISE documentation coverage. | 4: Core ISE endpoints only. |
| **Speed of Delivery** | 10% | 2: Significant refactor required. | 5: Targeted fixes only. |
| **Weighted Total** | | **4.5** | **3.2** |

## Architecture

### Component Diagram

```text
[Trader CLI (Numeric Input)]
       |
[dashboard.py (Rich Hub)] <---> [ISE Intelligence Menu]
       |                               |
       |-- (Direct Imports)            |-- (ISEAdapter)
       |                               |
[Master DataPipeline] <----------------+
       |
       |-- [Adapters: Kite, ISE, FII]
       |-- [Cache: Parquet/SQL]
       |
[Core Engines (Refactored)]
       |-- [MARK5MLTrainer] (Calls Pipeline)
       |-- [MARK5Predictor] (Calls Pipeline)
       |-- [RobustBacktester] (Calls Pipeline)
```

### Data Flow

The system initializes by configuring a root `FileHandler` for logging. When `dashboard.py` requests data (e.g., for backtesting), it calls `DataPipeline.get_data()`. The pipeline checks for local cache; if missing, it routes to either `KiteAdapter` (for live OHLCV) or the new `ISEAdapter` (for historical stats/fundamentals). For the **ISE Intelligence Menu**, the `ISEAdapter` specifically triggers endpoints for trending stocks or news, returning a structured JSON that the dashboard renders into a `Rich.Table`. Every action is tracked in the health check test suite.

### Key Interfaces

```python
class ISEAdapter:
    def get_market_intelligence(self, endpoint: str, **params) -> Dict:
        # Maps Documentation Endpoints 1-14 to REST calls
        pass

class DataPipeline:
    def get_market_data(self, ticker: str, source: str = 'kite') -> pd.DataFrame:
        # Centralized router for all training and dashboard needs
        pass
```

[Master Data Router] *(considered: local function calls — rejected because they prevent cross-module reuse; global state — rejected because it risks data corruption)* — *[Necessary to satisfy the 'no redundancy' constraint, ensuring no two files implement the same fetch logic.]* Traces To: REQ-4

## Agent Team

| Phase | Agent(s) | Parallel | Deliverables |
|-------|----------|----------|--------------|
| 1     | data_engineer | No       | Feature-complete `ISEAdapter` and refactored `DataPipeline` master router. |
| 2     | coder    | No       | Refactored `dashboard.py` (menu bugs + ISE UI) and `MARK5MLTrainer` (pipeline integration). |
| 3     | refactor | No       | Verified deletion of all redundant scripts and fetchers in `deprecated/scripts/`. |
| 4     | tester   | No       | Automated `tests/dashboard_health_check.py` and 10-iteration stress test report. |
| 5     | code_reviewer | No  | 100% End-to-End audit of "Toyota engine" logic and non-synthetic flows. |

[Assigning `data_engineer`] — *[Required for the complex task of mapping 14 ISE endpoints into a robust, centralized data source.]* Traces To: REQ-2
[Assigning `tester`] — *[Addresses the specific request for 5-10 testing iterations to ensure absolute perfection.]* Traces To: REQ-7

## Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| Kite/ISE API Downtime | HIGH | MEDIUM | `data_engineer` will implement robust fallback to local cache; `tester` will use mock responses for stress cycles. |
| Circular Imports | MEDIUM | HIGH | `coder` will use Dependency Injection or move shared constants to `core/utils/` to break cycles. Traces To: REQ-4 |
| Data Incompatibility | LOW | MEDIUM | `data_engineer` will standardize all adapter outputs into a consistent format. |
| Regressions from Purge | MEDIUM | LOW | `refactor` will only delete files with zero references in the new centralized core. Traces To: REQ-5 |

## Success Criteria

1. **Unified Control Center**: `dashboard.py` runs as a numeric-entry menu loop with zero naming mismatches.
2. **ISE Market Intelligence**: `ISE Intelligence` menu provides detailed data for all 14 Documented Endpoints.
3. **Zero Redundancy**: All market data flows through the master `DataPipeline`.
4. **Factory Pristine Repo**: All redundant scripts and logic artifacts are permanently deleted.
5. **Toyota Reliability**: `tests/dashboard_health_check.py` passes 10 consecutive full-feature iterations.
