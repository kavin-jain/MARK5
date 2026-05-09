---
title: "ML Pipeline Verification"
created: "2026-04-17T00:00:00Z"
status: "draft"
authors: ["TechLead", "User"]
type: "design"
design_depth: "deep"
task_complexity: "complex"
---

# ML Pipeline Verification Design Document

## Problem Statement
The MARK5 machine learning system currently suffers from severe architectural flaws that compromise the authenticity of its reported results. The primary issue is a data leakage vulnerability stemming from a mismatch between the Combinatorial Purged Cross-Validation (CPCV) purging horizon (hardcoded to 24 bars) and the Triple Barrier labeling look-forward window (up to 70 bars) — *[Fixing this mathematical root cause ensures zero look-ahead bias, as selected in the Full Pipeline Audit]* Traces To: REQ-1.

Furthermore, the orchestration of model training (`trainer.py`) employs destructive directory operations that create race conditions, preventing the safe parallelization of training across the 150-stock universe. *(considered: sequential wrapper execution — rejected because it severely degrades performance on a 150-stock universe)* Traces To: REQ-2.

To restore trust and achieve a 'Toyota Engine' standard of reliability, we must execute a Deep Structural Refactor. This requires mathematically linking the CPCV horizons to the feature engineering bounds, isolating the rolling standardization logic to strictly post-split datasets, and refactoring file I/O to use UUID-based temporary directories for safe concurrent execution. Finally, to prove this integrity unequivocally to stakeholders, a self-contained `verify_mark5.py` orchestrator must be built to enforce CPU-only determinism, run rigorous Pytest assertions against these mathematical boundaries, and generate a transparent Markdown analytics matrix — *[A single master script satisfies the requirement for a portable, highly impressive executive review package]* Traces To: REQ-3, REQ-4, REQ-5, REQ-6.

## Requirements

### Functional Requirements
1. **REQ-1**: The system must dynamically link the CPCV `prediction_horizon` to the labeling look-forward window (max 70 bars) to guarantee mathematical isolation — *[Ensures absolute authenticity of the 'Toyota Engine' backtest metrics by proving zero look-ahead bias]*
2. **REQ-2**: The orchestration layer (`trainer.py`) must use isolated, UUID-based temporary directories for model artifact generation to enable safe parallel execution across the 150-stock universe. *(considered: enforcing strict sequential execution — rejected because a 150-stock CPU-only run would take an unacceptably long time)*
3. **REQ-3**: The `verify_mark5.py` script must support a `--fast` mode flag to run the full audit pipeline on just the Top 5 elite stocks for rapid executive review — *[Balances the need for deep verification with pragmatic time constraints]*
4. **REQ-4**: The `verify_mark5.py` orchestrator must automatically invalidate `mark5.db` metrics and halt deployment if any mathematical boundary leakage test fails — *[Satisfies the 'Hard Gate' safety requirement]*

### Non-Functional Requirements
1. **REQ-5**: The verification suite must use `pytest` to aggressively target mission-critical paths (CPCV purging, feature standardization isolation).
2. **REQ-6**: The audit execution must be forced to CPU-only to eliminate GPU floating-point non-determinism, ensuring 100% reproducibility.

### Constraints
- The final audit result must be generated natively as a static Markdown matrix in the `reports/` directory without requiring external dashboard UI dependencies.
- Modifications to the core ML ensemble weights or feature selection must be avoided; the focus is entirely on structural data integrity.

## Approach

### Selected Approach
**The Deep Structural Refactor**
This approach tackles the mathematical root causes of leakage and race conditions by refactoring the core logic of `cpcv.py` and `trainer.py`. We will implement dynamic coupling between the prediction horizon and the label window, and introduce UUID-based temporary directory management for artifact creation. This is wrapped in a self-contained `verify_mark5.py` test harness that leverages Pytest for mission-critical validation — *[Chosen because it provides a permanent 'Toyota Engine' solution that enables safe, high-speed parallelization without accumulating technical debt]* Traces To: REQ-1, REQ-2, REQ-6.

### Alternatives Considered
#### The Defensive Integrity Wrapper
- **Description**: Enforces safe sequential execution and hardcoded safety margins entirely via an orchestration wrapper, avoiding changes to core ML logic.
- **Pros**: Rapid implementation; achieves 100% safety.
- **Cons**: Sequential execution forces extremely long runtimes for the 150-stock universe; hardcoded margins are brittle.
- **Rejected Because**: Fails to address the underlying race conditions and creates technical debt through brittle hardcoding.

#### The Post-Mortem Shadow Audit
- **Description**: Leaves current code untouched and verifies authenticity via independent historical replay scripts.
- **Pros**: Zero risk to existing `MARK5MLTrainer` code.
- **Cons**: Ignores race conditions; wastes compute generating broken models before catching them.
- **Rejected Because**: It acts as a stopgap rather than a permanent fix.

### Decision Matrix
| Criterion | Weight | Deep Refactor | Defensive Wrapper | Shadow Audit |
|-----------|--------|---------------|-------------------|--------------|
| Leakage Eradication | 35% | 5: Fixes mathematical root cause | 4: Safe but brittle hardcoding | 2: Only catches leaks post-facto |
| Execution Reliability | 25% | 5: Fixes race conditions for safe parallel runs | 3: Enforces slow sequential safety | 1: Ignores race conditions |
| Reproducibility | 20% | 5: Clean architecture ensures 100% fidelity | 5: Wrapper enforces CPU-only | 4: Shadow replay verifies fidelity |
| Implementation Speed | 20% | 2: Complex surgical changes required | 4: Fast orchestration patch | 5: Scripts only, no core changes |
| **Weighted Total** | | **4.40** | **3.95** | **2.75** |

## Architecture

### Component Diagram
```text
[verify_mark5.py (Orchestrator)]
       |
       |---[Pytest Suite (tests/)]
       |       |--> Validates cpcv.py boundaries
       |       |--> Validates features.py standardization
       |
       |---[MARK5MLTrainer (Parallel Workers)]
               |--> Uses UUID-based temp dirs under models/tmp/
               |--> Integrates cpcv.py with dynamic horizon
               |--> Commits to models/{ticker}/ on success
```

### Data Flow
1. Initiation: User triggers `verify_mark5.py` (with optional `--fast` flag).
2. Verification: The script executes the mission-critical Pytest suite. If tests fail, execution halts and metrics are invalidated — *[Enforces the Hard Block requirement to guarantee authenticity]*
3. Training: The script dispatches parallel `trainer.py` subprocesses configured for CPU-only execution — *[Eliminates GPU floating-point non-determinism]*
4. Artifact Isolation: Each trainer instance writes to a uniquely identified temporary directory (e.g., `models/tmp/<uuid>`) — *[Solves the race condition vulnerability during concurrent execution]*
5. Consolidation: Upon successful validation of an artifact, the orchestrator atomically moves it to the final `models/<ticker>` destination.
6. Reporting: A static Markdown analytics matrix is synthesized from the results and saved to the `reports/` directory.

### Key Interfaces
```python
def generate_cpcv_splits(data: pd.DataFrame, label_horizon: int) -> List[Tuple]:
    # Returns purged and embargoed index arrays dynamically linked to the label horizon

def audit_and_verify(fast_mode: bool = False) -> None:
    # Master orchestrator function in verify_mark5.py
```

## Agent Team
| Phase | Agent(s) | Parallel | Deliverables |
|-------|----------|----------|--------------|
| 1     | `tester`   | No       | Mission-critical Pytest suite targeting CPCV boundaries and feature standardization leakage. |
| 2     | `refactor` | No       | Updated `cpcv.py` and `trainer.py` implementing dynamic horizons and UUID-based artifact isolation. |
| 3     | `coder`    | No       | The `verify_mark5.py` orchestrator script with `--fast` mode and Markdown analytics generation. |
| 4     | `code_reviewer` | No  | Final structural review to ensure zero look-ahead bias and 'Toyota Engine' compliance. |

## Risk Assessment
| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| Test suite fails to catch a secondary data leakage vector | HIGH | LOW | The `code_reviewer` agent will perform a dedicated analysis of the pandas rolling window boundaries and datetime indexes across `features.py` and `cpcv.py` — *[Mitigates the risk of 'fake' alpha slipping through the mathematical boundaries]* |
| Performance degradation due to CPU-only execution constraint | MEDIUM | HIGH | Implementation of the `--fast` flag ensures the Top 5 elite stocks can be audited rapidly — *[Directly addresses the trade-off of forcing determinism by providing a rapid executive review path]* |
| UUID temp directories leave orphaned artifacts on failure | LOW | MEDIUM | The `verify_mark5.py` orchestrator will implement strict `try/finally` cleanup blocks to guarantee garbage collection — *[Ensures 'no dangling threads' are left behind]* |

## Success Criteria
1. Execution of `pytest` natively succeeds, running at least 3 new mission-critical assertions that specifically validate CPCV purging horizons and standardisation boundary isolation.
2. Running `python verify_mark5.py --fast` successfully completes a parallel, CPU-only regeneration of the Top 5 elite stocks without triggering directory deletion race conditions.
3. The orchestrator successfully synthesizes and outputs a static Markdown analytics matrix to `reports/verify_mark5_report.md`, proving the authenticity of the generated models.
4. Execution of `verify_mark5.py` demonstrably halts, throwing an error, if any Pytest assertion in the leakage suite fails.
