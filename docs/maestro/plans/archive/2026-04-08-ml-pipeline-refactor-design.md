---
title: "ML Pipeline Refactor & Institutional Validation"
created: "2026-04-08T00:00:00Z"
status: "draft"
authors: ["TechLead", "User"]
type: "design"
design_depth: "deep"
task_complexity: "complex"
---

# ML Pipeline Refactor & Institutional Validation Design Document

## Problem Statement

The current machine learning pipeline and Temporal Convolutional Network (TCN) models located in `@core/models/` require a comprehensive audit to eliminate potential data leakage, overfitting, and accumulated technical debt (including unwanted/deprecated files). The existing setup lacks the institutional-level validation rigor—specifically Combinatorial Purged Cross-Validation (CPCV) and the Deflated Sharpe Ratio (DSR)—necessary to mathematically prove a model's robustness and ensure zero overfitting before live deployment. The project demands an end-to-end refactoring to make the pipeline "diamond solid," ensuring every line of code is verified and accurate. Furthermore, models that fail this strict validation must be safely routed to a Shadow Deployment environment for out-of-sample monitoring rather than being summarily discarded or allowed to halt the pipeline entirely. The ultimate goal is a pristine, bug-free codebase with an unassailable training and evaluation methodology.

## Requirements

### Functional Requirements

1. **REQ-1**: The pipeline must comprehensively train and evaluate the Temporal Convolutional Network (TCN).
2. **REQ-2**: Implement Combinatorial Purged Cross-Validation (CPCV) to split the data while purging/embargoing to prevent data leakage.
3. **REQ-3**: Calculate the Deflated Sharpe Ratio (DSR) to mathematically validate model performance across multiple trials.
4. **REQ-4**: Failed models (that do not cross the DSR threshold) must be automatically routed to a Shadow Deployment mode rather than halting the pipeline.
5. **REQ-5**: The codebase must implement robust experiment tracking (e.g., MLflow/WandB) and enforce reproducible random seeds.

### Non-Functional Requirements

1. **REQ-6**: "Diamond solid" code quality, with every single file and line in `@core/models/` verified for accuracy and zero bugs.
2. **REQ-7**: Zero tolerance for overfitting.
3. **REQ-8**: Institutional-level robustness is prioritized heavily over training speed/latency.

### Constraints

- **REQ-9**: The refactor must utilize the current deep learning framework (e.g., PyTorch/TensorFlow) without migrating to new, specialized institutional frameworks.
- **REQ-10**: All deprecated, unused, or "unwanted" files in `@core/models/` must be conservatively isolated or removed to ensure a pristine active pipeline boundary.

## Approach

### Selected Approach

**Institutional TCN Refactor & Small-Data Aware CPCV**

The project will execute a comprehensive refactor of `@core/models/`, decoupling feature engineering from model state. We will implement CPCV and DSR natively, but carefully adapt them for a limited dataset (~2000 days of 60-minute data).

[Native CPCV and DSR Adapted for Limited Data] *(considered: Standard deep-purging CPCV — rejected because it would excessively shrink the available 2000-day dataset)* — *[We will implement a small-data-aware CPCV, carefully balancing embargo/purge sizes and validation folds. This extracts maximum learning while mathematically preventing overfitting without starving the model of data.]* Traces To: REQ-2, REQ-3, REQ-7, REQ-9

[Decoupling feature engineering from model state] *(considered: Wrapping existing code in a validation overlay — rejected because it leaves legacy technical debt intact)* — *[This allows us to audit every single file, guaranteeing 'diamond solid' code quality, and perfects the feature pipeline to squeeze every ounce of signal from the limited data.]* Traces To: REQ-6

[Conservatively Isolating Unwanted Files] *(considered: Aggressive deletion of all unused code — rejected because you opted for conservative isolation)* — *[This ensures the active pipeline boundary is pristine while keeping older models/scripts safely sandboxed.]* Traces To: REQ-10

[Routing Failures to Shadow Mode] *(considered: Hard failing the pipeline — rejected because Shadow Mode is safer for tracking)* — *[This preserves analytical visibility and safely gathers more live out-of-sample performance data over time, which is crucial given the limited historical set.]* Traces To: REQ-4

### Alternatives Considered

#### Pragmatic Wrapper & Validation Overlay
- **Description**: Wrap existing training engine in CPCV without refactoring the features or internal model structures.
- **Pros**: Fast to implement, low risk of breaking existing logic.
- **Cons**: Leaves technical debt; does not fulfill the zero-bug "perfect codebase" requirement.
- **Rejected Because**: The user explicitly requested an audit of "every single code file and every line" to ensure zero scope of error.

#### Complete Rewrite using Institutional Frameworks
- **Description**: Port the logic to high-level frameworks like PyTorch Lightning + Ray Tune.
- **Pros**: Clean slate, standardizes execution.
- **Cons**: High regression risk, replaces currently working code completely.
- **Rejected Because**: The constraint is to maintain the current stack and only refactor the logic.

### Decision Matrix

| Criterion | Weight | Approach 1 (Refactor & CPCV) | Approach 2 (Overlay) | Approach 3 (Rewrite) |
|-----------|--------|------------------------------|----------------------|----------------------|
| **Overfitting Prevention** | 40% | 5: Native CPCV/DSR integration ensures deep data isolation. | 4: Validates output strictly, but internal leakage risk remains. | 5: Clean slate with standardized institutional validation. |
| **Code "Perfection"** | 30% | 5: Every line in `@core/models/` is audited and cleaned. | 2: Leaves legacy code structures largely in place. | 4: Clean code, but introduces heavy 3rd-party abstraction. |
| **Regression Risk** | 20% | 3: Moderate risk from refactoring active training loops. | 5: Lowest risk since core logic is untouched. | 1: High risk from rewriting the entire pipeline from scratch. |
| **Training Speed** | 10% | 2: Slower due to combinatorial folds. | 3: Slightly faster than App 1 due to less overhead. | 4: Framework parallelization offsets CPCV overhead. |
| **Weighted Total** | | **4.3** | **3.5** | **3.6** |

## Architecture

### Component Diagram

```text
[Zerodha Kite Connect Data (~2000 days, 60m)]
          |
[Feature Engineering Module] -> (Decoupled, pristine feature sets)
          |
[Small-Data CPCV Splitter] -> (Purges/embargoes tailored to preserve data volume)
          |
[TCN Training Engine] <---+-> [MLflow/WandB Tracking]
          |               |
[DSR Evaluation] <--------+
          |
   (Pass DSR?)
    /       \
  Yes        No
  /           \
[Live]      [Shadow Deployment]
```

### Data Flow

Raw 60-minute data flows from Zerodha into the read-only Feature Engineering Module. Standardized feature tensors are passed to the Small-Data CPCV Splitter, which calculates validation fold indices with purging/embargoing, preventing leakage. The TCN Engine trains over these indices, tracking hyperparameters and metrics. Finally, the DSR evaluates the output, routing passing models to Live inference and failing models to a tracked Shadow Deployment.

### Key Interfaces

```python
def feature_engineer(data_df: pd.DataFrame) -> torch.Tensor:
    # Strictly decoupled transformation
    pass

def generate_cpcv_splits(features: torch.Tensor, labels: torch.Tensor, n_folds: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    # Returns index masks enforcing embargo to prevent leakage
    pass
```

[Splitter returning index masks] — *[Chosen to ensure memory efficiency and to strictly enforce isolation between training/validation paths without risking data duplication errors.]*

[Decoupled Feature Engineering] *(considered: inline feature engineering inside training loops — rejected because it increases leakage risk)* — *[Ensures 'diamond solid' verification boundary where features can be unit tested entirely independently of the TCN.]* Traces To: REQ-6

## Agent Team

| Phase | Agent(s) | Parallel | Deliverables |
|-------|----------|----------|--------------|
| 1     | data_engineer | No       | Decoupled feature engineering module and Small-Data CPCV split indices. |
| 2     | coder    | No       | Refactored TCN Engine, DSR calculation logic, isolated legacy files, MLflow integration. |
| 3     | performance_engineer | No | Validation of CPCV execution performance to avoid pipeline bottlenecks. |
| 4     | code_reviewer | No | Strict 100% file audit across `@core/models/` for bug detection and leakage assertion. |

[Assigning `data_engineer`] — *[Required to properly construct the index masks and purge/embargo limits tailored specifically for the 60m frequency dataset without destroying the limited data pool.]*
[Assigning `code_reviewer`] *(considered: skipping dedicated review — rejected because you demanded 'no scope of error' and 'perfect' quality)* — *[Guarantees the zero-bug tolerance and perfect execution required for this pipeline.]* Traces To: REQ-6

## Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| Data Starvation via Purging | HIGH | MEDIUM | The `data_engineer` will optimize purge/embargo limits tailored to the 60m frequency to maximize data retention. Traces To: REQ-2, REQ-7 |
| Over-Engineering for "Perfection" | MEDIUM | HIGH | The `code_reviewer` will block progress if changes lack unit tests verifying zero leakage and unchanged baseline accuracy. Traces To: REQ-6 |
| Shadow Deployment Clutter | LOW | HIGH | Models routed to Shadow Mode will be aggressively sandboxed and logged to an isolated tracking scope. |

## Success Criteria

1. **Zero Data Leakage**: The decoupled feature engineering logic and the `Small-Data CPCV Splitter` mathematically enforce strict purging and embargoing, verified mechanically by programmatic tests.
2. **Deflated Sharpe Ratio Enforcement**: The `TCN Training Engine` successfully evaluates trained models against the calculated DSR threshold to ensure non-spurious performance.
3. **Shadow Mode Fallback**: Models failing the DSR threshold are seamlessly routed to a Shadow Deployment sandbox and tracked independently on MLflow/WandB.
4. **Pristine Source Tree**: Unwanted and deprecated legacy files in `@core/models/` are conservatively isolated out of the active training path, leaving only an actively verified, bug-free codebase.
5. **No Model Regressions**: The existing TCN functionality and baseline predictive accuracy are preserved or improved under the new, rigorous validation regime.

[Mathematical Proof Criteria] *(considered: anecdotal backtest passing — rejected because it doesn't guarantee the 'institutional level' accuracy demanded)* — *[A strict programmatic assertion of zero data leakage provides the 'diamond solid' foundation requested.]* Traces To: REQ-3, REQ-7
