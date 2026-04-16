---
title: "Institutional Meta-Ensemble Training Optimization"
created: "2026-04-09T14:00:00Z"
status: "draft"
authors: ["TechLead", "User"]
type: "design"
design_depth: "deep"
task_complexity: "complex"
---

# Institutional Meta-Ensemble Training Optimization Design Document

## Problem Statement

The current MARK5 training pipeline is producing models with low predictive power (AUC ~0.50) and near-zero statistical significance (DSR ~0.0), causing all models to fail the production gate and remain in Shadow Mode. Two major technical flaws exist: (1) the TCN architecture is not integrated into the automated training loop, and (2) the current ensemble meta-learner produces negative weights, indicating unstable and anti-predictive base-model contributions. To achieve institutional-level "world's best" performance, the system requires a shift from raw binary labeling to a Meta-Labeling architecture, full 4-model ensemble integration (XGB, LGB, CAT, TCN) with Non-Negative Stacking, and rigorous testing across the 10-stock universe to ensure "diamond solid" reliability.

## Requirements

### Functional Requirements

1. **REQ-1**: Implement **Meta-Labeling** (Marcos Lopez de Prado). Train a primary model to identify trends, then use the ensemble to predict the probability of that trade being successful (binary classification of profit vs. loss).
2. **REQ-2**: Integrate **TCN** into the `MARK5MLTrainer`. It must train alongside the GBMs in every CPCV fold and contribute to the ensemble.
3. **REQ-3**: Implement **Non-Negative Stacking**. Replace the meta-learner with a constrained optimizer (e.g., Non-Negative Least Squares or a constrained SciPy optimizer) to ensure all model weights are ≥ 0.
4. **REQ-4**: Refactor the "Train All Universe" dashboard function to handle this two-stage training process sequentially for all 10 stocks.

### Non-Functional Requirements

1. **REQ-5**: Improved Metrics: Target an AUC > 0.60 and a DSR > 0.95 across the universe.
2. **REQ-6**: Reliability: Ensure the "Toyota engine" standard where training failures for one stock do not crash the entire universe loop.

### Constraints

- Must use purely real market data from Zerodha Kite Connect.
- Must preserve the `RobustModelRegistry` and `ModelVersionManager` infrastructure.

## Approach

### Selected Approach

**Approach 1: Institutional Meta-Ensemble**

The project will transform the `MARK5MLTrainer` into a two-stage pipeline. The first stage will train a simple trend-identifier (primary model). The second stage will train a 4-model ensemble (XGB, LGB, CAT, TCN) to perform **Meta-Labeling**—deciding if the primary signal should be taken.

[Implement Meta-Labeling] — *[This approach was chosen to maximize signal-to-noise ratio and reach the "world's best" predictive accuracy by separating trend identification from probability of success.]* Traces To: REQ-1, REQ-5

[Integrate TCN into CPCV] — *[Provides the deep learning temporal context requested, ensuring the TCN contributes its attention-based features to the final trade decision.]* Traces To: REQ-2

[Non-Negative Stacking] *(considered: Regularized Stacking — rejected because it still allows negative coefficients)* — *[Chosen to force institutional stability, ensuring every model in our "diamond solid" ensemble contributes positively or not at all.]* Traces To: REQ-3

### Alternatives Considered

#### Approach 2: TBM Refinement
- **Description**: Tighten Triple Barrier Method parameters and integrate TCN without meta-labeling.
- **Pros**: Lower implementation complexity.
- **Cons**: Less likely to resolve the AUC ~0.5 issue in noisy regimes.
- **Rejected Because**: The objective is "world's best" performance, which mandates the more robust meta-labeling methodology.

### Decision Matrix

| Criterion | Weight | Approach 1: Meta-Labeling | Approach 2: TBM Refinement |
|-----------|--------|---------------------------|----------------------------|
| **Predictive Edge (AUC)** | 40% | 5: Meta-labeling is state-of-the-art for noise. | 3: Tightening TBM has limits. |
| **Stability (Weights)** | 30% | 5: Non-negative stacking is rock solid. | 5: Non-negative stacking is rock solid. |
| **Feature Depth (TCN)** | 20% | 5: Native TCN integration in all folds. | 4: Standard ensemble integration. |
| **Complexity Risk** | 10% | 2: Requires two-stage training logic. | 4: Direct refactor of current loop. |
| **Weighted Total** | | **4.7** | **3.8** |

## Architecture

### Component Diagram

```text
[Zerodha Kite Market Data]
          |
[Financial Engineer] -> [Primary Model (Trend ID)] -> [Meta-Labeling Targets]
                                                              |
[4-Model CPCV Suite] <----------------------------------------+
  |-- XGBoost
  |-- LightGBM
  |-- CatBoost
  |-- TCN (Deep Learning)
          |
[Non-Negative Meta-Learner] -> [Final Signal probability]
          |
[DSR Production Gate] -> [Live / Shadow Deployment]
```

### Data Flow

1. The app starts, clears the screen, and renders the ASCII banner.
2. The `Main Menu` is displayed as a Rich `Table` within a `Panel`.
3. The user enters a number (e.g., `1` for Training).
4. The `Menu Switcher` identifies the choice and displays a `Sub-menu`.
5. For each ticker in the universe, the `MARK5MLTrainer` first trains a primary trend identification model.
6. The success or failure of these primary signals is recorded as "Meta-Labels" (1 for profit, 0 for loss).
7. A 4-model ensemble (XGB, LGB, CAT, TCN) is then trained via 28-fold CPCV to predict these Meta-Labels.
8. A Non-Negative Stacking meta-learner combines the probabilities.
9. Final metrics are evaluated against the DSR gate.

### Key Interfaces

```python
def train_meta_labels(ticker: str, data: pd.DataFrame):
    # Stage 1: Trend Identification
    # Stage 2: Train Ensemble on success/failure of Stage 1
    pass

class NonNegativeStacking:
    # Constrained optimizer ensuring weights[i] >= 0
    pass
```

[Two-Stage CPCV] — *[Necessary to correctly validate meta-labeling without lookahead bias, ensuring the "diamond solid" reliability requested.]*

[Non-Negative Stacking] — *[Uses SciPy's `nnls` or similar constrained solver to guarantee that no model in the ensemble can negatively impact the prediction.]* Traces To: REQ-3

## Agent Team

| Phase | Agent(s) | Parallel | Deliverables |
|-------|----------|----------|--------------|
| 1     | data_engineer | No | Refactored `financial_engineer.py` with Meta-Labeling target generation logic. |
| 2     | coder | No | Refactored `trainer.py` with TCN integration, two-stage CPCV, and Non-Negative Stacking. |
| 3     | performance_engineer | No | Optimized TCN training parameters and GPU utilization metrics. |
| 4     | code_reviewer | No | Final audit of the meta-labeling pipeline for reliability and "Toyota engine" standards. |

## Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| Computational Explosion | HIGH | HIGH | The `performance_engineer` will optimize TCN epochs and batch sizes, utilizing GPU-native training. |
| Meta-Label Data Sparsity | MEDIUM | MEDIUM | The `data_engineer` will tune the primary trend identification threshold to ensure sufficient samples. Traces To: REQ-1 |
| Artifact Bloat | LOW | LOW | Only the most performant "Passing Gate" models will be kept in full detail. |

## Success Criteria

1. **Meta-Labeling Active**: The `MARK5MLTrainer` successfully identifies trend signals and trains the ensemble to predict their success.
2. **Positive Coefficients Only**: Every meta-learner weight saved in `weights.json` is ≥ 0.
3. **TCN Contribution**: The TCN model is verified to train and provide OOF predictions in all 28 CPCV folds.
4. **Universe Loop Verified**: The "Train All Universe" dashboard function successfully iterates through 10 stocks, logging metrics.
5. **Statistical Significance**: At least 3 of the 10 stocks pass the production gate (DSR > 0.95, P(Sharpe>1.5) > 40%).
