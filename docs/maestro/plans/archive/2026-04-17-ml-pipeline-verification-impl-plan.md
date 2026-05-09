---
title: "ML Pipeline Verification Implementation Plan"
design_ref: "docs/maestro/plans/2026-04-17-ml-pipeline-verification-design.md"
created: "2026-04-17T00:00:00Z"
status: "draft"
total_phases: 4
estimated_files: 5
task_complexity: "complex"
---

# ML Pipeline Verification Implementation Plan

## Plan Overview
- **Total phases**: 4
- **Agents involved**: `tester`, `refactor`, `coder`, `code_reviewer`
- **Estimated effort**: Medium-High, resolving structural race conditions, mathematical leakage, and orchestrating the complete audit suite.

## Dependency Graph
```text
[Phase 1: Mission-Critical Test Suite] (tester)
                 |
                 v
[Phase 2: Deep Structural Refactor] (refactor)
                 |
                 v
[Phase 3: Self-Contained Orchestrator] (coder)
                 |
                 v
[Phase 4: Structural Review] (code_reviewer)
```

## Execution Strategy
| Stage | Phases | Execution | Agent Count | Notes |
|-------|--------|-----------|-------------|-------|
| 1     | Phase 1 | Sequential | 1 | Test boundaries |
| 2     | Phase 2 | Sequential | 1 | Core refactoring |
| 3     | Phase 3 | Sequential | 1 | Master script |
| 4     | Phase 4 | Sequential | 1 | Final review |

## Cost Estimation
| Phase | Agent | Model | Est. Input | Est. Output | Est. Cost |
|-------|-------|-------|-----------|------------|----------|
| 1 | tester | Flash | 1500 | 400 | $0.003 |
| 2 | refactor | Flash | 2000 | 600 | $0.004 |
| 3 | coder | Flash | 1500 | 500 | $0.003 |
| 4 | code_reviewer | Flash | 3000 | 200 | $0.004 |
| **Total** | | | **8000** | **1700** | **$0.014** |

## Phase 1: Mission-Critical Test Suite
### Objective
Define strict Pytest boundaries to mathematically prove the absence of data leakage before implementation begins.
### Agent: `tester`
### Parallel: No
### Files to Create
- `tests/test_cpcv_horizons.py` — Tests to enforce dynamic horizon linking
- `tests/test_feature_leakage.py` — Tests to enforce rolling window standardization isolation
### Validation
- `pytest tests/test_cpcv_horizons.py tests/test_feature_leakage.py` (Expected: failing initially)
### Dependencies
- Blocked by: None
- Blocks: 2

## Phase 2: Deep Structural Refactor
### Objective
Fix mathematical root causes of leakage and resolve parallel directory deletion race conditions.
### Agent: `refactor`
### Parallel: No
### Files to Modify
- `core/models/training/cpcv.py` — Dynamically couple `prediction_horizon`
- `core/models/training/trainer.py` — Implement UUID-based temp dirs under `models/tmp/`
### Validation
- `pytest tests/test_cpcv_horizons.py tests/test_feature_leakage.py` (Expected: passing)
### Dependencies
- Blocked by: 1
- Blocks: 3

## Phase 3: Self-Contained Orchestrator
### Objective
Create the master verification script to run tests, regenerate models safely, and output Markdown matrices.
### Agent: `coder`
### Parallel: No
### Files to Create
- `verify_mark5.py` — CLI orchestrator with `--fast` mode support.
### Validation
- `python verify_mark5.py --fast` (Expected: successful run and generation of `reports/verify_mark5_report.md`)
### Dependencies
- Blocked by: 2
- Blocks: 4

## Phase 4: Structural Review
### Objective
Review all modified files for safety, isolation, and 'Toyota Engine' compliance.
### Agent: `code_reviewer`
### Parallel: No
### Validation
- Review execution results.
### Dependencies
- Blocked by: 3
- Blocks: None

## File Inventory
| # | File | Phase | Purpose |
|---|------|-------|---------|
| 1 | `tests/test_cpcv_horizons.py` | 1 | Prevent leakage |
| 2 | `tests/test_feature_leakage.py` | 1 | Prevent leakage |
| 3 | `core/models/training/cpcv.py` | 2 | Fix mathematical isolation |
| 4 | `core/models/training/trainer.py` | 2 | Fix race conditions |
| 5 | `verify_mark5.py` | 3 | Verification orchestrator |

## Risk Classification
| Phase | Risk | Rationale |
|-------|------|-----------|
| 1 | LOW | Read-only test generation |
| 2 | HIGH | Modifies core ML engine math |
| 3 | MEDIUM | Creates orchestrator; risk of incomplete artifact cleanup |
| 4 | LOW | Read-only audit |

## Execution Profile
```
Execution Profile:
- Total phases: 4
- Parallelizable phases: 0 (in 0 batches)
- Sequential-only phases: 4
- Estimated parallel wall time: 0m
- Estimated sequential wall time: ~4 minutes

Note: Native subagents currently run without user approval gates.
All tool calls are auto-approved without user confirmation.
```