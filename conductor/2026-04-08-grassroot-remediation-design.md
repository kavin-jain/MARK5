---
design_depth: deep
task_complexity: complex
---

# Design Document: MARK5 Grassroot Remediation (Post-Audit Fixes)

## 1. Problem Statement
The recent audit of the MARK5 codebase identified several high-priority findings that threaten system stability, performance, and security. Specifically:
- **Volatile State**: Core trading state is stored in `/tmp/`, risking total data loss on system reboot.
- **Performance Bottlenecks**: Missing database indexes on the trade journal and slow Python loops in fractional differentiation will cause system latency as trade volume scales.
- **Reporting Discrepancies**: `ExecutionEngine` returns average instead of fill prices, poisoning P&L accuracy.
- **Logic Risks**: Trend extensions lack safety caps, and Kite tokens are persisted insecurely in-place within `.env`.
- **Security Gaps**: Insecure CORS and hardcoded default credentials in configuration.

Remediation requires a "grassroot" approach to fix these issues at the architectural and implementation levels, ensuring the system is production-ready.

**Traces To**: Audit Report Findings (LOG-01, PERF-01, LOG-02, LOG-03, SEC-02, SEC-03, SEC-04).

**Key Decision**: Holistic Remediation — *Rationale: Directly addresses the user's requirement to solve issues from the "grass root level" rather than applying surface-level patches (Traces To: User Request).*

## 2. Requirements

**Functional Requirements**:
- **REQ-1 (Persistent State)**: Relocate `mark5_trader_state.json` from `/tmp/` to a persistent path (e.g., `data/state/`) and ensure the directory is created if missing.
- **REQ-2 (DB Optimization)**: Add SQL indexes to the `trade_journal` table on `stock`, `status`, and `timestamp` columns to ensure O(log N) query performance.
- **REQ-3 (Feature Vectorization)**: Replace the Python loop in `_frac_diff_ffd` with a vectorized `numpy.convolve` implementation to improve calculation speed by ~10-100x.
- **REQ-4 (Accurate Reporting)**: Modify `ExecutionEngine.close_position` to return the actual execution price from the broker/OMS instead of the average position price.
- **REQ-5 (Safety Caps)**: Implement a cap on "Trend Extension" TP adjustments in `autonomous.py` based on a multiple of ATR (e.g., max 3 extensions).
- **REQ-6 (Security Hardening)**: 
  - Restrict CORS `allow_origins` to localhost/configured domain.
  - Load DB credentials via `os.getenv` with safe fallbacks.
  - Disable in-place `.env` token updates; tokens should be managed via environment variables or a secure local cache.

**Non-Functional Requirements**:
- **Adherence to Conventions**: All changes must preserve the existing v8.0/v9.0 headers and changelog styles.
- **Thread Safety**: Ensure all state-modifying changes remain protected by existing locks.

**Traces To**: Audit Report Findings.

**Key Decision**: Vectorize via NumPy — *Rationale: Essential for fulfilling REQ-3 and ensuring the system can handle large watchlists without cycle latency (Traces To: PERF-02).*

## 3. Approach

**Selected Approach: Holistic Remediation (Approach 1)**

**Remediation Logic**:
1. **Persistence Strategy**:
   - Update `AutonomousTrader.__init__` to default the `state` path to `data/state/`.
   - Ensure `os.makedirs` is called on the parent directory of the state path.
2. **Database Strategy**:
   - Update `MARK5DatabaseManager.init_database` to include `CREATE INDEX` statements for `trade_journal`.
   - Use `IF NOT EXISTS` to ensure idempotency.
3. **Algorithm Strategy**:
   - Refactor `_frac_diff_ffd` in `features.py`. Use `np.convolve` with the calculated weights array instead of the nested for-loop.
4. **Execution Strategy**:
   - Update `ExecutionEngine.execute_order` to capture the return value from `oms.place_order` (if it returns price) or pass the fill price from `_on_fill` back to the result object.
5. **Security Strategy**:
   - Update `dashboard/main.py` CORS list.
   - Refactor `validators.py` or `config_manager.py` to prioritize `os.getenv` for `TimescaleConfig` password.

**Decision Matrix (Deep Depth)**:
| Criterion | Weight | Holistic Approach | Targeted Approach |
|-----------|--------|-------------------|-------------------|
| **Audit Alignment** | 40% | 5: Addresses root causes. | 3: Surface fixes. |
| **Stability** | 30% | 5: Persistent and vectorized. | 4: Solves state loss. |
| **Security** | 20% | 5: Env-var driven. | 2: Leaves .env mutation. |
| **Weighted Total** | | **4.7** | **3.3** |

**Traces To**: REQ-1 through REQ-6.

**Key Decision**: Use `IF NOT EXISTS` for all schema updates — *Rationale: Ensures that the database can be re-initialized or migrated without errors on existing installations (Traces To: REQ-2).*

## 4. Architecture

**Architecture Overview**:
The remediation plan modifies components across all three layers of the MARK5 architecture (Infrastructure, Core, Dashboard) to improve baseline health.

**Impacted Components**:
1. **Infrastructure (Data/Persistence)**:
   - `MARK5DatabaseManager`: Updated schema with performance indexes.
   - `KiteFeedAdapter`: Removed insecure `.env` mutation helper.
2. **Core (Trading/Analytics)**:
   - `AutonomousTrader`: Relocated state persistence; implemented Trend Extension safety caps.
   - `AdvancedFeatureEngine`: Vectorized calculations for high-performance inference.
   - `ExecutionEngine`: Fixed P&L reporting by surfacing actual fill prices.
3. **Dashboard (Gateway)**:
   - `dashboard/main.py`: Hardened CORS middleware.

**Data Flow Changes**:
- **Trader State**: Memory -> `data/state/mark5_trader_state.json` (Persistent) *(considered: SQLite state table — rejected because the existing JSON structure is simple and sufficient for current state volume).*
- **P&L Reporting**: OMS Fill Price -> `OrderResult` -> Trade Journal (Accurate).

**Traces To**: REQ-1 through REQ-6.

**Key Decision**: Relocate state to `data/state/` — *Rationale: Standard project directory structure ensures state is backed up and survives system temp file cleanup (Traces To: REQ-1).*

## 5. Agent Team

**Agent Composition**:
- **Maestro (Orchestrator)**: Coordinates the sequential remediation phases.
- **Coder (Remediation)**: Primary agent for implementing logic changes in `AutonomousTrader`, `ExecutionEngine`, and `AdvancedFeatureEngine`. (Traces To: REQ-1, REQ-3, REQ-4, REQ-5)
- **Data Engineer (Persistence)**: Handles SQLite schema updates and path management in `MARK5DatabaseManager`. (Traces To: REQ-2)
- **Security Engineer (Hardening)**: Implements CORS restrictions and credential handling in `dashboard/main.py` and `KiteFeedAdapter`. (Traces To: REQ-6)
- **Tester (Validation)**: Verifies that vectorized calculations match original outputs and that state persists correctly after process restart. (Traces To: All REQs)

**Key Decision**: Assign `Data Engineer` to schema changes — *Rationale: Ensures indexes are applied with high precision and idempotency (Traces To: REQ-2).* *(considered: using Coder for everything — rejected because specialized Data/Security agents provide higher-quality hardening).*

## 6. Risk Assessment

**Risk Table**:
| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| **Data Corruption during Indexing** | High | Low | Use `PRAGMA integrity_check` before applying indexes; use `IF NOT EXISTS`. |
| **Logic Mismatch in Vectorization** | Medium | Medium | Implement parallel unit tests comparing vectorized output vs. original loop output before replacing. |
| **Path Access Errors** | Medium | Low | Use `os.path.expanduser` or absolute project root paths; ensure parent dirs exist via `os.makedirs`. |
| **OMS API Breaking Change** | Low | Low | Review the `OrderResult` interface to ensure compatibility with all broker adapters. |

**Traces To**: REQ-1 through REQ-6.

**Key Decision**: Test vectorization against original loop — *Rationale: Essential to ensure that `np.convolve` handles edge cases (like short history) identically to the original implementation (Traces To: REQ-3).*

## 7. Success Criteria

**Measurable Goals**:
- **Persistent State**: After a system reboot (simulated by process kill/restart), the trader resumes with all `active_signals` intact from `data/state/`. (Traces To: REQ-1)
- **DB Performance**: Queries on `trade_journal` for specific stocks or OPEN status show `EXPLAIN QUERY PLAN` usage of the new indexes. (Traces To: REQ-2)
- **Feature Accuracy**: Vectorized `_frac_diff_ffd` produces values within `1e-9` precision of the original loop output. (Traces To: REQ-3)
- **Reporting Fidelity**: Trade journal entries for closed positions match the actual execution prices reported by the OMS log. (Traces To: REQ-4)
- **Safe Extensions**: TP adjustments for `STRONG_BULL` regimes never exceed 3 extensions or 5% price distance. (Traces To: REQ-5)
- **CORS Restricted**: Browser requests from non-localhost domains (e.g., malicious-site.com) are blocked by the dashboard. (Traces To: REQ-6)

**Traces To**: All Functional Requirements.

**Key Decision**: Use `1e-9` precision for vectorization test — *Rationale: Ensures that ML model inputs remain numerically stable despite the algorithmic change (Traces To: REQ-3).*
