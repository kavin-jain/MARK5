## 2026-05-01 — Sharpe, Calmar Ratio Fix and Silent Failures

**Bug:** Sharpe ratio was missing the risk-free rate. Calmar ratio was missing altogether from outputs. Silent `except: pass` in paper execution adapter were eating runtime tick errors.
**Root cause:** Initial implementation optimized for naive relative metrics rather than true mathematical targets, leading to inflated Sharpe ratios. The paper adapter used naive error swallowing for missing keys without accounting for broader operational logging.
**Fix:** Deducted `0.065 / 252` from daily returns before calculating Sharpe and applied it consistently across evaluation systems. Added proper logging to `paper_exec.py`.
**Guard:** `assert -2 < sharpe < 5, f"Sharpe {sharpe:.2f} outside plausible range"` added to the Sharpe calculation logic.
