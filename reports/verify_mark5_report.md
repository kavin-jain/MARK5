# MARK5 Verification Report
Generated: 2026-05-02 17:52:37
Duration: 1410.57 seconds

## 1. Test Suite Results
Status: ✅ PASSED

<details>
<summary>Pytest Output</summary>

```
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.0.1, pluggy-1.6.0 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: /home/lynx/Documents/MARK5
configfile: pytest.ini
plugins: mock-3.15.1, anyio-4.11.0, cov-7.0.0
collecting ... collected 4 items

tests/test_cpcv_horizons.py::test_purging_horizon_covers_label_lookahead PASSED [ 25%]
tests/test_cpcv_horizons.py::test_cpcv_split_isolation_math PASSED       [ 50%]
tests/test_feature_leakage.py::test_feature_engineering_isolation_across_splits PASSED [ 75%]
tests/test_feature_leakage.py::test_global_standardization_leakage PASSED [100%]

=============================== warnings summary ===============================
tests/test_feature_leakage.py::test_feature_engineering_isolation_across_splits
tests/test_feature_leakage.py::test_feature_engineering_isolation_across_splits
  /home/lynx/Documents/MARK5/core/models/features.py:129: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
    df['amihud_ratio'] = (c.pct_change().abs() / (v * c + epsilon)).rolling(20).median() * 1e9

tests/test_feature_leakage.py::test_feature_engineering_isolation_across_splits
tests/test_feature_leakage.py::test_feature_engineering_isolation_across_splits
  /home/lynx/Documents/MARK5/core/models/features.py:150: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
    df['vol_adj_mom'] = c.pct_change(20) / (c.pct_change().rolling(20).std() * np.sqrt(20) + epsilon)

tests/test_feature_leakage.py::test_feature_engineering_isolation_across_splits
tests/test_feature_leakage.py::test_feature_engineering_isolation_across_splits
  /home/lynx/Documents/MARK5/core/models/features.py:158: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
    df['mfi_div'] = c.pct_change(mfi_period) - (mfi.pct_change(mfi_period) / 100.0)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
======================== 4 passed, 6 warnings in 3.04s =========================

```

</details>

## 2. Model Regeneration Summary
| Symbol | Status | Version | CPCV P(Sharpe>1.5) | Mean Sharpe | Passes Gate | Reason/Error |
|--------|--------|---------|-------------------|-------------|-------------|--------------|
| HDFCBANK.NS | success | v1 | 0.0% | -2.96 | ❌ | - |
| RELIANCE.NS | success | v1 | 0.0% | -99.00 | ❌ | - |
| ICICIBANK.NS | success | v2 | 50.0% | -0.40 | ❌ | - |
| INFY.NS | success | v2 | 0.0% | -1.30 | ❌ | - |
| TCS.NS | success | v3 | 22.2% | 0.10 | ✅ | - |
