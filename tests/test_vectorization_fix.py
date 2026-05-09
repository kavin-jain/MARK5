"""Tests vectorized fractional differentiation performance."""
import numpy as np
import pandas as pd
import time
from core.models.features import _frac_diff_ffd_vectorized


def test_vectorization():
    n = 1000
    data = pd.Series(np.random.randn(n), index=pd.bdate_range('2026-01-01', periods=n))
    d = 0.4

    start = time.time()
    result = _frac_diff_ffd_vectorized(data, d)
    elapsed = time.time() - start

    assert not result.dropna().empty, "Result should not be empty"
    assert len(result) == n, "Result length should match input"
    assert elapsed < 5.0, f"Vectorized FFD took {elapsed:.2f}s (should be <5s)"
