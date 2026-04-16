import numpy as np
import pandas as pd
import time
from core.models.features import AdvancedFeatureEngine

def test_vectorization():
    engine = AdvancedFeatureEngine()
    
    # Generate large synthetic data
    n = 1000
    data = pd.Series(np.random.randn(n), index=pd.date_range('2026-01-01', periods=n, freq='min'))
    d = 0.4
    
    print(f"Testing _frac_diff_ffd with n={n}, d={d}")
    
    # Time the execution
    start = time.time()
    result = engine._frac_diff_ffd(data, d)
    end = time.time()
    
    print(f"Execution time: {end - start:.4f}s")
    
    # Sanity check output
    print(f"First 5 non-NaN values:\n{result.dropna().head()}")
    print(f"Shape: {result.shape}")
    
    assert not result.dropna().empty, "Result should not be empty"
    assert len(result) == n, "Result length should match input"
    print("SUCCESS: Vectorization test passed.")

if __name__ == "__main__":
    test_vectorization()
