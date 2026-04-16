
import numpy as np
import pandas as pd
import time
from core.models.features import AdvancedFeatureEngine

def slow_frac_diff_ffd(series, d, thres=1e-4):
    """Non-vectorized version for comparison."""
    w = [1.0]
    k = 1
    while True:
        w_k = -w[-1] / k * (d - k + 1)
        if abs(w_k) < thres:
            break
        w.append(w_k)
        k += 1
    
    w = w[::-1]
    s = series.dropna().values
    out = np.full(len(s), np.nan)
    
    for i in range(len(w), len(s) + 1):
        out[i-1] = np.dot(w, s[i-len(w):i])
        
    return pd.Series(out, index=series.dropna().index)

def test_frac_diff():
    fe = AdvancedFeatureEngine()
    
    # Generate dummy data
    n = 10000
    data = pd.Series(np.random.randn(n).cumsum())
    d = 0.4
    
    print(f"Testing frac_diff_ffd with {n} points, d={d}")
    
    # Time vectorized version
    start = time.time()
    vec_result = fe._frac_diff_ffd(data, d)
    vec_time = time.time() - start
    print(f"Vectorized time: {vec_time:.4f}s")
    
    # Time slow version
    start = time.time()
    slow_result = slow_frac_diff_ffd(data, d)
    slow_time = time.time() - start
    print(f"Slow time: {slow_time:.4f}s")
    
    # Verify correctness
    # Compare non-NaN values
    vec_vals = vec_result.dropna().values
    slow_vals = slow_result.dropna().values
    
    np.testing.assert_allclose(vec_vals, slow_vals, rtol=1e-5, atol=1e-8)
    print("✅ Correctness verified: Vectorized matches Slow version.")
    
    if vec_time < slow_time:
        print(f"✅ Performance verified: Vectorized is {slow_time/vec_time:.1f}x faster.")
    else:
        print("❌ Performance FAILED: Vectorized is NOT faster.")

if __name__ == "__main__":
    test_frac_diff()
